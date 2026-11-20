import hashlib
from pathlib import Path
from typing import Any, Optional

import ml_inference
import ml_quest
import yaml
from fastapi import APIRouter, HTTPException, Request
from ml_quest.check import is_column_part
from ml_quest.handoff import decision_text
from ml_quest.problem import active_value, get_value
from ml_quest.questions import PART, ROLES, TARGET_KINDS
from pydantic import BaseModel, Field

from ml_provide.api import find_bundle, get_store
from ml_provide.cache import read_bundle
from ml_provide.units import unit_hint

router = APIRouter(prefix="/api")

# 問題の設定のファイルを読めないときの例外
READ_ERRORS = (OSError, ValueError, yaml.YAMLError)

# 問題の設定の形が崩れているときに ML-Quest から出る例外
SHAPE_ERRORS = (ValueError, TypeError, AttributeError, KeyError)

# 問題の設定のファイルの拡張子
SUFFIXES = (".yaml", ".yml")

# 一式の場所から問題の設定の候補を探す, 上のディレクトリの数
HINT_DEPTH = 4

# 受け持ちの短い名前 (ML-Quest の選択肢の名前の括弧の前) と, 正解の種類
# の名前 (表の列と合成の表を見分けるため, 括弧も残す)
ROLE_NAMES: dict[str, str] = {
    str(item["value"]): str(item["label"]).split(" (")[0] for item in ROLES
}
KIND_NAMES: dict[str, str] = {
    str(item["value"]): str(item["label"]) for item in TARGET_KINDS
}


class ProblemLink(BaseModel):
    path: Optional[str] = Field(
        default=None,
        description="problem.yaml か, それを持つディレクトリ (空で外す)"
    )


def problem_file(text: str) -> Path:
    """
    画面から受け取った問題の設定の場所を, problem.yaml の絶対パスにする
    関数

    Parameters
    ----------
    text: str
        problem.yaml か, それを持つディレクトリ (~ から書ける)

    Returns
    ----------
    path: Path
        problem.yaml の絶対パス

    Raises
    ----------
    HTTPException
        無い場合は 404, YAML のファイルでない場合は 422
    """
    # ディレクトリなら中の problem.yaml (読めない場所の形は 422)
    try:
        path = Path(text.strip()).expanduser().resolve()
        if path.is_dir():
            path = path / "problem.yaml"
        found = path.is_file()
    except (OSError, ValueError, RuntimeError) as error:
        raise HTTPException(
            422, f"問題の設定の場所として読めません: {error}"
        ) from error
    if not found:
        raise HTTPException(404, f"{path} に問題の設定がありません")
    if path.suffix.lower() not in SUFFIXES:
        raise HTTPException(
            422, f"{path.name} は YAML のファイル (.yaml) ではありません"
        )

    return path


def read_problem(path: Path) -> dict[str, Any]:
    """
    問題の設定を読み, 問題の設定のファイルであることを確かめる関数

    Parameters
    ----------
    path: Path
        problem.yaml

    Returns
    ----------
    problem: dict[str, Any]
        問題の設定

    Raises
    ----------
    HTTPException
        読めない場合と, 問題の設定のファイルでない場合 (一式の設定など)
        は 422
    """
    # 版の番号の他に, 名前と作った時刻 (ML-Quest が必ず書く) を持つこと
    try:
        problem: dict[str, Any] = ml_quest.load_problem(path)
    except READ_ERRORS as error:
        raise HTTPException(
            422, f"問題の設定を読めません: {error}"
        ) from error
    if not isinstance(problem.get("name"), str) or not problem.get(
        "created_at"
    ):
        raise HTTPException(
            422, f"{path.name} は問題の設定のファイルではありません"
        )

    return problem


def version_of(path: Path) -> str:
    """
    問題の設定のファイルの中身の印 (sha256) を返す関数

    Parameters
    ----------
    path: Path
        problem.yaml

    Returns
    ----------
    version: str
        中身の sha256 の 16 進 (画面が変わったことに気付くため)
    """
    # ファイルの中身の要約 (読んだ後に消えた場合も 422)
    try:
        version = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise HTTPException(
            422, f"問題の設定を読めません: {error}"
        ) from error

    return version


def is_usable(path: Path) -> bool:
    """
    推論に渡せる列の予測の問題の設定のファイルかを確かめる関数

    Parameters
    ----------
    path: Path
        problem.yaml

    Returns
    ----------
    usable: bool
        読めて, 受け渡しが通り, 受け持ちが列の予測なら True
    """
    # 読めないファイルと, 推論の結論を扱わない受け持ちは使わない
    try:
        if not path.is_file():
            return False
        problem = read_problem(path)
        ml_quest.handoff(problem)
        usable = bool(is_column_part(problem))
    except (HTTPException, OSError, *SHAPE_ERRORS):
        usable = False

    return usable


def origin_hint(copy: Path) -> tuple[str, str]:
    """
    写した問題の設定から, 候補に出す問題の設定と添える文を決める関数

    骨組みのプロジェクトと実験の problem.yaml は学習の時点の写しで, 元の
    問題の設定 (ML-Pathfinder の置き場) の場所を origin に持つ

    Parameters
    ----------
    copy: Path
        一式の上で見つけた problem.yaml

    Returns
    ----------
    hint: tuple[str, str]
        候補の場所と, 写しか元かを伝える文
    """
    # 元の場所があり, 推論に渡せるなら元. 無いか使えなければ写し
    try:
        origin = ml_quest.problem_origin(read_problem(copy))
    except HTTPException:
        origin = None
    if origin is None:
        hint = (
            str(copy),
            "学習の時点の写しで, 元の問題の設定の場所を持ちません. "
            "ML-Pathfinder で直した損得は届かないので, 元があれば元を"
            "つないでください"
        )
        return hint
    if Path(origin) != copy and is_usable(Path(origin)):
        hint = (
            origin,
            f"学習の時点の写し ({copy}) が指す元の問題の設定です. "
            "ML-Pathfinder で直した損得が推論に届きます"
        )
        return hint
    hint = (
        str(copy),
        f"写しが指す元の問題の設定 ({origin}) が無いか推論に渡せないので, "
        "学習の時点の写しを候補にしました. ML-Pathfinder で直した損得は"
        "届きません"
    )

    return hint


def find_hint(
    bundle_path: Path, home: Path
) -> tuple[Optional[str], Optional[str]]:
    """
    一式の場所から上にたどり, 問題の設定のファイルの候補を探す関数

    骨組みのプロジェクトは problem.yaml を持ち, 学習の実験の中に一式を
    作るため. 見つけたファイルが元の場所を持つ写しなら, 元を候補にする

    Parameters
    ----------
    bundle_path: Path
        一式のディレクトリ
    home: Path
        このアプリの置き場 (置き場の中の一式は探さない)

    Returns
    ----------
    found: tuple[Optional[str], Optional[str]]
        最初に見つけた, 推論に渡せる列の予測の問題の設定のファイルの場所
        (元の場所を持つ写しなら元) と, 写しか元かを伝える文 (無ければ
        どちらも None)
    """
    # 置き場の中とホームより上は探さない. 推論に渡せるものだけ
    found: tuple[Optional[str], Optional[str]] = (None, None)
    top = Path.home().resolve()
    place = bundle_path.resolve()
    if place.is_relative_to(home.resolve()):
        return found
    for _ in range(HINT_DEPTH + 1):
        if place == top or place == place.parent:
            break
        candidate = place / "problem.yaml"
        place = place.parent
        if not is_usable(candidate):
            continue
        found = origin_hint(candidate)
        return found

    return found


def training_columns(bundle: ml_inference.Bundle) -> Optional[list[str]]:
    """
    一式の学習に使ったデータのファイルの列の名前を読む関数 (推論する
    データも同じ列を持つ)

    Parameters
    ----------
    bundle: ml_inference.Bundle
        推論の一式

    Returns
    ----------
    columns: Optional[list[str]]
        列の名前 (データのファイルが無いか読めなければ None)
    """
    # データセットに残る元のデータの場所の 1 行目
    source = bundle.dataset.info.get("source") or {}
    columns: Optional[list[str]] = None
    try:
        columns = ml_quest.read_columns(str(source.get("path") or ""))
    except (OSError, *SHAPE_ERRORS):
        columns = None

    return columns


def issues_of(
    bundle: ml_inference.Bundle, target: Optional[str],
    decision: Optional[dict[str, Any]], conclusion: dict[str, Any]
) -> list[str]:
    """
    問題の設定から読んだ値と一式の食い違いを文で返す関数

    Parameters
    ----------
    bundle: ml_inference.Bundle
        推論の一式
    target: Optional[str]
        問題の設定の目的の列 (受け渡しの data.target)
    decision: Optional[dict[str, Any]]
        問題の設定の意思決定 (ML-Inference の形)
    conclusion: dict[str, Any]
        一式の既定値に問題の設定の値を重ねた結論の決め方

    Returns
    ----------
    issues: list[str]
        食い違いの文 (無ければ空)
    """
    # 目的の列, 意思決定の型と一式のタスクと列, 結論の決め方
    issues = []
    bundle_target = bundle.dataset.info.get("target")
    if target and bundle_target and target != bundle_target:
        issues.append(
            f"問題の設定の目的の列 {target} と, 一式の目的の列 "
            f"{bundle_target} が違います (別の問題の設定かもしれません)"
        )
    columns = training_columns(bundle)
    if decision:
        issues += [
            f"意思決定: {message}" for message in
            ml_inference.check_decision(
                decision, bundle, columns, with_current=False
            )
        ]

    # 今のやり方の決まりの列は学習のデータで確かめる (たどり直しで初めて
    # 誤りになる前に)
    rule = (decision or {}).get("current")
    if isinstance(rule, dict) and columns:
        issues += [
            f"今のやり方: {message}" for message in
            ml_inference.check_baseline(rule, columns)
            if "データにありません" in message
        ]
    issues += [
        f"結論の決め方: {message}" for message in
        ml_inference.check_rules(conclusion, bundle.task, bundle.classes)
    ]

    # まとめる列は推論するデータの列 (学習のデータで確かめる)
    group = conclusion.get("group_by")
    if columns and isinstance(group, str) and group not in columns:
        issues.append(
            f"結論の決め方: まとめる列 {group} が一式の学習のデータに"
            "ありません (このままでは推論が誤りになります)"
        )

    return issues


def bundle_time(bundle: ml_inference.Bundle) -> Optional[str]:
    """
    一式の過去の値の手順の時刻の列を返す関数

    Parameters
    ----------
    bundle: ml_inference.Bundle
        推論の一式

    Returns
    ----------
    time: Optional[str]
        時刻でずらす手順の時刻の列 (1 つだけの場合. 無ければ None)
    """
    # レシピの lag の手順の time
    times = {
        str(step["time"]) for step in bundle.dataset.recipe.get("steps", [])
        if step.get("op") == "lag" and step.get("time")
    }
    time = next(iter(times)) if len(times) == 1 else None

    return time


def with_baseline(
    problem: dict[str, Any], decision: Optional[dict[str, Any]],
    fallback_time: Optional[str] = None
) -> tuple[Optional[dict[str, Any]], list[str]]:
    """
    問題の設定の今のやり方 (action.current) を, たどり直しで比べる今のやり方
    の決まりとして意思決定の current に入れる関数

    量を決める型は, 列, k 期前, 平均を ML-Inference の決まりにする (決まり
    の値は置いた後の量なので, 今ある量があっても意味が同じ). 選ぶと振り分け
    は受け渡しの列の名前のまま

    Parameters
    ----------
    problem: dict[str, Any]
        問題の設定
    decision: Optional[dict[str, Any]]
        受け渡しの意思決定 (ML-Inference の形)
    fallback_time: Optional[str] = None
        問題の設定に時刻の列が無い場合の時刻の列 (一式の過去の値の手順の
        時刻の列. 系列を分ける列も一式の手順から ML-Inference が補う)

    Returns
    ----------
    result: tuple[Optional[dict[str, Any]], list[str]]
        今のやり方を入れた意思決定と, 入れられなかった理由の文
    """
    # 量を決める型で, 今のやり方を書いた場合だけ
    current = active_value(problem, "action.current")
    notes: list[str] = []
    if not decision or decision.get("type") != "quantity" or not (
        isinstance(current, dict)
    ):
        result = (decision, notes)
        return result
    if current.get("type") == "column":
        rule: dict[str, Any] = {"type": "column",
                                "column": current.get("column")}
        result = ({**decision, "current": rule}, notes)
        return result

    # k 期前と平均: 時刻の列 (予測の時刻の列, 分割の時刻の列, 一式の
    # 過去の値の手順の順), 系列を分ける列, 1 期の間隔 (データを入れた
    # ときに求めた値. 無ければ推論するデータの時刻の差から)
    time = active_value(problem, f"{PART}.time_column") or active_value(
        problem, f"{PART}.split_time_column"
    )
    rule = {"type": current.get("type"), "periods": current.get("periods")}
    if time:
        rule["time"] = time
        rule["by"] = list(active_value(problem, f"{PART}.series_by") or [])
    elif fallback_time:
        rule["time"] = fallback_time
    else:
        notes.append(
            "今のやり方 (k 期前と平均) は時刻の列で数えるので, 問題の設定にも"
            "一式の過去の値の手順にも時刻の列が無いと, たどり直しで比べません"
        )
        result = (decision, notes)
        return result
    seconds = get_value(problem, "data.interval_seconds")
    if isinstance(seconds, (int, float)) and seconds >= 1:
        rule["freq"] = f"{int(seconds)}s"
    result = ({**decision, "current": rule}, notes)

    return result


def part_name(part: dict[str, Any]) -> str:
    """
    受け渡しの受け持ちを, 画面に出す短い名前にする関数

    Parameters
    ----------
    part: dict[str, Any]
        受け渡しの part (role, target_kind など)

    Returns
    ----------
    name: str
        受け持ちの名前 (正解の種類があれば添える)
    """
    # 予測すると作るは正解の種類も (点群, 画像の領域など)
    role = part.get("role")
    name = ROLE_NAMES.get(str(role), str(role))
    kind = part.get("target_kind")
    if kind:
        name += f", 正解の種類は{KIND_NAMES.get(str(kind), str(kind))}"

    return name


def linked_view(
    entry: dict[str, Any], path: Path, problem: dict[str, Any]
) -> dict[str, Any]:
    """
    問題の設定から推論の値を読み, 画面に出す形にする関数

    列の予測でない受け持ちは推論の値を持たないため, 一式の既定値の
    結論の決め方と, まだ扱わないことの文を返す

    Parameters
    ----------
    entry: dict[str, Any]
        一式の登録
    path: Path
        problem.yaml
    problem: dict[str, Any]
        問題の設定 (版 2. 版 1 は読んだときに移したもの)

    Returns
    ----------
    view: dict[str, Any]
        name, path, version, created_at, role (受け持ちの名前), handled
        (推論の結論を扱う受け持ちか), decision, conclusion, horizon,
        basis (損得の根拠), text (意思決定の文), issues (一式との食い違い)

    Raises
    ----------
    HTTPException
        後ろの段階に渡せない問題の設定と, 読めない一式は 422
    """
    # 渡せる問題の設定かの確認 (機械学習を使わない, 直すところがある場合
    # は断り)
    try:
        stages = ml_quest.handoff(problem)
        handled = is_column_part(problem)
    except SHAPE_ERRORS as error:
        raise HTTPException(
            422, f"問題の設定を推論に渡せません: {error}"
        ) from error
    try:
        bundle = read_bundle(str(entry["path"]))
    except Exception as error:
        message = unit_hint(error) or f"一式を読めません: {error}"
        raise HTTPException(422, message) from error

    # 結論の決め方は一式の既定値に重ね, 意思決定は問題の設定のまま
    inference = stages["inference"]
    role = part_name(stages["part"])
    decision, notes = with_baseline(
        problem, inference["decision"] if handled else None,
        bundle_time(bundle)
    )
    conclusion = {
        **(bundle.info.get("conclusion") or {}), **inference["conclusion"]
    }
    if not handled:
        text = (
            f"この問題の設定の受け持ち ({role}) の推論の結論はまだ扱いません. "
            "意思決定と結論の決め方は一式の既定値のままです"
        )
    elif decision:
        text = decision_text(decision)
    else:
        text = "意思決定しない (予測を見て人が決める)"
    view = {
        "name": problem["name"], "path": str(path),
        "version": version_of(path),
        "created_at": str(problem["created_at"]),
        "role": role, "handled": handled,
        "decision": decision, "conclusion": conclusion,
        "horizon": inference["horizon"],
        "basis": active_value(problem, "action.basis") if handled else None,
        "text": text,
        "issues": notes + issues_of(
            bundle, stages["data"]["target"], decision, conclusion
        ) if handled else [],
    }

    return view


def problem_state(entry: dict[str, Any], home: Path) -> dict[str, Any]:
    """
    一式につないだ問題の設定の今の状態を返す関数

    Parameters
    ----------
    entry: dict[str, Any]
        一式の登録
    home: Path
        このアプリの置き場 (候補を探さない場所)

    Returns
    ----------
    state: dict[str, Any]
        path (つないだ場所. 無ければ None), linked (読んだ値. 読めなければ
        None), error (読めない理由), hint (つないでいない場合の候補),
        hint_note (候補が写しか元かを伝える文)
    """
    # つないでいなければ候補だけ
    text = entry.get("problem")
    state: dict[str, Any]
    if not isinstance(text, str):
        hint, note = find_hint(Path(str(entry["path"])), home)
        state = {"path": None, "linked": None, "error": None,
                 "hint": hint, "hint_note": note}
        return state

    # 置き換わったか読めなくなった場合は理由 (つなぎは外さない)
    state = {"path": text, "linked": None, "error": None, "hint": None,
             "hint_note": None}
    try:
        path = problem_file(text)
        problem = read_problem(path)
        created = entry.get("problem_created")
        if created and str(problem.get("created_at")) != created:
            raise HTTPException(
                422, f"{path} がつないだときと別の問題の設定に置き換わって"
                "います"
            )
        state["linked"] = linked_view(entry, path, problem)
    except HTTPException as error:
        state["error"] = str(error.detail)

    return state


@router.get("/bundles/{bundle_id}/problem")
def get_problem(request: Request, bundle_id: str) -> dict[str, Any]:
    """
    一式につないだ問題の設定から読んだ推論の値を返す関数

    Parameters
    ----------
    request: Request
        リクエスト
    bundle_id: str
        一式の id

    Returns
    ----------
    state: dict[str, Any]
        path, linked, error, hint, hint_note
    """
    # 推論のたびの読み直し (覚えない)
    state = problem_state(
        find_bundle(request, bundle_id), get_store(request).home
    )

    return state


@router.put("/bundles/{bundle_id}/problem")
def link_problem(
    request: Request, bundle_id: str, body: ProblemLink
) -> dict[str, Any]:
    """
    一式に問題の設定をつなぐか, つなぎを外す関数

    Parameters
    ----------
    request: Request
        リクエスト
    bundle_id: str
        一式の id
    body: ProblemLink
        問題の設定の場所 (空で外す)

    Returns
    ----------
    state: dict[str, Any]
        つないだ後の状態 (path, linked, error, hint, hint_note)

    Raises
    ----------
    HTTPException
        無い場所は 404, 推論に渡せない問題の設定は 422
    """
    # つなぐ前に, 読めて推論に渡せることの確認
    entry = find_bundle(request, bundle_id)
    store = get_store(request)
    if not (body.path or "").strip():
        try:
            entry = store.link_problem(bundle_id, None)
        except KeyError as error:
            raise HTTPException(404, "一式が見つかりません") from error
        state = problem_state(entry, store.home)
        return state
    path = problem_file(str(body.path))
    problem = read_problem(path)
    linked_view(entry, path, problem)
    try:
        entry = store.link_problem(
            bundle_id, str(path), str(problem["created_at"])
        )
    except KeyError as error:
        raise HTTPException(404, "一式が見つかりません") from error
    state = problem_state(entry, store.home)

    return state
