import hashlib
from pathlib import Path
from typing import Any, Optional

import ml_inference
import ml_quest
from fastapi import APIRouter, HTTPException, Request
from ml_quest import (
    PART,
    ROLES,
    TARGET_KINDS,
    active_value,
    decision_text,
    get_value,
    is_column_part,
)
from pydantic import BaseModel, Field

from ml_provide.api import find_bundle, get_store
from ml_provide.cache import describe, read_bundle
from ml_provide.problem_files import (
    SHAPE_ERRORS,
    problem_file,
    read_problem,
    uses_problem,
)
from ml_provide.units import unit_hint

router = APIRouter(prefix="/api")

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


def is_usable(path: Path, ready: bool = False) -> bool:
    """
    推論に渡せる問題の設定のファイルかを確かめる関数

    Parameters
    ----------
    path: Path
        problem.yaml
    ready: bool = False
        既製のモデルの一式に渡すか (作り方が既製のモデルをそのままの問題
        の設定を探す)

    Returns
    ----------
    usable: bool
        読めて, 受け渡しが通り, 受け持ちが列の予測 (既製のモデルの一式なら
        作り方が既製のモデルをそのまま) なら True
    """
    # 読めないファイルと, 推論の結論を扱わない受け持ちは使わない
    try:
        if not path.is_file():
            return False
        problem = read_problem(path)
        stages = ml_quest.handoff(problem)
        usable = (
            stages["inference"].get("ready") is not None if ready
            else bool(is_column_part(problem))
        )
    except (HTTPException, OSError, *SHAPE_ERRORS):
        usable = False

    return usable


def origin_hint(copy: Path, ready: bool = False) -> tuple[str, str]:
    """
    写した問題の設定から, 候補に出す問題の設定と添える文を決める関数

    骨組みのプロジェクトと実験の problem.yaml は学習の時点の写しで, 元の
    問題の設定 (ML-Pathfinder の置き場) の場所を origin に持つ

    Parameters
    ----------
    copy: Path
        一式の上で見つけた problem.yaml
    ready: bool = False
        既製のモデルの一式の候補か

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
            "この問題の設定は元の場所を持ちません (origin が無い). 写しの場合は"
            ", ML-Pathfinder で直した損得は届かないので, 元があれば元を"
            "つないでください"
        )
        return hint
    if Path(origin) != copy and is_usable(Path(origin), ready):
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
    bundle_path: Path, home: Path, ready: bool = False
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
    ready: bool = False
        既製のモデルの一式か (作り方が既製のモデルをそのままの問題の設定を
        探す)

    Returns
    ----------
    found: tuple[Optional[str], Optional[str]]
        最初に見つけた, 推論に渡せる問題の設定のファイルの場所
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
        if not is_usable(candidate, ready):
            continue
        found = origin_hint(candidate, ready)
        return found

    return found


def named_columns(value: Any) -> list[str]:
    """
    レシピの手順の引数や当てはめた値から, 列の名前を取り出す関数

    Parameters
    ----------
    value: Any
        列の名前 1 つか並び (auto などの文字もそのまま 1 つの名前)

    Returns
    ----------
    names: list[str]
        列の名前 (並びでない値と空は空)
    """
    # 文字 1 つか, 文字の並び
    if isinstance(value, str):
        names = [value] if value else []
    elif isinstance(value, list):
        names = [str(item) for item in value if isinstance(item, str)]
    else:
        names = []

    return names


def recipe_columns(bundle: ml_inference.Bundle) -> Optional[list[str]]:
    """
    一式のデータセットの説明とレシピから, 学習のデータの列の名前を集める
    関数

    学習に使う列, 目的の列, レシピの手順が読む列 (当てはめた列を含む),
    読み込みの設定の列を合わせる. 手順で作った列 (ワンホットの列など) も
    入るので, 実際の列より多いことがある

    Parameters
    ----------
    bundle: ml_inference.Bundle
        推論の一式

    Returns
    ----------
    columns: Optional[list[str]]
        列の名前 (表と時系列でない一式は None. 表の列が説明に残らないため)
    """
    # 表と時系列だけ (画像や文は学習に使う列を持たない)
    info = bundle.dataset.info
    if info.get("kind", "table") not in ("table", "timeseries"):
        return None
    names = [str(item.get("name")) for item in info.get("features") or []
             if isinstance(item, dict) and item.get("name") is not None]
    names += named_columns(info.get("target"))

    # 手順の引数と当てはめた値の列 (auto は列の名前でない)
    for step in bundle.dataset.recipe.get("steps") or []:
        if not isinstance(step, dict):
            continue
        state = step.get("state")
        if not isinstance(state, dict):
            state = {}
        for value in (step.get("columns"), step.get("by"), step.get("time"),
                      step.get("column"), state.get("columns")):
            names += [n for n in named_columns(value) if n != "auto"]
    split = bundle.dataset.recipe.get("split") or {}
    if isinstance(split, dict):
        names += named_columns(split.get("column"))

    # 読み込みの設定の列 (時刻の列など)
    source = info.get("source") or {}
    options = source.get("options") if isinstance(source, dict) else None
    for key, value in (options or {}).items():
        if str(key).endswith("_column"):
            names += named_columns(value)
    columns = list(dict.fromkeys(names))

    return columns


def training_columns(
    bundle: ml_inference.Bundle
) -> tuple[Optional[list[str]], bool]:
    """
    一式の学習に使ったデータの列の名前を返す関数 (推論するデータも同じ列
    を持つ)

    次の順で取る. 1. 元のデータの場所 (source.path. 手元の場所を残す版 1 の書き出し) が
    読めればそのファイルの列 2. 説明に残る入力の列 (source.columns. 今の
    ML-EDA は手元の場所の代わりに残す) 3. 説明とレシピから集めた列 (自動
    で見つけた時刻の列などが抜けるので, 列が全てとは限らない)

    Parameters
    ----------
    bundle: ml_inference.Bundle
        推論の一式

    Returns
    ----------
    found: tuple[Optional[list[str]], bool]
        列の名前 (どこからも分からなければ None) と, 列が全てか (1 と 2 は
        True, 3 は False)
    """
    # 元のデータの場所の 1 行目, 説明の入力の列, 説明とレシピの順
    source = bundle.dataset.info.get("source") or {}
    if not isinstance(source, dict):
        source = {}
    path = source.get("path")
    if path:
        try:
            found = (ml_quest.read_columns(str(path)), True)
            return found
        except (OSError, *SHAPE_ERRORS):
            pass
    listed = source.get("columns")
    if isinstance(listed, list) and listed:
        found = ([str(name) for name in listed], True)
        return found
    found = (recipe_columns(bundle), False)

    return found


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
    # 「データに無い」の知らせは列が全て分かる場合だけ (説明とレシピから
    # 集めた列では, 自動で見つけた列を誤って無いと知らせるため)
    known, complete = training_columns(bundle)
    columns = known if complete else None
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


def resolved_weights(library: str, model: str, weights: str) -> str:
    """
    重みの名前を, 版の決まった名前に直す関数 (DEFAULT を IMAGENET1K_V1 など)

    Parameters
    ----------
    library: str
        出どころ (READY_LIBRARIES の名前)
    model: str
        モデルの名前
    weights: str
        重みの名前

    Returns
    ----------
    name: str
        直した名前 (出どころが読めない名前なら, 渡した名前のまま)
    """
    # 出どころの読み方は ML-Inference の READY_LIBRARIES (重みは取りに行か
    # ない)
    reader = ml_inference.READY_LIBRARIES.get(library)
    if reader is None:
        return weights
    try:
        found = reader(model, weights)
    except Exception:
        return weights
    name = str((found.get("params") or {}).get("weights") or weights)

    return name


def ready_issues(
    bundle: ml_inference.Bundle, planned: dict[str, Any]
) -> list[str]:
    """
    問題の設定で決めておいた既製のモデルと, 一式の出どころの食い違いを
    文で返す関数

    Parameters
    ----------
    bundle: ml_inference.Bundle
        既製のモデルの一式
    planned: dict[str, Any]
        受け渡しの推論の ready (library, model, weights, sha256 (重みの
        ハッシュ値). 決めていない値は None)

    Returns
    ----------
    issues: list[str]
        食い違いの文 (無ければ空)
    """
    # 出どころ, モデル, 重みの名前 (重みは版の決まった名前で比べる)
    ready = bundle.info.get("ready") or {}
    labels = {"library": "出どころ", "model": "モデル", "weights": "重み"}
    issues = []
    for key, label in labels.items():
        want = planned.get(key)
        have = ready.get(key)
        if not isinstance(want, str) or not want:
            continue
        if key == "weights" and isinstance(have, str):
            want = resolved_weights(
                str(planned.get("library") or ready.get("library")),
                str(planned.get("model") or ready.get("model")), want
            )
        if want != have:
            issues.append(
                f"既製のモデル: 問題の設定の{label} {want} と, 一式の{label} "
                f"{have} が違います"
            )

    # 重みのハッシュ値 (一式が読んだ重みのハッシュ値と. どちらも小文字に
    # し, 一式に記録が無ければ比べられないと知らせる)
    want = planned.get("sha256")
    records = (bundle.info.get("model") or {}).get("pretrained") or []
    have = next((r.get("sha256") for r in records
                 if isinstance(r, dict) and isinstance(r.get("sha256"), str)
                 and r.get("sha256")), None)
    if not isinstance(want, str) or not want.strip():
        return issues
    if have is None:
        issues.append(
            "既製のモデル: 一式に重みのハッシュ値の記録が無いので, 問題の"
            "設定の重みのハッシュ値を確かめられません"
        )
    elif want.strip().lower() != have.strip().lower():
        issues.append(
            f"既製のモデル: 問題の設定の重みのハッシュ値 {want.strip()[:12]}… "
            f"と, 一式が読んだ重みのハッシュ値 {have.strip()[:12]}… が違います"
            " (別の重みかもしれません)"
        )

    return issues


def ready_view(
    bundle: ml_inference.Bundle, stages: dict[str, Any]
) -> dict[str, Any]:
    """
    既製のモデルの一式につないだ問題の設定から, 推論の値を読む関数

    作り方が既製のモデルをそのままの問題の設定だけ, 結論の決め方を既定に
    する. 意思決定は使えない (既製のモデルの一式は外れ方の記録を持たない)

    Parameters
    ----------
    bundle: ml_inference.Bundle
        既製のモデルの一式
    stages: dict[str, Any]
        受け渡しの値

    Returns
    ----------
    view: dict[str, Any]
        handled, decision (常に None), conclusion, basis, text, issues
    """
    # 一式の既定値に問題の設定の結論の決め方を重ねる
    inference = stages["inference"]
    planned = inference.get("ready")
    conclusion = {
        **(bundle.info.get("conclusion") or {}), **inference["conclusion"]
    }
    view: dict[str, Any]
    if planned is None:
        view = {
            "handled": False, "decision": None,
            "conclusion": dict(bundle.info.get("conclusion") or {}),
            "basis": None, "issues": [],
            "text": (
                "この問題の設定の作り方は既製のモデルをそのままではないので, "
                "既製のモデルの一式には使いません. 結論の決め方は一式の既定値"
                "のままです"
            ),
        }
        return view
    view = {
        "handled": True, "decision": None, "conclusion": conclusion,
        "basis": None,
        "text": (
            "既製のモデルの一式は意思決定を使えないので, 問題の設定の意思決定"
            "は使いません. 結論の決め方は問題の設定の値を既定にします"
        ),
        "issues": ready_issues(bundle, planned) + [
            f"結論の決め方: {message}" for message in
            ml_inference.check_rules(conclusion, bundle.task, bundle.classes)
        ],
    }

    return view


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
    結論の決め方と, まだ扱わないことの文を返す. 既製のモデルの一式は,
    作り方が既製のモデルをそのままの問題の設定の結論の決め方と, 重みの
    食い違いを返す (意思決定は使えない)

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

    # 既製のモデルの一式は結論の決め方だけ
    inference = stages["inference"]
    role = part_name(stages["part"])
    head = {
        "name": problem["name"], "path": str(path),
        "version": version_of(path),
        "created_at": str(problem["created_at"]), "role": role,
        "horizon": None,
    }
    if bundle.from_ready:
        view = {**head, **ready_view(bundle, stages)}
        return view

    # 結論の決め方は一式の既定値に重ね, 意思決定は問題の設定のまま
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
        **head, "handled": handled,
        "decision": decision, "conclusion": conclusion,
        "horizon": inference["horizon"],
        "basis": active_value(problem, "action.basis") if handled else None,
        "text": text,
        "issues": notes + issues_of(
            bundle, stages["data"]["target"], decision, conclusion
        ) if handled else [],
    }

    return view


def bundle_kind(path: str) -> Optional[str]:
    """
    一式の説明のデータの種類を返す関数

    Parameters
    ----------
    path: str
        一式のディレクトリ

    Returns
    ----------
    kind: Optional[str]
        データの種類 (NO_PROBLEM_KINDS は問題の設定を読まない一式, ready
        は既製のモデルの一式. 読めない一式は None)
    """
    # 一覧と同じ覚えた説明 (読めない一式は, 今までどおり読んで理由を出す)
    try:
        description, _ = describe(path)
    except Exception:
        return None
    found = description.get("kind")
    kind = found if isinstance(found, str) else None

    return kind


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
    # 問題の設定の欄を出さない一式は, 問題の設定の読み込みも候補の探索も
    # 無し (画面が一式と並べて要求しても, 結論と意思決定に使わない設定.
    # 並べる件数と反応の表の列, 返す件数と待ち時間の上限は, 一式が問題の
    # 設定から写した値. 群の数と列, 人の確認の件数の上限, 配る数の上限,
    # 値段の候補と上限, 発注の損と上限は実験の成果物から写した値)
    text = entry.get("problem")
    state: dict[str, Any]
    kind = bundle_kind(str(entry["path"]))
    if not uses_problem(kind):
        state = {"path": text if isinstance(text, str) else None,
                 "linked": None, "error": None, "hint": None,
                 "hint_note": None}
        return state

    # つないでいなければ候補だけ
    if not isinstance(text, str):
        hint, note = find_hint(
            Path(str(entry["path"])), home, kind == "ready"
        )
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
