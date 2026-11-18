import hashlib
from pathlib import Path
from typing import Any, Optional

import ml_inference
import ml_quest
import yaml
from fastapi import APIRouter, HTTPException, Request
from ml_quest.handoff import decision_text
from ml_quest.problem import active_value
from pydantic import BaseModel, Field

from ml_provide.api import find_bundle, get_store, open_bundle

router = APIRouter(prefix="/api")

# 問題の設定のファイルを読めないときの例外
READ_ERRORS = (OSError, ValueError, yaml.YAMLError)

# 問題の設定の形が崩れているときに ML-Quest から出る例外
SHAPE_ERRORS = (ValueError, TypeError, AttributeError, KeyError)

# 問題の設定のファイルの拡張子
SUFFIXES = (".yaml", ".yml")

# 一式の場所から問題の設定の候補を探す, 上のディレクトリの数
HINT_DEPTH = 4


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
    # ディレクトリなら中の problem.yaml
    path = Path(text.strip()).expanduser().resolve()
    if path.is_dir():
        path = path / "problem.yaml"
    if not path.is_file():
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
    # ファイルの中身の要約
    version = hashlib.sha256(path.read_bytes()).hexdigest()

    return version


def find_hint(bundle_path: Path) -> Optional[str]:
    """
    一式の場所から上にたどり, 問題の設定のファイルの候補を探す関数

    骨組みのプロジェクトは problem.yaml を持ち, 学習の実験の中に一式を
    作るため

    Parameters
    ----------
    bundle_path: Path
        一式のディレクトリ

    Returns
    ----------
    hint: Optional[str]
        最初に見つけた問題の設定のファイルの場所 (無ければ None)
    """
    # 読めて, 問題の設定のファイルであるものだけ
    hint: Optional[str] = None
    places = [bundle_path, *bundle_path.parents][:HINT_DEPTH + 1]
    for place in places:
        candidate = place / "problem.yaml"
        if not candidate.is_file():
            continue
        try:
            read_problem(candidate)
        except HTTPException:
            continue
        hint = str(candidate)
        return hint

    return hint


def issues_of(
    bundle: ml_inference.Bundle, problem: dict[str, Any],
    decision: Optional[dict[str, Any]], conclusion: dict[str, Any]
) -> list[str]:
    """
    問題の設定から読んだ値と一式の食い違いを文で返す関数

    Parameters
    ----------
    bundle: ml_inference.Bundle
        推論の一式
    problem: dict[str, Any]
        問題の設定
    decision: Optional[dict[str, Any]]
        問題の設定の意思決定 (ML-Inference の形)
    conclusion: dict[str, Any]
        一式の既定値に問題の設定の値を重ねた結論の決め方

    Returns
    ----------
    issues: list[str]
        食い違いの文 (無ければ空)
    """
    # 目的の列, 意思決定の型と一式のタスク, 結論の決め方
    issues = []
    target = active_value(problem, "ml.target")
    bundle_target = bundle.dataset.info.get("target")
    if target and bundle_target and target != bundle_target:
        issues.append(
            f"問題の設定の目的の列 {target} と, 一式の目的の列 "
            f"{bundle_target} が違います (別の問題の設定かもしれません)"
        )
    if decision:
        issues += [
            f"意思決定: {message}" for message in
            ml_inference.check_decision(decision, bundle)
        ]
    issues += [
        f"結論の決め方: {message}" for message in
        ml_inference.check_rules(conclusion, bundle.task, bundle.classes)
    ]

    return issues


def linked_view(
    entry: dict[str, Any], path: Path, problem: dict[str, Any]
) -> dict[str, Any]:
    """
    問題の設定から推論の値を読み, 画面に出す形にする関数

    Parameters
    ----------
    entry: dict[str, Any]
        一式の登録
    path: Path
        problem.yaml
    problem: dict[str, Any]
        問題の設定

    Returns
    ----------
    view: dict[str, Any]
        name, path, version, created_at, decision, conclusion, horizon,
        basis (損得の根拠), text (意思決定の文), issues (一式との食い違い)

    Raises
    ----------
    HTTPException
        後ろの段階に渡せない問題の設定と, 読めない一式は 422
    """
    # 渡せる問題の設定か (機械学習を使わない, 直すところがある場合は断る)
    try:
        inference = ml_quest.handoff(problem)["inference"]
    except SHAPE_ERRORS as error:
        raise HTTPException(
            422, f"問題の設定を推論に渡せません: {error}"
        ) from error
    try:
        bundle = open_bundle(str(entry["path"]))
    except Exception as error:
        raise HTTPException(422, f"一式を読めません: {error}") from error

    # 結論の決め方は一式の既定値に重ね, 意思決定は問題の設定のまま
    decision = inference["decision"]
    conclusion = {
        **(bundle.info.get("conclusion") or {}), **inference["conclusion"]
    }
    view = {
        "name": problem["name"], "path": str(path),
        "version": version_of(path),
        "created_at": str(problem["created_at"]),
        "decision": decision, "conclusion": conclusion,
        "horizon": inference["horizon"],
        "basis": active_value(problem, "action.basis"),
        "text": decision_text(decision) if decision
        else "意思決定しない (予測を見て人が決める)",
        "issues": issues_of(bundle, problem, decision, conclusion),
    }

    return view


def problem_state(entry: dict[str, Any]) -> dict[str, Any]:
    """
    一式につないだ問題の設定の今の状態を返す関数

    Parameters
    ----------
    entry: dict[str, Any]
        一式の登録

    Returns
    ----------
    state: dict[str, Any]
        path (つないだ場所. 無ければ None), linked (読んだ値. 読めなければ
        None), error (読めない理由), hint (つないでいない場合の候補)
    """
    # つないでいなければ候補だけ
    text = entry.get("problem")
    state: dict[str, Any]
    if not isinstance(text, str):
        state = {"path": None, "linked": None, "error": None,
                 "hint": find_hint(Path(str(entry["path"])))}
        return state

    # 置き換わったか読めなくなった場合は理由 (つなぎは外さない)
    state = {"path": text, "linked": None, "error": None, "hint": None}
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
        path, linked, error, hint
    """
    # 推論のたびに読み直すので, 覚えない
    state = problem_state(find_bundle(request, bundle_id))

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
        つないだ後の状態 (path, linked, error, hint)

    Raises
    ----------
    HTTPException
        無い場所は 404, 推論に渡せない問題の設定は 422
    """
    # つなぐ前に, 読めて推論に渡せることを確かめる
    entry = find_bundle(request, bundle_id)
    store = get_store(request)
    if not (body.path or "").strip():
        entry = store.link_problem(bundle_id, None)
        state = problem_state(entry)
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
    state = problem_state(entry)

    return state
