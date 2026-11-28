"""
一式につないだ問題の設定 (ML-Quest の problem.yaml) のファイルの読み込み

問題の設定の画面 (problems) と, 結果のたどり直し (api) の両方が読むので,
どちらにも依らない置き場に分ける
"""

from pathlib import Path
from typing import Any, Optional

import ml_quest
import yaml
from fastapi import HTTPException

# 問題の設定のファイルを読めないときの例外
READ_ERRORS = (OSError, ValueError, yaml.YAMLError)

# 問題の設定の形が崩れているときに ML-Quest から出る例外
SHAPE_ERRORS = (ValueError, TypeError, AttributeError, KeyError)

# 問題の設定のファイルの拡張子
SUFFIXES = (".yaml", ".yml")

# 問題の設定の欄を出さず, 問題の設定を読まない一式の種類 (雑音だけ,
# 並べる, 群分け, 異常の度合い, 効果, 値段, 需要, 検索. 結論と意思決定に
# 問題の設定を使わず, 件数や上限は一式が問題の設定か実験の成果物から写す)
NO_PROBLEM_KINDS = frozenset({
    "noise", "rank", "group", "anomaly", "uplift", "price", "demand",
    "search",
})


def uses_problem(kind: Optional[str]) -> bool:
    """
    一式の種類から, 問題の設定の欄を出すかを返す関数

    Parameters
    ----------
    kind: Optional[str]
        一式の説明のデータの種類 (読めない一式は None)

    Returns
    ----------
    uses: bool
        問題の設定の欄を出すか (種類の分からない一式は出す)
    """
    uses = kind not in NO_PROBLEM_KINDS

    return uses


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
            422, f"問題の場所として読めません: {error}"
        ) from error
    if not found:
        raise HTTPException(404, f"{path} に問題がありません")
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
            422, f"問題を読めません: {error}"
        ) from error
    if not isinstance(problem.get("name"), str) or not problem.get(
        "created_at"
    ):
        raise HTTPException(
            422, f"{path.name} は問題のファイルではありません"
        )

    return problem


def linked_guard(
    entry: dict[str, Any]
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """
    一式につないだ問題の設定から, たどり直しで数える守る指標を読む関数

    Parameters
    ----------
    entry: dict[str, Any]
        一式の登録

    Returns
    ----------
    found: tuple[Optional[dict[str, Any]], Optional[str]]
        ML-Quest の受け渡しの推論の守る指標 (name, unit, better, line,
        measure, counted. つないでいないか, 守る指標を答えていないか,
        有無を見られなければ None) と, 有無を見られなかったときの知らせ
        の文 (無ければ None)

    Raises
    ----------
    HTTPException
        守る指標を答えた問題の設定が推論に渡せない場合は 400
    """
    # たどり直しのたびに読み直す (推論の既定と同じく, 問題の設定の今の値).
    # 守る指標の有無を見られなければ, 守る指標の無い場合と同じに通す
    # (守る指標の無いたどり直しを, 問題の設定の崩れで止めないため)
    text = entry.get("problem")
    if not isinstance(text, str):
        return None, None
    try:
        path = problem_file(text)
        problem = read_problem(path)
    except HTTPException as error:
        return None, (
            "つないだ問題を読めないので, 守る指標は数えていません "
            f"({error.detail})"
        )
    created = entry.get("problem_created")
    if created and str(problem.get("created_at")) != created:
        return None, (
            "つないだ問題のファイルが別の問題で上書きされているので, "
            "守る指標は数えていません"
        )
    try:
        name = ml_quest.active_value(problem, "issue.guard.name")
    except SHAPE_ERRORS:
        return None, (
            "つないだ問題の形が崩れているので, 守る指標は数えて"
            "いません"
        )
    if not isinstance(name, str) or not name.strip():
        return None, None

    # 守る指標があれば, 推論に渡せない問題の設定は理由を添えて断る
    try:
        stages = ml_quest.handoff(problem)
    except SHAPE_ERRORS as error:
        raise HTTPException(
            400, f"つないだ問題の守る指標「{name.strip()}」を数える"
            f"には, 問題を推論に渡せる形に直してください ({error}). "
            "守る指標を数えずにたどり直すなら, 一式の画面でつなぎを外して"
            "ください"
        ) from error
    found = (stages.get("inference") or {}).get("guard")
    guard = found if isinstance(found, dict) else None

    return guard, None
