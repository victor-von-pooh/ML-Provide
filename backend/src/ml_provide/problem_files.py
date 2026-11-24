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


def linked_guard(entry: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    一式につないだ問題の設定から, たどり直しで数える守る指標を読む関数

    Parameters
    ----------
    entry: dict[str, Any]
        一式の登録

    Returns
    ----------
    guard: Optional[dict[str, Any]]
        ML-Quest の受け渡しの推論の守る指標 (name, unit, better, line,
        measure, counted). 問題の設定をつないでいないか, 守る指標を
        答えていなければ None

    Raises
    ----------
    HTTPException
        つないだ問題の設定が読めない場合, 別の問題の設定に置き換わった
        場合, 推論に渡せない場合は 400
    """
    # たどり直しのたびに読み直す (推論の既定と同じく, 問題の設定の今の値)
    text = entry.get("problem")
    if not isinstance(text, str):
        return None
    try:
        path = problem_file(text)
        problem = read_problem(path)
        created = entry.get("problem_created")
        if created and str(problem.get("created_at")) != created:
            raise HTTPException(
                422, f"{path} がつないだときと別の問題の設定に置き換わって"
                "います"
            )
        try:
            stages = ml_quest.handoff(problem)
        except SHAPE_ERRORS as error:
            raise HTTPException(
                422, f"問題の設定を推論に渡せません: {error}"
            ) from error
    except HTTPException as error:
        raise HTTPException(
            400, "つないだ問題の設定から守る指標を読めないので, たどり直せ"
            f"ません ({error.detail}). 問題の設定を直すか, 一式の画面で"
            "つなぎを外してください"
        ) from error
    found = (stages.get("inference") or {}).get("guard")
    guard = found if isinstance(found, dict) else None

    return guard
