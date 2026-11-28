from pathlib import Path
from typing import Any

import ml_inference
import yaml
from fastapi import APIRouter, HTTPException

router = APIRouter()

# データセットを読めないときの例外
READ_ERRORS = (
    OSError, ValueError, TypeError, KeyError, AttributeError, yaml.YAMLError
)


@router.get("/datasets/describe")
def describe_dataset(path: str) -> dict[str, Any]:
    """
    一式を作る画面のために, データセットの版とファイルの組の項目を返す
    関数

    Parameters
    ----------
    path: str
        データセットのディレクトリ (dataset.yaml を持つ. ~ から書ける)

    Returns
    ----------
    description: dict[str, Any]
        path, format_version. 版 2 と版 3 なら ML-Inference の
        describe_items の欄 (name, role, target, entries, splits, classes,
        normalized, items, problem)

    Raises
    ----------
    HTTPException
        dataset.yaml の無い場所は 404, 読めないデータセットは 400
    """
    # 版だけを見て, 版 1 は make_bundle に任せる (列のデータセット)
    root = Path(path).expanduser().resolve()
    file = root / "dataset.yaml"
    if not file.is_file():
        raise HTTPException(
            404, f"データセット {root} が見つかりません (dataset.yaml がありません)"
        )
    try:
        info = yaml.safe_load(file.read_text(encoding="utf-8"))
    except READ_ERRORS as error:
        message = f"dataset.yaml を読めません: {error}"
        raise HTTPException(400, message) from error
    version = info.get("format_version") if isinstance(info, dict) else None
    if version == 1:
        description: dict[str, Any] = {
            "path": str(root), "format_version": 1
        }
        return description

    # 版 2 と版 3 (と知らない版) は ML-Inference の説明 (ML-EDA の読み込み
    # で確かめる)
    try:
        found = ml_inference.describe_items(root)
    except READ_ERRORS as error:
        message = str(error) or type(error).__name__
        raise HTTPException(400, message) from error
    description = {"path": str(root), **found}

    return description
