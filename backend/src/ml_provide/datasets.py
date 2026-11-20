from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException
from ml_eda import load_items

router = APIRouter(prefix="/api")

# 点を選ぶ扱いになる入力の項目の種類 (ML-Inference の capture_points と
# points. 選ぶ点の数が要る)
POINT_KINDS = ("capture", "point_cloud")

# データセットを読めないときの例外
READ_ERRORS = (
    OSError, ValueError, TypeError, KeyError, AttributeError, yaml.YAMLError
)


def entry_rows(info: dict[str, Any]) -> list[dict[str, Any]]:
    """
    ファイルの組のデータセット (版 2 と版 3) の項目を, 画面のための行に
    する関数

    Parameters
    ----------
    info: dict[str, Any]
        dataset.yaml の中身 (load_items で確かめたもの)

    Returns
    ----------
    rows: list[dict[str, Any]]
        name, kind, role, shape, points (点を選ぶ数が要るか), steps (書き
        出しでかけた中身の手順. 版 2 と手順の無い項目は空) の並び
    """
    # 書き出した順のまま
    rows = []
    for name, entry in info["entries"].items():
        kind = entry.get("kind")
        rows.append({
            "name": str(name), "kind": kind, "role": entry.get("role"),
            "shape": entry.get("shape"), "points": kind in POINT_KINDS,
            "steps": list(entry.get("steps") or []),
        })

    return rows


def mask_classes(info: dict[str, Any]) -> list[dict[str, Any]]:
    """
    正解の領域の項目の, 値と名前の並びを返す関数

    Parameters
    ----------
    info: dict[str, Any]
        dataset.yaml の中身

    Returns
    ----------
    classes: list[dict[str, Any]]
        value と name (名前の無い値は None) の並び. 領域でなければ空
    """
    # 正解の最初の項目の classes (無ければ values だけ)
    classes: list[dict[str, Any]] = []
    target = info.get("target") or {}
    names = target.get("entries") or []
    if target.get("kind") != "mask" or not names:
        return classes
    entry = info["entries"].get(names[0]) or {}
    if entry.get("classes"):
        classes = [
            {"value": item.get("value"), "name": item.get("name")}
            for item in entry["classes"]
        ]
    else:
        classes = [{"value": value, "name": None}
                   for value in entry.get("values") or []]

    return classes


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
        path, format_version. 版 2 と版 3 なら name, role, target (kind,
        entries, origin), entries (entry_rows), splits (分割ごとの件数),
        classes (領域の値と名前), normalized (撮影の正規化があるか)

    Raises
    ----------
    HTTPException
        dataset.yaml の無い場所は 404, 読めないデータセットは 400
    """
    # 版だけを見て, 版 1 は make_bundle に任せる (列のデータセット)
    root = Path(path).expanduser().resolve()
    file = root / "dataset.yaml"
    if not file.is_file():
        raise HTTPException(404, f"{root} に dataset.yaml がありません")
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

    # 版 2 と版 3 (と知らない版) は ML-EDA の読み込みで確かめる
    try:
        dataset = load_items(root)
    except READ_ERRORS as error:
        message = str(error) or type(error).__name__
        raise HTTPException(400, message) from error
    info = dataset.info
    target = info.get("target") or {}
    description = {
        "path": str(root), "format_version": info.get("format_version"),
        "name": info.get("name"), "role": info.get("role"),
        "target": {
            "kind": target.get("kind"),
            "entries": list(target.get("entries") or []),
            "origin": target.get("origin"),
        },
        "entries": entry_rows(info),
        "splits": {
            name: (split or {}).get("items")
            for name, split in (info.get("splits") or {}).items()
        },
        "classes": mask_classes(info),
        "normalized": bool(info.get("normalization")),
    }

    return description
