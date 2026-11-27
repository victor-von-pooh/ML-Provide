"""
写真の縮小画像を返す API (検索の結果と, 既製の埋め込みの群分けの結果)

画面から送る写真の場所は, 結果に記録した写真の場所の中だけを読む
"""

import io
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

from ml_provide.api import USER_ERRORS, find_result_meta, get_store
from ml_provide.cache import open_bundle

router = APIRouter()

# 縮小画像の長い辺の上限 (画素)
PHOTO_SIDE = 256

# 縮小画像を作る写真のファイルの大きさと画素の数の上限 (大きな画像と, 小さな
# ファイルで画素の多い画像で記憶を使い切らないため)
PHOTO_BYTES = 50_000_000
PHOTO_PIXELS = 25_000_000

# 縮小画像を作る写真の形式 (他の形式の読み込みの道具を呼ばないため)
PHOTO_FORMATS = ("PNG", "JPEG", "WEBP", "GIF", "BMP", "TIFF")


def inside_file(base: Path, given: str) -> Optional[Path]:
    """
    決まった場所の中のファイルだけを解決して返す関数

    Parameters
    ----------
    base: Path
        写真のディレクトリ
    given: str
        ディレクトリからの写真の場所

    Returns
    ----------
    path: Optional[Path]
        解決したファイル (外を指す, ファイルでない, 場所として読めない
        (NUL の文字を含むなど) 場合は None)
    """
    # リンクと .. を解決した後に, ディレクトリの中のファイルか
    try:
        root = base.expanduser().resolve()
        path = (root / given).resolve()
        found = path.is_relative_to(root) and path.is_file()
    except (OSError, ValueError):
        found = False
    if not found:
        return None

    return path


def source_file(meta: dict[str, Any], given: str) -> Optional[Path]:
    """
    結果の推論した写真の場所 (source_path) の中のファイルを返す関数

    Parameters
    ----------
    meta: dict[str, Any]
        result.yaml の中身
    given: str
        結果の表の path の値

    Returns
    ----------
    path: Optional[Path]
        解決したファイル (外を指す, ファイルでない場合は None)

    Raises
    ----------
    HTTPException
        写真の場所の記録が無い場合は 400
    """
    # ディレクトリならその中, ファイルならそのファイルだけ
    source = meta.get("source_path")
    if not source:
        raise HTTPException(400, "推論した写真の場所が記録にありません")
    base = Path(str(source))
    found = inside_file(base, given) if base.is_dir() else (
        base.resolve() if base.is_file() and given == base.name else None
    )

    return found


def photo_response(found: Optional[Path], given: str) -> Response:
    """
    解決した写真の縮小画像の応答を返す関数

    Parameters
    ----------
    found: Optional[Path]
        解決したファイル (場所の外なら None)
    given: str
        画面から送った写真の場所 (誤りの文に使う)

    Returns
    ----------
    response: Response
        縮小画像の PNG

    Raises
    ----------
    HTTPException
        場所の外を指す場合と, 画像として読めない場合は 400
    """
    # 縮小画像は同じ結果の同じ写真なら変わらないので手元に置かせる
    if found is None:
        raise HTTPException(
            400, f"{given} は写真の場所の中のファイルではありません"
        )
    response = Response(thumbnail(found), media_type="image/png",
                        headers={"Cache-Control": "private, max-age=3600"})

    return response


def thumbnail(path: Path) -> bytes:
    """
    写真を長い辺 PHOTO_SIDE 画素以下に縮めた PNG にする関数

    Parameters
    ----------
    path: Path
        写真のファイル

    Returns
    ----------
    data: bytes
        PNG の中身

    Raises
    ----------
    HTTPException
        大きすぎるファイルか画像と, 決めた形式の画像として読めない場合は
        400
    """
    # 画像の大きさを見てから中身を読み, 縮める (透明度は RGB にそろえる)
    from PIL import Image, UnidentifiedImageError

    if path.stat().st_size > PHOTO_BYTES:
        raise HTTPException(400, f"{path.name} が大きすぎて縮められません")
    try:
        with Image.open(path, formats=PHOTO_FORMATS) as image:
            width, height = image.size
            if width * height > PHOTO_PIXELS:
                raise HTTPException(
                    400, f"{path.name} の画素が多すぎて縮められません"
                )
            image.thumbnail((PHOTO_SIDE, PHOTO_SIDE))
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError,
            ValueError) as error:
        raise HTTPException(
            400, f"{path.name} を画像として読めません"
        ) from error
    data = buffer.getvalue()

    return data


@router.get("/results/{result_id}/search-photo")
def get_search_photo(
    request: Request, result_id: str,
    side: str = Query(..., pattern="^(query|item)$"),
    path: str = Query(..., min_length=1, max_length=4096)
) -> Response:
    """
    検索の結果の, 問いの写真か索引の写真の縮小画像を返す関数

    Parameters
    ----------
    request: Request
        リクエスト
    result_id: str
        結果の id
    side: str
        query (問いの写真) か item (索引の写真)
    path: str
        結果の表の path (問い) か item (索引) の値

    Returns
    ----------
    response: Response
        縮小画像の PNG

    Raises
    ----------
    HTTPException
        検索の結果でない場合, 一式か写真の場所が無い場合, 場所の外を指す
        場合, 画像として読めない場合は 400
    """
    # 検索の結果と, 問いの写真の場所 (ディレクトリかファイル)
    _, meta = find_result_meta(request, result_id)
    if not meta.get("search"):
        raise HTTPException(400, "検索の結果ではありません")
    if side == "query":
        found = source_file(meta, path)
    else:
        # 索引の写真は一式の search.root の中だけ
        entry = get_store(request).find(str(meta.get("bundle_id")))
        if entry is None:
            raise HTTPException(400, "結果の一式が登録されていません")
        try:
            bundle = open_bundle(str(entry["path"]))
        except (OSError, *USER_ERRORS) as error:
            raise HTTPException(400, f"一式を読めません: {error}") from error
        root = (bundle.info.get("search") or {}).get("root")
        if not root:
            raise HTTPException(400, "一式に索引の写真の場所がありません")
        found = inside_file(Path(str(root)), path)
    response = photo_response(found, path)

    return response


@router.get("/results/{result_id}/group-photo")
def get_group_photo(
    request: Request, result_id: str,
    path: str = Query(..., min_length=1, max_length=4096)
) -> Response:
    """
    既製の埋め込みの群分けの結果の, 推論した写真の縮小画像を返す関数

    Parameters
    ----------
    request: Request
        リクエスト
    result_id: str
        結果の id
    path: str
        結果の表の path の値

    Returns
    ----------
    response: Response
        縮小画像の PNG

    Raises
    ----------
    HTTPException
        既製の埋め込みの群分けの結果でない場合, 写真の場所が無い場合,
        場所の外を指す場合, 画像として読めない場合は 400
    """
    # 写真を群に分けた結果 (grouping.photos) の推論した写真の中だけ
    _, meta = find_result_meta(request, result_id)
    if meta.get("output_form") != "cluster" or not (
        (meta.get("grouping") or {}).get("photos")
    ):
        raise HTTPException(400, "写真を群に分けた結果ではありません")
    response = photo_response(source_file(meta, path), path)

    return response
