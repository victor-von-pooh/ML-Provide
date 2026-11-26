import io
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

from ml_provide.api import USER_ERRORS, find_result_meta, get_store
from ml_provide.cache import open_bundle

router = APIRouter(prefix="/api")

# 縮小画像の長い辺の上限 (画素)
PHOTO_SIDE = 256

# 縮小画像を作る写真のファイルの大きさの上限 (バイト. 大きな画像で記憶を
# 使い切らないため)
PHOTO_BYTES = 50_000_000


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
        解決したファイル (外を指す, ファイルでない場合は None)
    """
    # リンクと .. を解決した後に, ディレクトリの中のファイルか
    root = base.expanduser().resolve()
    path = (root / given).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        return None

    return path


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
        大きすぎるファイルと, 画像として読めない場合は 400
    """
    # 画像の読み込みと縮小 (向きと透明度は RGB にそろえる)
    from PIL import Image, UnidentifiedImageError

    if path.stat().st_size > PHOTO_BYTES:
        raise HTTPException(400, f"{path.name} が大きすぎて縮められません")
    try:
        with Image.open(path) as image:
            image.thumbnail((PHOTO_SIDE, PHOTO_SIDE))
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(
            400, f"{path.name} を画像として読めません ({error})"
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
        source = meta.get("source_path")
        if not source:
            raise HTTPException(400, "問いの写真の場所が記録にありません")
        base = Path(str(source))
        found = inside_file(base, path) if base.is_dir() else (
            base.resolve() if base.is_file() and path == base.name else None
        )
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
    if found is None:
        raise HTTPException(
            400, f"{path} は写真の場所の中のファイルではありません"
        )
    response = Response(thumbnail(found), media_type="image/png",
                        headers={"Cache-Control": "private, max-age=3600"})

    return response
