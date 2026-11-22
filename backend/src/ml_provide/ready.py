"""
学習しない既製のモデルの一式 (ML-Inference の一式の版 6) の下見と作り方

出どころ (ライブラリ), モデルの名前, 重みの名前から, 種類の名前の表,
前処理, 重みのファイルの名前とライセンスの情報を読み, 一式を作る. 重みは
一式に写さず, 使うときに出どころから読む (ML-Inference の決まり)
"""

import hashlib
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse
from urllib.request import urlopen

import ml_inference
import torch
from ml_modelcontrol import pretrained_info

# 名前の長さの上限 (モデルと重みの名前)
MAX_READY_NAME = 120

# 重みの hash の形 (sha256 の 16 進の 64 文字)
SHA256 = re.compile(r"^[0-9a-f]{64}$")

# 既製のモデルの一式を作る処理を 1 つずつにする鍵 (初回は重みを取りに
# 行くので, 同じファイルを同時に取りに行かないように)
READY_LOCK = threading.Lock()

# 重みの取得の時間の上限 (秒. 全体と, 1 回の読み込みで待つ長さ). 超えたら
# 止めて鍵を手放す (ぶら下がった取得で他の作成を待たせないため)
FETCH_SECONDS = 600.0
READ_SECONDS = 30.0

# このアプリが時間の上限を付けて重みを取る出どころ (重みを URL から
# PyTorch の置き場の checkpoints にファイルの名前で置き, そこから読むもの.
# 他の出どころはライブラリの読み方に任せる)
FETCHED_LIBRARIES = ("torchvision",)

# 重みを読む 1 回の大きさ (バイト)
CHUNK = 1 << 20

# torch.hub と同じ, ファイルの名前に入った hash の頭の形
HASH_PREFIX = re.compile(r"-([a-f0-9]*)\.")


class FetchError(Exception):
    """
    出どころから重みを取れない誤り (ネットワークの誤りと時間の上限)
    """

# 画面に返すライセンスの項目 (ML-Inference が ready.license に写すもの)
LICENSE_KEYS = (
    "license", "status", "redistributable", "gated", "alert", "message",
    "source", "url"
)


def ready_names(library: str, model: str, weights: str) -> tuple[str, ...]:
    """
    出どころ, モデル, 重みの名前を確かめて前後の空白を除く関数

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
    names: tuple[str, ...]
        前後の空白を除いた出どころ, モデル, 重みの名前

    Raises
    ----------
    ValueError
        出どころが無い場合と, 名前が空か長すぎる場合
    """
    # 出どころは登録簿の名前だけ
    names = (library.strip(), model.strip(), weights.strip())
    if names[0] not in ml_inference.READY_LIBRARIES:
        raise ValueError(
            f"既製のモデルの出どころ {names[0] or '(空)'} はありません "
            f"(出どころ: {', '.join(ml_inference.READY_LIBRARIES)})"
        )
    for label, name in (("モデル", names[1]), ("重み", names[2])):
        if not name:
            raise ValueError(f"{label}の名前を書いてください")
        if len(name) > MAX_READY_NAME:
            raise ValueError(
                f"{label}の名前は {MAX_READY_NAME} 文字までです"
            )

    return names


def ready_license(found: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    出どころの重みのライセンスの情報を読む関数

    ML-ModelControl の pretrained_info で読む (torchvision は機械で読める
    情報が無いので unknown). 読めなければ None (ML-Inference が読めな
    かったと記録する)

    Parameters
    ----------
    found: dict[str, Any]
        READY_LIBRARIES の関数の結果 (unit, params)

    Returns
    ----------
    info: Optional[dict[str, Any]]
        pretrained_info の結果 (読めなければ None)
    """
    # 読めない出どころは警告として画面に出す (作るのは止めない)
    try:
        info: Optional[dict[str, Any]] = pretrained_info(
            str(found["unit"]), dict(found["params"])
        )
    except Exception:
        return None

    return info


def license_view(info: Optional[dict[str, Any]]) -> dict[str, Any]:
    """
    ライセンスの情報を画面に返す形にする関数

    Parameters
    ----------
    info: Optional[dict[str, Any]]
        pretrained_info の結果

    Returns
    ----------
    view: dict[str, Any]
        LICENSE_KEYS の項目 (読めなければ警告の文)
    """
    # 読めないものは分からないものとして警告
    if info is None:
        view: dict[str, Any] = {
            "license": None, "status": "unknown", "redistributable": False,
            "gated": False, "alert": True,
            "message": "ライセンスの情報を読めませんでした. 使う前に配布元で"
            "確かめてください",
            "source": "読めなかった (ライセンスの情報を読めない)",
            "url": None,
        }
        return view
    view = {key: info.get(key) for key in LICENSE_KEYS}

    return view


def preview_ready(library: str, model: str, weights: str) -> dict[str, Any]:
    """
    既製のモデルの一式を作る前に, 種類の名前の表, 前処理, 重みのファイル,
    ライセンスを読む関数 (重みは取りに行かない)

    Parameters
    ----------
    library: str
        出どころ
    model: str
        モデルの名前
    weights: str
        重みの名前 (DEFAULT は版の決まった名前に直る)

    Returns
    ----------
    preview: dict[str, Any]
        library, model, weights (直した名前), classes, input, preprocess,
        url, file, license
    """
    # 出どころの読み方は ML-Inference の READY_LIBRARIES
    library, model, weights = ready_names(library, model, weights)
    found = ml_inference.READY_LIBRARIES[library](model, weights)
    preview = {
        "library": library,
        "model": model,
        "weights": str(found["params"].get("weights") or weights),
        "classes": [str(name) for name in found["classes"]],
        "input": found.get("input"),
        "preprocess": dict(found["preprocess"]),
        "url": found.get("url"),
        "file": found.get("file"),
        "license": license_view(ready_license(found)),
    }

    return preview


def make_ready(
    place: Path, library: str, model: str, weights: str,
    name: str, sha256: Optional[str] = None
) -> dict[str, Any]:
    """
    既製のモデルの一式を作る関数

    Parameters
    ----------
    place: Path
        作るディレクトリ (まだ無い場所)
    library: str
        出どころ
    model: str
        モデルの名前
    weights: str
        重みの名前
    name: str
        一式の名前
    sha256: Optional[str] = None
        決めておいた重みの hash (渡すと, 読んだ重みと違えば作らない)

    Returns
    ----------
    info: dict[str, Any]
        書き出した inference.yaml の中身

    Raises
    ----------
    ValueError
        名前か hash の形が違う場合と, ML-Inference が作れない場合
    FetchError
        重みを出どころから取れない場合と, 取得が時間の上限を超えた場合
    """
    # hash は小文字の 16 進の 64 文字にそろえる
    library, model, weights = ready_names(library, model, weights)
    pinned = (sha256 or "").strip().lower() or None
    if pinned is not None and not SHA256.match(pinned):
        raise ValueError("重みの hash は sha256 の 16 進の 64 文字で書いてください")

    # 下見と同じライセンスの情報を渡し, 重みを読んで hash を固定 (読めない
    # ライセンスも警告の付いた情報にして, 一式の画面で警告する)
    found = ml_inference.READY_LIBRARIES[library](model, weights)
    license = ready_license(found) or {
        "unit": found["unit"], **license_view(None)
    }
    with READY_LOCK:
        if library in FETCHED_LIBRARIES:
            fetch_weights(found)
        info: dict[str, Any] = ml_inference.make_ready_bundle(
            place, model, weights, library=library, sha256=pinned,
            name=name, license=license
        )

    return info


def fetch_weights(found: dict[str, Any]) -> None:
    """
    出どころの重みのファイルが PyTorch の置き場に無ければ, 時間の上限を
    付けて取りに行く関数

    置き場にあれば何もしない (ML-Inference はここから読む). 取ったファイル
    は, 名前に hash の頭があれば torch.hub と同じく比べ, 確かめてから置く

    Parameters
    ----------
    found: dict[str, Any]
        READY_LIBRARIES の関数の結果 (url, file)

    Returns
    ----------
    None

    Raises
    ----------
    FetchError
        取れない場合, 時間の上限を超えた場合, hash の頭が違う場合
    """
    # URL もファイルの名前も無い出どころは, ライブラリの読み方に任せる
    url = str(found.get("url") or "")
    name = Path(str(found.get("file") or "")).name
    if not url or not name:
        return
    target = Path(torch.hub.get_dir()) / "checkpoints" / name
    if target.is_file():
        return
    if urlparse(url).scheme != "https":
        raise FetchError(f"重みの URL {url} は https ではないので取りに行きません")

    # 隣の一時ファイルに書き, 確かめてから入れ替える (途中のファイルを
    # 重みとして読ませない)
    target.parent.mkdir(parents=True, exist_ok=True)
    writing = target.with_name(f".{name}.fetching")
    digest = hashlib.sha256()
    deadline = time.monotonic() + FETCH_SECONDS
    try:
        with urlopen(url, timeout=READ_SECONDS) as response, writing.open(
            "wb"
        ) as file:
            while chunk := response.read(CHUNK):
                if time.monotonic() > deadline:
                    raise TimeoutError
                digest.update(chunk)
                file.write(chunk)
    except TimeoutError as error:
        writing.unlink(missing_ok=True)
        raise FetchError(
            f"重み {name} の取得が時間の上限 (全体で {FETCH_SECONDS:g} 秒, "
            f"途切れて {READ_SECONDS:g} 秒) を超えました. ネットワークを確かめるか, {name} を PyTorch の置き場 "
            "(torch.hub.get_dir() の checkpoints) に置いてください"
        ) from error
    except OSError as error:
        writing.unlink(missing_ok=True)
        raise FetchError(
            f"重み {name} を {url} から取れません ({type(error).__name__}: "
            f"{error}). ネットワークを確かめるか, {name} を PyTorch の置き場 "
            "(torch.hub.get_dir() の checkpoints) に置いてください"
        ) from error
    match = HASH_PREFIX.search(name)
    if match and not digest.hexdigest().startswith(match.group(1)):
        writing.unlink(missing_ok=True)
        raise FetchError(
            f"取った重み {name} の hash が名前の hash の頭 {match.group(1)} "
            "と違います (取得の途中で壊れたか, 出どころが変わった)"
        )
    os.replace(writing, target)
