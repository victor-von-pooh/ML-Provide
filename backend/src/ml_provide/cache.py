"""
読み込んだ推論の一式と, その説明を覚えて使い回す仕組み

組み立てたモデルを持つ一式 (推論に使う) と, 一覧のための説明を分けて
覚える. 一覧を開いても, 推論に使うモデルを追い出さない
"""

import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import ml_inference

# 組み立てたモデルを覚える一式の数を変える環境変数
MODEL_CACHE_ENV = "ML_PROVIDE_MODEL_CACHE"

# 組み立てたモデルを覚える一式の数の既定 (モデルごとにメモリを使う)
MODEL_CACHE_DEFAULT = 4

# 説明を覚える一式の数 (説明は小さい)
DESCRIPTION_CACHE_SIZE = 256

# 一式の印 (一式のディレクトリのファイルと重みの, 更新時刻と大きさ)
Stamp = tuple[Any, ...]

# 場所ごとの (印, 一式)
MODELS: OrderedDict[str, tuple[Stamp, ml_inference.Bundle]] = OrderedDict()

# 場所ごとの (印, 重みの場所, 説明)
DESCRIPTIONS: OrderedDict[
    str, tuple[Stamp, Optional[Path], dict[str, Any]]
] = (
    OrderedDict()
)

# 場所ごとの (印, 学習済みの重みの警告. モデルを手放した後も出すため)
WARNINGS: dict[str, tuple[Stamp, list[str]]] = {}

LOCK = threading.Lock()


def model_cache_size() -> int:
    """
    組み立てたモデルを覚える一式の数を返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    size: int
        数 (環境変数 ML_PROVIDE_MODEL_CACHE. 1 以上の整数でなければ既定)
    """
    # 読めない値は既定の数
    value = os.environ.get(MODEL_CACHE_ENV, "")
    size = MODEL_CACHE_DEFAULT
    if value.strip().isdigit() and int(value) >= 1:
        size = int(value)

    return size


def file_stamp(path: Path) -> tuple[float, ...]:
    """
    ファイルの更新時刻と大きさを返す関数

    Parameters
    ----------
    path: Path
        ファイル

    Returns
    ----------
    stamp: tuple[float, ...]
        更新時刻と大きさ (無ければ空)
    """
    # 消えたファイルは空 (読み込みで誤りになる)
    stamp: tuple[float, ...] = ()
    try:
        status = path.stat()
    except OSError:
        return stamp
    stamp = (status.st_mtime, float(status.st_size))

    return stamp


def weights_of(bundle: ml_inference.Bundle) -> Optional[Path]:
    """
    一式の重みのファイルを返す関数

    Parameters
    ----------
    bundle: ml_inference.Bundle
        読んだ一式

    Returns
    ----------
    weights: Optional[Path]
        重みのファイル (モデルを持たない一式は None. 距離に基づく異常の
        度合いの一式など)
    """
    # model が null の一式 (版 11) は重みを持たない
    weights = None if bundle.info.get("model") is None else (
        bundle.weights_path
    )

    return weights


def bundle_stamp(path: str, weights: Optional[Path]) -> Stamp:
    """
    一式のディレクトリの全てのファイルと重みの印を返す関数

    定義, データセットの説明, レシピ, inference.yaml のどれが変わっても
    読み直すため

    Parameters
    ----------
    path: str
        一式のディレクトリ
    weights: Optional[Path]
        重みのファイル (一式の外を指す場合があるため. 分からなければ None)

    Returns
    ----------
    stamp: Stamp
        ファイルの相対パス, 更新時刻, 大きさの並び
    """
    # 一式のディレクトリの中のファイル (順をそろえたもの)
    root = Path(path)
    parts: list[Any] = []
    for folder, _, names in sorted(os.walk(root)):
        for name in sorted(names):
            file = Path(folder) / name
            parts.append((str(file.relative_to(root)), file_stamp(file)))
    if weights is not None:
        parts.append(("weights", str(weights), file_stamp(weights)))
    stamp = tuple(parts)

    return stamp


def fresh_model(path: str) -> Optional[ml_inference.Bundle]:
    """
    覚えた推論の一式が古くなければ返す関数

    Parameters
    ----------
    path: str
        一式のディレクトリ

    Returns
    ----------
    bundle: Optional[ml_inference.Bundle]
        一式 (覚えていないか古ければ None)
    """
    # 印の比べ (重みは覚えた一式の場所)
    with LOCK:
        cached = MODELS.get(path)
    bundle: Optional[ml_inference.Bundle] = None
    if cached is None:
        return bundle
    if cached[0] != bundle_stamp(path, weights_of(cached[1])):
        return bundle
    with LOCK:
        if path in MODELS:
            MODELS.move_to_end(path)
    bundle = cached[1]

    return bundle


def open_bundle(path: str) -> ml_inference.Bundle:
    """
    推論に使う一式を読む関数 (古くなければ覚えたものを使う)

    組み立てたモデルも一式と一緒に覚える. 覚える数は model_cache_size

    Parameters
    ----------
    path: str
        一式のディレクトリ

    Returns
    ----------
    bundle: ml_inference.Bundle
        一式
    """
    # 覚えた一式か, 読み直した一式
    found = fresh_model(path)
    if found is not None:
        return found
    bundle = ml_inference.load_bundle(path)
    stamp = bundle_stamp(path, weights_of(bundle))
    with LOCK:
        MODELS[path] = (stamp, bundle)
        MODELS.move_to_end(path)
        while len(MODELS) > model_cache_size():
            MODELS.popitem(last=False)

    return bundle


def read_bundle(path: str) -> ml_inference.Bundle:
    """
    モデルを使わない用途 (登録の確かめ, 問題の設定とのつなぎ) のために
    一式を読む関数 (覚えたモデルを追い出さない)

    Parameters
    ----------
    path: str
        一式のディレクトリ

    Returns
    ----------
    bundle: ml_inference.Bundle
        一式 (覚えた推論の一式が古くなければそれ)
    """
    # 推論に使う一式の覚えには入れない読み込み
    bundle = fresh_model(path) or ml_inference.load_bundle(path)

    return bundle


def remember_warnings(path: str, bundle: ml_inference.Bundle) -> None:
    """
    組み立てた一式の学習済みの重みの警告を覚える関数

    Parameters
    ----------
    path: str
        一式のディレクトリ
    bundle: ml_inference.Bundle
        一式

    Returns
    ----------
    None
    """
    # 組み立てる前は警告が分からないので覚えないこと
    if "model" not in bundle.cache:
        return
    warnings = list(bundle.cache.get("pretrained_warnings") or [])
    stamp = bundle_stamp(path, weights_of(bundle))
    with LOCK:
        WARNINGS[path] = (stamp, warnings)


def describe(path: str) -> tuple[dict[str, Any], bool]:
    """
    一式の説明と, モデルを組み立てたかを返す関数

    一覧と一式の画面のため. 推論に使う一式の覚えを動かさない

    Parameters
    ----------
    path: str
        一式のディレクトリ

    Returns
    ----------
    description: dict[str, Any]
        ML-Inference の describe_bundle (学習済みの重みの警告は, モデルを
        手放した後も覚えたものを出す)
    loaded: bool
        組み立てたモデルを今覚えているか
    """
    # 組み立てた一式があればその説明
    bundle = fresh_model(path)
    loaded = False
    if bundle is not None:
        remember_warnings(path, bundle)
        description = ml_inference.describe_bundle(bundle)
        loaded = "model" in bundle.cache
    else:
        description = cached_description(path)

    # 前に組み立てたときの警告 (一式が変わっていなければ)
    with LOCK:
        known = WARNINGS.get(path)
        weights = DESCRIPTIONS.get(path, (None, None))[1]
    if known is not None and not description.get("pretrained_warnings"):
        weights = weights_of(bundle) if bundle is not None else weights
        if known[0] == bundle_stamp(path, weights):
            description = {**description, "pretrained_warnings": known[1]}
    description = with_cached_weights(description)

    return description, loaded


def with_cached_weights(description: dict[str, Any]) -> dict[str, Any]:
    """
    既製のモデルの一式の, 出どころの重みのファイルがあるかを見直す関数

    出どころの重みは一式の外 (PyTorch の置き場) にあり, 一式の印に入らない
    ので, 覚えた説明でも毎回確かめる

    Parameters
    ----------
    description: dict[str, Any]
        一式の説明

    Returns
    ----------
    description: dict[str, Any]
        ready.weights_cached を今のファイルで直した説明 (既製のモデルの
        一式でなければそのまま)
    """
    # 置き場のファイルの有無だけ (中身のハッシュ値は読むときに確かめる)
    ready = description.get("ready")
    if not isinstance(ready, dict) or not ready.get("weights_file"):
        return description
    cached = Path(str(ready["weights_file"])).is_file()
    description = {**description, "ready": {**ready, "weights_cached": cached}}

    return description


def cached_description(path: str) -> dict[str, Any]:
    """
    覚えた説明が古くなければ返し, 古ければ読み直して覚える関数

    Parameters
    ----------
    path: str
        一式のディレクトリ

    Returns
    ----------
    description: dict[str, Any]
        説明
    """
    # 説明だけの覚え (モデルは組み立てないこと)
    with LOCK:
        cached = DESCRIPTIONS.get(path)
    if cached is not None and cached[0] == bundle_stamp(path, cached[1]):
        description = dict(cached[2])
        return description
    bundle = ml_inference.load_bundle(path)
    description = ml_inference.describe_bundle(bundle)
    stamp = bundle_stamp(path, weights_of(bundle))
    with LOCK:
        DESCRIPTIONS[path] = (stamp, weights_of(bundle), description)
        DESCRIPTIONS.move_to_end(path)
        while len(DESCRIPTIONS) > DESCRIPTION_CACHE_SIZE:
            DESCRIPTIONS.popitem(last=False)
    description = dict(description)

    return description


def forget(path: str) -> None:
    """
    一式の覚えたもの (モデル, 説明, 警告) を手放す関数

    Parameters
    ----------
    path: str
        一式のディレクトリ

    Returns
    ----------
    None
    """
    with LOCK:
        MODELS.pop(path, None)
        DESCRIPTIONS.pop(path, None)
        WARNINGS.pop(path, None)
