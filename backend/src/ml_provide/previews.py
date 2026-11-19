"""
配列の出力の 1 件を, 画面で見せる小さな形にする仕組み

配列は np.load の mmap_mode で開いたまま渡し, 先に間引いてから数に
変える (大きな配列を丸ごと写さないため)
"""

import math
from typing import Any, Optional

import numpy as np

# 見本の大きさ (絵の 1 辺の画素, 点群の点, ベクトルの値)
PREVIEW_SIDE = 256
PREVIEW_POINTS = 2000
PREVIEW_VALUES = 16

# 見せる配列の値の数の上限 (読むのは間引いた分と, 区切って数える分だけ)
PREVIEW_LIMIT = 200_000_000

# 長さと番号の数を, 区切って数えるときの 1 回の値の数
CHUNK = 1_000_000

# 見せられる型の種類 (真偽, 整数, 符号無しの整数, 小数)
NUMBER_KINDS = "biuf"

# 画像のチャンネルの数として扱う数
CHANNELS = (1, 3, 4)

# 番号ごとの画素を数える番号の数の上限
MAX_LABELS = 50


def dtype_problem(array: np.ndarray) -> Optional[str]:
    """
    見せられない型 (複素数, 構造のある型, 文字, オブジェクト) の理由を
    返す関数

    Parameters
    ----------
    array: np.ndarray
        配列

    Returns
    ----------
    problem: Optional[str]
        理由 (見せられれば None)
    """
    # 実数と整数と真偽だけ
    problem: Optional[str] = None
    if array.dtype.kind not in NUMBER_KINDS or array.dtype.fields:
        problem = (
            f"型が {array.dtype} の配列は見せられません (見せるのは実数, "
            "整数, 真偽の配列だけ). .npy を落として見てください"
        )

    return problem


def row_blocks(array: np.ndarray) -> list[slice]:
    """
    配列を先頭の次元で区切る範囲を返す関数

    Parameters
    ----------
    array: np.ndarray
        1 次元以上の配列

    Returns
    ----------
    blocks: list[slice]
        1 回に CHUNK 個ほどの値を読む範囲
    """
    # 先頭の次元の 1 つ分の値の数から決める, 1 回の行の数
    width = max(1, int(np.prod(array.shape[1:], dtype=np.int64)))
    step = max(1, CHUNK // width)
    blocks = [
        slice(start, start + step) for start in range(0, array.shape[0], step)
    ]

    return blocks


def plain(values: np.ndarray) -> np.ndarray:
    """
    配列を数の欠けと無限大の無い float64 にする関数

    Parameters
    ----------
    values: np.ndarray
        配列 (間引いた後の小さなもの)

    Returns
    ----------
    plain: np.ndarray
        float64 の配列 (欠けは 0, 無限大は float64 の端)
    """
    # 小さな配列だけの写し
    result = np.nan_to_num(np.asarray(values, dtype=np.float64))

    return result


def vector_norm(array: np.ndarray) -> Optional[float]:
    """
    配列の全ての値の長さ (L2) を区切って求める関数

    Parameters
    ----------
    array: np.ndarray
        配列

    Returns
    ----------
    norm: Optional[float]
        長さ (数で表せない大きさは None)
    """
    # 先頭の次元で区切った 2 乗の和
    total = 0.0
    if array.ndim == 0:
        total = float(plain(array) ** 2)
    else:
        for block in row_blocks(array):
            part = plain(array[block])
            total += float(np.sum(part * part))
    root = math.sqrt(total)
    norm = root if math.isfinite(root) else None

    return norm


def thin_grid(array: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
    """
    絵の縦と横の次元を 1 辺 PREVIEW_SIDE 以下に間引く関数

    Parameters
    ----------
    array: np.ndarray
        配列 (間引く前. mmap のままでよい)
    axes: tuple[int, int]
        縦と横の次元の位置

    Returns
    ----------
    thinned: np.ndarray
        間引いた配列 (読んだのはこの分だけ)
    """
    # 縦と横を同じ間隔で
    side = max(array.shape[axes[0]], array.shape[axes[1]])
    stride = max(1, -(-side // PREVIEW_SIDE))
    index = [slice(None)] * array.ndim
    index[axes[0]] = slice(None, None, stride)
    index[axes[1]] = slice(None, None, stride)
    thinned = np.asarray(array[tuple(index)])

    return thinned


def label_counts(array: np.ndarray) -> list[dict[str, int]]:
    """
    領域の番号ごとの画素の数を区切って数える関数

    Parameters
    ----------
    array: np.ndarray
        (高さ, 幅) の番号の配列

    Returns
    ----------
    labels: list[dict[str, int]]
        label, count (番号の小さい順に MAX_LABELS 個まで)
    """
    # 行で区切った数の和
    counts: dict[int, int] = {}
    for block in row_blocks(array):
        found, numbers = np.unique(
            np.asarray(array[block]).astype(np.int64), return_counts=True
        )
        for label, count in zip(found.tolist(), numbers.tolist(),
                                strict=True):
            counts[label] = counts.get(label, 0) + int(count)
    labels = [
        {"label": label, "count": counts[label]}
        for label in sorted(counts)[:MAX_LABELS]
    ]

    return labels


def image_grid(array: np.ndarray) -> Optional[np.ndarray]:
    """
    画像の配列を間引いて (高さ, 幅) か (高さ, 幅, 3) の 0 から 255 に
    する関数

    Parameters
    ----------
    array: np.ndarray
        1 件の画像 ((高さ, 幅), (高さ, 幅, チャンネル), (チャンネル,
        高さ, 幅))

    Returns
    ----------
    grid: Optional[np.ndarray]
        0 から 255 の整数の格子 (絵にできない形は None)
    """
    # 形から決めるチャンネルの位置と, 先に間引く縦と横
    grid: Optional[np.ndarray] = None
    if array.ndim == 2:
        thinned = thin_grid(array, (0, 1))
    elif array.ndim == 3 and array.shape[2] in CHANNELS:
        thinned = thin_grid(array, (0, 1))
    elif array.ndim == 3 and array.shape[0] in CHANNELS:
        thinned = np.moveaxis(thin_grid(array, (1, 2)), 0, -1)
    else:
        return grid
    if thinned.size == 0:
        return grid
    if thinned.ndim == 3:
        thinned = thinned[..., 0] if thinned.shape[2] == 1 else (
            thinned[..., :3]
        )

    # 最小と最大で 0 から 255 に (間引いた分での決定)
    values = plain(thinned).clip(-1e300, 1e300)
    low, high = float(np.min(values)), float(np.max(values))
    scale = (high - low) or 1.0
    grid = np.round((values - low) / scale * 255).astype(np.int64)

    return grid


def array_preview(array: np.ndarray, form: str) -> dict[str, Any]:
    """
    1 件の配列を, 出力の形ごとに画面で見せる形にする関数

    Parameters
    ----------
    array: np.ndarray
        1 件の配列 (mmap のままでよい. 型は dtype_problem で確かめた後)
    form: str
        出力の形 (mask, image, point_cloud, vector)

    Returns
    ----------
    preview: dict[str, Any]
        form, shape, dtype, view (grid, image, points, values), values
        (最初の値), size, norm と見せ方ごとの中身
    """
    # どの形も形と型, 最初の値, 長さ
    head = plain(np.asarray(array).reshape(-1)[:PREVIEW_VALUES]) if (
        array.flags.c_contiguous
    ) else plain(np.asarray(array.flat[:PREVIEW_VALUES]))
    preview: dict[str, Any] = {
        "form": form,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "view": "values",
        "values": head.tolist(),
        "size": int(array.size),
        "norm": vector_norm(array),
    }

    # 領域は間引いたクラスの番号の格子と, 番号ごとの画素の数
    if form == "mask" and array.ndim == 2 and array.size > 0:
        grid = plain(thin_grid(array, (0, 1))).astype(np.int64)
        preview.update(view="grid", grid=grid.tolist(),
                       labels=label_counts(array))
    # 画像は 0 から 255 の格子
    elif form == "image":
        image = image_grid(array)
        if image is not None:
            preview.update(view="image", image=image.tolist())
    # 点群は間引いた最初の 3 つの次元
    elif form == "point_cloud" and array.ndim == 2 and array.shape[1] >= 2:
        stride = max(1, -(-array.shape[0] // PREVIEW_POINTS))
        points = plain(array[::stride, :3])
        preview.update(view="points", points=points.tolist(),
                       total=int(array.shape[0]))

    return preview
