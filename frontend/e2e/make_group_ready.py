"""
画面の通しのテスト用に, 既製の埋め込みの群分けの骨組みと同じ形の成果物
(群の中心, 写真ごとの群, 群の安定性と群の要約) と, 既製の埋め込みの群分け
の一式 (群に分けた写真を渡すので, 群の中心から遠い行の上限を持つ版 17.
ML-Inference の make_group_bundle) を作り, 群に分ける新しい商品の写真を書く

埋め込みは作り物の出どころ (e2e_ready の e2e. チャンネルの平均) の埋め込みの
一式で求めるので, ネットワークを使わない. 群の中心は色ごとの埋め込みの平均
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ml_eda import Data, describe_groups
from ml_inference import (
    make_group_bundle,
    make_ready_bundle,
    predict,
    register_units,
)
from PIL import Image

# 商品の写真の色 (赤, 緑, 青の濃さ. 群の番号の順) と, 色ごとの写真の大きさ
# (幅, 高さ)
COLORS = {"red": (220, 30, 30), "green": (30, 200, 40),
          "blue": (30, 40, 210)}
SIZES = [(32, 24), (40, 30), (24, 24), (48, 32)]

# 学習に無い淡い赤 (赤の群に入るが, 群の中心から学習のときより遠い)
PALE = (255, 150, 150)

# 骨組みが書く比べる相手の名前と, 群の安定性の行
BASELINE = "写真を縮めた画素をそのまま k-means で分けた群"
STABILITY = {
    "k": 3, "method": "seed", "repeats": 5, "ari": 0.96,
    "ari_baseline": 0.81, "ari_diff": 0.15, "ari_min": 0.93,
    "ari_min_baseline": 0.74, "ari_min_diff": 0.19, "silhouette": 0.71,
    "silhouette_baseline": 0.48, "silhouette_diff": 0.23,
}


def write_image(
    path: Path, color: tuple[int, int, int], size: tuple[int, int],
    shift: int
) -> None:
    """
    1 色の写真を書く関数

    Parameters
    ----------
    path: Path
        書くファイル
    color: tuple[int, int, int]
        赤, 緑, 青の濃さ
    size: tuple[int, int]
        幅と高さ
    shift: int
        濃さのずらし (同じ色の写真を少しずつ違える)

    Returns
    ----------
    None
    """
    # 幅 × 高さの 1 色
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    array[:, :] = [max(0, min(255, value + shift)) for value in color]
    Image.fromarray(array).save(path)


def write_artifacts(root: Path, photos: Path) -> Path:
    """
    骨組みと同じ手順で, 既製の埋め込みの群分けの成果物を書く関数

    Parameters
    ----------
    root: Path
        書くディレクトリ (埋め込みの一式を root / "embedder" に作る)
    photos: Path
        群に分けた写真のディレクトリ

    Returns
    ----------
    folder: Path
        成果物のディレクトリ
    """
    # 埋め込みの一式での, 読めた写真だけの埋め込みと, 最も濃い色の群
    embedder = root / "embedder"
    info = make_ready_bundle(embedder, "colors", "V1", library="e2e",
                             form="vector", overwrite=True)
    frame = predict(embedder, photos)
    frame = frame[frame["status"] == "ok"].reset_index(drop=True)
    points = np.stack(frame["output"].to_list()).astype(np.float32)
    labels = points.argmax(1)
    centers = np.stack([points[labels == n].mean(0) for n in range(3)])

    # 群の中心, 割り当て (2 割を検証用), 群の安定性と群の要約
    folder = root / "artifacts"
    (folder / "evaluations").mkdir(parents=True, exist_ok=True)
    (folder / "group_centers.json").write_text(json.dumps({
        "k": 3,
        "ready": {"library": "e2e", "model": "colors", "weights": "V1",
                  "sha256": info["model"]["pretrained"][0]["sha256"]},
        "centers": centers.tolist(),
    }), encoding="utf-8")
    split = ["valid" if n % 5 == 4 else "train" for n in range(len(frame))]
    pd.DataFrame({"path": frame["path"], "group": labels,
                  "split": split}).to_csv(
        folder / "group_assignments.csv", index=False
    )
    summary = describe_groups(
        Data(kind="table", frame=frame[["width", "height"]]),
        columns=["width", "height"], labels=labels.tolist()
    )["groups"]
    for row, value in zip(summary[:-1], (0.95, 0.9, 0.85), strict=True):
        row["stability"] = value
    (folder / "evaluations" / "group_stability.json").write_text(
        json.dumps({"rows": [STABILITY], "epoch": None,
                    "baseline": BASELINE}, ensure_ascii=False),
        encoding="utf-8"
    )
    (folder / "evaluations" / "group_summary.json").write_text(
        json.dumps({"rows": summary}, ensure_ascii=False), encoding="utf-8"
    )

    return folder


def main(root: Path) -> None:
    """
    群に分けた写真, 成果物, 一式と新しい商品の写真を書く関数

    Parameters
    ----------
    root: Path
        書くディレクトリ

    Returns
    ----------
    None
    """
    # 作り物の出どころの部品 (起動の設定で読むのと同じモジュール)
    register_units(["e2e_ready"])

    # 色ごとに大きさの違う商品の写真と, 成果物と一式
    photos = root / "photos"
    for name, color in COLORS.items():
        for index, size in enumerate(SIZES):
            write_image(photos / f"{name}_{index}.png", color, size,
                        10 * index)
    artifacts = write_artifacts(root, photos)
    make_group_bundle(root / "bundle", artifacts, name="商品の写真の群",
                      data=photos, overwrite=True)

    # 新しい商品の写真 (赤 2 枚, 緑 1 枚, 青 2 枚, 学習に無い淡い赤 1 枚と,
    # 読めないファイル. 淡い赤は赤の群の中心から学習のときより遠い)
    new = root / "new"
    for index, name in enumerate(["red", "red", "green", "blue", "blue"]):
        write_image(new / f"n{index}_{name}.png", COLORS[name],
                    SIZES[index % len(SIZES)], 5 * index)
    write_image(new / "n5_pale.png", PALE, SIZES[0], 0)
    (new / "n9_broken.png").write_bytes(b"not an image")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
