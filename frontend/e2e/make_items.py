"""
画面の通しのテスト用に, 版 2 と版 3 のデータセットと, そこで学習した実験を
作る

ML-EDA の合成の見本から画像と領域 (領域の値が 0, 1, 2 と, 飛ぶ 0, 3, 7),
撮影と点群の版 2 のデータセットと, 大きさの違う画像を中身の手順で 16 x 16 にそろえた版 3 のデータセットを書き
出し, ML-ModelControl の見本の定義 (unet_segmentation,
point_cloud_predictor) を 2 エポックだけ学習した ML-Found の形の実験
(final_model.pt と run.json) を作る. 骨組みのプロジェクトの実験の中に
推論の一式 (版 4) を作ったもの (ML-Quest の骨組みの学習のスクリプトが
するように) も並べる
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import ml_inference
import numpy as np
import torch
import yaml
from ml_eda import export_capture_clouds, export_image_masks, load_items
from ml_eda.synthetic import write_image_masks, write_shape_captures
from ml_modelcontrol import build_model, load_example
from PIL import Image

# 学習のエポックの数
EPOCHS = 2

# 撮影の深度から戻した点から選ぶ数と, 正解の点群から選ぶ数
INPUT_POINTS = 512
TARGET_POINTS = 64

# 点を選ぶ乱数の種
SEED = 0


def write_datasets(root: Path) -> dict[str, Path]:
    """
    合成の見本から版 2 のデータセットを 3 つと, 版 3 のデータセットを 1 つ
    書き出す関数

    Parameters
    ----------
    root: Path
        作るディレクトリ

    Returns
    ----------
    paths: dict[str, Path]
        元の画像と撮影のディレクトリと, データセットの場所
    """
    # 画像と領域 (正解は合成), 撮影と点群. 検証用とテスト用の分割も
    split = {"method": "random", "valid": 0.25, "test": 0.25, "seed": 0}
    write_image_masks(root / "raw_m", count=8, size=(16, 16))
    export_image_masks(
        root / "raw_m" / "images", root / "raw_m" / "masks",
        root / "masks", split=split,
        classes={0: "background", 1: "circle", 2: "square"},
        origin={"type": "synthetic"}
    )
    write_shape_captures(root / "raw_c", count=6, views=3, size=(16, 16),
                         points=64)
    export_capture_clouds(
        root / "raw_c" / "captures", root / "raw_c" / "clouds",
        root / "clouds", split=split
    )

    # 24 x 24 の画像を中身の手順で 16 x 16 にそろえた版 3 (版 2 の領域の
    # 実験の重みがそのまま使える大きさ)
    write_image_masks(root / "raw_m3", count=8, size=(24, 24))
    export_image_masks(
        root / "raw_m3" / "images", root / "raw_m3" / "masks",
        root / "masks_v3", split=split,
        classes={0: "background", 1: "circle", 2: "square"},
        item_steps={"image": [{"op": "resize", "width": 16,
                               "height": 16}]}
    )
    # 領域の値が飛ぶ (0, 3, 7) 版 2 (同じ画像の領域の値を付け替え. 値で
    # 名前を引くことを確かめる)
    shutil.copytree(root / "raw_m", root / "raw_gap")
    for mask in (root / "raw_gap" / "masks").glob("*.png"):
        values = np.asarray(Image.open(mask))
        Image.fromarray(
            np.array([0, 3, 7], dtype=np.uint8)[values]
        ).save(mask)
    export_image_masks(
        root / "raw_gap" / "images", root / "raw_gap" / "masks",
        root / "masks_gap", split=split,
        classes={0: "background", 3: "circle", 7: "square"}
    )
    paths = {
        "images": root / "raw_m" / "images",
        "captures": root / "raw_c" / "captures",
        "masks": root / "masks", "clouds": root / "clouds",
        "masks_v3": root / "masks_v3", "masks_gap": root / "masks_gap",
    }

    return paths


def mask_batch(dataset: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """
    画像と領域の学習用の分割を, 画像とクラスの番号の組にする関数

    Parameters
    ----------
    dataset: Path
        版 2 の画像と領域のデータセット

    Returns
    ----------
    batch: tuple[torch.Tensor, torch.Tensor]
        (件数, 3, 高さ, 幅) の 0 から 1 の画像と, (件数, 高さ, 幅) の番号
        (領域の値 0, 1, 2 がそのままクラスの番号)
    """
    # 画素を 255 で割る (ML-Inference の image の扱いと同じ)
    images, masks = [], []
    for item in load_items(dataset).split("train"):
        images.append(np.asarray(item["image"]).transpose(2, 0, 1) / 255.0)
        masks.append(np.asarray(item["mask"]))
    batch = (torch.tensor(np.stack(images), dtype=torch.float32),
             torch.tensor(np.stack(masks), dtype=torch.long))

    return batch


def cloud_batch(dataset: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """
    撮影と点群の学習用の分割を, 選んだ入力の点と正解の点の組にする関数

    Parameters
    ----------
    dataset: Path
        版 2 の撮影と点群のデータセット

    Returns
    ----------
    batch: tuple[torch.Tensor, torch.Tensor]
        (件数, INPUT_POINTS, 3) と (件数, TARGET_POINTS, 3) の点
    """
    # 深度から点に戻し, 件ごとの乱数で決まった数を選ぶ
    observed, targets = [], []
    for position, item in enumerate(load_items(dataset).split("train")):
        rng = np.random.default_rng([SEED, position])
        points = ml_inference.backproject(item["capture"])
        cloud = np.asarray(item["cloud"])
        observed.append(points[rng.choice(len(points), INPUT_POINTS)])
        targets.append(cloud[rng.choice(len(cloud), TARGET_POINTS)])
    batch = (torch.tensor(np.stack(observed), dtype=torch.float32),
             torch.tensor(np.stack(targets), dtype=torch.float32))

    return batch


def chamfer_loss(predicted: torch.Tensor, target: torch.Tensor) -> Any:
    """
    2 乗の距離の Chamfer の損失を返す関数

    Parameters
    ----------
    predicted: torch.Tensor
        (件数, 点の数, 3) の予測
    target: torch.Tensor
        (件数, 点の数, 3) の正解

    Returns
    ----------
    loss: Any
        両向きの最も近い点までの 2 乗の距離の平均の和
    """
    # 全ての組の距離から両向きの最小
    distance = torch.cdist(predicted, target) ** 2
    loss = distance.min(dim=2).values.mean() + (
        distance.min(dim=1).values.mean()
    )

    return loss


def train(
    spec: dict, batch: tuple[torch.Tensor, torch.Tensor], kind: str,
    run: Path
) -> None:
    """
    定義から組み立てたモデルを数エポック学習し, 実験のディレクトリに書く
    関数

    Parameters
    ----------
    spec: dict
        ML-ModelControl の定義
    batch: tuple[torch.Tensor, torch.Tensor]
        学習用の入力と正解
    kind: str
        mask か cloud (損失の選び)
    run: Path
        実験のディレクトリ (final_model.pt と run.json)

    Returns
    ----------
    None
    """
    # 全ての件を 1 つのバッチで, 損失を監視した実験として記録
    torch.manual_seed(0)
    model = build_model(spec, seed=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inputs, target = batch
    loss = torch.tensor(0.0)
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        # 出力が 1 つのモデルは配列をそのまま返す
        output = model(inputs)
        if kind == "mask":
            loss = torch.nn.functional.cross_entropy(output, target)
        else:
            loss = chamfer_loss(output, target)
        loss.backward()
        optimizer.step()
    run.mkdir(parents=True)
    torch.save(model.state_dict(), run / "final_model.pt")
    state = {"format_version": 1, "status": "completed",
             "monitor": "train/loss", "best_value": float(loss.item()),
             "best_epoch": EPOCHS, "final_weights": "best"}
    (run / "run.json").write_text(json.dumps(state), encoding="utf-8")


def main(root: Path, project: Path) -> None:
    """
    データセット, 定義, 実験と, 骨組みのプロジェクトの実験の一式を作る
    関数

    Parameters
    ----------
    root: Path
        作るディレクトリ
    project: Path
        骨組みのプロジェクト (問題の設定に記録したもの)

    Returns
    ----------
    None
    """
    # データセットと見本の定義 (領域は背景を含む 3 クラス)
    root.mkdir(parents=True)
    paths = write_datasets(root)
    unet = load_example("unet_segmentation")
    unet["nodes"]["head"]["params"]["out_channels"] = 3
    specs = {"mask": unet, "cloud": load_example("point_cloud_predictor")}
    for name, spec in specs.items():
        (root / f"{name}_spec.yaml").write_text(
            yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8"
        )

    # 2 エポックの学習
    train(unet, mask_batch(paths["masks"]), "mask", root / "runs" / "mask")
    train(specs["cloud"], cloud_batch(paths["clouds"]), "cloud",
          root / "runs" / "cloud")

    # 骨組みの学習のスクリプトのように, 実験の中に推論の一式 (版 4)
    run = project / "outputs" / "masks" / "2026-01-03" / "09-00-00"
    train(unet, mask_batch(paths["masks"]), "mask", run)
    ml_inference.make_bundle(
        run / "inference", unet, paths["masks"], run, name="masks"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
