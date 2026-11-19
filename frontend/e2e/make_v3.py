"""
画面の通しのテスト用に, 推論の一式の版 3 (列の出力でない一式) を作る

正解の無い表から, 領域 (正解は今の処理の出力) とベクトルを出す小さなモデルの
一式を作る. 後から手順を登録簿に無いものに書き換える一式と, その
inference.yaml も並べる. 自作の部品 (e2e_units) を使う一式と, 部品を登録
されていない名前に書き換えたモデルの定義も並べる
"""

import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import ml_inference
import numpy as np
import pandas as pd
import torch
import yaml
from ml_eda import export_dataset, load_data
from ml_modelcontrol import build_model


def spec_of(outputs: int, grid: Optional[list[int]] = None) -> dict:
    """
    全結合の 1 層 (と形を戻す層) のモデルの定義を返す関数

    Parameters
    ----------
    outputs: int
        出力の幅
    grid: Optional[list[int]] = None
        出力を (クラス, 高さ, 幅) に戻す形

    Returns
    ----------
    spec: dict
        ML-ModelControl の定義
    """
    # 3 列から出力の幅へ
    nodes: dict[str, Any] = {
        "head": {"unit": "nn.Linear", "inputs": "x",
                 "params": {"in_features": 3, "out_features": outputs}}
    }
    last = "head"
    if grid:
        nodes["grid"] = {"unit": "nn.Unflatten", "inputs": "head",
                         "params": {"dim": 1, "unflattened_size": grid}}
        last = "grid"
    spec = {"format_version": 1, "inputs": {"x": {"shape": [2, 3]}},
            "nodes": nodes, "outputs": {"out": last}}

    return spec


def main(root: Path) -> None:
    """
    データ, データセット, 一式を作る関数

    Parameters
    ----------
    root: Path
        作るディレクトリ

    Returns
    ----------
    None
    """
    # 正解の無い表 (新しいデータの 3 行目は欠けて推論できない)
    root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.normal(size=30),
                          "b": rng.normal(size=30),
                          "c": rng.normal(size=30)})
    frame.to_csv(root / "train.csv", index=False)
    frame.loc[2, "a"] = np.nan
    frame.head(6).to_csv(root / "new.csv", index=False)
    recipe = {"format_version": 1, "target": None, "task": "none",
              "steps": [{"op": "standardize", "columns": "auto"}],
              "split": {"method": "random", "valid": 0.2, "seed": 0}}
    export_dataset(load_data(root / "train.csv"), recipe, root / "dataset")

    # 領域とベクトルの一式
    bundles = (
        ("mask", spec_of(128, [2, 8, 8]), {
            "form": "mask", "output_options": {"classes": ["背景", "物"]},
            "target": {"kind": "mask",
                       "origin": {"type": "process", "name": "今の判定"}}}),
        ("vector", spec_of(5), {"form": "vector", "role": "group"}),
    )
    for name, spec, options in bundles:
        weights = root / f"{name}.pt"
        torch.save(build_model(spec, seed=0).state_dict(), weights)
        ml_inference.make_bundle(
            root / name, spec, root / "dataset", weights,
            name=f"版 3 の{name}", **options
        )

    # 自作の部品を使う一式 (起動の設定と同じモジュールから登録)
    ml_inference.register_units(["e2e_units"])
    spec = spec_of(4)
    spec["nodes"]["lab"] = {"unit": "e2e.Twice", "inputs": "x",
                            "params": {"features": 3}}
    spec["nodes"]["head"]["inputs"] = "lab"
    torch.save(build_model(spec, seed=0).state_dict(), root / "unit.pt")
    ml_inference.make_bundle(
        root / "unit", spec, root / "dataset", root / "unit.pt",
        name="版 3 の自作の部品", form="vector", role="group"
    )
    spec["nodes"]["lab"]["unit"] = "e2e_missing.Block"
    (root / "unit-missing-spec.yaml").write_text(
        yaml.safe_dump(spec), encoding="utf-8"
    )

    # 後から手順を登録簿に無いものに書き換える一式
    shutil.copytree(root / "vector", root / "later")
    file = root / "later" / "inference.yaml"
    info = yaml.safe_load(file.read_text(encoding="utf-8"))
    info["name"] = "版 3 の手順の無い一式"
    file.write_text(yaml.safe_dump(info, allow_unicode=True),
                    encoding="utf-8")
    info["procedure"] = {"name": "my_sampler", "steps": 3, "seed": 0}
    (root / "later-inference.yaml").write_text(
        yaml.safe_dump(info, allow_unicode=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
