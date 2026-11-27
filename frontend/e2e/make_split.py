"""
画面の通しのテスト用に, 骨組みのプロジェクトに表の分類のデータセットと,
そのデータセットを config.yaml の data.dataset に記録した実験と, 実験の
中の推論の一式を作る

ML-EDA でワンホットと標準化をかけたデータセット (テスト用の分割を持つ) を
プロジェクトの data/split-dataset に書き出し, 学習のスクリプトがするように,
実験 (outputs/<名前>/<日付>/<時刻>) に run.json, config.yaml, 重みと推論の
一式 (inference) を置く. 分割のファイルは前処理の後の列なので, 場所として
推論すると前処理が 2 度かかり, 分割の名前で推論したときだけ推論できる
"""

import sys
from pathlib import Path

import ml_inference
import numpy as np
import pandas as pd
import torch
import yaml
from ml_eda import export_dataset, load_data
from ml_modelcontrol import build_model

# データセットの場所 (プロジェクトからの相対. 実験の config に書く)
DATASET = "data/split-dataset"

# 実験の場所 (出力先からの名前/日付/時刻)
RUN = Path("outputs") / "split" / "2026-01-04" / "09-00-00"


def main(project: Path) -> None:
    """
    データセット, 実験, 実験の中の一式を作る関数

    Parameters
    ----------
    project: Path
        骨組みのプロジェクトのディレクトリ

    Returns
    ----------
    None
    """
    # 契約の種類と 2 つの数の列から, 解約するかの表
    rng = np.random.default_rng(0)
    count = 120
    frame = pd.DataFrame({
        "plan": rng.choice(["月ごと", "年ごと"], size=count),
        "months": rng.integers(1, 48, size=count),
        "calls": rng.poisson(2, size=count),
        "label": rng.choice(["続く", "解約"], size=count),
    })
    place = project / "data"
    place.mkdir(parents=True, exist_ok=True)
    frame.to_csv(place / "split.csv", index=False)
    recipe = {
        "format_version": 1, "target": "label", "task": "classification",
        "steps": [{"op": "one_hot", "columns": ["plan"]},
                  {"op": "standardize", "columns": "auto"}],
        "split": {"method": "random", "valid": 0.2, "test": 0.25,
                  "seed": 0},
    }
    export_dataset(load_data(place / "split.csv"), recipe,
                   project / DATASET)

    # 実験の記録と重みと, 実験の中の一式
    run = project / RUN
    run.mkdir(parents=True)
    (run / "run.json").write_text('{"status": "completed"}',
                                  encoding="utf-8")
    (run / "config.yaml").write_text(
        yaml.safe_dump({"data": {"dataset": DATASET}}), encoding="utf-8"
    )
    spec = {
        "format_version": 1,
        "inputs": {"x": {"shape": [2, 4]}},
        "nodes": {"head": {"unit": "nn.Linear",
                           "params": {"in_features": 4, "out_features": 2},
                           "inputs": "x"}},
        "outputs": {"logits": "head"},
    }
    torch.save(build_model(spec, seed=0).state_dict(),
               run / "final_model.pt")
    ml_inference.make_bundle(
        run / "inference", spec, project / DATASET, run / "final_model.pt",
        name="解約の分割"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
