"""
画面の通しのテスト用に, 並べるの実験と, その中の並べるの一式 (版 8) と,
並べる利用者, 前の反応, 後の反応の表を作る

ML-Quest の並べるの骨組みの学習のスクリプトがするように, 骨組みの
プロジェクトの実験 (outputs/<名前>/<日付>/<時刻>) に, ML-Found が書く重み
(final_model.pt), 写した定義 (model_spec.yaml), 実験の状態 (run.json) と,
学習のスクリプトが書く利用者と品物の番号の表 (rank_index.json) を置き,
ML-Inference の make_rank_bundle で実験の中に一式 (inference) を作る.
モデルは ML-ModelControl の同梱の 2 塔を番号の表の大きさにしたもの
"""

import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from ml_inference import make_rank_bundle
from ml_modelcontrol import build_model, load_example

# 番号の表の利用者と品物 (0 で始まる利用者も)
USERS = ["u0", "u1", "u2", "007"]
ITEMS = [f"i{n}" for n in range(6)]

# 反応の表の列の名前 (問題の設定の rank の列)
COLUMNS = {"user": "member", "item": "item_id", "reaction": "qty"}


def main(project: Path, tables: Path) -> None:
    """
    骨組みのプロジェクトの中に並べるの実験と一式を作り, 推論に渡す表を書く
    関数

    Parameters
    ----------
    project: Path
        骨組みのプロジェクトのディレクトリ
    tables: Path
        利用者と反応の表を書くディレクトリ

    Returns
    ----------
    None
    """
    # 同梱の 2 塔を利用者 4 人と品物 6 件の大きさに
    spec = load_example("two_tower_retrieval")
    spec["inputs"]["user"]["shape"] = [4, len(USERS)]
    spec["inputs"]["item"]["shape"] = [4, len(ITEMS)]
    spec["nodes"]["user_fc1"]["params"]["in_features"] = len(USERS)
    spec["nodes"]["item_fc1"]["params"]["in_features"] = len(ITEMS)

    # ML-Found が書く重み, 定義, 実験の状態と, 学習のスクリプトが書く番号の表
    run = project / "outputs" / "shop" / "2026-01-06" / "09-00-00"
    run.mkdir(parents=True, exist_ok=True)
    model = build_model(spec, seed=0, load_weights=False)
    torch.save(model.state_dict(), run / "final_model.pt")
    (run / "model_spec.yaml").write_text(
        yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8"
    )
    (run / "rank_index.json").write_text(
        json.dumps({"users": USERS, "items": ITEMS}), encoding="utf-8"
    )
    state = {"format_version": 1, "status": "completed"}
    (run / "run.json").write_text(json.dumps(state), encoding="utf-8")

    # 骨組みの問題の設定の rank の件数と列で, 実験の中に一式
    make_rank_bundle(run / "inference", run, k=3, columns=COLUMNS,
                     name="shop")

    # 並べる利用者 (番号の表に無い利用者も), 前の反応, 後の反応
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"member": ["u0", "007", "nobody"]}).to_csv(
        tables / "members.csv", index=False
    )
    pd.DataFrame({
        "member": ["u0", "u0", "007"],
        "item_id": ["i0", "i1", "i2"],
        "qty": [1, 2, 1],
    }).to_csv(tables / "seen.csv", index=False)
    pd.DataFrame({
        "member": ["u0", "007", "007"],
        "item_id": ["i3", "i4", "i5"],
        "qty": [1, 1, 2],
    }).to_csv(tables / "later.csv", index=False)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
