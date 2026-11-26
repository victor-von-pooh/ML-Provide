"""
画面の通しのテスト用に, 時間をまたぐ量の判断の実験と, 需要の一式 (版 14) と,
発注の数を決める新しい日ごとの記録の表を作る

実験は ML-Quest の時間をまたぐ量の判断の学習のスクリプトと同じ成果物
(需要の決め方 demand_estimator.json, 当てはめた値を含む前処理のレシピ
demand_recipe.yaml, 発注の数の表 demand_orders.csv, 検証用とテスト用の評価
evaluations/demand_<分割>.json, 需要のモデルの定義と重み) を,
ML-Inference のテストの見本の作り方 (tests/test_demand_bundle.py の
daily_rows と write_demand_run. 在庫の数の列を持つ記録) で書いたもの.
ML-Inference の make_demand_bundle で一式を作る
"""

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
from ml_inference import make_demand_bundle

# 骨組みと同じ形の見本の作り方 (submodule の ML-Inference のテスト)
SAMPLE = (
    Path(__file__).resolve().parents[2] / "ML-Inference" / "tests"
    / "test_demand_bundle.py"
)


def load_sample() -> ModuleType:
    """
    ML-Inference のテストの, 日ごとの記録と実験の作り方を読む関数

    Parameters
    ----------
    None

    Returns
    ----------
    module: ModuleType
        daily_rows と write_demand_run を持つモジュール
    """
    # 見本の作り方を 2 か所に持たないための, 同じ関数
    spec = importlib.util.spec_from_file_location("demand_sample", SAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def main(work: Path) -> None:
    """
    時間をまたぐ量の判断の実験と需要の一式を作り, 発注の数を決める新しい
    日ごとの記録の表を書く関数

    Parameters
    ----------
    work: Path
        作業のディレクトリ (bundle に一式, daily.csv に新しい記録,
        records.csv に学習の元の記録)

    Returns
    ----------
    None
    """
    # 見本の実験と, 学習の元の表を渡して作った一式 (在庫の数の列を使う)
    sample = load_sample()
    work.mkdir(parents=True, exist_ok=True)
    for name in ("experiment", "bundle"):
        if (work / name).exists():
            shutil.rmtree(work / name)
    raw = work / "records.csv"
    sample.daily_rows(True).to_csv(raw, index=False)
    run = sample.write_demand_run(work / "experiment", raw, True)
    make_demand_bundle(work / "bundle", run, name="demand", data=raw,
                       overwrite=True)

    # 新しい記録 (4 商品の記録と, 発注を決める日の在庫の数が負の 1 商品.
    # 発注を決める日の売れた数は空)
    rows = sample.daily_rows(True)
    broken = rows[rows["product_id"] == "p00"].assign(
        product_id="p99",
        record_id=lambda frame: frame["record_id"].str.replace("r0_", "r9_")
    )
    broken.loc[broken.index[-1], "stock"] = -1.0
    new = pd.concat([rows, broken], ignore_index=True)
    last = new.groupby("product_id").tail(1).index
    new.loc[last, "units"] = None
    new.to_csv(work / "daily.csv", index=False)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
