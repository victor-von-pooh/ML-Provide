"""
画面の通しのテスト用に, 値段を変える判断の実験と, 値段の一式 (版 13) と,
値段を選ぶ新しい商品の表を作る

実験は ML-Quest の値段を変える判断の骨組みの学習を短く動かした成果物の見本
(ML-Inference のテストの見本 price_skeleton. 値段の決め方
price_estimator.json, 当てはめた値を含む前処理のレシピ price_recipe.yaml,
選んだ値段の表 price_choices.csv, 検証用とテスト用の評価
evaluations/price_<分割>.json, 売れ行きのモデルの定義と重み, 元の表
sales.csv) を写したもの. ML-Inference の make_price_bundle で一式を作る
"""

import shutil
import sys
from pathlib import Path

import pandas as pd
from ml_inference import make_price_bundle

# 骨組みと同じ形の見本 (submodule の ML-Inference のテストの見本)
SKELETON = (
    Path(__file__).resolve().parents[2] / "ML-Inference" / "tests"
    / "price_skeleton"
)


def main(work: Path) -> None:
    """
    値段を変える判断の実験と値段の一式を作り, 値段を選ぶ新しい商品の表を
    書く関数

    Parameters
    ----------
    work: Path
        作業のディレクトリ (bundle に一式, products.csv に新しい商品,
        sales.csv に売れた数の列を持つ記録)

    Returns
    ----------
    None
    """
    # 見本の実験と, 学習の元の表を渡して作った一式
    work.mkdir(parents=True, exist_ok=True)
    run = work / "experiment"
    if run.exists():
        shutil.rmtree(run)
    shutil.copytree(SKELETON, run)
    raw = work / "sales.csv"
    shutil.copy(SKELETON / "sales.csv", raw)
    make_price_bundle(work / "bundle", run, name="prices", data=raw,
                      overwrite=True)

    # 新しい商品 (商品ごとの最後の記録を今の状態にした 6 行と, 値段の
    # 読めない 1 行. 売れた数の列は無い)
    sales = pd.read_csv(raw)
    new = sales.groupby("product_id", sort=False).tail(1).drop(
        columns=["units"]
    ).reset_index(drop=True)
    broken = new.iloc[[0]].assign(record_id="r9999", product_id="p99",
                                  price=0.0)
    pd.concat([new, broken], ignore_index=True).to_csv(
        work / "products.csv", index=False
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
