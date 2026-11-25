"""
画面の通しのテスト用に, 異常の検知の実験と, 異常の度合いの一式 (版 11) と,
度合いを出す新しい行の表を作る

ML-Quest の異常の検知の骨組みが実験に書く成果物と同じ形で, 学習用の正常の
行の中心と精度行列 (距離に基づく度合いの決め方. anomaly_scorer.json), 当て
はめた値を含む前処理のレシピ (anomaly_recipe.yaml), 行ごとの度合いの表
(anomaly_scores.csv), 検証用の上位 K 件の評価 (evaluations/anomaly_valid.json)
を置き, ML-Inference の make_anomaly_bundle で一式を作る. 表は 3 つの列が
連れ立って動くセンサの行と, 振動だけ外れた少しの異常の行
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ml_eda import export_dataset, load_data, load_dataset
from ml_inference import make_anomaly_bundle

# 正常の行と異常の行の数, 人が 1 回に確かめる件数の上限
NORMAL = 60
ANOMALIES = 4
CAPACITY = 3

# 度合いに使う列
FEATURES = ["temp", "vibration", "current"]


def sensor_rows(seed: int = 0, normal: int = NORMAL,
                anomalies: int = ANOMALIES) -> pd.DataFrame:
    """
    正常の行と外れた少しの異常の行の, 合成のセンサの表を作る関数

    Parameters
    ----------
    seed: int = 0
        乱数の種
    normal: int = NORMAL
        正常の行の数
    anomalies: int = ANOMALIES
        異常の行の数 (表の最後の行)

    Returns
    ----------
    frame: pd.DataFrame
        reading_id, temp, vibration, current, is_anomaly の表 (異常の行
        にだけ印)
    """
    # 3 つの列が連れ立って動く正常の行と, 振動だけ外れた異常の行
    rng = np.random.default_rng(seed)
    rows = []
    for number in range(normal + anomalies):
        base = rng.normal(0, 1)
        broken = number >= normal
        rows.append({
            "reading_id": f"r{seed}{number:03d}",
            "temp": 50 + 5 * base + rng.normal(0, 0.3),
            "vibration": 2 + base + rng.normal(0, 0.1)
            + (6 if broken else 0),
            "current": 10 - 2 * base + rng.normal(0, 0.2),
            "is_anomaly": 1 if broken else None,
        })
    frame = pd.DataFrame(rows)

    return frame


def write_anomaly_run(root: Path, raw: Path) -> Path:
    """
    異常の検知の骨組みと同じ成果物を持つ, 距離に基づく度合いの実験の
    ディレクトリを作る関数

    Parameters
    ----------
    root: Path
        作業のディレクトリ (データセットと実験を置く)
    raw: Path
        元の表の CSV

    Returns
    ----------
    run: Path
        実験のディレクトリ
    """
    # 異常の検知の表を ML-EDA で書き出し, 分割ごとの行の特徴
    data_dir = root / "dataset"
    recipe = {"task": "anomaly", "target": "is_anomaly", "id": "reading_id",
              "steps": [{"op": "standardize", "columns": "auto"}],
              "split": {"method": "random", "valid": 0.2, "test": 0.2,
                        "seed": 0, "unlabeled": "normal"}}
    export_dataset(load_data(raw), recipe, data_dir)
    dataset = load_dataset(data_dir)
    run = root / "run"
    run.mkdir(parents=True)
    splits = {}
    for split in dataset.frames:
        x, y = dataset.arrays(split)
        splits[split] = {
            "x": np.array(x, dtype=np.float64),
            "y": np.zeros(len(x), dtype=np.int64) if y is None
            else np.asarray(y, dtype=np.int64),
            "ids": dataset.frames[split]["reading_id"].to_numpy(),
        }

    # 学習用の正常の行の中心と精度行列 (骨組みと同じ式の度合い)
    points = splits["train"]["x"]
    diff = points - points.mean(0)
    covariance = diff.T @ diff / (len(points) - 1) + np.eye(3) * 0.001
    distance = {"center": points.mean(0).tolist(),
                "precision": np.linalg.inv(covariance).tolist(),
                "ridge": 0.001}
    tables = []
    for split, part in splits.items():
        gap = part["x"].astype(np.float32).astype(np.float64) - np.asarray(
            distance["center"]
        )
        part["scores"] = (gap * (gap @ np.asarray(distance["precision"]))
                          ).mean(1)
        tables.append(pd.DataFrame({
            "reading_id": part["ids"], "split": split,
            "score": part["scores"], "anomaly": part["y"],
        }))
    pd.concat(tables).to_csv(run / "anomaly_scores.csv", index=False)

    # 度合いの決め方, レシピの写し, 評価
    train = splits["train"]["scores"]
    normal = {"rows": len(train), "mean": float(train.mean())}
    for point in (50, 90, 95, 99):
        normal[f"p{point}"] = float(np.percentile(train, point))
    normal["max"] = float(train.max())
    scorer = {
        "format_version": 1, "score": "distance",
        "recipe": "anomaly_recipe.yaml", "scores": "anomaly_scores.csv",
        "features": FEATURES, "id": "reading_id", "capacity": CAPACITY,
        "k": CAPACITY, "normal": normal, "model": None,
        "distance": distance,
    }
    (run / "anomaly_scorer.json").write_text(json.dumps(scorer, indent=2),
                                             encoding="utf-8")
    (run / "anomaly_recipe.yaml").write_text(
        (data_dir / "recipe.yaml").read_text(encoding="utf-8"),
        encoding="utf-8"
    )
    (run / "evaluations").mkdir()
    (run / "evaluations" / "anomaly_valid.json").write_text(json.dumps({
        "name": "anomaly", "split": "valid", "kind": "table",
        "rows": [{"rows": 13, "anomalies": 1, "k": CAPACITY,
                  "precision_at_k": 0.333, "average_precision": 0.9}],
    }), encoding="utf-8")

    return run


def main(work: Path) -> None:
    """
    異常の検知の実験と異常の度合いの一式を作り, 度合いを出す新しい行の表を
    書く関数

    Parameters
    ----------
    work: Path
        作業のディレクトリ (bundle に一式, readings.csv に新しい行)

    Returns
    ----------
    None
    """
    # 学習の元の表, 実験, 元の表で度合いを確かめた一式
    work.mkdir(parents=True, exist_ok=True)
    raw = work / "train.csv"
    sensor_rows().to_csv(raw, index=False)
    run = write_anomaly_run(work / "experiment", raw)
    make_anomaly_bundle(work / "bundle", run, name="sensors", data=raw)

    # 新しい行 (前処理の後に欠ける行を 1 つ含む. 正解の印の列は無い)
    new = sensor_rows(seed=1, normal=12, anomalies=2).drop(
        columns=["is_anomaly"]
    )
    new.loc[1, "temp"] = None
    new.to_csv(work / "readings.csv", index=False)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
