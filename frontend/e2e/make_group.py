"""
画面の通しのテスト用に, 群分けの実験と, その中の群分けの一式 (版 9) と,
群に割り当てる新しい行の表を作る

ML-Quest のまとめるの骨組みの学習のスクリプトがするように, 骨組みの
プロジェクトの実験 (outputs/<名前>/<日付>/<時刻>) に, ML-Found が書く重み
(final_model.pt), 写した定義 (model_spec.yaml), 実験の状態 (run.json) と,
学習のスクリプトが書く列の符号化と群の中心 (group_centers.json), 群の
割り当て (group_assignments.csv), 群の安定性と群の要約
(evaluations/group_stability.json, evaluations/group_summary.json) を置き,
ML-Inference の make_group_bundle で実験の中に一式 (inference) を作る.
モデルは ML-ModelControl の同梱の対照学習の符号化器を符号化した行の幅に
したもの. 群の中心は地域ごとの埋め込みの平均
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from ml_eda import Data, describe_groups
from ml_inference import make_group_bundle
from ml_modelcontrol import build_model, load_example

# 群の数 (地域の数) と, 群ごとの行の数
REGIONS = ["north", "south", "east"]
ROWS = 20

# 群を作る列 (数の列, 種類の列の順)
COLUMNS = ["spend", "visits", "region"]

# ML-Quest のまとめるの骨組みの比べる相手の名前
BASELINE = "列の特徴をそのまま k-means で分けた群"


def customer_rows(seed: int = 0, rows: int = ROWS) -> pd.DataFrame:
    """
    3 つのはっきりした群のある, 合成の顧客の表を作る関数

    Parameters
    ----------
    seed: int = 0
        乱数の種
    rows: int = ROWS
        群ごとの行の数

    Returns
    ----------
    frame: pd.DataFrame
        customer, spend, visits, region の表
    """
    # 群ごとに使う額, 来た回数, 地域が違う
    rng = np.random.default_rng(seed)
    items = []
    for group, region in enumerate(REGIONS):
        for number in range(rows):
            items.append({
                "customer": f"c{seed}_{group}_{number}",
                "spend": float(rng.normal(1000 * (group + 1), 50)),
                "visits": int(rng.poisson(2 + 6 * group)),
                "region": region,
            })
    frame = pd.DataFrame(items)

    return frame


def encoding_of(frame: pd.DataFrame) -> dict:
    """
    骨組みの make_encoding と同じ列の符号化を作る関数

    Parameters
    ----------
    frame: pd.DataFrame
        群を作る列

    Returns
    ----------
    encoding: dict
        numeric (mean と std) と categories (初めて出た順の値)
    """
    # 数の列は母の標準偏差, 種類の列は文字の値
    numeric, categories = {}, {}
    for name in frame.columns:
        column = frame[name]
        if pd.api.types.is_numeric_dtype(column):
            values = column.astype(float)
            std = float(values.std(ddof=0))
            numeric[name] = {"mean": float(values.mean()),
                             "std": std if std > 0 else 1.0}
        else:
            categories[name] = list(dict.fromkeys(column.map(str)))
    encoding = {"numeric": numeric, "categories": categories}

    return encoding


def features_of(frame: pd.DataFrame, encoding: dict) -> np.ndarray:
    """
    骨組みの encode_rows と同じ式で行を符号化する関数

    Parameters
    ----------
    frame: pd.DataFrame
        群を作る列
    encoding: dict
        列の符号化

    Returns
    ----------
    features: np.ndarray
        (行の数, 幅) の float32
    """
    # 数の列は標準化, 種類の列は値ごとの 0 と 1
    parts = []
    for name, stats in encoding["numeric"].items():
        values = (frame[name].astype(float) - stats["mean"]) / stats["std"]
        parts.append(values.fillna(0.0).to_numpy()[:, None])
    for name, values in encoding["categories"].items():
        text = frame[name].map(str).to_numpy()
        parts.append(np.stack([text == value for value in values], 1))
    features = np.concatenate(parts, 1).astype(np.float32)

    return features


def write_group_run(run: Path, frame: pd.DataFrame) -> Path:
    """
    まとめるの骨組みと同じ成果物を持つ実験のディレクトリを作る関数

    Parameters
    ----------
    run: Path
        実験のディレクトリ
    frame: pd.DataFrame
        学習の表 (customer_rows の形)

    Returns
    ----------
    run: Path
        実験のディレクトリ
    """
    # 同梱の符号化器を符号化した行の幅に
    run.mkdir(parents=True, exist_ok=True)
    columns = frame[COLUMNS]
    encoding = encoding_of(columns)
    features = features_of(columns, encoding)
    spec = load_example("contrastive_encoder")
    spec["inputs"]["x"]["shape"] = [8, features.shape[1]]
    spec["nodes"]["fc1"]["params"]["in_features"] = features.shape[1]

    # ML-Found が書く重み, 定義, 実験の状態
    model = build_model(spec, seed=0, load_weights=False)
    torch.save(model.state_dict(), run / "final_model.pt")
    (run / "model_spec.yaml").write_text(
        yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8"
    )
    state = {"format_version": 1, "status": "completed"}
    (run / "run.json").write_text(json.dumps(state), encoding="utf-8")

    # 群の中心は地域ごとの埋め込みの平均, 割り当ては最も近い中心
    model.eval()
    with torch.no_grad():
        points = model(x=torch.from_numpy(features))["embedding"]
    points = points.double().numpy()
    truth = frame["region"].map({r: n for n, r in enumerate(REGIONS)})
    centers = np.stack([points[truth == g].mean(0)
                        for g in range(len(REGIONS))])
    distance = ((points ** 2).sum(1)[:, None] - 2 * points @ centers.T
                + (centers ** 2).sum(1)[None, :])
    labels = distance.argmin(1)
    split = np.where(np.arange(len(frame)) % 5 == 0, "valid", "train")
    pd.DataFrame({"customer": frame["customer"], "group": labels,
                  "split": split}).to_csv(run / "group_assignments.csv",
                                          index=False)
    (run / "group_centers.json").write_text(json.dumps(
        {"k": len(REGIONS), "encoding": encoding,
         "centers": centers.tolist()}, ensure_ascii=False, indent=2
    ), encoding="utf-8")

    # 群の安定性と群の要約 (ML-Found の save_evaluation の形)
    evaluations = run / "evaluations"
    evaluations.mkdir(exist_ok=True)
    row: dict = {"k": len(REGIONS), "method": "seed", "repeats": 5}
    for name, value, base in (("ari", 0.97, 0.99), ("ari_min", 0.95, 0.98),
                              ("silhouette", 0.62, 0.55)):
        row.update({name: value, f"{name}_baseline": base,
                    f"{name}_diff": round(value - base, 6)})
    common = {"split": "all", "kind": "table", "x": None, "guard": False,
              "epoch": 12, "created_at": "2026-01-06T09:00:00+09:00"}
    (evaluations / "group_stability.json").write_text(json.dumps({
        "name": "group_stability", "columns": list(row), "rows": [row],
        "description": "群の安定性", "baseline": BASELINE, "better": {},
        **common
    }, ensure_ascii=False), encoding="utf-8")
    rows = describe_groups(Data(kind="table", frame=columns),
                           columns=COLUMNS,
                           labels=[int(v) for v in labels])["groups"]
    for item in rows[:-1]:
        item["stability"] = round(0.9 + 0.02 * item["group"], 6)
    (evaluations / "group_summary.json").write_text(json.dumps({
        "name": "group_summary", "columns": list(rows[0]), "rows": rows,
        "description": "群の要約", "baseline": None,
        "better": {"stability": "higher"}, **common
    }, ensure_ascii=False), encoding="utf-8")

    return run


def main(project: Path, tables: Path) -> None:
    """
    骨組みのプロジェクトの中に群分けの実験と一式を作り, 推論に渡す表を
    書く関数

    Parameters
    ----------
    project: Path
        骨組みのプロジェクトのディレクトリ
    tables: Path
        新しい行の表を書くディレクトリ

    Returns
    ----------
    None
    """
    # 実験と, 実験の中の一式
    run = project / "outputs" / "customers" / "2026-01-07" / "09-00-00"
    write_group_run(run, customer_rows())
    make_group_bundle(run / "inference", run, name="customers")

    # 新しい行 (数にできない行を 1 つ含む)
    tables.mkdir(parents=True, exist_ok=True)
    new = customer_rows(seed=1, rows=5).astype({"spend": object})
    new.loc[len(new)] = {"customer": "x", "spend": "?", "visits": 1,
                         "region": "north"}
    new.to_csv(tables / "customers.csv", index=False)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
