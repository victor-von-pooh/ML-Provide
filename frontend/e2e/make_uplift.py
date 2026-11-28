"""
画面の通しのテスト用に, 効果の見積もりの実験と, 効果の一式 (処置の群と
対照の群の 2 つの予測 (t_learner) は版 12, 処置の列を入力に入れる 1 つの
予測 (s_learner) は版 18) と, 効果を見積もる新しい顧客の表を作る

ML-Quest の効果の見積もりの骨組みが実験に書く成果物と同じ形で, 効果の
決め方 (uplift_estimator.json), 当てはめた値を含む前処理のレシピ
(uplift_recipe.yaml), 行ごとの効果の表 (uplift_effects.csv), 検証用と
テスト用の Qini の係数と上位 K 割の効果の差 (evaluations/uplift_<分割>.json),
検証用とテスト用の Qini の曲線 (evaluations/qini_curve_<分割>.json), 効果の
モデルの定義と重みを置き, ML-Inference の make_uplift_bundle で一式を
作る. 表はクーポンを無作為に配った記録で, 来店の多い顧客ほど配ると買う
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from ml_eda import export_dataset, load_data, load_dataset
from ml_inference import make_uplift_bundle
from ml_modelcontrol import build_model, load_example

# 学習の記録の顧客の数, 配る数の上限, 評価の上位の割合
CUSTOMERS = 400
BUDGET = 8
K = 0.3

# 見積もりに使う列
FEATURES = ["age", "visits"]


def coupon_rows(seed: int = 0, customers: int = CUSTOMERS) -> pd.DataFrame:
    """
    来店の多い顧客ほど配ると買う, 無作為に配った合成の記録を作る関数

    Parameters
    ----------
    seed: int = 0
        乱数の種
    customers: int = CUSTOMERS
        顧客の数

    Returns
    ----------
    frame: pd.DataFrame
        customer_id, age, visits, coupon, purchased の表
    """
    # 処置は無作為, 結果は来店の回数と処置で決める
    rng = np.random.default_rng(seed)
    rows = []
    for number in range(customers):
        visits = int(rng.integers(0, 10))
        coupon = int(rng.random() < 0.5)
        chance = 0.15 + (0.07 * visits if coupon else 0.0)
        rows.append({
            "customer_id": f"c{seed}{number:04d}",
            "age": float(rng.integers(20, 70)),
            "visits": float(visits),
            "coupon": coupon,
            "purchased": int(rng.random() < chance),
        })
    frame = pd.DataFrame(rows)

    return frame


def uplift_spec(learner: str = "t_learner") -> dict[str, Any]:
    """
    列の数に合わせた効果のモデルの定義を返す関数

    Parameters
    ----------
    learner: str = "t_learner"
        効果の決め方 (t_learner か s_learner)

    Returns
    ----------
    spec: dict[str, Any]
        ML-ModelControl の同梱の例 tabular_uplift (s_learner は
        tabular_uplift_s) の入力の幅を変えたもの
    """
    # s_learner は列の数と処置の列の 1 の和の幅の 1 つの枝
    if learner == "s_learner":
        spec = load_example("tabular_uplift_s")
        width = len(FEATURES) + 1
        spec["inputs"]["x"]["shape"] = [8, width]
        spec["nodes"]["outcome1"]["params"]["in_features"] = width
        return spec

    # 入力の幅と, 2 つの枝の最初の全結合の入力の幅
    spec = load_example("tabular_uplift")
    width = len(FEATURES)
    spec["inputs"]["x"]["shape"] = [8, width]
    spec["nodes"]["treated1"]["params"]["in_features"] = width
    spec["nodes"]["control1"]["params"]["in_features"] = width

    return spec


def with_treatment(x: np.ndarray, value: np.ndarray | float) -> np.ndarray:
    """
    行の特徴の最後に処置の列を足す関数 (s_learner の入力)

    Parameters
    ----------
    x: np.ndarray
        (行の数, 列の数) の float32 の行の特徴
    value: np.ndarray | float
        行ごとの処置の値か, 全ての行に入れる 1 か 0

    Returns
    ----------
    inputs: np.ndarray
        (行の数, 列の数 + 1) の float32
    """
    # 処置の値は前処理をかけずに小数で
    column = np.broadcast_to(np.asarray(value, dtype=np.float32), len(x))
    inputs = np.column_stack([x, column]).astype(np.float32)

    return inputs


def train_model(
    x: np.ndarray, t: np.ndarray, y: np.ndarray, learner: str = "t_learner"
) -> torch.nn.Module:
    """
    処置の群の行は treated, 対照の群の行は control の枝で学習する関数
    (s_learner は全ての行の処置の値を入力の最後に足した outcome で学習)

    Parameters
    ----------
    x: np.ndarray
        (行の数, 列の数) の float32 の学習用の行の特徴
    t: np.ndarray
        処置なら 1, 対照なら 0
    y: np.ndarray
        買ったかの 0 と 1
    learner: str = "t_learner"
        効果の決め方 (t_learner か s_learner)

    Returns
    ----------
    model: torch.nn.Module
        学習したモデル (評価の状態)
    """
    # 全ての行を 1 度に渡す, 群ごとの枝 (s_learner は 1 つの予測) の二値の
    # 交差エントロピー
    torch.manual_seed(0)
    model = build_model(uplift_spec(learner), seed=0, load_weights=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    single = learner == "s_learner"
    inputs = torch.tensor(with_treatment(x, t) if single else x)
    treated = torch.tensor(t == 1)
    target = torch.tensor(y, dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(300):
        optimizer.zero_grad()
        outputs = model(x=inputs)
        logits = outputs.reshape(-1) if single else torch.where(
            treated, outputs["treated"].reshape(-1),
            outputs["control"].reshape(-1)
        )
        loss_fn(logits, target).backward()
        optimizer.step()
    model.eval()

    return model


def estimate(
    model: torch.nn.Module, x: np.ndarray, learner: str = "t_learner"
) -> dict[str, np.ndarray]:
    """
    骨組みと同じ式で, 行ごとの 2 つの群の予測と効果を求める関数

    Parameters
    ----------
    model: torch.nn.Module
        効果のモデル
    x: np.ndarray
        (行の数, 列の数) の float32
    learner: str = "t_learner"
        効果の決め方 (s_learner は処置の列を 1 と 0 にして入力の最後に
        足した 2 回の予測)

    Returns
    ----------
    found: dict[str, np.ndarray]
        treated, control, effect の float64 の配列 (確率と確率の差)
    """
    # 2 つの予測と, それぞれをシグモイドで確率にしてからの差
    with torch.no_grad():
        if learner == "s_learner":
            treated, control = (
                model(x=torch.tensor(with_treatment(x, value))).reshape(-1)
                for value in (1.0, 0.0)
            )
        else:
            outputs = model(x=torch.tensor(x))
            treated = outputs["treated"].reshape(-1)
            control = outputs["control"].reshape(-1)
    treated = torch.sigmoid(treated.double())
    control = torch.sigmoid(control.double())
    found = {"treated": treated.numpy(), "control": control.numpy(),
             "effect": (treated - control).numpy()}

    return found


def qini_curve(
    effect: np.ndarray, t: np.ndarray, y: np.ndarray
) -> dict[str, list[float]]:
    """
    効果の高い順の Qini の曲線と, でたらめに配る線を求める関数

    Parameters
    ----------
    effect: np.ndarray
        行ごとの効果の見積もり
    t: np.ndarray
        処置なら 1, 対照なら 0
    y: np.ndarray
        結果

    Returns
    ----------
    curve: dict[str, list[float]]
        rows, fraction, qini, random の並び (先頭は 0 の点)
    """
    # 上位の行の群ごとの数と結果の和 (Y_t − Y_c × n_t / n_c)
    order = np.argsort(-effect, kind="stable")
    tt, yy = t[order].astype(float), y[order].astype(float)
    n_t, n_c = np.cumsum(tt), np.cumsum(1 - tt)
    y_t, y_c = np.cumsum(tt * yy), np.cumsum((1 - tt) * yy)
    ratio = np.divide(n_t, n_c, out=np.zeros_like(n_t), where=n_c > 0)
    values = np.append(0.0, np.where(n_c > 0, y_t - y_c * ratio, y_t))
    rows = np.arange(len(effect) + 1)
    curve = {"rows": rows.tolist(),
             "fraction": (rows / len(effect)).tolist(),
             "qini": values.tolist(),
             "random": (values[-1] * rows / len(effect)).tolist()}

    return curve


def evaluation(effect: np.ndarray, t: np.ndarray, y: np.ndarray,
               curve: dict[str, list[float]]) -> dict[str, Any]:
    """
    分割の Qini の係数と上位 K 割の効果の差を求める関数

    Parameters
    ----------
    effect: np.ndarray
        行ごとの効果の見積もり
    t: np.ndarray
        処置なら 1, 対照なら 0
    y: np.ndarray
        結果
    curve: dict[str, list[float]]
        qini_curve の曲線

    Returns
    ----------
    row: dict[str, Any]
        rows, treated, control, k, qini, uplift_at_k, effect, estimated
    """
    # 理想の並びは処置の群の結果, 対照の群の結果の符号を変えた値の並び
    def area(found: dict[str, list[float]]) -> float:
        values = np.asarray(found["qini"])
        return float(((values[1:] + values[:-1]) / 2).sum())

    random = len(effect) * curve["qini"][-1] / 2
    ideal = area(qini_curve(np.where(t == 1, y, -y), t, y))
    qini = (area(curve) - random) / (ideal - random)
    top = np.argsort(-effect, kind="stable")[:int(K * len(effect))]
    lift = None
    if (t[top] == 1).any() and (t[top] == 0).any():
        lift = float(y[top][t[top] == 1].mean() - y[top][t[top] == 0].mean())
    row = {"rows": len(effect), "treated": int((t == 1).sum()),
           "control": int((t == 0).sum()), "k": K, "qini": float(qini),
           "uplift_at_k": lift,
           "effect": float(y[t == 1].mean() - y[t == 0].mean()),
           "estimated": float(effect.mean())}

    return row


def write_uplift_run(
    root: Path, raw: Path, learner: str = "t_learner"
) -> Path:
    """
    効果の見積もりの骨組みと同じ成果物を持つ実験のディレクトリを作る関数

    Parameters
    ----------
    root: Path
        作業のディレクトリ (データセットと実験を置く)
    raw: Path
        元の表の CSV
    learner: str = "t_learner"
        効果の決め方 (t_learner か s_learner)

    Returns
    ----------
    run: Path
        実験のディレクトリ
    """
    # 介入の記録の表を ML-EDA で書き出し, 分割ごとの行の特徴
    data_dir = root / "dataset"
    recipe = {"task": "uplift", "target": "purchased", "treatment": "coupon",
              "outcome": "binary", "id": "customer_id",
              "steps": [{"op": "standardize", "columns": "auto"}],
              "split": {"method": "stratified", "valid": 0.25, "test": 0.25,
                        "seed": 0, "unlabeled": "drop"}}
    export_dataset(load_data(raw), recipe, data_dir)
    dataset = load_dataset(data_dir)
    run = root / "run"
    run.mkdir(parents=True)
    splits = {}
    for split in dataset.frames:
        x, y = dataset.arrays(split)
        splits[split] = {
            "x": np.array(x, dtype=np.float32),
            "t": np.asarray(dataset.treatment(split), dtype=np.int64),
            "y": np.asarray(y, dtype=np.float64),
            "ids": dataset.frames[split]["customer_id"].to_numpy(),
        }

    # 学習用の行で学んだ効果のモデル
    train = splits["train"]
    model = train_model(train["x"], train["t"], train["y"], learner)
    torch.save(model.state_dict(), run / "final_model.pt")
    (run / "model_spec.yaml").write_text(
        yaml.safe_dump(uplift_spec(learner), allow_unicode=True),
        encoding="utf-8"
    )

    # 行ごとの効果, 効果の決め方, レシピの写し, 評価と曲線
    tables = []
    for split, part in splits.items():
        part.update(estimate(model, part["x"], learner))
        tables.append(pd.DataFrame({
            "customer_id": part["ids"], "split": split,
            "effect": part["effect"], "treated": part["treated"],
            "control": part["control"], "treatment": part["t"],
            "outcome": part["y"],
        }))
    pd.concat(tables).to_csv(run / "uplift_effects.csv", index=False)
    effects = train["effect"]
    summary = {"rows": len(effects), "mean": float(effects.mean()),
               "min": float(effects.min())}
    for point in (10, 50, 90):
        summary[f"p{point}"] = float(np.percentile(effects, point))
    summary["max"] = float(effects.max())
    summary["positive"] = float((effects > 0).mean())
    names = ({"input": "x", "outcome": "outcome"} if learner == "s_learner"
             else {"input": "x", "treated": "treated", "control": "control"})
    estimator = {
        "format_version": 1, "learner": learner, "outcome": "binary",
        "recipe": "uplift_recipe.yaml", "effects": "uplift_effects.csv",
        "features": FEATURES, "id": "customer_id", "treatment": "coupon",
        "budget": BUDGET, "k": K,
        "model": {**names, "link": "sigmoid"},
        "scale": None, "train": summary,
    }
    (run / "uplift_estimator.json").write_text(
        json.dumps(estimator, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    (run / "uplift_recipe.yaml").write_text(
        (data_dir / "recipe.yaml").read_text(encoding="utf-8"),
        encoding="utf-8"
    )
    (run / "evaluations").mkdir()
    for split in ("valid", "test"):
        part = splits[split]
        curve = qini_curve(part["effect"], part["t"], part["y"])
        row = evaluation(part["effect"], part["t"], part["y"], curve)
        (run / "evaluations" / f"uplift_{split}.json").write_text(
            json.dumps({"name": f"uplift_{split}", "split": split,
                        "kind": "table", "rows": [row]}), encoding="utf-8"
        )
        points = [dict(zip(curve, values, strict=True))
                  for values in zip(*curve.values(), strict=True)]
        (run / "evaluations" / f"qini_curve_{split}.json").write_text(
            json.dumps({"name": f"qini_curve_{split}", "split": split,
                        "kind": "curve", "x": "fraction", "rows": points}),
            encoding="utf-8"
        )

    return run


def main(work: Path, learner: str = "t_learner") -> None:
    """
    効果の見積もりの実験と効果の一式を作り, 効果を見積もる新しい顧客の
    表を書く関数

    Parameters
    ----------
    work: Path
        作業のディレクトリ (bundle に一式, customers.csv に新しい顧客,
        trial.csv に無作為に配った試験の記録)
    learner: str = "t_learner"
        効果の決め方 (t_learner か s_learner)

    Returns
    ----------
    None
    """
    # 学習の元の表, 実験, 元の表で効果を確かめた一式
    work.mkdir(parents=True, exist_ok=True)
    raw = work / "train.csv"
    coupon_rows().to_csv(raw, index=False)
    run = write_uplift_run(work / "experiment", raw, learner)
    make_uplift_bundle(work / "bundle", run, name="coupons", data=raw)

    # 新しい顧客 (前処理の後に欠ける行を 1 つ含む. 処置と結果の列は無い)
    new = coupon_rows(seed=1, customers=40).drop(
        columns=["coupon", "purchased"]
    )
    new.loc[1, "visits"] = None
    new.to_csv(work / "customers.csv", index=False)

    # 配る相手を無作為に分けた試験の記録 (処置と結果の列を持つ)
    coupon_rows(seed=2, customers=120).to_csv(work / "trial.csv", index=False)


if __name__ == "__main__":
    main(Path(sys.argv[1]), *sys.argv[2:3])
