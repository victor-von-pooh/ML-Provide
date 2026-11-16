import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import ml_inference
import numpy as np
import pandas as pd
import torch
import yaml
from ml_eda import Dataset, export_dataset, load_data, load_dataset
from ml_modelcontrol import build_model
from torch import nn, optim

# 見本ごとの学習の回数 (全ての行を 1 度に使う)
STEPS = 300

# 見本の作り方の版 (作り方を変えたら上げ, 置き場の古い見本を作り直す)
SAMPLES_VERSION = 1

# 推論を試す新しいデータの行の数
FRESH_ROWS = 50


def churn_frame(rng: np.random.Generator, count: int) -> pd.DataFrame:
    """
    契約の解約の見本の表を作る関数

    Parameters
    ----------
    rng: np.random.Generator
        乱数
    count: int
        行の数

    Returns
    ----------
    frame: pd.DataFrame
        id, plan, months, calls, fee, label (解約 か 継続) の表
    """
    # 月ごとの契約, 短い利用, 多い問い合わせで高まる解約の点数
    plan = rng.choice(["月ごと", "年ごと"], size=count, p=[0.6, 0.4])
    months = rng.integers(1, 60, size=count)
    calls = rng.poisson(2, size=count)
    fee = rng.normal(5000, 1500, size=count).round(-1)
    score = (
        1.2 * (plan == "月ごと") - 0.05 * months + 0.5 * calls
        + rng.normal(0, 0.8, size=count) - 0.5
    )
    frame: pd.DataFrame = pd.DataFrame({
        "id": np.arange(1, count + 1), "plan": plan, "months": months,
        "calls": calls, "fee": fee,
        "label": np.where(score > 0, "解約", "継続"),
    })

    return frame


def visitors_frame(rng: np.random.Generator, count: int) -> pd.DataFrame:
    """
    店の来店数の見本の表を作る関数

    Parameters
    ----------
    rng: np.random.Generator
        乱数
    count: int
        行の数

    Returns
    ----------
    frame: pd.DataFrame
        shop, temperature, holiday, promo, visitors の表
    """
    # 店ごとの水準, 休日, 販促, 気温 (25 度で最も多い)
    shop = rng.choice(["駅前店", "郊外店", "本店"], size=count)
    level = pd.Series(shop).map({"駅前店": 400, "郊外店": 250, "本店": 600})
    temperature = rng.uniform(0, 35, size=count).round(1)
    holiday = rng.integers(0, 2, size=count)
    promo = rng.integers(0, 2, size=count)
    visitors = (
        level.to_numpy() * (1 + 0.3 * holiday + 0.2 * promo)
        - 0.4 * (temperature - 25) ** 2 + rng.normal(0, 30, size=count)
    ).round()
    frame: pd.DataFrame = pd.DataFrame({
        "shop": shop, "temperature": temperature, "holiday": holiday,
        "promo": promo, "visitors": visitors,
    })

    return frame


def review_frame(rng: np.random.Generator, count: int) -> pd.DataFrame:
    """
    レビューの見本の表を作る関数

    Parameters
    ----------
    rng: np.random.Generator
        乱数
    count: int
        行の数

    Returns
    ----------
    frame: pd.DataFrame
        review, label (良い か 悪い) の表
    """
    # 良い言葉か悪い言葉を含む短い文
    things = ["本体", "配送", "梱包", "説明書", "画面", "音"]
    good = ["とても良い", "満足です", "最高でした", "気に入りました"]
    bad = ["ひどい", "不満です", "最悪でした", "がっかりです"]
    labels = rng.choice(["良い", "悪い"], size=count)
    reviews = [
        f"{rng.choice(things)}は"
        f"{rng.choice(good if label == '良い' else bad)}。"
        for label in labels
    ]
    frame: pd.DataFrame = pd.DataFrame(
        {"review": reviews, "label": labels}
    )

    return frame


def daily_frame(rng: np.random.Generator, count: int) -> pd.DataFrame:
    """
    2 つの店の日々の売上の見本の表を作る関数

    Parameters
    ----------
    rng: np.random.Generator
        乱数
    count: int
        店ごとの日の数

    Returns
    ----------
    frame: pd.DataFrame
        date, shop, sales の表 (店ごとに日付の順)
    """
    # 店ごとの水準, 週末の倍率, 前の日からの揺れの続き
    days = pd.date_range("2025-01-01", periods=count, freq="D")
    parts = []
    for shop, level in (("駅前店", 100.0), ("郊外店", 60.0)):
        noise = np.zeros(count)
        for day in range(1, count):
            noise[day] = 0.7 * noise[day - 1] + rng.normal(0, 5)
        weekend = np.where(days.day_of_week >= 5, 1.4, 1.0)
        parts.append(pd.DataFrame({
            "date": days.strftime("%Y-%m-%d"), "shop": shop,
            "sales": (level * weekend + noise).round(1),
        }))
    frame: pd.DataFrame = pd.concat(parts, ignore_index=True)

    return frame


def daily_fresh(rng: np.random.Generator, count: int) -> pd.DataFrame:
    """
    日々の売上の見本の, 過去の 30 日と売上の空の先の 7 日の表を作る関数

    Parameters
    ----------
    rng: np.random.Generator
        乱数
    count: int
        使わない (日の数は決まっている)

    Returns
    ----------
    frame: pd.DataFrame
        date, shop, sales の表 (先の 7 日の売上は空)
    """
    # 学習のデータの続きの期間を作り, 最後の 7 日の売上を空に
    whole = daily_frame(rng, DAILY_DAYS + 37)
    whole["date"] = pd.to_datetime(whole["date"])
    later = whole[whole["date"] >= whole["date"].max() - pd.Timedelta(
        days=36
    )].copy()
    ahead = later["date"] > later["date"].max() - pd.Timedelta(days=7)
    later.loc[ahead, "sales"] = np.nan
    later["date"] = later["date"].dt.strftime("%Y-%m-%d")
    frame: pd.DataFrame = later.reset_index(drop=True)

    return frame


# 日々の売上の見本の, 店ごとの日の数
DAILY_DAYS = 300

# 見本の名前から, データを作る関数と設定 (目的の列, タスク, 前処理, 分割,
# 読み込みの設定, 結論の決め方の既定値)
SAMPLES: dict[str, tuple[Callable[..., pd.DataFrame], dict[str, Any]]] = {
    "見本: 解約の予測 (表の分類)": (churn_frame, {
        "target": "label", "task": "classification",
        "steps": [
            {"op": "drop_columns", "columns": ["id"]},
            {"op": "one_hot", "columns": ["plan"]},
            {"op": "standardize", "columns": "auto"},
        ],
        "split": {"method": "stratified", "valid": 0.2, "seed": 0},
        "options": {}, "conclusion": {"positive": "解約", "threshold": 0.4},
    }),
    "見本: 来店数の予測 (表の回帰)": (visitors_frame, {
        "target": "visitors", "task": "regression",
        "steps": [
            {"op": "one_hot", "columns": ["shop"]},
            {"op": "standardize", "columns": "auto"},
        ],
        "split": {"method": "random", "valid": 0.2, "seed": 0},
        "options": {}, "conclusion": {
            "bands": [{"label": "少ない", "upper": 300},
                      {"label": "ふつう", "upper": 600},
                      {"label": "多い"}],
            "group_by": "shop",
        },
    }),
    "見本: 日々の売上の予測 (時系列, 先の予測)": (daily_frame, {
        "target": "sales", "task": "forecasting",
        "steps": [
            {"op": "lag", "columns": ["sales"], "lags": [1, 7],
             "by": ["shop"]},
            {"op": "drop_missing"},
            {"op": "datetime_parts", "columns": ["date"],
             "parts": ["weekday"]},
            {"op": "one_hot", "columns": ["shop"]},
            {"op": "standardize", "columns": "auto"},
        ],
        "split": {"method": "time", "valid": 0.2},
        "options": {}, "rows": DAILY_DAYS, "fresh": daily_fresh,
        "conclusion": {"group_by": "shop"},
    }),
    "見本: レビューの分類 (文)": (review_frame, {
        "target": "label", "task": "classification", "steps": [],
        "item_steps": [{"op": "vocabulary", "unit": "char", "length": 16}],
        "split": {"method": "stratified", "valid": 0.2, "seed": 0},
        "options": {"text_column": "review"},
        "conclusion": {"min_confidence": 0.8},
    }),
}


def model_spec(dataset: Dataset, outputs: int) -> dict[str, Any]:
    """
    見本のデータセットに合わせた小さなモデルの定義を返す関数

    Parameters
    ----------
    dataset: Dataset
        ML-EDA のデータセット
    outputs: int
        出力の幅 (分類はクラスの数, 回帰は 1)

    Returns
    ----------
    spec: dict[str, Any]
        ML-ModelControl の定義 (文は埋め込みの平均, 表は全結合 2 層)
    """
    # 文は番号の並び, 表は学習に使う列
    if dataset.info["kind"] == "text":
        length = int(dataset.info["splits"]["train"]["shape"][1])
        vocabulary = next(
            len(step["state"]["vocabulary"])
            for step in dataset.recipe["item_steps"]
            if step["op"] == "vocabulary"
        )
        nodes: dict[str, Any] = {
            "bag": {"unit": "nn.EmbeddingBag",
                    "params": {"num_embeddings": vocabulary,
                               "embedding_dim": 16, "mode": "mean",
                               "padding_idx": 0},
                    "inputs": {"input": "ids"}},
            "head": {"unit": "nn.Linear",
                     "params": {"in_features": 16, "out_features": outputs},
                     "inputs": "bag"},
        }
        inputs = {"ids": {"shape": [2, length], "dtype": "int64",
                          "fill": "ones"}}
    else:
        width = len(dataset.features)
        nodes = {
            "fc": {"unit": "nn.Linear",
                   "params": {"in_features": width, "out_features": 16},
                   "inputs": "x"},
            "act": {"unit": "nn.ReLU", "inputs": "fc"},
            "head": {"unit": "nn.Linear",
                     "params": {"in_features": 16, "out_features": outputs},
                     "inputs": "act"},
        }
        inputs = {"x": {"shape": [2, width]}}
    spec = {"format_version": 1, "name": "sample", "inputs": inputs,
            "nodes": nodes, "outputs": {"y": "head"}}

    return spec


def train(
    spec: dict[str, Any], dataset: Dataset
) -> tuple[nn.Module, dict[str, float]]:
    """
    見本のモデルを学習用の分割で学習する関数

    Parameters
    ----------
    spec: dict[str, Any]
        モデルの定義
    dataset: Dataset
        ML-EDA のデータセット

    Returns
    ----------
    result: tuple[nn.Module, dict[str, float]]
        学習したモデルと, 目的の値の中心と幅 (回帰. 分類は空)
    """
    # 回帰の目的の値の標準化 (一式の target_scale で戻す)
    model = build_model(spec, seed=0)
    x, y = dataset.tensors("train")
    scale: dict[str, float] = {}
    if dataset.info["task"] == "classification":
        loss_fn: nn.Module = nn.CrossEntropyLoss()
    else:
        scale = {"center": y.mean().item(), "scale": y.std().item()}
        y = ((y - scale["center"]) / scale["scale"]).unsqueeze(1)
        loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    model.train()
    for _ in range(STEPS):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    result = (model, scale)

    return result


def write_sample(
    root: Path, name: str, number: int, seed: int
) -> tuple[Path, Path]:
    """
    見本の 1 つ (データ, データセット, 一式, 新しいデータ) を書く関数

    Parameters
    ----------
    root: Path
        見本の置き場
    name: str
        見本の名前
    number: int
        見本の番号 (ディレクトリの名前と乱数に使う)
    seed: int
        乱数のシード

    Returns
    ----------
    paths: tuple[Path, Path]
        一式のディレクトリと, 推論を試す新しいデータのファイル
    """
    # 学習用のデータと, 目的の列の無い新しいデータ
    maker, settings = SAMPLES[name]
    place = root / f"sample{number}"
    shutil.rmtree(place, ignore_errors=True)
    place.mkdir(parents=True)
    rng = np.random.default_rng(seed + number)
    maker(rng, settings.get("rows", 600)).to_csv(
        place / "train.csv", index=False
    )
    if "fresh" in settings:
        fresh = settings["fresh"](rng, FRESH_ROWS)
    else:
        fresh = maker(rng, FRESH_ROWS).drop(columns=[settings["target"]])
    fresh.to_csv(place / "new.csv", index=False)

    # 前処理して書き出し, 小さなモデルを学習して一式に
    recipe = {
        "format_version": 1, "target": settings["target"],
        "task": settings["task"], "steps": settings["steps"],
        "item_steps": settings.get("item_steps", []),
        "split": settings["split"],
    }
    data = load_data(place / "train.csv", **settings["options"])
    export_dataset(data, recipe, place / "dataset")
    dataset = load_dataset(place / "dataset")
    classes = dataset.info["classes"]
    spec = model_spec(dataset, len(classes) if classes else 1)
    model, scale = train(spec, dataset)
    torch.save(model.state_dict(), place / "weights.pt")
    ml_inference.make_bundle(
        place / "bundle", spec, place / "dataset", place / "weights.pt",
        target_scale=scale or None, conclusion=settings["conclusion"],
        name=name
    )
    (place / "sample.yaml").write_text(
        yaml.safe_dump({"version": SAMPLES_VERSION, "name": name},
                       allow_unicode=True),
        encoding="utf-8"
    )
    paths = (place / "bundle", place / "new.csv")

    return paths


def current(place: Path, name: str) -> bool:
    """
    置き場の見本が, 今の作り方で作り終えたものかを確かめる関数

    Parameters
    ----------
    place: Path
        見本のディレクトリ
    name: str
        見本の名前

    Returns
    ----------
    ready: bool
        印のファイルの版と名前が今と同じで, 一式と新しいデータがあるか
    """
    # 印のファイル, 一式, 新しいデータ
    try:
        mark = yaml.safe_load(
            (place / "sample.yaml").read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError):
        return False
    ready = (
        isinstance(mark, dict) and mark.get("version") == SAMPLES_VERSION
        and mark.get("name") == name
        and (place / "bundle" / "inference.yaml").is_file()
        and (place / "new.csv").is_file()
    )

    return ready


def write_samples(root: Path, seed: int = 0) -> dict[str, tuple[Path, Path]]:
    """
    操作を試すための見本の一式を, 無いものだけ書く関数

    Parameters
    ----------
    root: Path
        見本の置き場
    seed: int = 0
        乱数のシード

    Returns
    ----------
    samples: dict[str, tuple[Path, Path]]
        見本の名前から, 一式のディレクトリと新しいデータのファイル
    """
    # 同じ版で作り終えた見本の再利用 (印のファイルは最後に書く)
    samples = {}
    for number, name in enumerate(SAMPLES):
        place = root / f"sample{number}"
        bundle, fresh = place / "bundle", place / "new.csv"
        if not current(place, name):
            bundle, fresh = write_sample(root, name, number, seed)
        samples[name] = (bundle, fresh)

    return samples
