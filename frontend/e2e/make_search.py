"""
画面の通しのテスト用に, 似た写真を探す骨組みと同じ形の成果物 (索引の説明と
埋め込み, 検証用の評価) と, 検索の一式 (版 15. ML-Inference の
make_search_bundle) を作り, 問いの写真を書く

埋め込みは作り物の出どころ (e2e_ready の e2e. チャンネルの平均) の埋め込みの
一式で求めるので, ネットワークを使わない
"""

import json
import sys
from pathlib import Path

import numpy as np
from ml_inference import (
    make_ready_bundle, make_search_bundle, predict, register_units
)
from PIL import Image

# 索引の写真の種類と色 (赤, 緑, 青の濃さ) と, 種類ごとの枚数
COLORS = {"red": (220, 30, 30), "green": (30, 200, 40),
          "blue": (30, 40, 210)}
PER_COLOR = 3

# 骨組みが書く比べる相手の名前
BASELINE = "写真を縮めた画素をそのまま余弦の近さで並べたもの"


def write_image(path: Path, color: tuple[int, int, int], shift: int) -> None:
    """
    1 色の写真を書く関数

    Parameters
    ----------
    path: Path
        書くファイル
    color: tuple[int, int, int]
        赤, 緑, 青の濃さ
    shift: int
        濃さのずらし (同じ種類の写真を少しずつ違える)

    Returns
    ----------
    None
    """
    # 24 × 32 の 1 色
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((24, 32, 3), dtype=np.uint8)
    array[:, :] = [max(0, min(255, value + shift)) for value in color]
    Image.fromarray(array).save(path)


def main(root: Path) -> None:
    """
    索引の写真, 成果物, 検索の一式と問いの写真を書く関数

    Parameters
    ----------
    root: Path
        書くディレクトリ

    Returns
    ----------
    None
    """
    # 作り物の出どころの部品 (起動の設定で読むのと同じモジュール)
    register_units(["e2e_ready"])

    # 種類のディレクトリの写真と, 作り物の出どころの埋め込み
    photos = root / "photos"
    for name, color in COLORS.items():
        for index in range(PER_COLOR):
            write_image(photos / name / f"{name}_{index}.png", color,
                        10 * index)
    embedder = root / "embedder"
    info = make_ready_bundle(embedder, "colors", "V1", library="e2e",
                             form="vector", overwrite=True)
    frame = predict(embedder, photos)
    points = np.stack(frame["output"].to_list()).astype(np.float32)

    # 索引の説明, 埋め込み, 評価
    artifacts = root / "artifacts"
    (artifacts / "evaluations").mkdir(parents=True, exist_ok=True)
    (artifacts / "search_index.json").write_text(json.dumps({
        "ready": {"library": "e2e", "model": "colors", "weights": "V1",
                  "sha256": info["model"]["pretrained"][0]["sha256"]},
        "width": int(points.shape[1]), "k": 3, "max_latency_ms": 5000,
        "root": str(photos.resolve()),
        "items": [{"path": row["path"], "label": row["path"].split("/")[0],
                   "sha256": row["digest"]} for _, row in frame.iterrows()],
    }, ensure_ascii=False), encoding="utf-8")
    np.save(artifacts / "search_embeddings.npy", points)
    (artifacts / "evaluations" / "search_valid.json").write_text(json.dumps({
        "rows": [{"k": 3, "queries": 3, "recall_at_k": 1.0,
                  "ndcg_at_k": 1.0, "recall_at_k_baseline": 0.67,
                  "ndcg_at_k_baseline": 0.7, "recall_at_k_diff": 0.33,
                  "ndcg_at_k_diff": 0.3}],
        "baseline": BASELINE,
    }, ensure_ascii=False), encoding="utf-8")
    make_search_bundle(root / "bundle", artifacts, overwrite=True)

    # 問いの写真 (赤と青と, 読めないファイル)
    queries = root / "queries"
    write_image(queries / "q_red.png", COLORS["red"], 5)
    write_image(queries / "q_blue.png", COLORS["blue"], 5)
    (queries / "q_broken.png").write_bytes(b"not an image")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
