import json
from pathlib import Path

import ml_inference
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from test_ready_bundle import LICENSE, client, tiny_weights  # noqa: F401

# 索引の写真の種類, 名前, 色 (赤, 緑, 青の濃さ)
PHOTOS = [
    ("red", "r1", (220, 20, 20)), ("red", "r2", (200, 40, 30)),
    ("red", "r3", (180, 30, 50)), ("blue", "b1", (20, 20, 220)),
    ("blue", "b2", (40, 30, 200)), ("green", "g1", (20, 210, 30)),
]

# 骨組みが書く評価の行
EVALUATION = {
    "k": 2, "queries": 3, "recall_at_k": 0.75, "ndcg_at_k": 0.7,
    "recall_at_k_baseline": 0.5, "ndcg_at_k_baseline": 0.45,
    "recall_at_k_diff": 0.25, "ndcg_at_k_diff": 0.25,
}


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    """
    1 色の写真を書く関数

    Parameters
    ----------
    path: Path
        書くファイル
    color: tuple[int, int, int]
        赤, 緑, 青の濃さ

    Returns
    ----------
    None
    """
    # 12 × 10 の 1 色
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((10, 12, 3), dtype=np.uint8)
    array[:, :] = color
    Image.fromarray(array).save(path)


def search_bundle(tmp_path: Path) -> Path:
    """
    骨組みと同じ形の成果物から, 作り物の出どころの検索の一式を作る関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    bundle: Path
        一式のディレクトリ
    """
    # 索引の写真と埋め込み
    photos = tmp_path / "photos"
    for label, name, color in PHOTOS:
        write_image(photos / label / f"{name}.png", color)
    embedder = tmp_path / "embedder"
    info = ml_inference.make_ready_bundle(
        embedder, "tiny", "V1", library="test", license=LICENSE,
        form="vector"
    )
    frame = ml_inference.predict(embedder, photos)
    points = np.stack(frame["output"].to_list()).astype(np.float32)

    # 索引の説明, 埋め込み, 評価と一式
    artifacts = tmp_path / "artifacts"
    (artifacts / "evaluations").mkdir(parents=True)
    (artifacts / "search_index.json").write_text(json.dumps({
        "ready": {"library": "test", "model": "tiny", "weights": "V1",
                  "sha256": info["model"]["pretrained"][0]["sha256"]},
        "width": 3, "k": 2, "max_latency_ms": 10000.0,
        "root": str(photos),
        "items": [{"path": row["path"], "label": row["path"].split("/")[0],
                   "sha256": row["digest"]} for _, row in frame.iterrows()],
    }), encoding="utf-8")
    np.save(artifacts / "search_embeddings.npy", points)
    (artifacts / "evaluations" / "search_valid.json").write_text(
        json.dumps({"rows": [EVALUATION],
                    "baseline": "写真を縮めた画素をそのまま余弦の近さで"
                                "並べたもの"}),
        encoding="utf-8"
    )
    bundle = tmp_path / "bundle"
    ml_inference.make_search_bundle(bundle, artifacts, license=LICENSE)

    return bundle


def test_search_predict_and_photos(
    client: TestClient, tmp_path: Path  # noqa: F811
) -> None:
    """
    検索の一式を登録して問いの写真を推論し, 上位 K 件, 待ち時間, 評価と,
    問いと索引の写真の縮小画像が返ることを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 登録と説明
    bundle = search_bundle(tmp_path)
    created = client.post("/api/bundles", json={"path": str(bundle)})
    assert created.status_code == 201, created.text
    bundle_id = created.json()["id"]
    described = client.get(f"/api/bundles/{bundle_id}").json()
    assert described["description"]["kind"] == "search"
    assert described["description"]["search"]["items"] == 6

    # 問題の設定の欄は出さず, 問題の設定も読まない (件数と待ち時間の上限
    # は一式が問題の設定から写して持つため)
    assert described["uses_problem"] is False
    listed = client.get("/api/bundles").json()
    assert [item["uses_problem"] for item in listed] == [False]
    state = client.get(f"/api/bundles/{bundle_id}/problem").json()
    assert state == {"path": None, "problem_id": None, "linked": None,
                     "error": None, "hint": None, "hint_note": None,
                     "hint_project": False}
    refused = client.put(f"/api/bundles/{bundle_id}/problem",
                         json={"path": str(tmp_path / "problem.yaml")})
    assert refused.status_code == 400
    assert "問題の設定を使わない" in refused.json()["detail"]

    # 赤い問いの写真で, 件数を変えて推論
    queries = tmp_path / "queries"
    write_image(queries / "q_red.png", (210, 30, 25))
    write_image(queries / "q_blue.png", (30, 25, 210))
    done = client.post(f"/api/bundles/{bundle_id}/predict",
                       json={"source": str(queries), "k": 3})
    assert done.status_code == 201, done.text
    result_id = done.json()["id"]
    result = client.get(f"/api/results/{result_id}").json()
    assert result["search"]["k"] == 3
    assert result["search"]["items"] == 6
    assert result["search"]["evaluation"]["recall_at_k"] == 0.75
    assert result["search"]["over_latency"] == 0
    assert result["summary"]["predicted"] == 2
    rows = client.get(f"/api/results/{result_id}/rows",
                      params={"limit": 50}).json()["rows"]
    red = [row for row in rows if row["path"] == "q_red.png"]
    assert [row["label"] for row in red] == ["red", "red", "red"]

    # 問いと索引の写真の縮小画像 (PNG)
    for side, path in (("query", "q_red.png"), ("item", red[0]["item"])):
        photo = client.get(f"/api/results/{result_id}/search-photo",
                           params={"side": side, "path": path})
        assert photo.status_code == 200, photo.text
        assert photo.headers["content-type"] == "image/png"
        assert photo.content[:8] == b"\x89PNG\r\n\x1a\n"

    # 場所の外, 無いファイル, 画像でないファイル, 知らない側
    (tmp_path / "photos" / "note.txt").write_text("x", encoding="utf-8")
    for side, path in (("item", "../queries/q_red.png"),
                       ("query", "../photos/red/r1.png"),
                       ("item", "red/none.png"), ("item", "note.txt")):
        photo = client.get(f"/api/results/{result_id}/search-photo",
                           params={"side": side, "path": path})
        assert photo.status_code == 400, (side, path)
    photo = client.get(f"/api/results/{result_id}/search-photo",
                       params={"side": "other", "path": "q_red.png"})
    assert photo.status_code == 422

    # 画素の多すぎる写真 (小さなファイル) は中身を読まずに断る
    Image.new("L", (6000, 5000)).save(tmp_path / "photos" / "huge.png")
    photo = client.get(f"/api/results/{result_id}/search-photo",
                       params={"side": "item", "path": "huge.png"})
    assert photo.status_code == 400
    assert "画素が多すぎ" in photo.json()["detail"]


def test_search_predict_errors(
    client: TestClient, tmp_path: Path  # noqa: F811
) -> None:
    """
    検索の一式に使えない入力と範囲の外の件数を断ることを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 一式と問いの写真
    bundle = search_bundle(tmp_path)
    bundle_id = client.post(
        "/api/bundles", json={"path": str(bundle)}
    ).json()["id"]
    queries = tmp_path / "queries"
    write_image(queries / "q.png", (210, 30, 25))
    for body in ({"source": str(queries), "k": 0},
                 {"source": str(queries), "k": 2.5},
                 {"source": str(queries), "k": True},
                 {"source": str(queries), "users": ["a"]},
                 {"source": str(queries), "rules": {"min_confidence": 0.5}},
                 {"source": str(queries),
                  "decision": {"type": "search", "k": 2}},
                 {"texts": ["a"]}, {}):
        done = client.post(f"/api/bundles/{bundle_id}/predict", json=body)
        assert done.status_code == 400, body

    # 無い結果の縮小画像は 404
    photo = client.get("/api/results/none/search-photo",
                       params={"side": "query", "path": "q.png"})
    assert photo.status_code == 404
