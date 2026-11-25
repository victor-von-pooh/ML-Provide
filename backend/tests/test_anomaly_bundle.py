import importlib.util
from datetime import datetime
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from test_bundle_v3 import register

from ml_provide.app import create_app

# テストで使うトークン
TOKEN = "test-token"


def load_maker() -> ModuleType:
    """
    画面の通しのテストが使う, 異常の検知の実験と一式を作るスクリプトを読む
    関数

    Parameters
    ----------
    None

    Returns
    ----------
    module: ModuleType
        frontend/e2e/make_anomaly.py (骨組みと同じ成果物を書く関数を持つ)
    """
    # 骨組みの成果物の書き方を 2 か所に持たないため, 同じスクリプトを使う
    path = Path(__file__).parents[2] / "frontend" / "e2e" / "make_anomaly.py"
    spec = importlib.util.spec_from_file_location("make_anomaly", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """
    一時ディレクトリを置き場にしたアプリのクライアントを作る関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    client: TestClient
        トークン付きのクライアント
    """
    app = create_app(
        data_dir=tmp_path / "home", token=TOKEN, allowed_hosts=["testserver"],
        procedures=[]
    )
    client = TestClient(app, headers={"x-provide-token": TOKEN})

    return client


@pytest.fixture
def anomaly_files(tmp_path: Path) -> tuple[Path, Path]:
    """
    異常の度合いの一式と, 度合いを出す新しい行の表を作る関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    files: tuple[Path, Path]
        一式のディレクトリと, 新しい行の表 (前処理の後に欠ける行を 1 つ
        含む 14 行)
    """
    load_maker().main(tmp_path / "anomaly")
    files = (tmp_path / "anomaly" / "bundle",
             tmp_path / "anomaly" / "readings.csv")

    return files


def test_anomaly_bundle(
    client: TestClient, anomaly_files: tuple[Path, Path]
) -> None:
    """
    異常の度合いの一式を登録して推論し, 度合いの結論と振り分けを残し,
    人の確認に回した行を度合いの高い順に読めることを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    anomaly_files: tuple[Path, Path]
        一式と新しい行の表

    Returns
    ----------
    None
    """
    # 版 11 の一式を登録すると, 説明に異常の欄があり, 問題の設定を読まない
    path, table = anomaly_files
    bundle = register(client, path)
    description = bundle["description"]
    assert (description["kind"], description["format_version"]) == (
        "anomaly", 11
    )
    assert description["column_output"] is False
    assert description["anomaly"]["capacity"] == 3
    assert description["decision"] == {"type": "triage", "capacity": 3}
    state = client.get(f"/api/bundles/{bundle['id']}/problem").json()
    assert state["linked"] is None and state["hint"] is None

    # 既定の振り分けで推論すると, 度合いの結論と振り分けの結論のまま残す
    url = f"/api/bundles/{bundle['id']}/predict"
    response = client.post(url, json={"source": str(table)})
    assert response.status_code == 201, response.text
    result = response.json()
    summary = result["summary"]
    assert (summary["rows"], summary["predicted"], summary["skipped"]) == (
        14, 13, 1
    )
    assert summary["id"] == "reading_id" and summary["score"] == "distance"
    assert set(summary["distribution"]) == {"p50", "p90", "p99", "max"}
    assert len(summary["top"]) == 3
    assert summary["evaluations"]["valid"]["precision_at_k"] == 0.333
    assert result["kind"] == "anomaly" and result["column_output"] is False
    assert "異常の度合い" in result["sentences"][0]
    decided = result["decision"]
    assert decided["decision"] == {"type": "triage", "capacity": 3}
    assert decided["summary"]["acted"] == 3

    # 待ち行列は人の確認に回した行を順位の順に (度合いの上位と同じ行)
    queue = client.get(f"/api/results/{result['id']}/queue").json()
    assert (queue["id"], queue["capacity"]) == ("reading_id", 3)
    rows = queue["rows"]
    assert [row["rank"] for row in rows] == [1, 2, 3]
    assert [row["row"] for row in rows] == [
        item["row"] for item in summary["top"]
    ]
    assert rows[0]["key"] == rows[0]["id"] == "r1013"
    assert rows[0]["column"] == "vibration"
    assert all(row["judgment"] is None for row in rows)

    # 件数の上限を変えると, その数だけ回す
    response = client.post(url, json={
        "source": str(table), "decision": {"type": "triage", "capacity": 2}
    })
    assert response.status_code == 201, response.text
    other = response.json()
    assert other["decision"]["summary"]["acted"] == 2
    queue = client.get(f"/api/results/{other['id']}/queue").json()
    assert len(queue["rows"]) == 2

    # 振り分けない結果の待ち行列は空
    response = client.post(url, json={"source": str(table), "decision": {}})
    assert response.status_code == 201, response.text
    plain = response.json()
    assert plain["decision"] is None
    queue = client.get(f"/api/results/{plain['id']}/queue").json()
    assert queue["rows"] == [] and queue["capacity"] is None


def test_anomaly_refusals(
    client: TestClient, anomaly_files: tuple[Path, Path]
) -> None:
    """
    異常の度合いの一式に, 振り分けでない意思決定, 結論の決め方, 何段も先の
    予測, 大きすぎる上限を渡すと断り, 列の出力だけの機能も断ることを
    確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    anomaly_files: tuple[Path, Path]
        一式と新しい行の表

    Returns
    ----------
    None
    """
    # 振り分けの他の型と項目は ML-Inference が断る
    path, table = anomaly_files
    bundle = register(client, path)
    url = f"/api/bundles/{bundle['id']}/predict"
    for extra in ({"rules": {"min_confidence": 0.5}}, {"forecast": True},
                  {"decision": {"type": "quantity"}},
                  {"decision": {"type": "triage", "capacity": 0}},
                  {"decision": {"type": "triage", "capacity": 1001}},
                  {"decision": {"type": "triage", "capacity": 3,
                                "miss_cost": 1}},
                  {"k": 3}):
        response = client.post(url, json={"source": str(table), **extra})
        assert response.status_code == 400, extra
    response = client.post(url, json={
        "source": str(table), "decision": {"type": "triage", "capacity": 1001}
    })
    assert "1000 件まで" in response.json()["detail"]

    # 結論の出し直し, 決め直し, たどり直しは列の出力の結果だけ
    result = client.post(url, json={"source": str(table)}).json()
    for name, body in (("conclude", {"rules": {}}),
                       ("decide", {"decision": None}), ("backtest", None)):
        response = client.post(f"/api/results/{result['id']}/{name}",
                               json=body)
        assert response.status_code == 400, name


def test_anomaly_reviews(
    tmp_path: Path, client: TestClient, anomaly_files: tuple[Path, Path]
) -> None:
    """
    人の確認の記録を 1 件ずつ足し, 書いた順に読めて, 待ち行列の今の判断が
    最後の記録になることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    client: TestClient
        クライアント
    anomaly_files: tuple[Path, Path]
        一式と新しい行の表

    Returns
    ----------
    None
    """
    # 記録の無い結果は 0 件
    path, table = anomaly_files
    bundle = register(client, path)
    result = client.post(f"/api/bundles/{bundle['id']}/predict",
                         json={"source": str(table)}).json()
    reviews = f"/api/results/{result['id']}/reviews"
    assert client.get(reviews).json() == {"records": []}

    # 1 件ずつ足すと, 時刻は今の時刻 (時差付き)
    queue = client.get(f"/api/results/{result['id']}/queue").json()["rows"]
    first, second = queue[0]["key"], queue[1]["key"]
    response = client.post(reviews, json={
        "row": first, "judgment": "anomaly", "reviewer": " 佐藤 "
    })
    assert response.status_code == 201, response.text
    record = response.json()
    assert (record["id"], record["judgment"], record["reviewer"]) == (
        first, "anomaly", "佐藤"
    )
    assert datetime.fromisoformat(record["reviewed_at"]).tzinfo is not None
    for row, judgment in ((second, "normal"), (first, "normal")):
        response = client.post(reviews, json={
            "row": row, "judgment": judgment, "reviewer": "鈴木"
        })
        assert response.status_code == 201, response.text

    # 書いた順の記録と, 同じ行の最後の記録が今の判断
    records = client.get(reviews).json()["records"]
    assert [(r["id"], r["judgment"]) for r in records] == [
        (first, "anomaly"), (second, "normal"), (first, "normal")
    ]
    queue = client.get(f"/api/results/{result['id']}/queue").json()["rows"]
    assert (queue[0]["judgment"], queue[0]["reviewer"]) == ("normal", "鈴木")
    assert queue[1]["judgment"] == "normal" and queue[2]["judgment"] is None

    # 記録は結果の隣の reviews.csv (ML-Inference の形)
    place = tmp_path / "home" / "results" / result["id"]
    found = pd.read_csv(place / "reviews.csv",
                        dtype=str, encoding="utf-8-sig")
    assert list(found.columns) == ["id", "judgment", "reviewer",
                                   "reviewed_at"]
    assert len(found) == 3

    # 判断, 名前, 行の誤りは断る
    for body, status in (
        ({"row": first, "judgment": "maybe", "reviewer": "a"}, 422),
        ({"row": first, "judgment": "normal", "reviewer": " "}, 400),
        ({"row": first, "judgment": "normal", "reviewer": "a\nb"}, 400),
        ({"row": first, "judgment": "normal", "reviewer": "a" * 81}, 400),
        ({"row": "nothing", "judgment": "normal", "reviewer": "a"}, 400),
    ):
        response = client.post(reviews, json=body)
        assert response.status_code == status, body
    assert len(client.get(reviews).json()["records"]) == 3


def test_reviews_only_anomaly(client: TestClient) -> None:
    """
    異常の度合いでない結果の待ち行列と記録を断ることを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント

    Returns
    ----------
    None
    """
    # 見本の分類の一式の結果
    first = client.post("/api/samples").json()[0]
    other = client.post(
        f"/api/bundles/{first['id']}/predict",
        json={"source": first["sample_input"]}
    ).json()
    base = f"/api/results/{other['id']}"
    for response in (
        client.get(f"{base}/queue"), client.get(f"{base}/reviews"),
        client.post(f"{base}/reviews", json={
            "row": 0, "judgment": "normal", "reviewer": "a"
        }),
    ):
        assert response.status_code == 400
        assert "異常の度合い" in response.json()["detail"]
