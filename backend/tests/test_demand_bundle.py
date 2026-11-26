import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from fastapi.testclient import TestClient
from test_bundle_v3 import register

from ml_provide.app import create_app

# テストで使うトークン
TOKEN = "test-token"

# 見本の一式の学んだ水準 (ML-Inference のテストの見本)
LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]


def load_maker() -> ModuleType:
    """
    画面の通しのテストが使う, 時間をまたぐ量の判断の実験と一式を作る
    スクリプトを読む関数

    Parameters
    ----------
    None

    Returns
    ----------
    module: ModuleType
        frontend/e2e/make_demand.py (骨組みと同じ形の見本から一式を作る
        関数を持つ)
    """
    # 一式と新しい記録の表の作り方を 2 か所に持たないための, 同じスクリプト
    path = Path(__file__).parents[2] / "frontend" / "e2e" / "make_demand.py"
    spec = importlib.util.spec_from_file_location("make_demand", path)
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


@pytest.fixture(scope="module")
def demand_files(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    需要の一式と, 発注の数を決める新しい日ごとの記録の表を作る関数

    Parameters
    ----------
    tmp_path_factory: pytest.TempPathFactory
        pytest が用意する一時ディレクトリの作り手

    Returns
    ----------
    work: Path
        bundle (一式), daily.csv (在庫の数の読めない商品を 1 つ含む 5 商品
        の記録) を持つディレクトリ
    """
    # このファイルのテストで 1 度だけ
    work = tmp_path_factory.mktemp("demand")
    load_maker().main(work)

    return work


def order_of(
    client: TestClient, result_id: str
) -> dict[str, float]:
    """
    結果の商品ごとの発注の数を返す関数

    Parameters
    ----------
    client: TestClient
        クライアント
    result_id: str
        需要の結果の id

    Returns
    ----------
    orders: dict[str, float]
        商品の id ごとの発注の数
    """
    # 発注の数を決めた全ての行
    rows = client.get(f"/api/results/{result_id}/orders").json()["rows"]
    orders = {str(row["product_id"]): float(row["order"]) for row in rows}

    return orders


def test_demand_bundle(client: TestClient, demand_files: Path) -> None:
    """
    需要の一式を登録して推論し, 需要の結論と発注の判断と分位点の要約を
    残し, 発注の数を大きい順に読め, 損と上限を変えると発注の数が変わる
    ことを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    demand_files: Path
        一式と新しい記録の表を持つディレクトリ

    Returns
    ----------
    None
    """
    # 版 14 の一式の登録. 説明の需要の欄と, 読まない問題の設定
    bundle = register(client, demand_files / "bundle")
    description = bundle["description"]
    assert (description["kind"], description["format_version"]) == (
        "demand", 14
    )
    assert description["column_output"] is False
    assert description["decision"] == {
        "type": "order", "shortage_cost": 3, "holding_cost": 1,
        "max_order": 60
    }
    demand = description["demand"]
    assert (demand["product"], demand["date"], demand["stock"]) == (
        "product_id", "date", "stock"
    )
    assert demand["levels"] == LEVELS and demand["horizon"] == 7
    assert set(demand["evaluations"]) == {"valid", "test"}
    state = client.get(f"/api/bundles/{bundle['id']}/problem").json()
    assert state["linked"] is None and state["hint"] is None

    # 既定の発注の判断での推論. 需要の結論と発注の判断の結論はそのまま
    url = f"/api/bundles/{bundle['id']}/predict"
    table = str(demand_files / "daily.csv")
    response = client.post(url, json={"source": table})
    assert response.status_code == 201, response.text
    result = response.json()
    summary = result["summary"]
    assert (summary["rows"], summary["predicted"], summary["skipped"]) == (
        5, 4, 1
    )
    assert (summary["product"], summary["stock"], summary["horizon"]) == (
        "product_id", "stock", 7
    )
    assert set(summary["evaluations"]) == {"valid", "test"}
    assert result["kind"] == "demand" and result["column_output"] is False
    assert "先の 7 日の需要の分位点を予測した" in result["sentences"][0]
    found = result["demand"]
    assert found["levels"] == LEVELS and found["horizon"] == 7
    assert (found["low"], found["high"]) == (0.1, 0.9)
    assert found["width"]["counted"] == 4
    assert 0 <= found["width"]["min"] <= found["width"]["max"]
    assert found["quantiles"] == sorted(found["quantiles"])
    assert found["quantiles"][2] == pytest.approx(
        summary["prediction"]["mean"]
    )
    decided = result["decision"]
    assert decided["decision"]["type"] == "order"
    totals = decided["summary"]
    assert totals["rows"] == 4 and totals["level"] == 0.75
    assert totals["max_order"] == 60 and totals["capped"] == 0
    assert any("τ = 0.75" in note for note in decided["notes"])

    # 発注の数は大きい順で, 予測できなかった商品は無い
    base = f"/api/results/{result['id']}/orders"
    orders = client.get(base).json()
    assert (orders["id"], orders["product"], orders["total"]) == (
        "record_id", "product_id", 4
    )
    assert orders["levels"] == LEVELS
    assert (orders["shortage_cost"], orders["holding_cost"]) == (3, 1)
    assert orders["max_order"] == 60
    rows = orders["rows"]
    counts = [row["order"] for row in rows]
    assert counts == sorted(counts, reverse=True)
    assert "p99" not in {row["product_id"] for row in rows}
    for row in rows:
        assert len(row["quantiles"]) == len(LEVELS)
        assert row["quantile"] == pytest.approx(row["quantiles"][3])
        assert float(row["order"]).is_integer() and row["order"] >= 0
        assert row["expected_cost"] >= 0
    assert sum(row["order"] for row in rows) == totals["total"]
    assert sum(row["expected_cost"] for row in rows) == pytest.approx(
        totals["expected_cost"]
    )
    page = client.get(f"{base}?offset=1&limit=2").json()
    assert [row["row"] for row in page["rows"]] == [
        row["row"] for row in rows[1:3]
    ]
    assert page["total"] == 4 and page["offset"] == 1

    # 読み直した結果の分位点の要約
    again = client.get(f"/api/results/{result['id']}").json()
    assert again["demand"] == result["demand"]
    assert again["summary"]["evaluations"] == summary["evaluations"]

    # 品切れの損を上げると水準が上がり, 発注の数は減らない
    response = client.post(url, json={"source": table, "decision": {
        "type": "order", "shortage_cost": 9, "holding_cost": 1,
        "max_order": 60
    }})
    assert response.status_code == 201, response.text
    high = response.json()
    assert high["decision"]["summary"]["level"] == 0.9
    before, after = order_of(client, result["id"]), order_of(
        client, high["id"]
    )
    assert all(after[key] >= before[key] for key in before)
    assert sum(after.values()) > sum(before.values())

    # 上限を 1 にすると, 上限を超える商品は 1 で切り, 印を付ける
    response = client.post(url, json={"source": table, "decision": {
        "type": "order", "shortage_cost": 9, "holding_cost": 1,
        "max_order": 1
    }})
    assert response.status_code == 201, response.text
    low = response.json()
    capped = [key for key, value in after.items() if value > 1]
    assert low["decision"]["summary"]["capped"] == len(capped) > 0
    rows = client.get(f"/api/results/{low['id']}/orders").json()["rows"]
    assert all(row["order"] <= 1 for row in rows)
    assert {str(row["product_id"]) for row in rows if row["capped"]} == set(
        capped
    )

    # 上限 0 は発注しない
    response = client.post(url, json={"source": table, "decision": {
        "type": "order", "shortage_cost": 3, "holding_cost": 1,
        "max_order": 0
    }})
    assert response.status_code == 201, response.text
    assert response.json()["decision"]["summary"]["total"] == 0

    # 発注の数を決めない結果の発注の数は空
    response = client.post(url, json={"source": table, "decision": {}})
    assert response.status_code == 201, response.text
    plain = response.json()
    assert plain["decision"] is None
    empty = client.get(f"/api/results/{plain['id']}/orders").json()
    assert empty["rows"] == [] and empty["max_order"] is None
    assert empty["total"] == 0
    assert plain["demand"]["width"]["counted"] == 4


def test_demand_refusals(client: TestClient, demand_files: Path) -> None:
    """
    需要の一式に, 発注の判断でない意思決定, 形の違う損と上限, 結論の
    決め方, 何段も先の予測, 並べるの欄を渡すと断り, 列の出力と異常の
    度合いと値段だけの機能も断ることを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    demand_files: Path
        一式と新しい記録の表を持つディレクトリ

    Returns
    ----------
    None
    """
    # 発注の判断の他の型と項目は ML-Inference での断り
    bundle = register(client, demand_files / "bundle")
    url = f"/api/bundles/{bundle['id']}/predict"
    table = str(demand_files / "daily.csv")
    good = {"type": "order", "shortage_cost": 3, "holding_cost": 1,
            "max_order": 60}
    for extra in ({"rules": {"min_confidence": 0.5}}, {"forecast": True},
                  {"decision": {"type": "treat", "budget": 3}},
                  {"decision": {**good, "shortage_cost": 0}},
                  {"decision": {**good, "holding_cost": -1}},
                  {"decision": {**good, "shortage_cost": True}},
                  {"decision": {**good, "max_order": -1}},
                  {"decision": {**good, "max_order": "60"}},
                  {"decision": {k: v for k, v in good.items()
                                if k != "max_order"}},
                  {"decision": {**good, "budget": 1}},
                  {"k": 3}):
        response = client.post(url, json={"source": table, **extra})
        assert response.status_code == 400, extra
    response = client.post(url, json={"source": table, "forecast": True})
    assert "発注する判断" in response.json()["detail"]

    # 使えない欄 (結論の出し直し, 決め直し, たどり直し, 人の確認, 配る行,
    # 選んだ値段)
    result = client.post(url, json={"source": table}).json()
    for name, body in (("conclude", {"rules": {}}),
                       ("decide", {"decision": None}), ("backtest", None)):
        response = client.post(f"/api/results/{result['id']}/{name}",
                               json=body)
        assert response.status_code == 400, name
    for name in ("queue", "treated", "prices"):
        response = client.get(f"/api/results/{result['id']}/{name}")
        assert response.status_code == 400, name
    assert client.get(
        f"/api/results/{result['id']}/orders?limit=501"
    ).status_code == 422
    response = client.get("/api/results/missing-1/orders")
    assert response.status_code == 404


def test_orders_only_demand(client: TestClient) -> None:
    """
    需要でない結果の発注の数を断ることを確かめる関数

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
    response = client.get(f"/api/results/{other['id']}/orders")
    assert response.status_code == 400
    assert "需要の一式" in response.json()["detail"]
