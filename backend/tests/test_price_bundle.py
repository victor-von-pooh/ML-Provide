import importlib.util
from decimal import Decimal
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
    画面の通しのテストが使う, 値段を変える判断の実験と一式を作るスクリプト
    を読む関数

    Parameters
    ----------
    None

    Returns
    ----------
    module: ModuleType
        frontend/e2e/make_price.py (骨組みと同じ形の見本から一式を作る関数
        を持つ)
    """
    # 一式と新しい商品の表の作り方を 2 か所に持たないための, 同じスクリプト
    path = Path(__file__).parents[2] / "frontend" / "e2e" / "make_price.py"
    spec = importlib.util.spec_from_file_location("make_price", path)
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
def price_files(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    値段の一式と, 値段を選ぶ新しい商品の表を作る関数

    Parameters
    ----------
    tmp_path_factory: pytest.TempPathFactory
        pytest が用意する一時ディレクトリの作り手

    Returns
    ----------
    work: Path
        bundle (一式), products.csv (値段の読めない行を 1 つ含む 7 行),
        sales.csv (売れた数の列を持つ記録) を持つディレクトリ
    """
    # このファイルのテストで 1 度だけ
    work = tmp_path_factory.mktemp("price")
    load_maker().main(work)

    return work


def within(change: float, limit: float) -> bool:
    """
    選んだ割合が変える幅の上限の中かを 10 進の表記どおりに比べる関数

    Parameters
    ----------
    change: float
        選んだ割合
    limit: float
        変える幅の上限

    Returns
    ----------
    inside: bool
        割合の絶対値が上限以下なら真
    """
    # 0.1 と -0.1 を上限 0.1 の中とする (ML-Inference と同じ比べ方)
    inside = abs(Decimal(repr(change))) <= Decimal(repr(limit))

    return inside


def test_price_bundle(client: TestClient, price_files: Path) -> None:
    """
    値段の一式を登録して推論し, 値段の結論と値段の判断と弾力性の分布を
    残し, 選んだ値段を利益の差の大きい順に読め, 上限を変えると選ぶ値段が
    上限の中に収まることを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    price_files: Path
        一式と新しい商品の表を持つディレクトリ

    Returns
    ----------
    None
    """
    # 版 13 の一式の登録. 説明の値段の欄と, 読まない問題の設定
    bundle = register(client, price_files / "bundle")
    description = bundle["description"]
    assert (description["kind"], description["format_version"]) == (
        "price", 13
    )
    assert description["column_output"] is False
    assert description["decision"] == {
        "type": "price", "candidates": [-0.1, 0.0, 0.1, 0.2],
        "max_change": 0.1
    }
    price = description["price"]
    assert (price["price"], price["cost"], price["product"]) == (
        "price", "unit_cost", "product_id"
    )
    assert set(price["evaluations"]) == {"valid", "test"}
    state = client.get(f"/api/bundles/{bundle['id']}/problem").json()
    assert state["linked"] is None and state["hint"] is None

    # 既定の値段の判断での推論. 値段の結論と値段の判断の結論はそのまま
    url = f"/api/bundles/{bundle['id']}/predict"
    table = str(price_files / "products.csv")
    response = client.post(url, json={"source": table})
    assert response.status_code == 201, response.text
    result = response.json()
    summary = result["summary"]
    assert (summary["rows"], summary["predicted"], summary["skipped"]) == (
        7, 6, 1
    )
    assert (summary["id"], summary["product"], summary["cost"]) == (
        "record_id", "product_id", "unit_cost"
    )
    assert summary["elasticity"]["counted"] == 6
    assert set(summary["evaluations"]) == {"valid", "test"}
    assert result["kind"] == "price" and result["column_output"] is False
    assert "売れた数を今の値段" in result["sentences"][0]
    found = result["price"]
    assert found["step"] == 0.01
    assert found["elasticity"]["counted"] == 6
    assert found["elasticity"]["mean"] == pytest.approx(
        summary["elasticity"]["mean"]
    )
    assert found["elasticity"]["min"] <= found["elasticity"]["p50"] <= (
        found["elasticity"]["max"]
    )
    decided = result["decision"]
    assert decided["decision"]["type"] == "price"
    totals = decided["summary"]
    assert totals["rows"] == 6 and totals["max_change"] == 0.1
    assert totals["total"] >= totals["current_total"]
    assert isinstance(totals["limited"], int)
    assert any("試しに値段を変えた記録" in note for note in decided["notes"])

    # 選んだ値段は見込みの利益の差の大きい順で, 上限の中
    base = f"/api/results/{result['id']}/prices"
    prices = client.get(base).json()
    assert (prices["id"], prices["product"], prices["total"]) == (
        "record_id", "product_id", 6
    )
    assert prices["candidates"] == [-0.1, 0.0, 0.1, 0.2]
    assert prices["max_change"] == 0.1
    rows = prices["rows"]
    gains = [row["profit"] - row["current_profit"] for row in rows]
    assert gains == sorted(gains, reverse=True)
    assert all(within(row["change"], 0.1) for row in rows)
    assert "p99" not in {row["product_id"] for row in rows}
    for row in rows:
        assert row["price"] == pytest.approx(
            row["current_price"] * (1 + row["change"])
        )
        assert row["profit"] == pytest.approx(
            (row["price"] - row["cost"]) * row["price_units"]
        )
    assert sum(row["profit"] for row in rows) == pytest.approx(
        totals["total"]
    )
    page = client.get(f"{base}?offset=2&limit=3").json()
    assert [row["row"] for row in page["rows"]] == [
        row["row"] for row in rows[2:5]
    ]
    assert page["total"] == 6 and page["offset"] == 2

    # 読み直した結果の弾力性の分布
    again = client.get(f"/api/results/{result['id']}").json()
    assert again["price"] == result["price"]

    # 上限を広げると, 上限の中で選び, 上限を超える候補の方が良い行は減る
    response = client.post(url, json={
        "source": table, "decision": {
            "type": "price", "candidates": [-0.1, 0, 0.1, 0.2],
            "max_change": 0.2
        }
    })
    assert response.status_code == 201, response.text
    wide = response.json()
    assert wide["decision"]["summary"]["limited"] == 0
    assert wide["decision"]["summary"]["total"] >= totals["total"]
    rows = client.get(f"/api/results/{wide['id']}/prices").json()["rows"]
    assert all(within(row["change"], 0.2) for row in rows)

    # 上限 0 は今の値段のまま
    response = client.post(url, json={
        "source": table, "decision": {
            "type": "price", "candidates": [-0.1, 0, 0.1], "max_change": 0
        }
    })
    assert response.status_code == 201, response.text
    still = response.json()
    assert still["decision"]["summary"]["changed"] == 0
    rows = client.get(f"/api/results/{still['id']}/prices").json()["rows"]
    assert {row["change"] for row in rows} == {0}

    # 値段を決めない結果の選んだ値段は空
    response = client.post(url, json={"source": table, "decision": {}})
    assert response.status_code == 201, response.text
    plain = response.json()
    assert plain["decision"] is None
    empty = client.get(f"/api/results/{plain['id']}/prices").json()
    assert empty["rows"] == [] and empty["max_change"] is None
    assert empty["total"] == 0


def test_price_measured(client: TestClient, price_files: Path) -> None:
    """
    売れた数の列を持つ表を渡すと, 誤差を数え直すことを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    price_files: Path
        一式と売れた数の列を持つ記録を持つディレクトリ

    Returns
    ----------
    None
    """
    # 記録の値段での予測の誤差
    bundle = register(client, price_files / "bundle")
    response = client.post(
        f"/api/bundles/{bundle['id']}/predict",
        json={"source": str(price_files / "sales.csv"), "decision": {}}
    )
    assert response.status_code == 201, response.text
    measured = response.json()["summary"]["measured"]
    rows = len(pd.read_csv(price_files / "sales.csv"))
    assert measured["rows"] == rows and measured["excluded"] == 0
    assert measured["mae"] <= measured["rmse"]


def test_price_refusals(client: TestClient, price_files: Path) -> None:
    """
    値段の一式に, 値段の判断でない意思決定, 形の違う候補と上限, 多すぎる
    候補, 結論の決め方, 何段も先の予測, 並べるの欄を渡すと断り, 列の出力と
    異常の度合いだけの機能も断ることを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    price_files: Path
        一式と新しい商品の表を持つディレクトリ

    Returns
    ----------
    None
    """
    # 値段の判断の他の型と項目は ML-Inference での断り
    bundle = register(client, price_files / "bundle")
    url = f"/api/bundles/{bundle['id']}/predict"
    table = str(price_files / "products.csv")
    good = {"type": "price", "candidates": [-0.1, 0, 0.1]}
    many = [round(-0.5 + 0.05 * i, 2) for i in range(21)]
    for extra in ({"rules": {"min_confidence": 0.5}}, {"forecast": True},
                  {"decision": {"type": "treat", "budget": 3}},
                  {"decision": good},
                  {"decision": {**good, "max_change": -0.1}},
                  {"decision": {**good, "max_change": True}},
                  {"decision": {**good, "max_change": "0.1"}},
                  {"decision": {"type": "price", "candidates": [0.1],
                                "max_change": 0.1}},
                  {"decision": {"type": "price", "candidates": [0, -1],
                                "max_change": 1}},
                  {"decision": {"type": "price", "candidates": [0, 0.1, 0.1],
                                "max_change": 1}},
                  {"decision": {"type": "price", "candidates": many,
                                "max_change": 1}},
                  {"decision": {**good, "max_change": 0.1, "cost": 1}},
                  {"k": 3}):
        response = client.post(url, json={"source": table, **extra})
        assert response.status_code == 400, extra
    response = client.post(url, json={"source": table, "forecast": True})
    assert "値段を選ぶ判断" in response.json()["detail"]
    response = client.post(url, json={
        "source": table,
        "decision": {"type": "price", "candidates": many, "max_change": 1}
    })
    assert "20 個まで" in response.json()["detail"]

    # 候補の数の境界の 20 個は通る
    response = client.post(url, json={
        "source": table,
        "decision": {"type": "price", "candidates": many[:20],
                     "max_change": 1}
    })
    assert response.status_code == 201, response.text

    # 結論の出し直し, 決め直し, たどり直し, 人の確認, 配る行は使えない
    result = client.post(url, json={"source": table}).json()
    for name, body in (("conclude", {"rules": {}}),
                       ("decide", {"decision": None}), ("backtest", None)):
        response = client.post(f"/api/results/{result['id']}/{name}",
                               json=body)
        assert response.status_code == 400, name
    for name in ("queue", "treated"):
        response = client.get(f"/api/results/{result['id']}/{name}")
        assert response.status_code == 400, name
    assert client.get(
        f"/api/results/{result['id']}/prices?limit=501"
    ).status_code == 422

    # 英数字とハイフンの外の結果の id と, 無い結果の id
    for name in ("..", "a_b", "x.y"):
        response = client.get(f"/api/results/{name}/prices")
        assert response.status_code in (404, 400), name
    response = client.get("/api/results/missing-1/prices")
    assert response.status_code == 404


def test_prices_only_price(client: TestClient) -> None:
    """
    値段でない結果の選んだ値段を断ることを確かめる関数

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
    response = client.get(f"/api/results/{other['id']}/prices")
    assert response.status_code == 400
    assert "値段の一式" in response.json()["detail"]
