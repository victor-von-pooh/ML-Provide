import importlib.util
import shutil
from pathlib import Path
from types import ModuleType

import ml_inference
import ml_quest
import pytest
from fastapi.testclient import TestClient
from ml_quest import example_problem
from test_bundle_v3 import register

from ml_provide.app import create_app

# テストで使うトークン
TOKEN = "test-token"


def load_maker() -> ModuleType:
    """
    画面の通しのテストが使う, 効果の見積もりの実験と一式を作るスクリプトを
    読む関数

    Parameters
    ----------
    None

    Returns
    ----------
    module: ModuleType
        frontend/e2e/make_uplift.py (骨組みと同じ成果物を書く関数を持つ)
    """
    # 骨組みの成果物の書き方を 2 か所に持たないための, 同じスクリプト
    path = Path(__file__).parents[2] / "frontend" / "e2e" / "make_uplift.py"
    spec = importlib.util.spec_from_file_location("make_uplift", path)
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
def uplift_files(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    効果の一式と, 効果を見積もる新しい顧客の表を作る関数

    Parameters
    ----------
    tmp_path_factory: pytest.TempPathFactory
        pytest が用意する一時ディレクトリの作り手

    Returns
    ----------
    work: Path
        bundle (一式), customers.csv (前処理の後に欠ける行を 1 つ含む
        40 行), trial.csv (処置と結果の列を持つ 120 行) を持つディレクトリ
    """
    # 学習は数秒かかるので, このファイルのテストで 1 度だけ
    work = tmp_path_factory.mktemp("uplift")
    load_maker().main(work)

    return work


def test_uplift_bundle(client: TestClient, uplift_files: Path) -> None:
    """
    効果の一式を登録して推論し, 効果の結論と配る判断と Qini の曲線を残し,
    配る行を効果の高い順に読めることを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    uplift_files: Path
        一式と新しい顧客の表を持つディレクトリ

    Returns
    ----------
    None
    """
    # 版 12 の一式の登録. 説明の効果の欄と, 読まない問題の設定
    bundle = register(client, uplift_files / "bundle")
    description = bundle["description"]
    assert (description["kind"], description["format_version"]) == (
        "uplift", 12
    )
    assert description["column_output"] is False
    assert description["decision"] == {"type": "treat", "budget": 8}
    uplift = description["uplift"]
    assert (uplift["outcome"], uplift["k"], uplift["id"]) == (
        "binary", 0.3, "customer_id"
    )
    assert len(uplift["curves"]["valid"]["fraction"]) == 101
    state = client.get(f"/api/bundles/{bundle['id']}/problem").json()
    assert state["linked"] is None and state["hint"] is None

    # 既定の配る判断での推論. 効果の結論と配る判断の結論はそのまま
    url = f"/api/bundles/{bundle['id']}/predict"
    table = str(uplift_files / "customers.csv")
    response = client.post(url, json={"source": table})
    assert response.status_code == 201, response.text
    result = response.json()
    summary = result["summary"]
    assert (summary["rows"], summary["predicted"], summary["skipped"]) == (
        40, 39, 1
    )
    assert summary["id"] == "customer_id" and summary["outcome"] == "binary"
    assert summary["top_k"]["k"] == 0.3
    assert summary["top_k"]["rows"] == int(0.3 * 39)
    assert len(summary["top"]) == 10 and summary["measured"] is None
    assert set(summary["evaluations"]) == {"valid", "test"}
    assert result["kind"] == "uplift" and result["column_output"] is False
    assert "効果 (処置したときの結果 − しないときの結果" in result[
        "sentences"
    ][0]
    assert result["uplift"]["outcome"] == "binary"
    assert result["uplift"]["k"] == 0.3
    assert set(result["uplift"]["curves"]) == {"valid", "test"}
    decided = result["decision"]
    assert decided["decision"] == {"type": "treat", "budget": 8}
    assert decided["summary"]["acted"] == 8
    assert any("無作為" in note for note in decided["notes"])

    # 配る行は効果の高い順 (効果の上位と同じ行)
    base = f"/api/results/{result['id']}/treated"
    treated = client.get(base).json()
    assert (treated["id"], treated["budget"], treated["total"]) == (
        "customer_id", 8, 8
    )
    rows = treated["rows"]
    assert [row["rank"] for row in rows] == list(range(1, 9))
    assert [row["row"] for row in rows] == [
        item["row"] for item in summary["top"][:8]
    ]
    assert rows[0]["id"] == summary["top"][0]["id"]
    assert all(row["effect"] > 0 for row in rows)
    assert rows[0]["effect"] == pytest.approx(
        rows[0]["treated"] - rows[0]["control"]
    )
    page = client.get(f"{base}?offset=2&limit=3").json()
    assert [row["rank"] for row in page["rows"]] == [3, 4, 5]
    assert page["total"] == 8 and page["offset"] == 2

    # 読み直した結果の曲線
    again = client.get(f"/api/results/{result['id']}").json()
    assert again["uplift"] == result["uplift"]

    # 一式の Qini の曲線の値のままの写し
    found = ml_inference.describe_bundle(
        ml_inference.load_bundle(uplift_files / "bundle")
    )["uplift"]
    assert result["uplift"]["curves"] == found["curves"]

    # 上限を変えた後の配る行
    response = client.post(url, json={
        "source": table, "decision": {"type": "treat", "budget": 3}
    })
    assert response.status_code == 201, response.text
    other = response.json()
    assert other["decision"]["summary"]["acted"] == 3
    fewer = client.get(f"/api/results/{other['id']}/treated").json()
    assert [row["row"] for row in fewer["rows"]] == [
        row["row"] for row in rows[:3]
    ]
    assert fewer["budget"] == 3

    # 上限が効果の正の行より多い場合の配る行
    response = client.post(url, json={
        "source": table, "decision": {"type": "treat", "budget": 1000}
    })
    assert response.status_code == 201, response.text
    every = response.json()
    positive = round(
        every["summary"]["distribution"]["positive"] * 39
    )
    assert every["decision"]["summary"]["acted"] == positive
    assert every["decision"]["summary"]["nonpositive"] == 39 - positive
    found = client.get(f"/api/results/{every['id']}/treated").json()
    assert found["total"] == positive and len(found["rows"]) == positive

    # 配らない結果の配る行は空
    response = client.post(url, json={"source": table, "decision": {}})
    assert response.status_code == 201, response.text
    plain = response.json()
    assert plain["decision"] is None
    empty = client.get(f"/api/results/{plain['id']}/treated").json()
    assert empty["rows"] == [] and empty["budget"] is None
    assert empty["total"] == 0


def test_uplift_measured(client: TestClient, uplift_files: Path) -> None:
    """
    処置と結果の列を持つ表を渡すと, Qini の係数と上位 K 割の効果の差を
    数え直すことを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    uplift_files: Path
        一式と試験の記録を持つディレクトリ

    Returns
    ----------
    None
    """
    # 無作為に分けた試験の記録での数え直し
    bundle = register(client, uplift_files / "bundle")
    response = client.post(
        f"/api/bundles/{bundle['id']}/predict",
        json={"source": str(uplift_files / "trial.csv")}
    )
    assert response.status_code == 201, response.text
    measured = response.json()["summary"]["measured"]
    assert measured["rows"] == 120 and measured["k"] == 0.3
    assert measured["treated"] + measured["control"] == 120
    assert isinstance(measured["qini"], float)


def test_uplift_refusals(client: TestClient, uplift_files: Path) -> None:
    """
    効果の一式に, 配る判断でない意思決定, 結論の決め方, 何段も先の予測,
    並べるの欄を渡すと断り, 列の出力と異常の度合いだけの機能も断ることを
    確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    uplift_files: Path
        一式と新しい顧客の表を持つディレクトリ

    Returns
    ----------
    None
    """
    # 配る判断の他の型と項目は ML-Inference での断り
    bundle = register(client, uplift_files / "bundle")
    url = f"/api/bundles/{bundle['id']}/predict"
    table = str(uplift_files / "customers.csv")
    for extra in ({"rules": {"min_confidence": 0.5}}, {"forecast": True},
                  {"decision": {"type": "triage", "capacity": 3}},
                  {"decision": {"type": "treat", "budget": 0}},
                  {"decision": {"type": "treat", "budget": 2.5}},
                  {"decision": {"type": "treat", "budget": "3"}},
                  {"decision": {"type": "treat", "budget": True}},
                  {"decision": {"type": "treat", "budget": 100_001}},
                  {"decision": {"type": "treat", "budget": 1e21}},
                  {"decision": {"type": "treat", "budget": 3,
                                "cost": 1}},
                  {"k": 3}):
        response = client.post(url, json={"source": table, **extra})
        assert response.status_code == 400, extra
    response = client.post(url, json={"source": table, "forecast": True})
    assert "配る判断" in response.json()["detail"]
    response = client.post(url, json={
        "source": table, "decision": {"type": "treat", "budget": 100_001}
    })
    assert "100,000 件まで" in response.json()["detail"]

    # 上限の境界の 100,000 件は通る
    response = client.post(url, json={
        "source": table, "decision": {"type": "treat", "budget": 100_000}
    })
    assert response.status_code == 201, response.text
    assert response.json()["decision"]["summary"]["budget"] == 100_000

    # 結論の出し直し, 決め直し, たどり直しと人の確かめは使えない
    result = client.post(url, json={"source": table}).json()
    for name, body in (("conclude", {"rules": {}}),
                       ("decide", {"decision": None}), ("backtest", None)):
        response = client.post(f"/api/results/{result['id']}/{name}",
                               json=body)
        assert response.status_code == 400, name
    assert client.get(
        f"/api/results/{result['id']}/queue"
    ).status_code == 400
    assert client.get(
        f"/api/results/{result['id']}/treated?limit=501"
    ).status_code == 422

    # 英数字とハイフンの外の結果の id と, 無い結果の id
    for name in ("..", "a_b", "x.y"):
        response = client.get(f"/api/results/{name}/treated")
        assert response.status_code in (404, 400), name
    response = client.get("/api/results/missing-1/treated")
    assert response.status_code == 404


def test_treated_only_uplift(client: TestClient) -> None:
    """
    効果でない結果の配る行を断ることを確かめる関数

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
    response = client.get(f"/api/results/{other['id']}/treated")
    assert response.status_code == 400
    assert "効果の一式" in response.json()["detail"]


@pytest.fixture(scope="module")
def single_files(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    処置の列を入力の終わりに足した 1 つの予測 (s_learner) の効果の一式と,
    効果を見積もる新しい顧客の表を作る関数

    Parameters
    ----------
    tmp_path_factory: pytest.TempPathFactory
        pytest が用意する一時ディレクトリの作り手

    Returns
    ----------
    work: Path
        experiment (実験), bundle (版 18 の一式), customers.csv,
        trial.csv を持つディレクトリ
    """
    # 学習は数秒かかるので, このファイルのテストで 1 度だけ
    work = tmp_path_factory.mktemp("single")
    load_maker().main(work, "s_learner")

    return work


def test_s_learner_bundle(client: TestClient, single_files: Path) -> None:
    """
    S-learner の効果の一式 (版 18) を登録して推論し, 効果の上位から配る
    結論と, 結果に写す効果の決め方を確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    single_files: Path
        一式と新しい顧客の表を持つディレクトリ

    Returns
    ----------
    None
    """
    # 版 18 の一式の登録. 入力は見積もりに使う列と処置の列, 出力は 1 つ
    bundle = register(client, single_files / "bundle")
    description = bundle["description"]
    assert (description["kind"], description["format_version"]) == (
        "uplift", 18
    )
    assert description["column_output"] is False
    assert description["decision"] == {"type": "treat", "budget": 8}
    uplift = description["uplift"]
    assert uplift["learner"] == "s_learner"
    assert (uplift["input"], uplift["output"]) == ("x", "outcome")
    assert "treated" not in uplift and "control" not in uplift
    assert uplift["features"] == ["age", "visits"]
    assert uplift["check"]["agreement"] == 1.0
    assert set(uplift["curves"]) == {"valid", "test"}

    # 既定の配る判断での推論. 効果の上位から上限まで配る
    url = f"/api/bundles/{bundle['id']}/predict"
    response = client.post(
        url, json={"source": str(single_files / "customers.csv")}
    )
    assert response.status_code == 201, response.text
    result = response.json()
    summary = result["summary"]
    assert (summary["rows"], summary["predicted"], summary["skipped"]) == (
        40, 39, 1
    )
    assert summary["outcome"] == "binary" and len(summary["top"]) == 10
    assert result["uplift"]["learner"] == "s_learner"
    assert result["uplift"]["k"] == 0.3
    decided = result["decision"]
    positive = round(summary["distribution"]["positive"] * 39)
    assert decided["summary"]["acted"] == min(8, positive) > 0
    treated = client.get(f"/api/results/{result['id']}/treated").json()
    rows = treated["rows"]
    assert len(rows) == decided["summary"]["acted"]
    assert [row["row"] for row in rows] == [
        item["row"] for item in summary["top"][:len(rows)]
    ]
    assert all(row["effect"] > 0 for row in rows)
    assert rows[0]["effect"] == pytest.approx(
        rows[0]["treated"] - rows[0]["control"]
    )
    again = client.get(f"/api/results/{result['id']}").json()
    assert again["uplift"] == result["uplift"]

    # 処置と結果のある試験の記録での数え直し
    response = client.post(
        url, json={"source": str(single_files / "trial.csv")}
    )
    assert response.status_code == 201, response.text
    assert response.json()["summary"]["measured"]["rows"] == 120


def test_s_learner_experiment(
    tmp_path: Path, single_files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    学習の実験の中に作った S-learner の効果の一式も実験の一覧に出て,
    一覧から登録できることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    single_files: Path
        一式を持つディレクトリ
    monkeypatch: pytest.MonkeyPatch
        ML-Pathfinder の置き場の差し替え

    Returns
    ----------
    None
    """
    # 記録したプロジェクトの実験の中の, 版 18 の一式 (inference)
    project = tmp_path / "project"
    run = project / "outputs" / "coupons" / "2026-01-02" / "09-00-00"
    shutil.copytree(single_files / "bundle", run / "inference")
    ml_quest.save_problem(
        {**example_problem("解約の引き止め"),
         "project": {"path": str(project.resolve())}},
        tmp_path / "pathfinder" / "problems" / "a"
    )
    monkeypatch.setenv("ML_PATHFINDER_HOME", str(tmp_path / "pathfinder"))
    app = create_app(
        data_dir=tmp_path / "home", token=TOKEN, allowed_hosts=["testserver"],
        procedures=[]
    )
    client = TestClient(app, headers={"x-provide-token": TOKEN})

    # 一覧に出て, 場所で登録すると S-learner の効果の一式
    rows = client.get("/api/experiments").json()["experiments"]
    assert [row["experiment"] for row in rows] == [
        "coupons/2026-01-02/09-00-00"
    ]
    response = client.post("/api/bundles", json={"path": rows[0]["path"]})
    assert response.status_code == 201, response.text
    description = response.json()["description"]
    assert description["format_version"] == 18
    assert description["uplift"]["learner"] == "s_learner"
