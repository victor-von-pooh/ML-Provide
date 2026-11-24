import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import ml_inference
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from test_bundle_v3 import register

from ml_provide.app import create_app

# テストで使うトークン
TOKEN = "test-token"

# 数にできない値の行の理由 (ML-Inference の文)
NOT_NUMBER = "列 spend の値 '?' を数にできません"


def load_maker() -> ModuleType:
    """
    画面の通しのテストが使う, 群分けの実験と一式を作るスクリプトを読む関数

    Parameters
    ----------
    None

    Returns
    ----------
    module: ModuleType
        frontend/e2e/make_group.py (骨組みと同じ成果物を書く関数を持つ)
    """
    # 骨組みの成果物の書き方を 2 か所に持たないため, 同じスクリプトを使う
    path = Path(__file__).parents[2] / "frontend" / "e2e" / "make_group.py"
    spec = importlib.util.spec_from_file_location("make_group", path)
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
def group_files(tmp_path: Path) -> tuple[Path, Path]:
    """
    骨組みのプロジェクトの実験の中の群分けの一式と, 新しい行の表を作る
    関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    files: tuple[Path, Path]
        一式のディレクトリと, 新しい行の表 (数にできない行を 1 つ含む)
    """
    load_maker().main(tmp_path / "project", tmp_path / "tables")
    bundle = next((tmp_path / "project" / "outputs").glob("*/*/*/inference"))
    files = (bundle, tmp_path / "tables" / "customers.csv")

    return files


def test_group_bundle(
    client: TestClient, group_files: tuple[Path, Path]
) -> None:
    """
    群分けの一式を登録して推論し, 群の要約と群の安定性を結論に残し,
    行ごとの群を読めて群で絞れることを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    group_files: tuple[Path, Path]
        一式と新しい行の表

    Returns
    ----------
    None
    """
    # 版 9 の一式を登録すると, 説明に群の欄があり, 問題の設定を読まない
    path, table = group_files
    bundle = register(client, path)
    description = bundle["description"]
    assert (description["kind"], description["format_version"]) == (
        "group", 9
    )
    assert description["output"]["form"] == "cluster"
    assert description["group"]["columns"] == ["spend", "visits", "region"]
    state = client.get(f"/api/bundles/{bundle['id']}/problem").json()
    assert state["linked"] is None and state["hint"] is None

    # 結論は ML-Inference の群分けの結論のまま (読み直して数え直さない)
    response = client.post(
        f"/api/bundles/{bundle['id']}/predict", json={"source": str(table)}
    )
    assert response.status_code == 201, response.text
    result: dict[str, Any] = response.json()
    summary = result["summary"]
    assert (summary["rows"], summary["predicted"], summary["skipped"],
            summary["k"]) == (16, 15, 1, 3)
    groups = summary["groups"]
    assert [row["group"] for row in groups] == [0, 1, 2, "全体"]
    assert list(groups[0]) == ["group", "rows", "share", "stability",
                               "spend", "visits", "region"]
    assert [row["rows"] for row in groups] == [5, 5, 5, 15]
    assert groups[0]["region"] == "north (100 %)"
    assert summary["stability"]["ari"] == 0.97
    assert summary["stability"]["baseline"] == (
        "列の特徴をそのまま k-means で分けた群"
    )
    assert result["sentences"][0] == (
        "16 行のうち 15 行を 3 個の群に割り当てた (1 行は割り当てられず)"
    )
    assert "学習の群の安定性" in result["sentences"][2]
    assert result["overall_count"] == len(result["sentences"])
    assert (result["output_form"], result["column_output"]) == (
        "cluster", False
    )

    # ML-Inference の group_summary に同じ表を渡した行と同じ
    frame = ml_inference.predict(str(path), table)
    assert groups == ml_inference.group_summary(str(path), frame)

    # 結果に群を作る列と学習のときの群の要約を残す (開き直しても同じ)
    grouping = result["grouping"]
    assert grouping["k"] == 3
    assert grouping["columns"] == ["spend", "visits", "region"]
    assert [row["group"] for row in grouping["learned"]] == [
        0, 1, 2, "全体"
    ]
    assert grouping["learned"][0]["rows"] == 20
    again = client.get(f"/api/results/{result['id']}").json()
    assert again["summary"] == summary and again["grouping"] == grouping

    # 行ごとの群は整数で読み, 割り当てない行は欠損と理由
    rows = client.get(
        f"/api/results/{result['id']}/rows", params={"limit": 100}
    ).json()
    assert rows["total"] == 16
    assert rows["rows"][0]["group"] == 0
    assert isinstance(rows["rows"][0]["distance"], float)
    last = rows["rows"][-1]
    assert (last["status"], last["group"], last["reason"]) == (
        "skipped", None, NOT_NUMBER
    )

    # 群の番号で絞る
    one = client.get(
        f"/api/results/{result['id']}/rows", params={"group": 1}
    ).json()
    assert one["total"] == 5
    assert {row["group"] for row in one["rows"]} == {1}
    assert {row["region"] for row in one["rows"]} == {"south"}


def test_group_refusals(
    tmp_path: Path, client: TestClient, group_files: tuple[Path, Path]
) -> None:
    """
    群分けの一式に結論の決め方, 意思決定, 並べるの欄を渡すと断り,
    群分けでない結果を群で絞ると断ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    client: TestClient
        クライアント
    group_files: tuple[Path, Path]
        一式と新しい行の表

    Returns
    ----------
    None
    """
    # 列の出力でない一式と同じく, 決め方と意思決定と並べるの欄は使えない
    path, table = group_files
    bundle = register(client, path)
    url = f"/api/bundles/{bundle['id']}/predict"
    for extra in ({"rules": {"min_confidence": 0.5}},
                  {"decision": {"type": "act_threshold"}},
                  {"forecast": True}, {"k": 3}):
        response = client.post(url, json={"source": str(table), **extra})
        assert response.status_code == 400, extra

    # 群を作る列の無い表は, 無い列を挙げて断る
    lacking = tmp_path / "lacking.csv"
    pd.read_csv(table).drop(columns=["region"]).to_csv(lacking, index=False)
    response = client.post(url, json={"source": str(lacking)})
    assert response.status_code == 400
    assert "region" in response.json()["detail"]

    # 群分けの結果だけを群で絞れる (結論の出し直しも断る)
    result = client.post(url, json={"source": str(table)}).json()
    response = client.post(f"/api/results/{result['id']}/conclude",
                           json={"rules": {}})
    assert response.status_code == 400
    first = client.post("/api/samples").json()[0]
    other = client.post(
        f"/api/bundles/{first['id']}/predict",
        json={"source": first["sample_input"]}
    ).json()
    response = client.get(f"/api/results/{other['id']}/rows",
                          params={"group": 0})
    assert response.status_code == 400
    assert response.json()["detail"] == "群で絞れるのは群分けの結果だけです"
