import shutil
from pathlib import Path
from typing import Any

import ml_quest
import pytest
from fastapi.testclient import TestClient
from ml_quest.examples import example_problem

from ml_provide.app import create_app

# テストで使うトークン
TOKEN = "test-token"


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
        data_dir=tmp_path / "home", token=TOKEN, allowed_hosts=["testserver"]
    )
    client = TestClient(app, headers={"x-provide-token": TOKEN})

    return client


def shop_problem() -> dict[str, Any]:
    """
    総菜の見本の問題の設定を, 日々の売上の見本の列 (日付, 店, 売上) に
    合わせる関数

    Parameters
    ----------
    None

    Returns
    ----------
    problem: dict[str, Any]
        店ごとに売上を予測し, 量を決める問題の設定
    """
    # 品と客数の無い表. 系列は店だけ
    problem = example_problem("総菜の作る量")
    for key, value in (
        ("information.columns", ["date", "shop", "sales"]),
        ("ml.series_by", ["shop"]),
        ("information.known", ["shop"]),
        ("information.unknown", None),
    ):
        problem = ml_quest.answer(problem, key, value)

    return problem


def project_bundle(
    client: TestClient, tmp_path: Path, source: str
) -> tuple[Path, dict[str, Any]]:
    """
    骨組みのプロジェクトのように, 問題の設定を持つディレクトリの実験の
    中に一式を写して登録する関数

    Parameters
    ----------
    client: TestClient
        クライアント
    tmp_path: Path
        一時ディレクトリ
    source: str
        写す一式の場所

    Returns
    ----------
    made: tuple[Path, dict[str, Any]]
        プロジェクトのディレクトリと, 登録した一式
    """
    # プロジェクト/problem.yaml と プロジェクト/experiments/run/bundle
    project = tmp_path / "project"
    ml_quest.save_problem(shop_problem(), project)
    place = project / "experiments" / "run" / "bundle"
    shutil.copytree(source, place)
    bundle = client.post("/api/bundles", json={"path": str(place)}).json()
    made = (project, bundle)

    return made


def test_link_and_predict(client: TestClient, tmp_path: Path) -> None:
    """
    一式の場所から問題の設定の候補を見つけ, つないで読んだ損得と結論の
    決め方で推論できることを確かめる関数

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
    # 見本の一式 (置き場の中) には候補が無い. プロジェクトの中なら候補
    added = client.post("/api/samples").json()
    daily = next(b for b in added if "日々の売上" in b["name"])
    state = client.get(f"/api/bundles/{daily['id']}/problem").json()
    assert state == {"path": None, "linked": None, "error": None,
                     "hint": None}
    project, bundle = project_bundle(client, tmp_path, daily["path"])
    url = f"/api/bundles/{bundle['id']}/problem"
    state = client.get(url).json()
    assert state["hint"] == str((project / "problem.yaml").resolve())

    # つなぐと, 問題の設定の損得と, 一式の既定値に重ねた結論の決め方
    state = client.put(url, json={"path": state["hint"]}).json()
    linked = state["linked"]
    assert linked["decision"] == {"type": "quantity", "under_cost": 200,
                                  "over_cost": 150}
    assert linked["conclusion"] == {"group_by": "shop"}
    assert linked["basis"] == "欠品は粗利 200 円, 廃棄は原価 150 円"
    assert "量を決める" in linked["text"] and linked["issues"] == []
    assert client.get("/api/bundles").json()[0]["problem"] == str(
        (project / "problem.yaml").resolve()
    )

    # 読んだ値で推論できる
    result = client.post(f"/api/bundles/{bundle['id']}/predict", json={
        "source": daily["sample_past"], "decision": linked["decision"],
        "rules": linked["conclusion"],
    })
    assert result.status_code == 201, result.text
    assert result.json()["decision"]["decision"]["under_cost"] == 200

    # 問題の設定を直すと, 次に読むときから使う (中身の印も変わる)
    problem = ml_quest.answer(
        ml_quest.load_problem(project), "action.decision",
        {"type": "quantity", "under_cost": 300, "over_cost": 150},
    )
    ml_quest.save_problem(problem, project)
    again = client.get(url).json()["linked"]
    assert again["decision"]["under_cost"] == 300
    assert again["version"] != linked["version"]

    # 外す
    state = client.put(url, json={"path": None}).json()
    assert state["path"] is None and state["hint"] is not None
    assert "problem" not in client.get("/api/bundles").json()[0]


def test_link_problems(client: TestClient, tmp_path: Path) -> None:
    """
    一式と食い違う問題の設定, 推論に渡せない問題の設定, 置き換わった
    問題の設定の扱いを確かめる関数

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
    # 来店数の回帰の一式に, 売上の量の問題の設定をつなぐと目的の列が違う
    added = client.post("/api/samples").json()
    visitors = next(b for b in added if "来店数" in b["name"])
    churn = next(b for b in added if "解約" in b["name"])
    path = tmp_path / "shop"
    ml_quest.save_problem(shop_problem(), path)
    url = f"/api/bundles/{visitors['id']}/problem"
    linked = client.put(url, json={"path": str(path)}).json()["linked"]
    assert any("目的の列" in text for text in linked["issues"])

    # まとめる列 (総菜の店の列 store) が学習のデータに無い
    daily = next(b for b in added if "日々の売上" in b["name"])
    sozai = tmp_path / "sozai"
    ml_quest.save_problem(example_problem("総菜の作る量"), sozai)
    linked = client.put(f"/api/bundles/{daily['id']}/problem", json={
        "path": str(sozai)
    }).json()["linked"]
    assert any("まとめる列 store" in text for text in linked["issues"])

    # 場所として読めない文字は 422
    assert client.put(url, json={"path": "/tmp/a\x00b"}).status_code == 422

    # 分類の一式に量の意思決定は合わない
    other = f"/api/bundles/{churn['id']}/problem"
    linked = client.put(other, json={"path": str(path)}).json()["linked"]
    assert any(text.startswith("意思決定:") for text in linked["issues"])

    # 機械学習を使わない問題の設定と, 無い場所は断る
    plain = tmp_path / "plain"
    ml_quest.save_problem(
        ml_quest.answer(example_problem("総菜の作る量"), "ml.use", False),
        plain,
    )
    refused = client.put(url, json={"path": str(plain)})
    assert refused.status_code == 422
    assert "推論に渡せません" in refused.json()["detail"]
    assert client.put(url, json={
        "path": str(tmp_path / "none")
    }).status_code == 404

    # 別の問題の設定に置き換わると理由を出し, つなぎは残す (作った時刻
    # で見分ける)
    replaced = dict(shop_problem(), created_at="2020-01-01T00:00:00+09:00")
    ml_quest.save_problem(replaced, path)
    state = client.get(url).json()
    assert state["linked"] is None and "置き換わって" in state["error"]
    assert state["path"] == str((path / "problem.yaml").resolve())
