import os
import shutil
from pathlib import Path

import ml_quest
import pytest
from fastapi.testclient import TestClient
from ml_quest.examples import example_problem

from ml_provide.app import create_app

# テストで使うトークン
TOKEN = "test-token"


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """
    一時ディレクトリを置き場にし, ML-Pathfinder の置き場も一時ディレクトリ
    にしたアプリのクライアントを作る関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        環境変数の差し替え

    Returns
    ----------
    client: TestClient
        トークン付きのクライアント
    """
    monkeypatch.setenv("ML_PATHFINDER_HOME", str(tmp_path / "pathfinder"))
    app = create_app(
        data_dir=tmp_path / "home", token=TOKEN, allowed_hosts=["testserver"]
    )
    client = TestClient(app, headers={"x-provide-token": TOKEN})

    return client


def test_list_experiments(client: TestClient, tmp_path: Path) -> None:
    """
    問題の設定に記録したプロジェクトの, 一式を持つ実験だけを新しい順に
    出し, 登録済みの一式を見分けることを確かめる関数

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
    # 置き場が無ければ空
    listing = client.get("/api/experiments").json()
    assert listing["experiments"] == [] and listing["projects"] == 0
    assert listing["problems"] == str(
        (tmp_path / "pathfinder" / "problems").resolve()
    )

    # 記録したプロジェクトに, 一式のある実験 2 つと一式の無い実験 1 つ
    added = client.post("/api/samples").json()
    project = tmp_path / "project"
    runs = project / "outputs" / "sales"
    old = runs / "2026-01-01" / "10-00-00" / "inference"
    new = runs / "2026-01-02" / "09-00-00" / "inference"
    for place in (old, new):
        shutil.copytree(added[0]["path"], place)
    (runs / "2026-01-03" / "08-00-00").mkdir(parents=True)
    os.utime(old / "inference.yaml", (1_700_000_000, 1_700_000_000))
    problems = tmp_path / "pathfinder" / "problems"
    problem = example_problem("総菜の作る量")
    ml_quest.save_problem(problem, problems / "a")
    ml_quest.save_problem(
        {**problem, "project": {"path": str(project.resolve())}},
        problems / "b"
    )
    ml_quest.save_problem(
        {**problem, "project": {"path": str(tmp_path / "gone")}},
        problems / "c"
    )
    (problems / "d").mkdir()
    (problems / "d" / "problem.yaml").write_text("[", encoding="utf-8")

    listing = client.get("/api/experiments").json()
    assert listing["projects"] == 1 and not listing["truncated"]
    rows = listing["experiments"]
    assert [row["experiment"] for row in rows] == [
        "sales/2026-01-02/09-00-00", "sales/2026-01-01/10-00-00",
    ]
    assert rows[0]["path"] == str(new.resolve())
    assert rows[0]["problem"] == problem["name"]
    assert rows[0]["project"] == str(project.resolve())
    assert rows[0]["bundle_id"] is None

    # 一覧の場所で登録すると, 登録済みとして見分ける
    bundle = client.post("/api/bundles", json={"path": rows[0]["path"]})
    assert bundle.status_code == 201
    rows = client.get("/api/experiments").json()["experiments"]
    assert rows[0]["bundle_id"] == bundle.json()["id"]
    assert rows[1]["bundle_id"] is None
