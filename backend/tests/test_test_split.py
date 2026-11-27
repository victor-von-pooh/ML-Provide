import shutil
from pathlib import Path
from typing import Any

import ml_inference
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from fastapi.testclient import TestClient
from ml_eda import export_dataset, load_data
from ml_modelcontrol import build_model

from ml_provide.app import create_app
from ml_provide.splits import find_test_split

# テストで使うトークン
TOKEN = "test-token"

# 実験の場所 (出力先からの名前/日付/時刻)
RUN = Path("outputs") / "churn" / "2026-11-28" / "10-00-00"


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


def linear_spec(width: int, outputs: int) -> dict[str, Any]:
    """
    全結合の 1 層のモデルの定義を返す関数

    Parameters
    ----------
    width: int
        入力の列の数
    outputs: int
        出力の幅

    Returns
    ----------
    spec: dict[str, Any]
        ML-ModelControl の定義
    """
    spec = {
        "format_version": 1,
        "inputs": {"x": {"shape": [2, width]}},
        "nodes": {
            "head": {
                "unit": "nn.Linear",
                "params": {"in_features": width, "out_features": outputs},
                "inputs": "x",
            }
        },
        "outputs": {"logits": "head"},
    }

    return spec


def write_experiment(project: Path, dataset: str = "data/dataset") -> Path:
    """
    学習のプロジェクト, 書き出したデータセット, 実験と実験の中の一式を
    作る関数

    Parameters
    ----------
    project: Path
        プロジェクトのディレクトリ
    dataset: str = "data/dataset"
        実験の config.yaml の data.dataset に書く場所

    Returns
    ----------
    bundle: Path
        実験の中の一式 (inference)
    """
    # ML-Found のプロジェクトの印と, ワンホットと標準化のデータセット
    # (分割のファイルは前処理の後の列なので, 前処理を 2 度かけると止まる)
    (project / "configs" / "default").mkdir(parents=True)
    (project / "configs" / "default" / "train.yaml").write_text(
        "data: {}\n", encoding="utf-8"
    )
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "plan": rng.choice(["月", "年"], size=120),
        "a": rng.normal(size=120),
        "label": rng.choice(["続く", "解約"], size=120),
    })
    (project / "data").mkdir()
    frame.to_csv(project / "data" / "churn.csv", index=False)
    recipe = {
        "format_version": 1, "target": "label", "task": "classification",
        "steps": [{"op": "one_hot", "columns": ["plan"]},
                  {"op": "standardize", "columns": "auto"}],
        "split": {"method": "random", "valid": 0.2, "test": 0.25,
                  "seed": 0},
    }
    export_dataset(
        load_data(project / "data" / "churn.csv"), recipe,
        project / "data" / "dataset"
    )

    # 実験の記録と, 実験の中の一式
    run = project / RUN
    run.mkdir(parents=True)
    (run / "run.json").write_text("{}", encoding="utf-8")
    (run / "config.yaml").write_text(
        yaml.safe_dump({"data": {"dataset": dataset}}), encoding="utf-8"
    )
    spec = linear_spec(3, 2)
    torch.save(build_model(spec, seed=0).state_dict(), run / "w.pt")
    bundle = run / "inference"
    ml_inference.make_bundle(
        bundle, spec, project / "data" / "dataset", run / "w.pt",
        name="解約"
    )

    return bundle


def register(client: TestClient, bundle: Path) -> dict[str, Any]:
    """
    場所を指定して一式を登録する関数

    Parameters
    ----------
    client: TestClient
        クライアント
    bundle: Path
        一式のディレクトリ

    Returns
    ----------
    entry: dict[str, Any]
        登録と説明
    """
    response = client.post("/api/bundles", json={"path": str(bundle)})
    assert response.status_code == 201, response.text
    entry: dict[str, Any] = response.json()

    return entry


def test_predict_test_split(client: TestClient, tmp_path: Path) -> None:
    """
    実験から登録した一式で, 元のデータセットのテスト用の分割を見つけて
    推論し, 結果から測り直せることを確かめる関数

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
    # 一式から実験, config の data.dataset, dataset.yaml の順に見つける
    project = tmp_path / "project"
    entry = register(client, write_experiment(project))
    found = client.get(f"/api/bundles/{entry['id']}/test-split").json()
    dataset = (project / "data" / "dataset").resolve()
    assert found["split"] == "test" and found["reason"] is None
    assert found["dataset"] == str(dataset)
    assert found["experiment"] == str((project / RUN).resolve())
    assert found["rows"] == 30

    # 名前だけを渡して推論する (前処理を 2 度かけない)
    response = client.post(
        f"/api/bundles/{entry['id']}/predict",
        json={"split": "test", "decision": {}},
    )
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["summary"]["rows"] == 30
    assert result["source"] == (
        f"元のデータセットのテスト用の分割 ({dataset} の test)"
    )
    meta = yaml.safe_load(
        (tmp_path / "home" / "results" / result["id"] / "result.yaml")
        .read_text(encoding="utf-8")
    )
    assert meta["format_version"] == 5
    assert meta["split"] == {"dataset": str(dataset), "name": "test"}
    assert meta["source_path"] is None
    rows = client.get(f"/api/results/{result['id']}/rows",
                      params={"offset": 0, "limit": 50}).json()
    assert set(r["label"] for r in rows["rows"]) <= {"続く", "解約"}

    # 結果の分割で測り直せ, 列の効きと 1 行の理由は理由付きで断る
    measured = client.post(f"/api/results/{result['id']}/remeasure",
                           json={"write": False})
    assert measured.status_code == 200, measured.text
    assert measured.json()["rows"] == 30
    explained = client.post(f"/api/results/{result['id']}/explain")
    assert explained.status_code == 400
    assert "前処理の前の列" in explained.json()["detail"]
    reason = client.post(f"/api/results/{result['id']}/rows/0/reason")
    assert reason.status_code == 400

    # 同じ分割の結果どうしは同じ入力として比べる
    again = client.post(
        f"/api/bundles/{entry['id']}/predict",
        json={"split": "test", "decision": {}},
    ).json()
    compared = client.get("/api/results/compare",
                          params={"ids": f"{result['id']},{again['id']}"})
    assert "source" in compared.json()["basis"]

    # 書き戻して置き場に写した一式も, 写した元の実験から見つける
    saved = client.post(f"/api/results/{result['id']}/remeasure",
                        json={"write": True}).json()
    assert saved["copied_to"]
    moved = client.get(f"/api/bundles/{entry['id']}/test-split").json()
    assert moved["dataset"] == str(dataset)


def test_split_request_errors(client: TestClient, tmp_path: Path) -> None:
    """
    選べない分割の名前と, 分割と他の入力を一緒に渡す要求を断ることを
    確かめる関数

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
    # 画面からは test の名前だけ (場所は受け取らない)
    project = tmp_path / "project"
    entry = register(client, write_experiment(project))
    url = f"/api/bundles/{entry['id']}/predict"
    response = client.post(url, json={"split": "train"})
    assert response.status_code == 400
    assert "test だけ" in response.json()["detail"]
    response = client.post(
        url, json={"split": "test", "source": str(project / "data")}
    )
    assert response.status_code == 400
    response = client.post(url, json={"split": "../x"})
    assert response.status_code == 400
    response = client.post(url, json={"split": "test", "texts": ["a"]})
    assert response.status_code == 400
    response = client.post(url, json={"split": "test", "forecast": True})
    assert response.status_code == 400


def test_split_not_found(client: TestClient, tmp_path: Path) -> None:
    """
    実験の外の一式, データセットの記録の無い実験, プロジェクトの外を指す
    データセットでは, 理由を返して推論もしないことを確かめる関数

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
    # .. でプロジェクトの外 (同じ中身の写し) を指すデータセット
    project = tmp_path / "project"
    bundle = write_experiment(project, dataset="../outside")
    outside = tmp_path / "outside"
    shutil.copytree(project / "data" / "dataset", outside)
    entry = register(client, bundle)
    found = client.get(f"/api/bundles/{entry['id']}/test-split").json()
    assert found["split"] is None
    assert "プロジェクトの外" in found["reason"]
    response = client.post(f"/api/bundles/{entry['id']}/predict",
                           json={"split": "test"})
    assert response.status_code == 400
    assert "プロジェクトの外" in response.json()["detail"]

    # リンクで外を指すデータセットと, 絶対パスで外を指すデータセット
    run = project / RUN
    (project / "data" / "link").symlink_to(outside)
    for place in ("data/link", str(outside)):
        (run / "config.yaml").write_text(
            yaml.safe_dump({"data": {"dataset": place}}), encoding="utf-8"
        )
        assert "プロジェクトの外" in str(find_test_split(entry)["reason"])

    # たどれない場所 (無い利用者の ~, リンクの輪) は 500 でなく理由
    (project / "data" / "loop").symlink_to(project / "data" / "loop")
    for place in ("~nosuchuser-for-test/x", "data/loop"):
        (run / "config.yaml").write_text(
            yaml.safe_dump({"data": {"dataset": place}}), encoding="utf-8"
        )
        response = client.get(f"/api/bundles/{entry['id']}/test-split")
        assert response.status_code == 200
        assert response.json()["split"] is None
        assert response.json()["reason"]

    # リンクの dataset.yaml は読まない
    linked = project / "data" / "linked"
    linked.mkdir()
    (linked / "dataset.yaml").symlink_to(
        project / "data" / "dataset" / "dataset.yaml"
    )
    (run / "config.yaml").write_text(
        yaml.safe_dump({"data": {"dataset": "data/linked"}}),
        encoding="utf-8"
    )
    assert "読めません" in str(find_test_split(entry)["reason"])

    # 記録の無い実験 (task.yaml の dataset.path も無い)
    (run / "config.yaml").write_text("data: {}\n", encoding="utf-8")
    assert "data.dataset" in str(find_test_split(entry)["reason"])

    # task.yaml の dataset.path で見つける
    (run / "task.yaml").write_text(
        yaml.safe_dump({"dataset": {"path": "data/dataset"}}),
        encoding="utf-8"
    )
    assert find_test_split(entry)["split"] == "test"

    # テスト用の分割の無いデータセットと, 分割のファイルが外を指す説明
    info_file = project / "data" / "dataset" / "dataset.yaml"
    info = yaml.safe_load(info_file.read_text(encoding="utf-8"))
    info_file.write_text(yaml.safe_dump({**info, "splits": {
        k: v for k, v in info["splits"].items() if k != "test"
    }}), encoding="utf-8")
    assert "テスト用の分割がありません" in str(
        find_test_split(entry)["reason"]
    )
    info_file.write_text(yaml.safe_dump({**info, "splits": {
        **info["splits"], "test": {"file": "../churn.csv", "rows": 30}
    }}), encoding="utf-8")
    assert "読めません" in str(find_test_split(entry)["reason"])


def test_split_outside_experiment(
    client: TestClient, tmp_path: Path
) -> None:
    """
    実験の外 (run.json の無い場所) の一式では, テスト用の分割を選べない
    ことを確かめる関数

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
    # 実験の run.json を消した一式
    project = tmp_path / "project"
    bundle = write_experiment(project)
    (project / RUN / "run.json").unlink()
    entry = register(client, bundle)
    found = client.get(f"/api/bundles/{entry['id']}/test-split").json()
    assert found["split"] is None
    assert "実験" in found["reason"]
