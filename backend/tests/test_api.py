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


def write_project(root: Path) -> dict[str, Path]:
    """
    表の分類のデータ, データセット, 定義, 実験のディレクトリを作る関数

    Parameters
    ----------
    root: Path
        作るディレクトリ

    Returns
    ----------
    paths: dict[str, Path]
        csv, dataset, spec, run の場所
    """
    # 2 クラスの表と, 書き出したデータセット
    rng = np.random.default_rng(0)
    root.mkdir(parents=True)
    frame = pd.DataFrame({
        "shop": rng.choice(["東", "西"], size=40),
        "a": rng.normal(size=40),
        "b": rng.normal(size=40),
        "label": rng.choice(["良", "悪"], size=40),
    })
    csv = root / "data.csv"
    frame.to_csv(csv, index=False)
    recipe = {
        "format_version": 1, "target": "label", "task": "classification",
        "steps": [{"op": "drop_columns", "columns": ["shop"]},
                  {"op": "standardize", "columns": "auto"}],
        "split": {"method": "random", "valid": 0.2, "seed": 0},
    }
    dataset = root / "dataset"
    export_dataset(load_data(csv), recipe, dataset)

    # 定義と, 重みを持つ実験のディレクトリ
    spec = root / "model.yaml"
    spec.write_text(yaml.safe_dump(linear_spec(2, 2)), encoding="utf-8")
    run = root / "run"
    run.mkdir()
    torch.save(
        build_model(linear_spec(2, 2), seed=0).state_dict(),
        run / "final_model.pt"
    )
    paths = {"csv": csv, "dataset": dataset, "spec": spec, "run": run}

    return paths


def create(client: TestClient, paths: dict[str, Path]) -> dict[str, Any]:
    """
    画面と同じ API で一式を作る関数

    Parameters
    ----------
    client: TestClient
        クライアント
    paths: dict[str, Path]
        write_project の場所

    Returns
    ----------
    bundle: dict[str, Any]
        登録と説明
    """
    response = client.post("/api/bundles/create", json={
        "name": "reviews", "spec": str(paths["spec"]),
        "dataset": str(paths["dataset"]), "weights": str(paths["run"]),
        "conclusion": {"min_confidence": 0.55},
    })
    assert response.status_code == 201, response.text
    bundle: dict[str, Any] = response.json()

    return bundle


def test_token_required(tmp_path: Path) -> None:
    """
    トークンの無い呼び出しを拒否することを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    app = create_app(
        data_dir=tmp_path / "home", token=TOKEN, allowed_hosts=["testserver"]
    )
    assert TestClient(app).get("/api/bundles").status_code == 401


def test_create_predict_conclude(tmp_path: Path, client: TestClient) -> None:
    """
    一式を作り, ファイルを推論し, 結論を出し直す流れを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    client: TestClient
        クライアント

    Returns
    ----------
    None
    """
    # 一式の作成と一覧
    paths = write_project(tmp_path / "project")
    bundle = create(client, paths)
    assert bundle["ok"] and bundle["managed"]
    assert bundle["description"]["classes"] == ["悪", "良"]
    assert [b["id"] for b in client.get("/api/bundles").json()] == [
        bundle["id"]
    ]

    # 推論と結果
    response = client.post(
        f"/api/bundles/{bundle['id']}/predict",
        json={"source": str(paths["csv"]), "rules": {"group_by": "shop"}},
    )
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["summary"]["rows"] == 40
    assert result["rules"] == {"min_confidence": 0.55, "group_by": "shop"}
    assert "decision" in result["columns"]
    listed = client.get(f"/api/results?bundle={bundle['id']}").json()
    assert [r["id"] for r in listed] == [result["id"]]

    # 表の一部と絞り込み
    rows = client.get(
        f"/api/results/{result['id']}/rows?offset=5&limit=10"
    ).json()
    assert rows["total"] == 40 and len(rows["rows"]) == 10
    assert rows["rows"][0]["index"] == 5
    held = client.get(
        f"/api/results/{result['id']}/rows?decision=保留"
    ).json()
    assert all(r["decision"] == "保留" for r in held["rows"])

    # 決め方を変えて出し直す (保留が無くなる)
    again = client.post(
        f"/api/results/{result['id']}/conclude", json={"rules": {}}
    ).json()
    assert again["rules"] == {}
    assert all(d["decision"] != "保留" for d in again["summary"]["decisions"])
    assert client.post(
        f"/api/results/{result['id']}/conclude",
        json={"rules": {"min_confidence": 3}},
    ).status_code == 400

    # CSV と削除
    csv = client.get(f"/api/results/{result['id']}/download")
    assert csv.status_code == 200 and "decision" in csv.text
    assert client.delete(f"/api/results/{result['id']}").status_code == 204
    assert client.get(f"/api/results/{result['id']}").status_code == 404


def test_register_and_delete(tmp_path: Path, client: TestClient) -> None:
    """
    場所で登録した一式は, 登録を消しても一式を消さないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    client: TestClient
        クライアント

    Returns
    ----------
    None
    """
    # ライブラリで作った一式を場所で登録
    paths = write_project(tmp_path / "project")
    place = tmp_path / "bundle"
    ml_inference.make_bundle(
        place, paths["spec"], paths["dataset"], paths["run"]
    )
    response = client.post("/api/bundles", json={"path": str(place)})
    assert response.status_code == 201, response.text
    entry = response.json()
    assert entry["name"] == "bundle" and not entry["managed"]
    again = client.post("/api/bundles", json={"path": str(place)})
    assert again.status_code == 400

    # 消しても一式は残る. 置き場で作った一式は消える
    assert client.delete(f"/api/bundles/{entry['id']}").status_code == 204
    assert (place / "inference.yaml").is_file()
    made = create(client, paths)
    made_dir = Path(made["path"])
    assert made_dir.is_dir()
    assert client.delete(f"/api/bundles/{made['id']}").status_code == 204
    assert not made_dir.exists()


def test_errors_are_explained(tmp_path: Path, client: TestClient) -> None:
    """
    作れない一式, 推論できない入力の理由を返すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    client: TestClient
        クライアント

    Returns
    ----------
    None
    """
    # 出力の幅がクラスの数と合わない定義
    paths = write_project(tmp_path / "project")
    paths["spec"].write_text(
        yaml.safe_dump(linear_spec(2, 3)), encoding="utf-8"
    )
    response = client.post("/api/bundles/create", json={
        "name": "bad", "spec": str(paths["spec"]),
        "dataset": str(paths["dataset"]), "weights": str(paths["run"]),
    })
    assert response.status_code == 400
    assert "見本の推論" in response.json()["detail"] or (
        "クラスの数" in response.json()["detail"]
    )
    assert client.get("/api/bundles").json() == []

    # 一式でない場所, 無い入力, 表の一式への文
    assert client.post(
        "/api/bundles", json={"path": str(tmp_path)}
    ).status_code == 400
    paths["spec"].write_text(
        yaml.safe_dump(linear_spec(2, 2)), encoding="utf-8"
    )
    bundle = create(client, paths)
    url = f"/api/bundles/{bundle['id']}/predict"
    assert client.post(url, json={}).status_code == 400
    assert client.post(
        url, json={"source": str(tmp_path / "none.csv")}
    ).status_code == 400
    response = client.post(url, json={"texts": ["良い"]})
    assert response.status_code == 400
    assert "文の一式" in response.json()["detail"]
    frame = pd.read_csv(paths["csv"]).drop(columns=["a"])
    frame.to_csv(tmp_path / "lack.csv", index=False)
    response = client.post(url, json={"source": str(tmp_path / "lack.csv")})
    assert response.status_code == 400 and "a" in response.json()["detail"]
    assert client.get("/api/results/..%2Fx").status_code == 404


def test_index_column_and_stable_groups(
    tmp_path: Path, client: TestClient
) -> None:
    """
    index の列を持つ入力と, 出し直しでまとめの名前が変わらないことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    client: TestClient
        クライアント

    Returns
    ----------
    None
    """
    # index と日付の列を足した入力
    paths = write_project(tmp_path / "project")
    bundle = create(client, paths)
    frame = pd.read_csv(paths["csv"])
    frame["index"] = range(100, 100 + len(frame))
    frame["day"] = ["2024-01-01", "2024-01-02"] * (len(frame) // 2)
    source = tmp_path / "new.csv"
    frame.to_csv(source, index=False)
    result = client.post(
        f"/api/bundles/{bundle['id']}/predict",
        json={"source": str(source), "rules": {"group_by": "day"}},
    ).json()
    rows = client.get(f"/api/results/{result['id']}/rows?limit=2").json()
    assert rows["rows"][1]["index"] == 1
    assert rows["rows"][1]["index (入力)"] == 101

    # 推論の直後と出し直しで, まとめの名前が同じ
    groups = [g["group"] for g in result["summary"]["groups"]]
    again = client.post(
        f"/api/results/{result['id']}/conclude",
        json={"rules": {"group_by": "day"}},
    ).json()
    assert [g["group"] for g in again["summary"]["groups"]] == groups


def test_samples(client: TestClient) -> None:
    """
    見本の一式を作り, 見本の新しいデータを推論できることを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント

    Returns
    ----------
    None
    """
    # 3 つの見本 (2 回目は登録し直さない)
    added = client.post("/api/samples").json()
    assert [b["name"] for b in added][0] == "見本: レビューの分類 (文)"
    assert len(added) == 3 and all(b["ok"] for b in added)
    assert client.post("/api/samples").json() == []

    # 見本の新しいデータの推論
    for bundle in added:
        response = client.post(
            f"/api/bundles/{bundle['id']}/predict",
            json={"source": bundle["sample_input"]},
        )
        assert response.status_code == 201, response.text
        assert response.json()["summary"]["predicted"] == 50
