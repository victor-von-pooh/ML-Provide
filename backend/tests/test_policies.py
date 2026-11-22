import hashlib
import json
import shutil
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ml_provide.api import PREDICT_LOCK
from ml_provide.app import create_app

# テストで使うトークン
TOKEN = "test-token"

# ML-Inference のテストの方策の一式 (ML-Found で書き出したもの) と期待の結果
POLICIES = (
    Path(__file__).resolve().parents[2] / "ML-Inference" / "tests" / "policies"
)
EXPECTED = json.loads((POLICIES / "expected.json").read_text(encoding="utf-8"))


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


def copy_policy(name: str, tmp_path: Path) -> Path:
    """
    テストの方策の一式を一時ディレクトリに写す関数

    Parameters
    ----------
    name: str
        一式の名前 (cartpole か box)
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    directory: Path
        写した一式のディレクトリ
    """
    directory = tmp_path / name
    shutil.copytree(POLICIES / name, directory)

    return directory


def decode(item: dict) -> Any:
    """
    期待の結果のファイルの観測と行動を元の値に戻す関数

    Parameters
    ----------
    item: Any
        array (数の並び. 有限でない値は文字) か raw (そのまま) か, 行動の
        数の並び

    Returns
    ----------
    value: Any
        元の値 (数の並びは形の付いた入れ子のリスト)
    """
    # 有限でない値の文字を数に戻して形を付ける
    if item.get("kind") == "raw":
        value = item["value"]
        return value
    values = np.array([float(number) for number in item["values"]])
    value = values.reshape(item["shape"]).tolist()

    return value


def register(client: TestClient, directory: Path) -> dict[str, Any]:
    """
    方策の一式を登録する関数

    Parameters
    ----------
    client: TestClient
        クライアント
    directory: Path
        方策の一式のディレクトリ

    Returns
    ----------
    view: dict[str, Any]
        登録と中身
    """
    response = client.post("/api/policies", json={"path": str(directory)})
    assert response.status_code == 201, response.text
    view: dict[str, Any] = response.json()

    return view


@pytest.mark.parametrize("name", ["cartpole", "box"])
def test_act_matches_ml_inference(
    client: TestClient, tmp_path: Path, name: str
) -> None:
    """
    登録した方策の一式で, 観測から ML-Inference と同じ行動と守りの結果を
    返すことを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    name: str
        一式の名前

    Returns
    ----------
    None
    """
    view = register(client, copy_policy(name, tmp_path))
    assert view["ok"] and view["spec"]["kind"] == "policy"
    for case in EXPECTED[name]:
        # 有限でない観測 (NaN, Infinity) もそのまま送る
        observation = decode(case["observation"])
        response = client.post(
            f"/api/policies/{view['id']}/act",
            content=json.dumps({"observation": observation}),
            headers={"content-type": "application/json"}
        )
        assert response.status_code == 200, response.text
        result = response.json()
        expected = case["result"]
        assert result["status"] == expected["status"]
        assert result["reasons"] == expected["reasons"]
        action = expected["action"]
        if isinstance(action, dict):
            assert np.allclose(
                result["action"], decode(action), atol=1e-5
            )
        else:
            assert result["action"] == action


def test_list_detail_and_delete(client: TestClient, tmp_path: Path) -> None:
    """
    一覧, 中身 (読まなかった重み), 二重の登録, 消すことを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    directory = copy_policy("box", tmp_path)
    view = register(client, directory)
    assert view["unused_keys"] == ["log_std", "value.bias", "value.weight"]
    listed = client.get("/api/policies").json()
    assert [item["id"] for item in listed] == [view["id"]]
    assert listed[0]["spec"]["action_space"]["type"] == "box"
    detail = client.get(f"/api/policies/{view['id']}").json()
    assert detail["spec"]["guards"]["fallback"]["kind"] == "action"

    # 同じ場所の二重の登録は断り, 消しても一式のファイルは残す
    again = client.post("/api/policies", json={"path": str(directory)})
    assert again.status_code == 400
    assert client.delete(f"/api/policies/{view['id']}").status_code == 204
    assert client.get("/api/policies").json() == []
    assert (directory / "policy.json").is_file()
    assert client.get(f"/api/policies/{view['id']}").status_code == 404


def test_rejects_unreadable_policy(
    client: TestClient, tmp_path: Path
) -> None:
    """
    読めない方策の一式を登録せず, 後で読めなくなった一式は理由付きで
    出すことを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # policy.json の無いディレクトリと, 推論の一式の取り違え
    response = client.post("/api/policies", json={"path": str(tmp_path)})
    assert response.status_code == 400
    directory = copy_policy("cartpole", tmp_path)
    spec_path = directory / "policy.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec_path.write_text(
        json.dumps({**spec, "kind": "inference"}), encoding="utf-8"
    )
    response = client.post("/api/policies", json={"path": str(directory)})
    assert response.status_code == 400
    assert "kind" in response.json()["detail"]

    # 登録した後に重みが変わった一式は, 一覧で理由を出して推論を断る
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    view = register(client, directory)
    with open(directory / "policy.pt", "ab") as file:
        file.write(b"0")
    listed = client.get("/api/policies").json()
    assert listed[0]["ok"]
    detail = client.get(f"/api/policies/{view['id']}").json()
    assert not detail["ok"] and "sha256" in detail["error"]
    response = client.post(
        f"/api/policies/{view['id']}/act",
        json={"observation": [0.0, 0.0, 0.0, 0.0]}
    )
    assert response.status_code == 400
    assert "sha256" in response.json()["detail"]


def test_bad_observation_falls_back(
    client: TestClient, tmp_path: Path
) -> None:
    """
    形の違う観測と数でない観測は, 外れたときの動きを返すことを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    view = register(client, copy_policy("cartpole", tmp_path))
    url = f"/api/policies/{view['id']}/act"
    for observation in ([0.0, 0.0], ["a", "b", "c", "d"], [[0.0], [0.0, 1.0]]):
        result = client.post(url, json={"observation": observation}).json()
        assert result["action"] is None
        assert result["status"] == "handover"
        assert result["reasons"]

    # 大きすぎる観測は読まずに断る
    response = client.post(url, json={"observation": [0.0] * 100_001})
    assert response.status_code == 400


def test_needs_model_spec(client: TestClient, tmp_path: Path) -> None:
    """
    モデルの定義を写していない方策の一式 (組み立て方がクラスの名前だけ) を,
    理由付きで登録しないことを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # クラスの名前と引数だけの組み立て方 (名前からコードを読み込まない)
    directory = copy_policy("cartpole", tmp_path)
    spec_path = directory / "policy.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["metadata"]["model"] = {"class": "model.PolicyNet", "args": {}}
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    response = client.post("/api/policies", json={"path": str(directory)})
    assert response.status_code == 400
    assert "metadata.model.spec" in response.json()["detail"]
    assert client.get("/api/policies").json() == []


def test_huge_integer_falls_back(client: TestClient, tmp_path: Path) -> None:
    """
    float にできない桁の大きな整数の観測は, 外れたときの動きを返すことを
    確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    view = register(client, copy_policy("cartpole", tmp_path))
    body = '{"observation": [1, 1e0, 0, 1' + "0" * 400 + "]}"
    response = client.post(
        f"/api/policies/{view['id']}/act", content=body,
        headers={"content-type": "application/json"}
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["action"] is None and result["status"] == "handover"
    assert result["reasons"][0].startswith("観測を数の配列にできない")


def test_nested_observation_limit(
    client: TestClient, tmp_path: Path
) -> None:
    """
    入れ子の観測も要素の数で上限を当てることを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    view = register(client, copy_policy("cartpole", tmp_path))
    url = f"/api/policies/{view['id']}/act"
    ragged = [[0.0] * 50_001, [0.0] * 50_000]
    assert client.post(url, json={"observation": ragged}).status_code == 400
    small = [[0.0] * 2, [0.0] * 2]
    assert client.post(url, json={"observation": small}).status_code == 200
    # 空のリストを並べた観測もリストを 1 つと数えて上限に掛かる
    empties: list[list[float]] = [[] for _ in range(100_001)]
    assert client.post(url, json={"observation": empties}).status_code == 400


def test_missing_policy_before_limit(client: TestClient) -> None:
    """
    無い一式は大きすぎる観測でも 404 にすることを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント

    Returns
    ----------
    None
    """
    response = client.post(
        "/api/policies/missing/act", json={"observation": [0.0] * 100_001}
    )
    assert response.status_code == 404, response.text


def test_act_does_not_wait_for_predict(
    client: TestClient, tmp_path: Path
) -> None:
    """
    推論の鍵を持っている間も方策の行動を返すことを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    view = register(client, copy_policy("cartpole", tmp_path))
    url = f"/api/policies/{view['id']}/act"
    observation = [0.0, 0.0, 0.0, 0.0]
    results: list[int] = []

    # 推論の鍵を持ったまま別の糸で送り, 待たずに返ることを確かめる
    def send() -> None:
        response = client.post(url, json={"observation": observation})
        results.append(response.status_code)

    with PREDICT_LOCK:
        thread = threading.Thread(target=send)
        thread.start()
        thread.join(timeout=30)
    assert results == [200]


def test_reloads_changed_files(client: TestClient, tmp_path: Path) -> None:
    """
    写したモデルの定義が変わると読み直し, 読めない重みは 400 にすることを
    確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 定義を壊すと, 覚えた方策を使わずに読み直して理由を出す
    directory = copy_policy("cartpole", tmp_path)
    view = register(client, directory)
    (directory / "model_spec.yaml").write_text("[", encoding="utf-8")
    detail = client.get(f"/api/policies/{view['id']}").json()
    assert not detail["ok"]

    # sha256 の合う壊れた重みは, 読めない一式として 400
    other = copy_policy("box", tmp_path)
    weights = b"not a pickle"
    (other / "policy.pt").write_bytes(weights)
    spec_path = other / "policy.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["weights_sha256"] = hashlib.sha256(weights).hexdigest()
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    response = client.post("/api/policies", json={"path": str(other)})
    assert response.status_code == 400
