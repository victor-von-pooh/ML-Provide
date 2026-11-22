from collections.abc import Iterator
from pathlib import Path
from typing import Any

import ml_inference
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from ml_modelcontrol import register_unit
from PIL import Image
from torch import nn

import ml_provide.ready as ready_module
from ml_provide.app import create_app

# テストで使うトークン
TOKEN = "test-token"

# 作り物の出どころの種類の名前の表と前処理
CLASSES = ["cat", "dog", "bird"]
PREPROCESS = {
    "resize": 10, "crop": 8, "interpolation": "bilinear",
    "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
}

# 作り物の出どころの重みのファイルの名前
FILE = "tiny-0123abcd.pth"

# 作り物のライセンスの情報 (pretrained_info の形. ネットワークを使わない)
LICENSE = {
    "unit": "test.ProvideReadyNet", "model": "tiny", "hub_id": None,
    "revision": None, "pinned_revision": None, "license": None,
    "status": "unknown", "redistributable": False, "gated": False,
    "alert": True, "message": "ライセンスを確かめてください",
    "source": "テストの情報", "url": None,
}


class ProvideReadyNet(nn.Module):
    def __init__(self, name: str = "tiny", weights: str = "V1"):
        # 親クラスのコンストラクタを呼び出す
        super().__init__()

        # 出どころの重み (チャンネルの平均をそのまま種類の logit に)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3, 3)
        with torch.no_grad():
            self.fc.weight.copy_(torch.eye(3) * 4.0)
            self.fc.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        チャンネルの平均から種類の logit を求める関数

        Parameters
        ----------
        x: torch.Tensor
            (件数, 3, 高さ, 幅) の画像

        Returns
        ----------
        out: torch.Tensor
            (件数, 3) の logit
        """
        # 平均して全結合
        out = self.fc(self.flatten(self.pool(x)))

        return out


register_unit(ProvideReadyNet, name="test.ProvideReadyNet", overwrite=True)


def tiny_weights(model: str, weights: str) -> dict[str, Any]:
    """
    作り物の出どころの重みの説明を返す関数

    Parameters
    ----------
    model: str
        モデルの名前
    weights: str
        重みの名前 (DEFAULT は V1 に直す)

    Returns
    ----------
    found: dict[str, Any]
        READY_LIBRARIES の関数の結果の形
    """
    # 名前は 1 つだけ
    chosen = "V1" if weights == "DEFAULT" else weights
    if model != "tiny" or chosen != "V1":
        raise ValueError(f"{model} の重みに {weights} がありません (重み: V1)")
    found = {
        "unit": "test.ProvideReadyNet",
        "params": {"name": model, "weights": chosen},
        "classes": list(CLASSES), "input": "image",
        "preprocess": dict(PREPROCESS),
        "url": f"https://example.com/{FILE}", "file": FILE,
    }

    return found


@pytest.fixture
def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """
    作り物の出どころを足し, 一時ディレクトリを置き場にしたクライアントを
    作る関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        出どころとライセンスの読み方の差し替え

    Returns
    ----------
    client: Iterator[TestClient]
        トークン付きのクライアント
    """
    # 出どころ, ライセンス, 重みの置き場を差し替える (ネットワークを使わない)
    monkeypatch.setitem(ml_inference.READY_LIBRARIES, "test", tiny_weights)
    monkeypatch.setattr(
        ready_module, "pretrained_info", lambda unit, params: dict(LICENSE)
    )
    hub = torch.hub.get_dir()
    torch.hub.set_dir(str(tmp_path / "hub"))
    app = create_app(
        data_dir=tmp_path / "home", token=TOKEN, allowed_hosts=["testserver"],
        procedures=[]
    )
    client = TestClient(app, headers={"x-provide-token": TOKEN})
    yield client
    torch.hub.set_dir(hub)


def write_images(folder: Path) -> Path:
    """
    赤, 緑, 青の画像と読めないファイルを書く関数

    Parameters
    ----------
    folder: Path
        書くディレクトリ

    Returns
    ----------
    folder: Path
        書いたディレクトリ
    """
    # 大きさの違う 3 枚と, 壊れた 1 枚
    folder.mkdir()
    for index, (name, size) in enumerate(
        [("a_red", (20, 12)), ("b_green", (9, 30)), ("c_blue", (10, 10))]
    ):
        array = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        array[:, :, index] = 255
        Image.fromarray(array).save(folder / f"{name}.png")
    (folder / "d_broken.png").write_bytes(b"not an image")

    return folder


def create_ready(client: TestClient, **extra: Any) -> Any:
    """
    画面と同じ API で作り物の出どころの既製のモデルの一式を作る関数

    Parameters
    ----------
    client: TestClient
        クライアント
    **extra: Any
        上書きする欄

    Returns
    ----------
    response: Any
        応答
    """
    body = {
        "name": "tiny", "library": "test", "model": "tiny", "weights": "V1",
        **extra,
    }
    response = client.post("/api/bundles/ready", json=body)

    return response


def test_ready_preview(client: TestClient) -> None:
    """
    下見が種類の名前, 前処理, ライセンスの警告を返し, 誤りを理由付きで
    返すことを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント

    Returns
    ----------
    None
    """
    # 出どころの一覧と下見 (DEFAULT は版の決まった名前に直る)
    assert "test" in client.get("/api/info").json()["ready_libraries"]
    response = client.get(
        "/api/ready/preview",
        params={"library": "test", "model": " tiny ", "weights": "DEFAULT"},
    )
    assert response.status_code == 200
    preview = response.json()
    assert preview["model"] == "tiny"
    assert preview["weights"] == "V1"
    assert preview["classes"] == CLASSES
    assert preview["preprocess"] == PREPROCESS
    assert preview["file"] == FILE
    assert preview["license"]["alert"] is True
    assert preview["license"]["message"] == LICENSE["message"]
    assert "unit" not in preview["license"]

    # 無い出どころ, 無い重み, 空の名前
    for params, word in [
        ({"library": "nothing", "model": "tiny", "weights": "V1"}, "出どころ"),
        ({"library": "test", "model": "tiny", "weights": "V9"}, "V9"),
        ({"library": "test", "model": " ", "weights": "V1"}, "モデルの名前"),
    ]:
        response = client.get("/api/ready/preview", params=params)
        assert response.status_code == 400
        assert word in response.json()["detail"]


def test_ready_preview_unknown_license(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ライセンスの情報を読めない出どころでも, 警告を付けて下見できることを
    確かめ, 作った一式も警告を持つことを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    monkeypatch: pytest.MonkeyPatch
        ライセンスの読み方の差し替え

    Returns
    ----------
    None
    """
    # 読めないライセンスは警告
    def broken(unit: str, params: dict[str, Any]) -> dict[str, Any]:
        raise KeyError(unit)

    monkeypatch.setattr(ready_module, "pretrained_info", broken)
    response = client.get(
        "/api/ready/preview",
        params={"library": "test", "model": "tiny", "weights": "V1"},
    )
    assert response.status_code == 200
    license = response.json()["license"]
    assert license["alert"] is True
    assert "読めません" in license["message"]

    # 作った一式も警告を持つ
    response = create_ready(client)
    assert response.status_code == 201, response.text
    recorded = response.json()["description"]["ready"]["license"]
    assert recorded["alert"] is True
    assert "読めません" in recorded["message"]


def test_ready_bundle_flow(tmp_path: Path, client: TestClient) -> None:
    """
    既製のモデルの一式を作って登録し, 説明と重みの置き場の有無を出し,
    画像から種類の名前と確信度を推論することを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    client: TestClient
        クライアント

    Returns
    ----------
    None
    """
    # 作って登録 (重みは写さず, ライセンスの警告と hash を持つ)
    response = create_ready(client)
    assert response.status_code == 201, response.text
    view = response.json()
    assert view["managed"] is True
    description = view["description"]
    assert description["kind"] == "ready"
    assert description["classes"] == CLASSES
    ready = description["ready"]
    assert ready["license"]["alert"] is True
    assert ready["weights_cached"] is False
    record = description["pretrained"][0]
    assert record["place"] == "source"
    assert len(record["sha256"]) == 64
    assert any("出どころから読み" in note for note in description["notes"])

    # 置き場に重みのファイルを置くと, 覚えた説明でも置いたと出す
    place = Path(ready["weights_file"])
    assert place.name == FILE
    place.parent.mkdir(parents=True, exist_ok=True)
    place.write_bytes(b"weights")
    again = client.get(f"/api/bundles/{view['id']}").json()
    assert again["description"]["ready"]["weights_cached"] is True

    # 画像のディレクトリを推論 (読めない画像は理由を付けて飛ばす)
    images = write_images(tmp_path / "images")
    response = client.post(
        f"/api/bundles/{view['id']}/predict",
        json={"source": str(images), "rules": {"min_confidence": None}},
    )
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["kind"] == "ready"
    assert result["summary"]["predicted"] == 3
    assert result["summary"]["skipped"] == 1
    rows = client.get(f"/api/results/{result['id']}/rows").json()["rows"]
    predicted = {row["path"]: row for row in rows if row["status"] == "ok"}
    assert [
        predicted[name]["prediction"] for name in sorted(predicted)
    ] == CLASSES
    assert all(0.5 < row["confidence"] <= 1 for row in predicted.values())

    # 意思決定は使えない (データセットと外れ方の記録が無い)
    response = client.post(
        f"/api/bundles/{view['id']}/predict",
        json={"source": str(images), "decision": {"type": "triage"}},
    )
    assert response.status_code == 400
    response = client.post(f"/api/results/{result['id']}/explain")
    assert response.status_code == 400

    # 消すと置き場の一式も消える
    folder = Path(view["path"])
    assert client.delete(f"/api/bundles/{view['id']}").status_code == 204
    assert not folder.exists()


def test_ready_bundle_refuses(client: TestClient) -> None:
    """
    hash が違う場合と形が違う場合, 名前の無い場合に作らず登録もしない
    ことを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント

    Returns
    ----------
    None
    """
    # 決めておいた hash と違う重み, hash の形の誤り, 無い重み
    for extra, word in [
        ({"sha256": "0" * 64}, "hash"),
        ({"sha256": "abc"}, "64 文字"),
        ({"weights": "V9"}, "V9"),
        ({"library": "nothing"}, "出どころ"),
    ]:
        response = create_ready(client, **extra)
        assert response.status_code == 400, extra
        assert word in response.json()["detail"]
    assert client.get("/api/bundles").json() == []

    # 読んだ hash を決めておけば作れる (大文字と前後の空白はそろえる)
    first = create_ready(client).json()
    sha256 = first["description"]["pretrained"][0]["sha256"]
    response = create_ready(client, sha256=f" {sha256.upper()} ")
    assert response.status_code == 201, response.text
