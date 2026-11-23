import hashlib
import http.client
import io
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.error import URLError

import ml_inference
import ml_quest
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from ml_modelcontrol import register_unit
from ml_quest import example_problem
from PIL import Image
from torch import nn

import ml_provide.problems as problems_module
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
    # 作って登録 (重みは写さず, ライセンスの警告とハッシュ値を持つ)
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
    ハッシュ値が違う場合と形が違う場合, 名前の無い場合に作らず登録もしない
    ことを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント

    Returns
    ----------
    None
    """
    # 決めておいたハッシュ値と違う重み, ハッシュ値の形の誤り, 無い重み
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

    # 読んだハッシュ値を決めておけば作れる (大文字と前後の空白はそろえる)
    first = create_ready(client).json()
    sha256 = first["description"]["pretrained"][0]["sha256"]
    response = create_ready(client, sha256=f" {sha256.upper()} ")
    assert response.status_code == 201, response.text


class FakeResponse:
    """
    open_url の応答の作り物 (決めた塊を返し, 尽きたら空)
    """

    def __init__(
        self, chunks: list[Any], wait: float = 0.0,
        url: str = f"https://example.com/{FILE}"
    ):
        # 返す塊 (例外なら投げる), 1 回の読み込みで待つ秒, 転送の後の URL
        self.chunks = list(chunks)
        self.wait = wait
        self.url = url

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def geturl(self) -> str:
        """
        転送の後の URL を返す関数

        Returns
        ----------
        url: str
            URL
        """
        return self.url

    def read1(self, size: int) -> bytes:
        """
        次の塊を返す関数 (届いた分だけ返す読み方. read は持たない)

        Parameters
        ----------
        size: int
            読む大きさ (使わない)

        Returns
        ----------
        chunk: bytes
            次の塊 (尽きたら空)
        """
        # 小さく読むこと. 待ってから次の塊 (例外は投げる. 無限の塊は ...)
        assert size <= 1 << 16
        time.sleep(self.wait)
        if not self.chunks:
            return b""
        chunk = self.chunks[0]
        if chunk is ...:
            return b"x"
        self.chunks.pop(0)
        if isinstance(chunk, Exception):
            raise chunk
        assert isinstance(chunk, bytes)

        return chunk


def use_fetch(
    monkeypatch: pytest.MonkeyPatch, answer: Any, file: str = FILE
) -> list[str]:
    """
    作り物の出どころの重みを取りに行かせ, open_url と重みのファイルの名前を
    差し替える関数

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        差し替え
    answer: Any
        open_url が返す応答 (例外なら投げる)
    file: str = FILE
        出どころの重みのファイルの名前

    Returns
    ----------
    calls: list[str]
        開いた URL
    """
    calls: list[str] = []

    def fake_open(url: str, timeout: float) -> Any:
        # 読み込みの上限を渡すことも確かめる
        calls.append(url)
        assert timeout == ready_module.READ_SECONDS
        if isinstance(answer, Exception):
            raise answer
        return answer

    def named(model: str, weights: str) -> dict[str, Any]:
        found = tiny_weights(model, weights)
        found.update(url=f"https://example.com/{file}", file=file)
        return found

    monkeypatch.setattr(ready_module, "FETCHED_LIBRARIES", ("test",))
    monkeypatch.setattr(ready_module, "open_url", fake_open)
    monkeypatch.setitem(ml_inference.READY_LIBRARIES, "test", named)

    return calls


def test_ready_fetch_timeout(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    重みの取得が時間の上限を超えると 502 で断り, 鍵を手放して途中の
    ファイルを残さないことを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    monkeypatch: pytest.MonkeyPatch
        取得と上限の差し替え

    Returns
    ----------
    None
    """
    # 細く流れ続ける取得 (1 バイトずつ尽きない) を短い上限で止める
    monkeypatch.setattr(ready_module, "FETCH_SECONDS", 0.05)
    use_fetch(monkeypatch, FakeResponse([...], wait=0.01))
    response = create_ready(client)
    assert response.status_code == 502, response.text
    assert "時間の上限" in response.json()["detail"]
    assert not ready_module.READY_LOCK.locked()
    assert client.get("/api/bundles").json() == []
    checkpoints = Path(torch.hub.get_dir()) / "checkpoints"
    assert list(checkpoints.iterdir()) == []

    # 読み込みが途切れたもの (socket の上限) も同じ誤り
    use_fetch(monkeypatch, FakeResponse([b"x", TimeoutError("timed out")]))
    response = create_ready(client)
    assert response.status_code == 502
    assert "時間の上限" in response.json()["detail"]
    assert not ready_module.READY_LOCK.locked()


def test_ready_fetch_network_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ネットワークの誤りと壊れた重みを 502 で断り, 重みを取れれば置き場に
    置いて作り, 次からは取りに行かないことを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    monkeypatch: pytest.MonkeyPatch
        取得の差し替え

    Returns
    ----------
    None
    """
    # つながらない場合
    use_fetch(monkeypatch, URLError("no route"))
    response = create_ready(client)
    assert response.status_code == 502
    assert "取れません" in response.json()["detail"]

    # 途中で切れた応答 (OSError でない誤り) と, http への転送
    for answer in (
        FakeResponse([b"x", http.client.IncompleteRead(b"x", 10)]),
        FakeResponse([b"x"], url=f"http://example.com/{FILE}"),
    ):
        use_fetch(monkeypatch, answer)
        response = create_ready(client)
        assert response.status_code == 502
        assert "取れません" in response.json()["detail"] or "https" in (
            response.json()["detail"]
        )
        checkpoints = Path(torch.hub.get_dir()) / "checkpoints"
        assert list(checkpoints.iterdir()) == []

    # 名前のハッシュ値の頭と違う中身 (一時ファイルも残さない)
    use_fetch(monkeypatch, FakeResponse([b"weights"]))
    response = create_ready(client)
    assert response.status_code == 502
    assert "ハッシュ値の頭" in response.json()["detail"]
    assert list(checkpoints.iterdir()) == []
    assert client.get("/api/bundles").json() == []

    # 名前のハッシュ値の頭と合う中身は置き場に置く
    content = b"tiny weights"
    file = f"tiny-{hashlib.sha256(content).hexdigest()[:8]}.pth"
    calls = use_fetch(monkeypatch, FakeResponse([content[:4], content[4:]]),
                      file=file)
    response = create_ready(client)
    assert response.status_code == 201, response.text
    ready = response.json()["description"]["ready"]
    assert ready["weights_cached"] is True
    assert Path(ready["weights_file"]).read_bytes() == content
    assert calls == [f"https://example.com/{file}"]

    # 置き場にあれば取りに行かない
    response = create_ready(client)
    assert response.status_code == 201
    assert len(calls) == 1


def test_ready_fetch_refuses_plain_http(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    https でない重みの URL を取りに行かないことを確かめる関数

    Parameters
    ----------
    monkeypatch: pytest.MonkeyPatch
        置き場の差し替え
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 置き場に無いファイルを http で
    hub = torch.hub.get_dir()
    torch.hub.set_dir(str(tmp_path / "hub"))
    try:
        with pytest.raises(ready_module.FetchError, match="https"):
            ready_module.fetch_weights({"url": f"http://example.com/{FILE}", "file": FILE})
    finally:
        torch.hub.set_dir(hub)


def test_ready_fetch_refuses_plain_redirect() -> None:
    """
    https でない URL への転送を断り, https への転送には従うことを確かめる
    関数

    Returns
    ----------
    None
    """
    # 転送の扱いだけを直に呼ぶ (ネットワークを使わない)
    handler = ready_module.HttpsOnlyRedirect()
    request = urllib.request.Request(f"https://example.com/{FILE}")
    headers = http.client.HTTPMessage()
    with pytest.raises(ready_module.FetchError, match="https でない"):
        handler.redirect_request(
            request, io.BytesIO(), 302, "Found", headers,
            f"http://example.com/{FILE}"
        )
    moved = handler.redirect_request(
        request, io.BytesIO(), 302, "Found", headers,
        f"https://mirror.example.com/{FILE}"
    )
    assert moved is not None
    assert moved.full_url == f"https://mirror.example.com/{FILE}"


# 作り方が既製のモデルをそのままの問題の設定の見本
READY_EXAMPLE = "写真の仕分け (既製のモデル)"


def test_ready_problem_link(
    tmp_path: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    既製のモデルの一式に問題の設定をつなぐと, 結論の決め方が既定になり,
    意思決定は使えないと書き, 決めておいた重みとの食い違いを出すことを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    client: TestClient
        クライアント
    monkeypatch: pytest.MonkeyPatch
        受け渡しの結論の決め方の差し替え

    Returns
    ----------
    None
    """
    # 骨組みと同じく, 問題の設定を持つプロジェクトの中に一式を作って登録
    project = tmp_path / "project"
    ml_quest.save_problem(example_problem(READY_EXAMPLE), project)
    ml_inference.make_ready_bundle(
        project / "outputs" / "inference", "tiny", "V1", library="test",
        conclusion={"min_confidence": 0.5}
    )
    response = client.post(
        "/api/bundles", json={"path": str(project / "outputs" / "inference")}
    )
    assert response.status_code == 201, response.text
    bundle_id = response.json()["id"]

    # つないでいなければ, 上のディレクトリの問題の設定が候補
    state = client.get(f"/api/bundles/{bundle_id}/problem").json()
    assert state["hint"] == str((project / "problem.yaml").resolve())

    # 受け渡しの結論の決め方を一式の既定値に重ねる
    handoff = ml_quest.handoff

    def with_rules(problem: dict) -> dict:
        stages = handoff(problem)
        stages["inference"]["conclusion"] = {"min_confidence": 0.8}
        return stages

    monkeypatch.setattr(ml_quest, "handoff", with_rules)
    response = client.put(
        f"/api/bundles/{bundle_id}/problem", json={"path": state["hint"]}
    )
    assert response.status_code == 200, response.text
    linked = response.json()["linked"]
    assert linked["handled"] is True and linked["decision"] is None
    assert linked["conclusion"] == {"min_confidence": 0.8}
    assert "意思決定を使えない" in linked["text"]

    # 見本は torchvision の ResNet18 なので, 出どころ, モデル, 重み, ハッシュ値が
    # 一式と違う
    issues = " / ".join(linked["issues"])
    for word in ("出どころ torchvision", "モデル resnet18",
                 "重み IMAGENET1K_V1", "重みのハッシュ値 8ec1484cdbb8"):
        assert word in issues, issues


def test_ready_problem_not_pretrained(client: TestClient) -> None:
    """
    作り方が既製のモデルをそのままでない問題の設定は, 既製のモデルの一式
    につなぐと使わないと書き, 結論の決め方は一式の既定値のままにする
    ことを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント

    Returns
    ----------
    None
    """
    # 列の予測の問題の設定 (総菜の作る量)
    response = create_ready(client)
    assert response.status_code == 201, response.text
    bundle_id = response.json()["id"]
    folder = Path(response.json()["path"]).parent / "problem"
    ml_quest.save_problem(example_problem("総菜の作る量"), folder)
    linked = client.put(
        f"/api/bundles/{bundle_id}/problem", json={"path": str(folder)}
    ).json()["linked"]
    assert linked["handled"] is False and linked["decision"] is None
    assert "既製のモデルをそのままではない" in linked["text"]
    assert linked["issues"] == []


def test_ready_issues_match(tmp_path: Path, client: TestClient) -> None:
    """
    決めておいた重みが一式と同じなら, 重みの名前の DEFAULT とハッシュ値の
    大文字も食い違いにしないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    client: TestClient
        作り物の出どころを足したクライアント

    Returns
    ----------
    None
    """
    # 一式が読んだ重みのハッシュ値を大文字で決めておく
    ml_inference.make_ready_bundle(
        tmp_path / "bundle", "tiny", "V1", library="test"
    )
    bundle = ml_inference.load_bundle(tmp_path / "bundle")
    sha = bundle.info["model"]["pretrained"][0]["sha256"]
    planned = {"library": "test", "model": "tiny", "weights": "DEFAULT",
               "sha256": f" {sha.upper()} "}
    assert problems_module.ready_issues(bundle, planned) == []
    other = problems_module.ready_issues(
        bundle, {**planned, "sha256": "0" * 64, "model": None}
    )
    assert len(other) == 1 and "重みのハッシュ値" in other[0]

    # 一式の記録の大文字も小文字にして比べる
    bundle.info["model"]["pretrained"][0]["sha256"] = sha.upper()
    assert problems_module.ready_issues(bundle, planned) == []

    # 一式にハッシュ値の記録が無ければ, 違うとは言わずに確かめられないと
    # 出す (決めていなければ何も出さない)
    bundle.info["model"]["pretrained"][0].pop("sha256")
    unknown = problems_module.ready_issues(bundle, planned)
    assert len(unknown) == 1 and "確かめられません" in unknown[0]
    assert "違います" not in unknown[0]
    assert problems_module.ready_issues(
        bundle, {**planned, "sha256": None}
    ) == []
