import dataclasses
import io
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
from ml_inference import PROCEDURES
from ml_modelcontrol import build_model, load_example
from test_bundle_v3 import head_spec, make_v3, register, write_unlabeled

import ml_provide.api as api_module
import ml_provide.problems as problems_module
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
        data_dir=tmp_path / "home", token=TOKEN, allowed_hosts=["testserver"],
        procedures=[]
    )
    client = TestClient(app, headers={"x-provide-token": TOKEN})

    return client


def diffusion_files(root: Path) -> dict[str, str]:
    """
    拡散のモデルの雑音の予測器の定義と重みを書く関数

    Parameters
    ----------
    root: Path
        書くディレクトリ

    Returns
    ----------
    paths: dict[str, str]
        spec (定義のファイル) と weights (重みのファイル)
    """
    # ML-ModelControl の見本の定義と, 乱数の初期値のままの重み
    root.mkdir(parents=True, exist_ok=True)
    spec = load_example("diffusion_denoiser")
    spec_file = root / "denoiser.yaml"
    spec_file.write_text(yaml.safe_dump(spec), encoding="utf-8")
    weights = root / "denoiser.pt"
    torch.save(build_model(spec, seed=0).state_dict(), weights)
    paths = {"spec": str(spec_file), "weights": str(weights)}

    return paths


def create_noise(
    client: TestClient, paths: dict[str, str], **extra: Any
) -> dict[str, Any]:
    """
    画面と同じ API で雑音だけの一式を作る関数

    Parameters
    ----------
    client: TestClient
        クライアント
    paths: dict[str, str]
        diffusion_files の場所
    **extra: Any
        要求に重ねる欄

    Returns
    ----------
    bundle: dict[str, Any]
        登録と説明
    """
    # 雑音の中身 x と時刻 t, DDIM の 5 回, 1 件の形は 16 の値
    body = {
        "name": "diffusion", "spec": paths["spec"],
        "weights": paths["weights"],
        "inputs": {"x": {"source": "state"}, "t": {"source": "time"}},
        "form": "vector", "output_options": {"shape": [16]},
        "procedure": {"name": "ddim", "steps": 5, "seed": 3},
        **extra,
    }
    response = client.post("/api/bundles/create", json=body)
    assert response.status_code == 201, response.text
    bundle: dict[str, Any] = response.json()

    return bundle


def outputs(client: TestClient, result_id: str, count: int) -> np.ndarray:
    """
    結果の件ごとの配列を並べる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    result_id: str
        結果の id
    count: int
        件数

    Returns
    ----------
    arrays: np.ndarray
        (件数, 16) の配列
    """
    # 行の位置の .npy を順に
    arrays = np.stack([
        np.load(io.BytesIO(client.get(
            f"/api/results/{result_id}/artifacts/{row}/file"
        ).content))
        for row in range(count)
    ])

    return arrays


def test_noise_bundle(tmp_path: Path, client: TestClient) -> None:
    """
    データセットを省いて雑音だけの一式を作り, 件数と種で推論できることを
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
    # データセットを省くか空にすると雑音だけの一式 (版 5)
    paths = diffusion_files(tmp_path / "model")
    bundle = create_noise(client, paths)
    assert create_noise(client, paths, dataset="")["ok"] is True
    description = bundle["description"]
    assert description["kind"] == "noise"
    assert description["format_version"] == 5
    assert description["role"] == "generate"
    assert description["output"] == {"form": "vector", "shape": [16]}
    assert description["target"] is None
    assert description["procedure"]["name"] == "ddim"

    # 件数と種で推論し, 件ごとの配列を持つ (結論は件数だけ)
    url = f"/api/bundles/{bundle['id']}/predict"
    response = client.post(url, json={"count": 4, "seed": 7})
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["source"] == "件数 4, 種 7"
    assert result["kind"] == "noise"
    assert result["column_output"] is False
    assert result["summary"] == {"rows": 4, "predicted": 4, "skipped": 0}
    assert result["artifacts"]["form"] == "vector"
    rows = client.get(f"/api/results/{result['id']}/rows").json()["rows"]
    assert [row["sample"] for row in rows] == [0, 1, 2, 3]
    assert {row["seed"] for row in rows} == {7}
    first = outputs(client, result["id"], 4)
    assert first.shape == (4, 16)

    # 同じ種なら同じ出力, 種を省くと一式の手順の種
    again = client.post(url, json={"count": 4, "seed": 7}).json()
    assert np.array_equal(outputs(client, again["id"], 4), first)
    default = client.post(url, json={"count": 2}).json()
    assert default["source"] == "件数 2, 種 3"

    # 件数が無い, 1 より小さい, 上限を超える, ファイルや文を渡すと断る
    for body in ({}, {"seed": 1}, {"count": 0},
                 {"count": api_module.MAX_SAMPLES + 1},
                 {"count": 2, "source": str(tmp_path)},
                 {"count": 2, "texts": ["a"]},
                 {"source": str(tmp_path)}):
        response = client.post(url, json=body)
        assert response.status_code == 400, body
    # 真偽値, 文字, 小数の件数と種は整数にせずに断る
    for body in ({"count": True}, {"count": "3"}, {"count": 3.0},
                 {"count": 2, "seed": True}, {"count": 2, "seed": "1"},
                 {"count": 2, "seed": 1.0}):
        response = client.post(url, json=body)
        assert response.status_code == 422, body
    assert client.post(url, json={"count": 2, "seed": 2 ** 63}).status_code \
        == 201
    detail = client.post(url, json={"count": 0}).json()["detail"]
    assert "1 以上" in detail
    big = client.post(url, json={"count": api_module.MAX_SAMPLES + 1})
    assert str(api_module.MAX_SAMPLES) in big.json()["detail"]

    # 記憶の上限は 1 件の形の大きさから件数を決める
    info = client.get("/api/info").json()
    assert info["max_samples"] == api_module.MAX_SAMPLES
    ddim = next(p for p in info["procedures"] if p["name"] == "ddim")
    assert ddim["start"] == "noise" and ddim["time"] is True
    assert "eta" in ddim["options"]


def test_noise_problem_not_read(
    tmp_path: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    雑音だけの一式では, 問題の設定を読まず候補も探さないことを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    client: TestClient
        クライアント
    monkeypatch: pytest.MonkeyPatch
        問題の設定を読む関数と候補を探す関数の差し替え

    Returns
    ----------
    None
    """
    # 読むか探すと落ちるようにして, 空の状態が返る
    def fail(*args: Any) -> Any:
        raise AssertionError("問題の設定を読みました")

    monkeypatch.setattr(problems_module, "find_hint", fail)
    monkeypatch.setattr(problems_module, "read_problem", fail)
    bundle = create_noise(client, diffusion_files(tmp_path / "model"))
    response = client.get(f"/api/bundles/{bundle['id']}/problem")
    assert response.status_code == 200, response.text
    assert response.json() == {
        "path": None, "linked": None, "error": None, "hint": None,
        "hint_note": None,
    }


def test_noise_limits(
    tmp_path: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    件数と種を雑音だけの一式にだけ受け付け, 値の上限で件数を絞ることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    client: TestClient
        クライアント
    monkeypatch: pytest.MonkeyPatch
        上限を小さくする

    Returns
    ----------
    None
    """
    # 1 件 16 の値で 40 個までなら 2 件まで
    paths = diffusion_files(tmp_path / "model")
    bundle = create_noise(client, paths)
    monkeypatch.setattr(api_module, "MAX_SAMPLE_VALUES", 40)
    url = f"/api/bundles/{bundle['id']}/predict"
    response = client.post(url, json={"count": 3})
    assert response.status_code == 400
    assert "2 件まで" in response.json()["detail"]
    assert client.post(url, json={"count": 2}).status_code == 201

    # 1 件だけで上限を超える形は別の文で断る
    monkeypatch.setattr(api_module, "MAX_SAMPLE_VALUES", 10)
    detail = client.post(url, json={"count": 1}).json()["detail"]
    assert "大きすぎます" in detail and "[16]" in detail

    # 作るときも, 確かめの 2 件が上限を超える形は作る前に断る
    # (int64 であふれる大きな形も)
    monkeypatch.setattr(api_module, "MAX_SAMPLE_VALUES", 40)
    homes = tmp_path / "home" / "bundles"
    before = sorted(homes.iterdir())
    for shape in ([32], [2 ** 40, 2 ** 40]):
        response = client.post("/api/bundles/create", json={
            "name": "huge", "spec": paths["spec"],
            "weights": paths["weights"],
            "inputs": {"x": {"source": "state"},
                       "t": {"source": "time"}},
            "form": "vector", "output_options": {"shape": shape},
            "procedure": {"name": "ddim", "steps": 5, "seed": 3},
        })
        assert response.status_code == 400, shape
        assert "大きすぎます" in response.json()["detail"]
    assert sorted(homes.iterdir()) == before

    # データセットのある一式には, 作るときの形の上限を当てない
    data = write_unlabeled(tmp_path / "shaped")
    spec_file = tmp_path / "head.yaml"
    spec_file.write_text(yaml.safe_dump(head_spec(32)), encoding="utf-8")
    torch.save(build_model(head_spec(32), seed=0).state_dict(),
               tmp_path / "head.pt")
    response = client.post("/api/bundles/create", json={
        "name": "shaped", "spec": str(spec_file),
        "dataset": str(data["dataset"]), "weights": str(tmp_path / "head.pt"),
        "form": "vector", "role": "group",
        "output_options": {"shape": [32]},
    })
    assert response.status_code == 201, response.text

    # データセットのある一式には件数と種を渡せない
    data = write_unlabeled(tmp_path / "data")
    path = make_v3(
        tmp_path, data, "embed", head_spec(5), form="vector", role="group"
    )
    other = register(client, path)
    url = f"/api/bundles/{other['id']}/predict"
    for body in ({"count": 2}, {"seed": 1, "source": str(data["csv"])}):
        response = client.post(url, json=body)
        assert response.status_code == 400
        assert "雑音だけの一式" in response.json()["detail"]

    # 手順が雑音から始まらないか, 1 件の形が無い一式は作らない
    response = client.post("/api/bundles/create", json={
        "name": "broken", "spec": paths["spec"],
        "weights": paths["weights"],
        "inputs": {"x": {"source": "state"}, "t": {"source": "time"}},
        "form": "vector",
        "procedure": {"name": "ddim", "steps": 5, "seed": 3},
    })
    assert response.status_code == 400
    assert "shape" in response.json()["detail"]

    # 1 件の形がモデルの入力と合わなければ, 1 件の形の欄へ案内
    response = client.post("/api/bundles/create", json={
        "name": "wrong", "spec": paths["spec"],
        "weights": paths["weights"],
        "inputs": {"x": {"source": "state"}, "t": {"source": "time"}},
        "form": "vector", "output_options": {"shape": [5]},
        "procedure": {"name": "ddim", "steps": 5, "seed": 3},
    })
    assert response.status_code == 400
    assert "1 件の形" in response.json()["detail"], response.text

    # 重みが定義と合わない誤りには, 1 件の形の案内を添えない
    other = tmp_path / "other.pt"
    torch.save(build_model(head_spec(5), seed=0).state_dict(), other)
    response = client.post("/api/bundles/create", json={
        "name": "weights", "spec": paths["spec"], "weights": str(other),
        "inputs": {"x": {"source": "state"}, "t": {"source": "time"}},
        "form": "vector", "output_options": {"shape": [16]},
        "procedure": {"name": "ddim", "steps": 5, "seed": 3},
    })
    assert response.status_code == 400
    assert "state_dict" in response.json()["detail"]
    assert "1 件の形" not in response.json()["detail"]

    # 手順の回数は上限まで (train_steps を大きくして, ML-Inference の範囲
    # の内側で画面の上限だけに当てる. ちょうど上限は通る)
    for steps, status in ((api_module.MAX_PROCEDURE_STEPS + 1, 400),
                          (api_module.MAX_PROCEDURE_STEPS, 201)):
        response = client.post("/api/bundles/create", json={
            "name": f"long{steps}", "spec": paths["spec"],
            "weights": paths["weights"],
            "inputs": {"x": {"source": "state"}, "t": {"source": "time"}},
            "form": "vector", "output_options": {"shape": [16]},
            "procedure": {"name": "ddim", "steps": steps, "seed": 3,
                          "options": {"train_steps": 2000}},
        })
        assert response.status_code == status, response.text
    assert response.json()["description"]["procedure"]["steps"] == 1000
    detail = client.post("/api/bundles/create", json={
        "name": "longer", "spec": paths["spec"],
        "weights": paths["weights"],
        "form": "vector", "output_options": {"shape": [16]},
        "procedure": {"name": "ddim", "steps": 1001, "seed": 3,
                      "options": {"train_steps": 2000}},
    }).json()["detail"]
    assert detail.startswith("手順の回数は 1,000 回まで")
    info = client.get("/api/info").json()
    assert info["max_procedure_steps"] == api_module.MAX_PROCEDURE_STEPS

    # 出力の形の誤りなど, 1 件の形でない誤りには案内を添えない
    response = client.post("/api/bundles/create", json={
        "name": "form", "spec": paths["spec"],
        "weights": paths["weights"],
        "inputs": {"x": {"source": "state"}, "t": {"source": "time"}},
        "form": "class", "output_options": {"shape": [16]},
        "procedure": {"name": "ddim", "steps": 5, "seed": 3},
    })
    assert response.status_code == 400
    assert "1 件の形" not in response.json()["detail"], response.text


def test_procedure_options(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    手順の設定の名前と既定値を ML-Inference の procedure_options から読み,
    引数を読めない手順でも登録簿の一覧を返すことを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    monkeypatch: pytest.MonkeyPatch
        登録簿に引数を読めない手順を置く

    Returns
    ----------
    None
    """
    # 設定の無い関数は空, **kwargs と引数を読めない関数は None, *args の
    # 名前は設定に入らない. 既定値は有限の数, 真偽値, 文字だけ (inf, NaN,
    # None は除く)
    def plain(model: Any, state: Any, steps: int, generator: Any) -> Any:
        return state

    def anything(
        model: Any, state: Any, steps: int, generator: Any, **options: Any
    ) -> Any:
        return state

    def spread(
        model: Any, state: Any, steps: int, generator: Any, *rest: Any,
        scale: float = 1.0
    ) -> Any:
        return state

    def limits(
        model: Any, state: Any, steps: int, generator: Any,
        low: float = float("-inf"), rate: float = float("nan"),
        mode: str = "fast", flag: bool = True, size: Any = None,
        step: float = 0.5,
    ) -> Any:
        return state

    # 登録簿の一覧 (/api/info) は ML-Inference の procedure_options から
    ddim = PROCEDURES["ddim"]
    for name, function in (("plain", plain), ("anything", anything),
                           ("spread", spread), ("limits", limits),
                           ("broken", max)):
        monkeypatch.setitem(PROCEDURES, name, dataclasses.replace(
            ddim, name=name, function=function
        ))
    response = client.get("/api/info")
    assert response.status_code == 200
    listed = {p["name"]: p for p in response.json()["procedures"]}
    assert listed["plain"]["options"] == []
    assert listed["anything"]["options"] is None
    assert listed["spread"]["options"] == ["scale"]
    assert listed["limits"]["defaults"] == {
        "mode": "fast", "flag": True, "step": 0.5
    }
    assert listed["broken"]["options"] is None
    assert listed["ddim"]["options"] == [
        "train_steps", "beta_start", "beta_end", "eta"
    ]
    assert listed["ddim"]["defaults"] == {
        "train_steps": 1000, "beta_start": 1e-4, "beta_end": 0.02,
        "eta": 0.0,
    }
    assert listed["broken"]["defaults"] == {}


def test_text_output_rows(tmp_path: Path, client: TestClient) -> None:
    """
    文を出す一式の結果の文と番号を, 数に見えても文字で読み, 予測した行の
    空の文を空の文字で返すことを確かめる関数

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
    # 文字の語彙で長さ 4 の並びにする文のデータセット
    texts = ["12", "34", "5.5", "NA"] * 5
    source = tmp_path / "texts.csv"
    pd.DataFrame({"text": texts, "label": ["a", "b"] * 10}).to_csv(
        source, index=False
    )
    recipe = {
        "format_version": 1, "target": "label", "task": "classification",
        "steps": [],
        "item_steps": [{"op": "vocabulary", "unit": "char", "length": 4}],
        "split": {"method": "random", "valid": 0.2, "test": 0.0, "seed": 0},
    }
    dataset = tmp_path / "dataset"
    export_dataset(load_data(source, text_column="text"), recipe, dataset)

    # 番号ごとの埋め込みから次の番号の logits で, 番号を 2 つ足す
    size = 12
    spec = {
        "format_version": 1,
        "inputs": {"ids": {"shape": [2, 4], "dtype": "int64"}},
        "nodes": {
            "embed": {"unit": "nn.Embedding", "inputs": "ids",
                      "params": {"num_embeddings": size,
                                 "embedding_dim": 4}},
            "head": {"unit": "nn.Linear", "inputs": "embed",
                     "params": {"in_features": 4, "out_features": size}},
        },
        "outputs": {"logits": "head"},
    }
    weights = tmp_path / "lm.pt"
    torch.save(build_model(spec, seed=0).state_dict(), weights)
    path = tmp_path / "lm"
    ml_inference.make_bundle(
        path, spec, dataset, weights, form="text", role="generate",
        inputs={"ids": {"source": "state"}},
        procedure={"name": "greedy", "steps": 2, "seed": 0}
    )
    bundle = register(client, path)
    response = client.post(f"/api/bundles/{bundle['id']}/predict",
                           json={"texts": ["12", "NA", "5"]})
    assert response.status_code == 201, response.text
    result = response.json()
    assert "tokens" in result["columns"]

    # 数に見える文と NA の文, 予測した行の空の文と番号は文字のまま
    place = tmp_path / "home" / "results" / result["id"]
    table = pd.read_csv(place / "predictions.csv", encoding="utf-8-sig",
                        dtype=str, keep_default_na=False)
    table.loc[0, ["prediction", "tokens"]] = ["123", "4"]
    table.loc[1, "prediction"] = "NA"
    table.loc[2, ["prediction", "tokens"]] = ["", ""]
    table.to_csv(place / "predictions.csv", index=False, encoding="utf-8-sig")
    rows = client.get(f"/api/results/{result['id']}/rows").json()["rows"]
    assert [row["prediction"] for row in rows] == ["123", "NA", ""]
    assert rows[0]["tokens"] == "4" and rows[2]["tokens"] == ""
    assert [row["status"] for row in rows] == ["ok"] * 3

    # tokens の列の無い古い文の結果も, 予測を文字のまま読む (全ての行が
    # 番号 1 つに見えても数にしない. 予測した行の空の文は空の文字)
    table.loc[0, "prediction"] = "5"
    table.loc[1, "prediction"] = "07"
    table.drop(columns=["tokens"]).to_csv(
        place / "predictions.csv", index=False, encoding="utf-8-sig"
    )
    rows = client.get(f"/api/results/{result['id']}/rows").json()["rows"]
    assert [row["prediction"] for row in rows] == ["5", "07", ""]
    assert "tokens" not in rows[0]


def test_table_output(tmp_path: Path, client: TestClient) -> None:
    """
    表の出力の雑音だけの一式 (版 7) を登録して推論し, 列の符号化の順の列
    と型を結果に残し, 表と CSV で返すことを確かめる関数

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
    # 見本の雑音の予測器を符号化した行の幅 (数の列 2 つと種類の 3 つ) に
    # して, 列の符号化を写した一式を ML-Inference の make_bundle で作る
    spec = load_example("diffusion_denoiser")
    spec["inputs"]["x"]["shape"] = [8, 5]
    spec["nodes"]["x_fc"]["params"]["in_features"] = 5
    spec["nodes"]["head"]["params"]["out_features"] = 5
    weights = tmp_path / "rows.pt"
    torch.save(build_model(spec, seed=0).state_dict(), weights)
    columns = [
        {"name": "price", "type": "number", "mean": 100.0, "std": 20.0,
         "integer": False},
        {"name": "count", "type": "number", "mean": 3.0, "std": 1.5,
         "integer": True},
        {"name": "shop", "type": "category", "categories": ["01", "02", "10"]},
    ]
    path = tmp_path / "rows"
    ml_inference.make_bundle(
        path, spec, None, weights, form="table",
        output_options={"columns": columns},
        inputs={"x": {"source": "state"}, "t": {"source": "time"}},
        procedure={"name": "ddim", "steps": 3, "seed": 0}
    )
    bundle = register(client, path)
    description = bundle["description"]
    assert description["format_version"] == 7
    assert description["kind"] == "noise"
    assert description["output"]["form"] == "table"
    assert description["output"]["columns"] == columns
    assert description["column_output"] is False

    # 件数と種で推論すると, 配列を持たない結果で, 列の名前と型を残す (数に
    # 見える種類の値の型を残す版 3)
    response = client.post(f"/api/bundles/{bundle['id']}/predict",
                           json={"count": 3, "seed": 1})
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["output_form"] == "table"
    assert result["artifacts"] is None
    assert result["format_version"] == 3
    assert result["summary"] == {"rows": 3, "predicted": 3, "skipped": 0}
    assert result["table_columns"] == [
        {"name": "price", "type": "number", "integer": False},
        {"name": "count", "type": "number", "integer": True},
        {"name": "shop", "type": "category", "categories": 3},
    ]
    assert result["columns"][:7] == [
        "sample", "seed", "status", "reason", "price", "count", "shop"
    ]
    assert "output" not in result["columns"]
    assert "artifact" not in result["columns"]
    reloaded = client.get(f"/api/results/{result['id']}").json()
    assert reloaded["table_columns"] == result["table_columns"]

    # 種類の列は数に見えても文字, 整数の列は整数のまま
    rows = client.get(f"/api/results/{result['id']}/rows").json()["rows"]
    assert [row["sample"] for row in rows] == [0, 1, 2]
    assert {row["shop"] for row in rows} <= {"01", "02", "10"}
    assert all(isinstance(row["count"], int) for row in rows)
    assert all(isinstance(row["price"], float) for row in rows)

    # 列の型の無い古い結果 (版 1) も, 残した列の名前と型で文字のまま読む
    place = tmp_path / "home" / "results" / result["id"]
    meta = yaml.safe_load((place / "result.yaml").read_text(encoding="utf-8"))
    assert meta["column_types"] == {"shop": "text"}
    old = {**meta, "format_version": 1}
    del old["column_types"]
    (place / "result.yaml").write_text(yaml.safe_dump(old), encoding="utf-8")

    # 推論しなかった行の値は欠損 (整数の列も小数にしない)
    table = pd.read_csv(place / "predictions.csv", encoding="utf-8-sig",
                        dtype=str, keep_default_na=False)
    table.loc[1, ["status", "reason", "price", "count", "shop"]] = [
        "skipped", "予測の値を表に戻せません", "", "", ""
    ]
    table.to_csv(place / "predictions.csv", index=False, encoding="utf-8-sig")
    table.loc[0, "shop"] = "NA"
    table.loc[2, "shop"] = "null"
    table.to_csv(place / "predictions.csv", index=False, encoding="utf-8-sig")
    rows = client.get(f"/api/results/{result['id']}/rows").json()["rows"]
    assert rows[1]["count"] is None and rows[1]["shop"] is None
    # 欠損に見える種類の値 (NA, null) は文字のまま
    assert rows[0]["shop"] == "NA" and rows[2]["shop"] == "null"
    assert isinstance(rows[0]["count"], int)
    skipped = client.get(f"/api/results/{result['id']}/rows",
                         params={"status": "skipped"}).json()
    assert skipped["total"] == 1

    # CSV は表のまま (列の符号化の順の列)
    text = client.get(f"/api/results/{result['id']}/download").text
    header = text.lstrip("﻿").splitlines()[0].split(",")
    assert header[:7] == [
        "sample", "seed", "status", "reason", "price", "count", "shop"
    ]
