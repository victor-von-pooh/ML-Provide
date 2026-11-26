import io
import shutil
import sys
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

# 起動の設定で読ませる自作の手順のモジュール (登録簿は 1 つの処理で共有)
MODULE = "provide_test_procedures"

# 自作の手順のモジュールの中身 (雑音からモデルの出力を少しずつ足す)
MODULE_CODE = '''
from ml_inference import register_procedure


def walk(model, state, steps, generator):
    """
    雑音にモデルの出力を回数で割って足していく手順
    """
    for _ in range(steps):
        state = state + model(state) / steps
    return state


register_procedure(
    "provide_walk", walk, start="noise", time=False, description="試し"
)
'''


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


def write_unlabeled(root: Path) -> dict[str, Path]:
    """
    正解の無い表のデータとデータセットを作る関数

    Parameters
    ----------
    root: Path
        作るディレクトリ

    Returns
    ----------
    paths: dict[str, Path]
        csv, dataset の場所
    """
    # 数の列 3 つ (1 行は欠けて推論できない)
    rng = np.random.default_rng(0)
    root.mkdir(parents=True)
    frame = pd.DataFrame({
        "a": rng.normal(size=30), "b": rng.normal(size=30),
        "c": rng.normal(size=30),
    })
    train = root / "train.csv"
    frame.to_csv(train, index=False)
    frame.loc[2, "a"] = np.nan
    csv = root / "new.csv"
    frame.head(6).to_csv(csv, index=False)
    recipe = {
        "format_version": 1, "target": None, "task": "none",
        "steps": [{"op": "standardize", "columns": "auto"}],
        "split": {"method": "random", "valid": 0.2, "seed": 0},
    }
    dataset = root / "dataset"
    export_dataset(load_data(train), recipe, dataset)
    paths = {"csv": csv, "dataset": dataset}

    return paths


def head_spec(outputs: int, grid: list[int] | None = None) -> dict[str, Any]:
    """
    全結合の 1 層 (と形を戻す層) のモデルの定義を返す関数

    Parameters
    ----------
    outputs: int
        出力の幅
    grid: list[int] | None = None
        出力を (クラス, 高さ, 幅) に戻す形 (領域の一式)

    Returns
    ----------
    spec: dict[str, Any]
        ML-ModelControl の定義
    """
    # 3 列から出力の幅へ
    nodes: dict[str, Any] = {
        "head": {
            "unit": "nn.Linear",
            "params": {"in_features": 3, "out_features": outputs},
            "inputs": "x",
        }
    }
    last = "head"
    if grid is not None:
        nodes["grid"] = {
            "unit": "nn.Unflatten",
            "params": {"dim": 1, "unflattened_size": grid},
            "inputs": "head",
        }
        last = "grid"
    spec = {
        "format_version": 1,
        "inputs": {"x": {"shape": [2, 3]}},
        "nodes": nodes,
        "outputs": {"out": last},
    }

    return spec


def make_v3(
    root: Path, paths: dict[str, Path], name: str, spec: dict[str, Any],
    **options: Any
) -> Path:
    """
    ML-Inference で版 3 の一式を作る関数

    Parameters
    ----------
    root: Path
        作るディレクトリの親
    paths: dict[str, Path]
        write_unlabeled の場所
    name: str
        一式の名前
    spec: dict[str, Any]
        モデルの定義
    **options: Any
        make_bundle に渡す引数

    Returns
    ----------
    path: Path
        一式のディレクトリ
    """
    # 乱数の初期値のままの重み
    weights = root / f"{name}.pt"
    torch.save(build_model(spec, seed=0).state_dict(), weights)
    path = root / name
    ml_inference.make_bundle(
        path, spec, paths["dataset"], weights, name=name, **options
    )

    return path


def register(client: TestClient, path: Path) -> dict[str, Any]:
    """
    場所で一式を登録する関数

    Parameters
    ----------
    client: TestClient
        クライアント
    path: Path
        一式のディレクトリ

    Returns
    ----------
    bundle: dict[str, Any]
        登録と説明
    """
    response = client.post("/api/bundles", json={"path": str(path)})
    assert response.status_code == 201, response.text
    bundle: dict[str, Any] = response.json()

    return bundle


def test_vector_bundle(tmp_path: Path, client: TestClient) -> None:
    """
    ベクトルの出力の一式を推論し, 件数だけの結論と配列を読めることを
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
    # 説明の版 3 の項目
    paths = write_unlabeled(tmp_path / "data")
    path = make_v3(
        tmp_path, paths, "embed", head_spec(5), form="vector", role="group"
    )
    bundle = register(client, path)
    description = bundle["description"]
    assert description["format_version"] == 3
    assert description["role"] == "group"
    assert description["output"] == {"form": "vector"}
    assert description["column_output"] is False
    assert description["target"] is None
    assert description["target_info"] is None
    assert description["procedure"] is None
    assert description["pretrained"] == []
    assert description["pretrained_warnings"] == []
    assert bundle["model_loaded"] is False
    # 問題の設定を読む種類の一式 (ベクトルの出力) は, 問題の設定の欄を出す
    assert bundle["uses_problem"] is True

    # 決め方と意思決定は断る
    url = f"/api/bundles/{bundle['id']}/predict"
    for extra in ({"rules": {"min_confidence": 0.5}},
                  {"decision": {"type": "triage"}}, {"forecast": True}):
        response = client.post(url, json={"source": str(paths["csv"]),
                                          **extra})
        assert response.status_code == 400
        assert "列の出力でない" in response.json()["detail"]

    # 画面と同じく, 空の決め方と空の意思決定は通す
    response = client.post(url, json={
        "source": str(paths["csv"]), "rules": {"positive": None},
        "decision": {},
    })
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["column_output"] is False
    assert result["output_form"] == "vector" and result["role"] == "group"
    assert result["format_version"] == 2
    assert result["summary"] == {"rows": 6, "predicted": 5, "skipped": 1}
    assert result["sentences"] == ["6 件のうち 5 件を推論 (1 件は推論できず)"]
    assert result["rules"] == {}
    assert result["artifacts"]["form"] == "vector"
    assert "artifact" in result["columns"]
    assert "output" not in result["columns"]
    assert client.get(f"/api/bundles/{bundle['id']}").json()["model_loaded"]

    # 表は相対パス, 配列はモデルの出力と同じ
    rid = result["id"]
    rows = client.get(f"/api/results/{rid}/rows").json()
    assert rows["rows"][0]["artifact"] == "artifacts/000000.npy"
    assert rows["rows"][2]["status"] == "skipped"
    preview = client.get(f"/api/results/{rid}/artifacts/0").json()
    assert preview["view"] == "values" and preview["shape"] == [5]
    loaded = ml_inference.load_bundle(path)
    frame = pd.read_csv(paths["csv"]).head(1)
    x = loaded.dataset.transform(frame).to_numpy(np.float32)
    with torch.no_grad():
        expected = loaded.model()(torch.from_numpy(x))[0].numpy()
    assert np.allclose(preview["values"], expected, atol=1e-5)
    assert preview["norm"] == pytest.approx(float(np.linalg.norm(expected)),
                                            rel=1e-4)
    file = client.get(f"/api/results/{rid}/artifacts/0/file")
    assert file.status_code == 200
    assert np.allclose(np.load(io.BytesIO(file.content)), expected,
                       atol=1e-5)
    assert client.get(f"/api/results/{rid}/artifacts/2").status_code == 404
    assert client.get(f"/api/results/{rid}/artifacts/99").status_code == 404

    # 数でない配列と, 外を指すパスは断る
    place = tmp_path / "home" / "results" / rid
    np.save(place / "artifacts" / "000001.npy", np.array(["a", "b"]))
    assert client.get(f"/api/results/{rid}/artifacts/1").status_code == 400
    table = pd.read_csv(place / "predictions.csv", encoding="utf-8-sig")
    table.loc[0, "artifact"] = "../result.yaml"
    table.to_csv(place / "predictions.csv", index=False, encoding="utf-8-sig")
    assert client.get(f"/api/results/{rid}/artifacts/0").status_code == 400

    # 列の出力だけの操作は断る
    for action, body in (("conclude", {"rules": {}}),
                         ("decide", {"decision": None}),
                         ("explain", {}), ("backtest", {}),
                         ("remeasure", {"write": False})):
        response = client.post(f"/api/results/{rid}/{action}", json=body)
        assert response.status_code == 400, action
        assert "列の出力でない" in response.json()["detail"]

    # 知らない新しい版の結果は止まり, 一覧では飛ばす
    meta = yaml.safe_load((place / "result.yaml").read_text(encoding="utf-8"))
    meta["format_version"] = ml_inference.RESULT_FORMAT_VERSION + 1
    (place / "result.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")
    response = client.get(f"/api/results/{rid}")
    assert response.status_code == 400
    assert "読めません" in response.json()["detail"]
    assert client.get("/api/results").json() == []


def test_mask_bundle(tmp_path: Path, client: TestClient) -> None:
    """
    領域の出力の一式を推論し, 格子を見せ, 正解の出どころの文を添える
    ことを確かめる関数

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
    # 今の処理の出力を正解にした領域の一式
    paths = write_unlabeled(tmp_path / "data")
    origin = {"type": "process", "name": "今の判定"}
    path = make_v3(
        tmp_path, paths, "mask", head_spec(32, [2, 4, 4]), form="mask",
        output_options={"classes": ["背景", "物"]},
        target={"kind": "mask", "origin": origin}
    )
    bundle = register(client, path)
    description = bundle["description"]
    assert description["output"]["classes"] == ["背景", "物"]
    assert description["target_info"] == {"kind": "mask", "origin": origin}

    # 結論は件数. 出どころの文は結論の文と分けて origin_note に
    response = client.post(f"/api/bundles/{bundle['id']}/predict",
                           json={"source": str(paths["csv"])})
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["summary"]["predicted"] == 5
    assert "今の判定 の出力" in result["origin_note"]
    assert result["sentences"] == ["6 件のうち 5 件を推論 (1 件は推論できず)"]
    assert result["overall_count"] == 1
    assert result["target_info"]["origin"] == origin

    # 格子はクラスの番号
    preview = client.get(
        f"/api/results/{result['id']}/artifacts/1"
    ).json()
    assert preview["view"] == "grid" and preview["shape"] == [4, 4]
    grid = np.array(preview["grid"])
    assert grid.shape == (4, 4) and set(grid.ravel()) <= {0, 1}
    assert sum(item["count"] for item in preview["labels"]) == 16

    # 書き直しても配列は残る (件数の数え直しは read_result → write_result)
    place = tmp_path / "home" / "results" / result["id"]
    assert len(list((place / "artifacts").glob("*.npy"))) == 5
    back = ml_inference.read_result(place)
    assert back["artifacts"]["count"] == 5


def test_previews_of_points_and_images() -> None:
    """
    点群と画像の配列の見せ方を確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    from ml_provide.api import array_preview

    # 点群は間引いた最初の 3 つの次元
    cloud = np.random.default_rng(0).normal(size=(5000, 4))
    preview = array_preview(cloud, "point_cloud")
    assert preview["view"] == "points" and preview["total"] == 5000
    assert len(preview["points"]) <= 2000 and len(preview["points"][0]) == 3

    # 画像はチャンネルを後ろにそろえて 0 から 255
    image = np.zeros((3, 600, 10))
    image[0, 0, 0] = 2.0
    preview = array_preview(image, "image")
    assert preview["view"] == "image"
    pixels = np.array(preview["image"])
    assert pixels.shape[2] == 3 and pixels.shape[0] <= 256
    assert pixels.max() == 255 and pixels.min() == 0

    # 絵にできない形は値だけ
    assert array_preview(np.zeros((2, 3, 5, 7)), "image")["view"] == "values"

    # 大きな領域は間引いた格子と, 全ての画素の数
    mask = np.zeros((1000, 700), dtype=np.int64)
    mask[:10] = 3
    preview = array_preview(mask, "mask")
    assert len(preview["grid"]) <= 256 and len(preview["grid"][0]) <= 256
    assert preview["labels"] == [{"label": 0, "count": 693000},
                                 {"label": 3, "count": 7000}]
    assert preview["norm"] == pytest.approx(3 * np.sqrt(7000))


def test_unsupported_arrays(tmp_path: Path, client: TestClient) -> None:
    """
    複素数, 構造のある型, オブジェクトの配列と外を指す行を断ることを
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
    # ベクトルの結果の配列を差し替える
    paths = write_unlabeled(tmp_path / "data")
    path = make_v3(tmp_path, paths, "embed", head_spec(5), form="vector")
    bundle = register(client, path)
    result = client.post(f"/api/bundles/{bundle['id']}/predict",
                         json={"source": str(paths["csv"])}).json()
    rid = result["id"]
    place = tmp_path / "home" / "results" / rid
    folder = place / "artifacts"
    cases = {
        0: np.array([1 + 2j, 3j]),
        1: np.zeros(3, dtype=[("a", "f4"), ("b", "i4")]),
        3: np.array(["a", "b"]),
    }
    for row, array in cases.items():
        np.save(folder / f"{row:06d}.npy", array)
        response = client.get(f"/api/results/{rid}/artifacts/{row}")
        assert response.status_code == 400
        assert "見せられません" in response.json()["detail"]

    # オブジェクトの配列 (pickle) は読まない
    np.save(folder / "000004.npy", np.array([{"a": 1}], dtype=object),
            allow_pickle=True)
    response = client.get(f"/api/results/{rid}/artifacts/4")
    assert response.status_code == 400
    assert "読めません" in response.json()["detail"]

    # 表の外を指す行
    table = pd.read_csv(place / "predictions.csv", encoding="utf-8-sig")
    table.loc[5, "artifact"] = "../../outside.npy"
    table.to_csv(place / "predictions.csv", index=False, encoding="utf-8-sig")
    assert client.get(f"/api/results/{rid}/artifacts/5").status_code == 400
    assert client.get(
        f"/api/results/{rid}/artifacts/5/file"
    ).status_code == 400


def test_group_sentences_and_origin(
    tmp_path: Path, client: TestClient
) -> None:
    """
    まとめる列のある列の出力の結果で, 出どころの文が全体の文とまとめる
    列ごとの文の数え方を崩さないことを確かめる関数

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
    # 今の処理の出力を正解にした分類の一式
    csv, dataset = labeled(tmp_path / "data")
    origin = {"type": "process", "name": "手作業の判定"}
    path = make_v3(
        tmp_path, {"dataset": dataset}, "cls", head_spec(2),
        target={"kind": "column", "origin": origin}
    )
    bundle = register(client, path)
    response = client.post(f"/api/bundles/{bundle['id']}/predict", json={
        "source": str(csv), "rules": {"group_by": "label"},
    })
    assert response.status_code == 201, response.text
    result = response.json()

    # 全体の文は件数と内訳, 残りはまとめる列ごと, 出どころは別
    sentences = result["sentences"]
    assert result["overall_count"] == 2
    assert sentences[0].startswith("40 件のうち 40 件を予測")
    assert all(s.startswith(("良:", "悪:")) for s in sentences[2:])
    assert len(sentences) == 4
    assert not any("手作業の判定" in s for s in sentences)
    assert "手作業の判定 の出力" in result["origin_note"]

    # 出し直しても同じ
    again = client.post(f"/api/results/{result['id']}/conclude",
                        json={"rules": {"group_by": "label",
                                        "min_confidence": 0.9}}).json()
    assert again["overall_count"] == 2
    assert "手作業の判定 の出力" in again["origin_note"]
    assert not any("手作業の判定" in s for s in again["sentences"])


def test_model_cache(
    tmp_path: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    一覧で組み立てたモデルを追い出さず, 一式のファイルが変われば読み直し,
    登録を消すと手放すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    client: TestClient
        クライアント
    monkeypatch: pytest.MonkeyPatch
        覚える数の環境変数を替える

    Returns
    ----------
    None
    """
    from ml_provide import cache

    # 覚えるモデルは 1 つにし, 2 つの一式を一覧に並べる
    monkeypatch.setenv(cache.MODEL_CACHE_ENV, "1")
    assert cache.model_cache_size() == 1
    monkeypatch.setenv(cache.MODEL_CACHE_ENV, "zero")
    assert cache.model_cache_size() == cache.MODEL_CACHE_DEFAULT
    monkeypatch.setenv(cache.MODEL_CACHE_ENV, "1")
    paths = write_unlabeled(tmp_path / "data")
    first = register(client, make_v3(tmp_path, paths, "first", head_spec(5),
                                     form="vector"))
    second = register(client, make_v3(tmp_path, paths, "second",
                                      head_spec(4), form="vector"))
    response = client.post(f"/api/bundles/{first['id']}/predict",
                           json={"source": str(paths["csv"])})
    assert response.status_code == 201

    # 一覧と別の一式の画面を開いても, 推論した一式のモデルは残る
    for _ in range(2):
        listed = {b["id"]: b for b in client.get("/api/bundles").json()}
        assert listed[first["id"]]["model_loaded"] is True
        assert listed[second["id"]]["model_loaded"] is False
        client.get(f"/api/bundles/{second['id']}")
    assert str(Path(first["path"])) in cache.MODELS

    # 定義のファイルが変わると読み直す (覚えたモデルは使わない)
    spec = Path(first["path"]) / "model_spec.yaml"
    spec.write_text(spec.read_text(encoding="utf-8") + "\n",
                    encoding="utf-8")
    assert client.get(
        f"/api/bundles/{first['id']}"
    ).json()["model_loaded"] is False

    # 登録を消すと, 覚えたものを手放す
    client.post(f"/api/bundles/{first['id']}/predict",
                json={"source": str(paths["csv"])})
    assert str(Path(first["path"])) in cache.MODELS
    assert client.delete(f"/api/bundles/{first['id']}").status_code == 204
    assert str(Path(first["path"])) not in cache.MODELS
    assert str(Path(first["path"])) not in cache.DESCRIPTIONS


def test_failed_inference_leaves_nothing(
    tmp_path: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    推論の後の書き出しに失敗すると, 結果を残さないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    client: TestClient
        クライアント
    monkeypatch: pytest.MonkeyPatch
        書き出しを失敗させる

    Returns
    ----------
    None
    """
    import ml_provide.api as api

    # 書き出しの途中の誤り
    paths = write_unlabeled(tmp_path / "data")
    bundle = register(client, make_v3(tmp_path, paths, "embed", head_spec(5),
                                      form="vector"))
    monkeypatch.setattr(api, "count_into", broken_count)
    response = client.post(f"/api/bundles/{bundle['id']}/predict",
                           json={"source": str(paths["csv"])})
    assert response.status_code == 400
    assert "書けません" in response.json()["detail"]
    assert client.get("/api/results").json() == []
    assert list((tmp_path / "home" / "results").iterdir()) == []


def broken_count(place: Path, meta: dict[str, Any]) -> None:
    """
    書き出しに失敗する数え直し (テストのため)

    Parameters
    ----------
    place: Path
        結果のディレクトリ
    meta: dict[str, Any]
        result.yaml の中身

    Returns
    ----------
    None
    """
    raise ValueError("書けません")


def labeled(root: Path) -> tuple[Path, Path]:
    """
    2 クラスの分類の表とデータセットを作る関数

    Parameters
    ----------
    root: Path
        作るディレクトリ

    Returns
    ----------
    paths: tuple[Path, Path]
        CSV とデータセット
    """
    # 3 つの数の列と 2 つのクラス
    rng = np.random.default_rng(1)
    root.mkdir(parents=True)
    frame = pd.DataFrame({
        "a": rng.normal(size=40), "b": rng.normal(size=40),
        "c": rng.normal(size=40), "label": rng.choice(["良", "悪"], size=40),
    })
    csv = root / "table.csv"
    frame.to_csv(csv, index=False)
    recipe = {
        "format_version": 1, "target": "label", "task": "classification",
        "steps": [{"op": "standardize", "columns": "auto"}],
        "split": {"method": "random", "valid": 0.2, "seed": 0},
    }
    export_dataset(load_data(csv), recipe, root / "dataset")
    paths = (csv, root / "dataset")

    return paths


def test_version_two_bundle_unchanged(
    tmp_path: Path, client: TestClient
) -> None:
    """
    版 2 の列の予測の一式が前と同じに動くことを確かめる関数

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
    # 分類のデータセットと版 3 の一式から, 足した項目を除いた版 2 の一式
    rng = np.random.default_rng(1)
    root = tmp_path / "data"
    root.mkdir()
    frame = pd.DataFrame({
        "a": rng.normal(size=40), "b": rng.normal(size=40),
        "c": rng.normal(size=40), "label": rng.choice(["良", "悪"], size=40),
    })
    csv = root / "table.csv"
    frame.to_csv(csv, index=False)
    recipe = {
        "format_version": 1, "target": "label", "task": "classification",
        "steps": [{"op": "standardize", "columns": "auto"}],
        "split": {"method": "random", "valid": 0.2, "seed": 0},
    }
    export_dataset(load_data(csv), recipe, root / "dataset")
    paths = {"csv": csv, "dataset": root / "dataset"}
    made = make_v3(tmp_path, paths, "v3", head_spec(2),
                   conclusion={"min_confidence": 0.55})
    old = tmp_path / "v2"
    shutil.copytree(made, old)
    info = yaml.safe_load((old / "inference.yaml").read_text(encoding="utf-8"))
    for key in ("role", "output", "target", "procedure"):
        info.pop(key)
    info["model"].pop("pretrained")
    info["format_version"] = 2
    (old / "inference.yaml").write_text(
        yaml.safe_dump(info, allow_unicode=True), encoding="utf-8"
    )

    # 説明は受け持ちが予測する, 出力の形がクラス, 正解の種類が列
    bundle = register(client, old)
    description = bundle["description"]
    assert description["format_version"] == 2
    assert description["role"] == "predict"
    assert description["column_output"] is True
    assert description["target"] == "label"
    assert description["target_info"] == {"kind": "column"}

    # 推論, 出し直し, 意思決定は前と同じ (結果の形は版 1)
    response = client.post(f"/api/bundles/{bundle['id']}/predict", json={
        "source": str(csv), "rules": {"group_by": "label"},
    })
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["format_version"] == 1 and result["artifacts"] is None
    assert result["column_output"] is True
    assert result["rules"] == {"min_confidence": 0.55, "group_by": "label"}
    assert "decision" in result["columns"]
    place = tmp_path / "home" / "results" / result["id"]
    assert not (place / "artifacts").exists()
    again = client.post(f"/api/results/{result['id']}/conclude",
                        json={"rules": {}})
    assert again.status_code == 200 and again.json()["rules"] == {}
    decided = client.post(f"/api/results/{result['id']}/decide", json={
        "decision": {"type": "triage", "positive": "良", "miss_cost": 5,
                     "false_alarm_cost": 1},
    })
    assert decided.status_code == 200, decided.text
    assert client.get(
        f"/api/results/{result['id']}/artifacts/0"
    ).status_code == 404


def test_procedures_from_startup_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    自作の手順は起動の設定で明示したモジュールだけから登録し, 登録簿に
    無い手順の一式は設定を案内して止まることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        import の場所と環境変数を替える

    Returns
    ----------
    None
    """
    # 登録簿は 1 つの処理で共有するので, 終わったら外す
    # (無い名前を置いてから消すと, 終わりに無い状態へ戻る)
    monkeypatch.setitem(ml_inference.PROCEDURES, "provide_walk", None)
    monkeypatch.delitem(ml_inference.PROCEDURES, "provide_walk")

    # 読めないモジュールは起動を止める
    with pytest.raises(RuntimeError, match="no_such_module"):
        create_app(data_dir=tmp_path / "home", token=TOKEN,
                   procedures=["no_such_module"])

    # 環境変数で明示したモジュールを起動のときに読む
    folder = tmp_path / "modules"
    folder.mkdir()
    (folder / f"{MODULE}.py").write_text(MODULE_CODE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(folder))
    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    monkeypatch.setenv("ML_PROVIDE_PROCEDURES", f" {MODULE} ,")
    app = create_app(data_dir=tmp_path / "home", token=TOKEN,
                     allowed_hosts=["testserver"])
    client = TestClient(app, headers={"x-provide-token": TOKEN})
    info = client.get("/api/info").json()
    assert info["procedure_modules"] == [MODULE]
    names = {item["name"]: item["builtin"] for item in info["procedures"]}
    assert names["provide_walk"] is False and names["ddim"] is True

    # 自作の手順の一式を推論する (手順の名前, 回数, 種を見せる)
    paths = write_unlabeled(tmp_path / "data")
    spec = head_spec(3)
    procedure = {"name": "provide_walk", "steps": 3, "seed": 7}
    path = make_v3(
        tmp_path, paths, "walk", spec, form="vector", role="generate",
        inputs={"x": {"source": "state"}}, output_options={"shape": [3]},
        procedure=procedure
    )
    bundle = register(client, path)
    assert bundle["description"]["procedure"] == procedure
    response = client.post(f"/api/bundles/{bundle['id']}/predict",
                           json={"source": str(paths["csv"])})
    assert response.status_code == 201, response.text

    # 登録簿に無い手順は, 起動の設定に足すよう案内 (一式のコードは読まない)
    file = path / "inference.yaml"
    info = yaml.safe_load(file.read_text(encoding="utf-8"))
    info["procedure"]["name"] = "never_registered"
    (path / "inference.yaml").write_text(
        yaml.safe_dump(info, allow_unicode=True), encoding="utf-8"
    )
    view = client.get(f"/api/bundles/{bundle['id']}").json()
    assert view["ok"] is False
    assert "never_registered" in view["error"]
    assert "ML_PROVIDE_PROCEDURES" in view["error"]
    response = client.post(f"/api/bundles/{bundle['id']}/predict",
                           json={"source": str(paths["csv"])})
    assert response.status_code == 400
    assert "--procedures" in response.json()["detail"]
    response = client.post("/api/bundles", json={"path": str(path),
                                                 "name": "again"})
    assert response.status_code == 400
    assert "起動の設定" in response.json()["detail"]


# 自作の部品を宣言するモジュールの中身 ({unit} は部品の名前)
UNIT_CODE = '''
from torch import nn


class Twice(nn.Module):
    def __init__(self, features: int = 3):
        super(Twice, self).__init__()
        self.fc = nn.Linear(features, features)

    def forward(self, x):
        out = self.fc(x) * 2.0
        return out


UNITS = [{{"cls": Twice, "name": "{unit}"}}]
'''


def test_units_from_startup_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    自作の部品は起動の設定で明示したモジュールだけから登録し, 登録されて
    いない部品の一式は設定を案内して止まることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        import の場所と環境変数を替える

    Returns
    ----------
    None
    """
    # 読めないモジュールと, 部品を宣言していないモジュールは起動を止める
    folder = tmp_path / "modules"
    folder.mkdir()
    monkeypatch.syspath_prepend(str(folder))
    module = f"provide_units_{tmp_path.name.replace('-', '_')}"
    (folder / f"{module}_empty.py").write_text("VALUE = 1\n")
    with pytest.raises(RuntimeError, match="no_such_units"):
        create_app(data_dir=tmp_path / "home", token=TOKEN,
                   units=["no_such_units"])
    with pytest.raises(RuntimeError, match=f"{module}_empty"):
        create_app(data_dir=tmp_path / "home", token=TOKEN,
                   units=[f"{module}_empty"])

    # 環境変数で明示したモジュールを起動のときに読む
    unit = f"{module}.Twice"
    (folder / f"{module}.py").write_text(
        UNIT_CODE.format(unit=unit), encoding="utf-8"
    )
    monkeypatch.setenv("ML_PROVIDE_UNITS", f" {module} ,")
    app = create_app(data_dir=tmp_path / "home", token=TOKEN,
                     allowed_hosts=["testserver"], procedures=[])
    client = TestClient(app, headers={"x-provide-token": TOKEN})
    info = client.get("/api/info").json()
    assert info["unit_modules"] == [{"name": module, "units": [unit]}]

    # 自作の部品の一式を登録して推論する
    paths = write_unlabeled(tmp_path / "data")
    spec = head_spec(4)
    spec["nodes"]["lab"] = {"unit": unit, "params": {"features": 3},
                            "inputs": "x"}
    spec["nodes"]["head"]["inputs"] = "lab"
    path = make_v3(tmp_path, paths, "lab", spec, form="vector",
                   role="group")
    bundle = register(client, path)
    assert bundle["ok"] is True
    url = f"/api/bundles/{bundle['id']}/predict"
    response = client.post(url, json={"source": str(paths["csv"])})
    assert response.status_code == 201, response.text

    # 登録されていない部品は, 名前と起動の設定を案内 (定義の名前は読まない)
    file = path / "model_spec.yaml"
    written = yaml.safe_load(file.read_text(encoding="utf-8"))
    written["nodes"]["lab"]["unit"] = "never_lab.Block"
    file.write_text(yaml.safe_dump(written), encoding="utf-8")
    view = client.get(f"/api/bundles/{bundle['id']}").json()
    assert view["ok"] is False
    assert "'never_lab.Block'" in view["error"]
    assert "ML_PROVIDE_UNITS" in view["error"]
    listed = client.get("/api/bundles").json()
    assert any("ML_PROVIDE_UNITS" in (item["error"] or "")
               for item in listed)
    response = client.post(url, json={"source": str(paths["csv"])})
    assert response.status_code == 400
    assert "--units" in response.json()["detail"]
    response = client.post("/api/bundles", json={"path": str(path),
                                                 "name": "again"})
    assert response.status_code == 400
    assert "never_lab.Block" in response.json()["detail"]
    assert "never_lab" not in sys.modules

    # 画面で作る一式も, 登録されていない部品なら同じ案内
    spec_file = tmp_path / "missing.yaml"
    spec_file.write_text(yaml.safe_dump(written), encoding="utf-8")
    response = client.post("/api/bundles/create", json={
        "name": "missing", "spec": str(spec_file),
        "dataset": str(paths["dataset"]), "weights": str(tmp_path / "lab.pt"),
        "form": "vector", "role": "group",
    })
    assert response.status_code == 400
    assert "ML_PROVIDE_UNITS" in response.json()["detail"]
