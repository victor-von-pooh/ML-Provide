import json
import shutil
from pathlib import Path
from typing import Any

import ml_inference
import ml_quest
import numpy as np
import pytest
import torch
import yaml
from fastapi.testclient import TestClient
from ml_eda import export_capture_clouds, export_image_masks, synthetic
from ml_modelcontrol import build_model, load_example
from ml_quest.examples import example_problem
from PIL import Image

from ml_provide.app import create_app

# テストで使うトークン
TOKEN = "test-token"

# 選ぶ点の数 (学習と推論で同じ値)
POINTS = 512


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """
    一時ディレクトリを置き場にしたアプリのクライアントを作る関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        環境変数の差し替え (ML-Pathfinder の置き場)

    Returns
    ----------
    client: TestClient
        トークン付きのクライアント
    """
    monkeypatch.setenv("ML_PATHFINDER_HOME", str(tmp_path / "pathfinder"))
    app = create_app(
        data_dir=tmp_path / "home", token=TOKEN, allowed_hosts=["testserver"],
        procedures=[]
    )
    client = TestClient(app, headers={"x-provide-token": TOKEN})

    return client


@pytest.fixture(scope="module")
def items(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """
    ML-EDA の合成の見本から版 2 のデータセット 2 つと, ML-ModelControl の
    見本の定義と重みを作る関数

    Parameters
    ----------
    tmp_path_factory: pytest.TempPathFactory
        pytest が用意する一時ディレクトリの作り手

    Returns
    ----------
    paths: dict[str, Path]
        元の画像と撮影, データセット, 定義, 重みの場所
    """
    # 画像と領域, 撮影と点群 (検証用とテスト用も)
    root = tmp_path_factory.mktemp("items")
    split = {"method": "random", "valid": 0.25, "test": 0.25, "seed": 0}
    synthetic.write_image_masks(root / "raw_m", count=8, size=(16, 16))
    export_image_masks(
        root / "raw_m" / "images", root / "raw_m" / "masks",
        root / "masks", split=split,
        classes={0: "background", 1: "circle", 2: "square"},
        origin={"type": "synthetic"}
    )
    synthetic.write_shape_captures(root / "raw_c", count=6, views=3,
                                   size=(16, 16), points=64)
    export_capture_clouds(
        root / "raw_c" / "captures", root / "raw_c" / "clouds",
        root / "clouds", split=split
    )

    # 版 3: 24 x 24 の画像を中身の手順で 16 x 16 にそろえたデータセット
    synthetic.write_image_masks(root / "raw_m3", count=8, size=(24, 24))
    export_image_masks(
        root / "raw_m3" / "images", root / "raw_m3" / "masks",
        root / "masks_v3", split=split,
        classes={0: "background", 1: "circle", 2: "square"},
        item_steps={"image": [{"op": "resize", "width": 16,
                               "height": 16}]}
    )

    # 領域の値が飛ぶ (0, 3, 7) データセット (同じ画像の領域の値を付け替え)
    shutil.copytree(root / "raw_m", root / "raw_gap")
    for mask in (root / "raw_gap" / "masks").glob("*.png"):
        values = np.asarray(Image.open(mask))
        Image.fromarray(
            np.array([0, 3, 7], dtype=np.uint8)[values]
        ).save(mask)
    export_image_masks(
        root / "raw_gap" / "images", root / "raw_gap" / "masks",
        root / "masks_gap", split=split,
        classes={0: "background", 3: "circle", 7: "square"}
    )

    # 見本の定義 (領域は背景を含む 3 クラス) と乱数の初期値の重み
    paths = {
        "images": root / "raw_m" / "images", "masks": root / "masks",
        "captures": root / "raw_c" / "captures", "clouds": root / "clouds",
        "images_v3": root / "raw_m3" / "images",
        "masks_v3": root / "masks_v3", "masks_gap": root / "masks_gap",
    }
    unet = load_example("unet_segmentation")
    unet["nodes"]["head"]["params"]["out_channels"] = 3
    for name, spec in (("mask", unet),
                       ("cloud", load_example("point_cloud_predictor"))):
        paths[f"{name}_spec"] = root / f"{name}_spec.yaml"
        paths[f"{name}_spec"].write_text(yaml.safe_dump(spec),
                                         encoding="utf-8")
        paths[f"{name}_weights"] = root / f"{name}.pt"
        torch.save(build_model(spec, seed=0).state_dict(),
                   paths[f"{name}_weights"])

    return paths


def fake_run(path: Path, weights: Path, monitor: str) -> Path:
    """
    重みと run.json だけを持つ ML-Found の実験のディレクトリを作る関数

    Parameters
    ----------
    path: Path
        実験のディレクトリ
    weights: Path
        写す重み
    monitor: str
        監視した指標

    Returns
    ----------
    path: Path
        実験のディレクトリ
    """
    path.mkdir(parents=True)
    shutil.copy(weights, path / "final_model.pt")
    state = {"format_version": 1, "status": "completed", "monitor": monitor,
             "best_value": 0.5, "best_epoch": 1, "final_weights": "best"}
    (path / "run.json").write_text(json.dumps(state), encoding="utf-8")

    return path


def create(
    client: TestClient, items: dict[str, Path], name: str,
    weights: Path, extra: dict[str, Any]
) -> Any:
    """
    学習の結果から一式を作る API を呼ぶ関数

    Parameters
    ----------
    client: TestClient
        クライアント
    items: dict[str, Path]
        見本の場所
    name: str
        mask か cloud
    weights: Path
        重みか実験のディレクトリ
    extra: dict[str, Any]
        足す欄 (items など)

    Returns
    ----------
    response: Any
        応答
    """
    dataset = items["masks"] if name == "mask" else items["clouds"]
    response = client.post("/api/bundles/create", json={
        "name": f"版 2 の{name}", "spec": str(items[f"{name}_spec"]),
        "dataset": str(dataset), "weights": str(weights), **extra,
    })

    return response


def test_describe_dataset(
    client: TestClient, items: dict[str, Path], tmp_path: Path
) -> None:
    """
    データセットの説明が, 版 2 の項目と点を選ぶ欄の要否を返し, 版 1 は
    版だけ, 無い場所と知らない版は理由を返すことを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    items: dict[str, Path]
        見本の場所
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 画像と領域: 入力は画像 (点は選ばない), 値と名前
    masks = client.get("/api/datasets/describe",
                       params={"path": str(items["masks"])}).json()
    assert masks["format_version"] == 2 and masks["role"] == "predict"
    assert masks["target"]["kind"] == "mask"
    assert masks["target"]["origin"]["type"] == "synthetic"
    assert [(e["name"], e["kind"], e["role"], e["points"])
            for e in masks["entries"]] == [
        ("image", "image", "input", False), ("mask", "mask", "target", False)
    ]
    assert masks["classes"][1] == {"value": 1, "name": "circle"}
    assert masks["splits"] == {"train": 4, "valid": 2, "test": 2}
    assert masks["normalized"] is False

    assert [e["steps"] for e in masks["entries"]] == [[], []]

    # 撮影と点群: 入力の撮影は点を選ぶ数が要る. 正規化あり
    clouds = client.get("/api/datasets/describe",
                        params={"path": str(items["clouds"])}).json()
    assert clouds["target"]["kind"] == "point_cloud"
    assert clouds["entries"][0]["points"] is True
    assert clouds["classes"] == [] and clouds["normalized"] is True

    # 版 3: 版 2 と同じ項目に, 書き出しでかけた中身の手順 (領域は ML-EDA
    # が画像に合わせた resize_mask)
    resized = client.get("/api/datasets/describe",
                         params={"path": str(items["masks_v3"])}).json()
    assert resized["format_version"] == 3
    assert resized["target"]["kind"] == "mask"
    steps = {e["name"]: e["steps"] for e in resized["entries"]}
    assert steps["image"] == [{"op": "resize", "width": 16, "height": 16}]
    assert [step["op"] for step in steps["mask"]] == ["resize_mask"]

    # 版 1 は版だけ, 無い場所は 404, 知らない版は ML-EDA の文で 400
    old = tmp_path / "old"
    old.mkdir()
    (old / "dataset.yaml").write_text("format_version: 1\n")
    response = client.get("/api/datasets/describe",
                          params={"path": str(old)})
    assert response.json() == {"path": str(old.resolve()),
                               "format_version": 1}
    response = client.get("/api/datasets/describe",
                          params={"path": str(tmp_path / "none")})
    assert response.status_code == 404
    (old / "dataset.yaml").write_text("format_version: 9\n")
    response = client.get("/api/datasets/describe",
                          params={"path": str(old)})
    assert response.status_code == 400
    assert "format_version 9" in response.json()["detail"]


def test_mask_bundle_from_run(
    client: TestClient, items: dict[str, Path], tmp_path: Path
) -> None:
    """
    版 2 の画像と領域のデータセットと実験から版 4 の一式を作り, 元の画像の
    ディレクトリを推論して領域の値の格子を見られることを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    items: dict[str, Path]
        見本の場所
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 実験から作ると, 版 4 で 1 件の扱いと実験の状態と IoU の確かめを持つ
    run = fake_run(tmp_path / "run", items["mask_weights"], "valid/loss")
    response = create(client, items, "mask", run, {})
    assert response.status_code == 201, response.text
    description = response.json()["description"]
    assert description["format_version"] == 4
    assert description["kind"] == "items"
    assert description["output"] == {
        "form": "mask", "classes": ["background", "circle", "square"]
    }
    assert description["items"]["values"] == [0, 1, 2]
    assert description["items"]["input_kind"] == "image"
    assert description["run"]["status"] == "completed"
    assert description["check"]["metric"] == "iou"

    # 元の画像のディレクトリを推論すると, 件数だけの結論と領域の格子
    bundle_id = response.json()["id"]
    response = client.post(f"/api/bundles/{bundle_id}/predict",
                           json={"source": str(items["images"])})
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["summary"] == {"rows": 8, "predicted": 8, "skipped": 0}
    assert result["output_form"] == "mask"
    assert result["column_output"] is False
    preview = client.get(
        f"/api/results/{result['id']}/artifacts/0"
    ).json()
    assert preview["view"] == "grid" and preview["shape"] == [16, 16]
    assert {item["label"] for item in preview["labels"]} <= {0, 1, 2}

    # 版 2 のデータセットのディレクトリも推論できる (全ての分割の件)
    response = client.post(f"/api/bundles/{bundle_id}/predict",
                           json={"source": str(items["masks"])})
    assert response.status_code == 201, response.text
    assert response.json()["summary"]["rows"] == 8

    # 結論の決め方と意思決定は断る
    response = client.post(
        f"/api/bundles/{bundle_id}/predict",
        json={"source": str(items["images"]),
              "rules": {"min_confidence": 0.5}}
    )
    assert response.status_code == 400


def test_mask_bundle_gap_values(
    client: TestClient, items: dict[str, Path], tmp_path: Path
) -> None:
    """
    領域の値が飛ぶ (0, 3, 7) データセットの一式で, 推論した領域が値で出て,
    値から名前を引けることを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    items: dict[str, Path]
        見本の場所
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # クラスの番号 0, 1, 2 は値 0, 3, 7 の順. 重みは全ての画素を番号 2
    # (値 7) にするように, 最後の層の偏りを変えたもの
    state = torch.load(items["mask_weights"])
    state["nodes.head.bias"] = torch.tensor([0.0, 0.0, 100.0])
    torch.save(state, tmp_path / "square.pt")
    run = fake_run(tmp_path / "run", tmp_path / "square.pt", "valid/loss")
    response = create(client, {**items, "masks": items["masks_gap"]},
                      "mask", run, {})
    assert response.status_code == 201, response.text
    description = response.json()["description"]
    assert description["items"]["values"] == [0, 3, 7]
    assert description["output"]["classes"] == [
        "background", "circle", "square"
    ]

    # 領域の格子は番号 (2) でなく値 (7) で, 値の並びと名前の並びから名前
    # を引ける (画面の凡例と同じ引き方)
    response = client.post(
        f"/api/bundles/{response.json()['id']}/predict",
        json={"source": str(items["images"])}
    )
    assert response.status_code == 201, response.text
    preview = client.get(
        f"/api/results/{response.json()['id']}/artifacts/0"
    ).json()
    assert [item["label"] for item in preview["labels"]] == [7]
    names = dict(zip(description["items"]["values"],
                     description["output"]["classes"], strict=True))
    assert names[7] == "square"


def test_mask_bundle_version3(
    client: TestClient, items: dict[str, Path], tmp_path: Path
) -> None:
    """
    版 3 (中身の手順で大きさをそろえた) のデータセットからも版 4 の一式を
    作り, 大きさの違う元の画像を推論すると元の画像の大きさに戻した領域が
    返ることを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    items: dict[str, Path]
        見本の場所
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 版 2 と同じ欄で作れる
    run = fake_run(tmp_path / "run", items["mask_weights"], "valid/loss")
    response = client.post("/api/bundles/create", json={
        "name": "版 3 の領域", "spec": str(items["mask_spec"]),
        "dataset": str(items["masks_v3"]), "weights": str(run),
    })
    assert response.status_code == 201, response.text
    description = response.json()["description"]
    assert description["format_version"] == 4
    assert description["kind"] == "items"
    assert description["check"]["metric"] == "iou"

    # 24 x 24 の元の画像は, 一式が手順をかけて 16 x 16 で推論し, 領域を
    # 元の 24 x 24 に戻して返す
    bundle_id = response.json()["id"]
    response = client.post(f"/api/bundles/{bundle_id}/predict",
                           json={"source": str(items["images_v3"])})
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["summary"] == {"rows": 8, "predicted": 8, "skipped": 0}
    preview = client.get(
        f"/api/results/{result['id']}/artifacts/0"
    ).json()
    assert preview["view"] == "grid" and preview["shape"] == [24, 24]
    assert {item["label"] for item in preview["labels"]} <= {0, 1, 2}

    # データセットのディレクトリの件は, そろえた 16 x 16 のまま
    response = client.post(f"/api/bundles/{bundle_id}/predict",
                           json={"source": str(items["masks_v3"])})
    assert response.status_code == 201, response.text
    result = response.json()
    preview = client.get(
        f"/api/results/{result['id']}/artifacts/0"
    ).json()
    assert preview["shape"] == [16, 16]


def test_cloud_bundle_needs_points(
    client: TestClient, items: dict[str, Path]
) -> None:
    """
    撮影と点群の一式は選ぶ点の数が要り, 作った一式で元の撮影を推論すると
    元の座標の点群を返すことを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    items: dict[str, Path]
        見本の場所

    Returns
    ----------
    None
    """
    # 選ぶ点の数が無ければ ML-Inference の理由で断る
    response = create(client, items, "cloud", items["cloud_weights"], {})
    assert response.status_code == 400
    assert "points" in response.json()["detail"]

    # 点の数と種を渡すと, 正規化を戻す一式 (Chamfer の確かめ)
    response = create(client, items, "cloud", items["cloud_weights"], {
        "items": {"options": {"points": POINTS, "target_points": 64,
                              "seed": 1}}
    })
    assert response.status_code == 201, response.text
    description = response.json()["description"]
    assert description["format_version"] == 4
    assert description["items"]["prepare"] == "capture_points"
    assert description["items"]["options"] == {
        "points": POINTS, "target_points": 64, "seed": 1
    }
    assert description["items"]["restore"] == "shift"
    assert description["check"]["metric"] == "chamfer"

    # 元の撮影を推論すると, 正規化の量と元の座標の点
    bundle_id = response.json()["id"]
    response = client.post(f"/api/bundles/{bundle_id}/predict",
                           json={"source": str(items["captures"])})
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["summary"]["predicted"] == 6
    assert {"shift_x", "shift_y", "shift_z"} <= set(result["columns"])
    rows = client.get(f"/api/results/{result['id']}/rows").json()["rows"]
    preview = client.get(
        f"/api/results/{result['id']}/artifacts/0"
    ).json()
    assert preview["view"] == "points" and preview["shape"] == [256, 3]

    # 元の座標 = 正規化した座標 - shift (正規化した座標は原点の近く)
    points = np.asarray(preview["points"])
    shift = np.array([rows[0][f"shift_{axis}"] for axis in "xyz"])
    assert np.allclose(points.mean(axis=0) + shift, 0.0, atol=0.5)


def test_experiments_list_v4_bundle(
    client: TestClient, items: dict[str, Path], tmp_path: Path
) -> None:
    """
    骨組みの学習が実験の中に作った版 4 の一式も実験の一覧に出て, 一覧から
    登録できることを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    items: dict[str, Path]
        見本の場所
    tmp_path: Path
        一時ディレクトリ

    Returns
    ----------
    None
    """
    # 記録したプロジェクトの実験の中に, 版 4 の一式 (骨組みの inference)
    project = tmp_path / "project"
    run = fake_run(project / "outputs" / "masks" / "2026-01-02" / "09-00-00",
                   items["mask_weights"], "valid/iou")
    ml_inference.make_bundle(
        run / "inference", items["mask_spec"], items["masks"], run,
        name="masks"
    )
    ml_quest.save_problem(
        {**example_problem("総菜の作る量"),
         "project": {"path": str(project.resolve())}},
        tmp_path / "pathfinder" / "problems" / "a"
    )

    # 一覧に出て, 場所で登録すると版 4 の一式
    rows = client.get("/api/experiments").json()["experiments"]
    assert [row["experiment"] for row in rows] == [
        "masks/2026-01-02/09-00-00"
    ]
    response = client.post("/api/bundles", json={"path": rows[0]["path"]})
    assert response.status_code == 201, response.text
    assert response.json()["description"]["format_version"] == 4
