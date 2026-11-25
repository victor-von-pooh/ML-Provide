import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import ml_inference
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient
from test_bundle_v3 import register

from ml_provide.app import create_app

# テストで使うトークン
TOKEN = "test-token"

# 数にできない値の行の理由 (ML-Inference の文)
NOT_NUMBER = "列 spend の値 '?' を数にできません"


def load_maker() -> ModuleType:
    """
    画面の通しのテストが使う, 群分けの実験と一式を作るスクリプトを読む関数

    Parameters
    ----------
    None

    Returns
    ----------
    module: ModuleType
        frontend/e2e/make_group.py (骨組みと同じ成果物を書く関数を持つ)
    """
    # 骨組みの成果物の書き方を 2 か所に持たないため, 同じスクリプトを使う
    path = Path(__file__).parents[2] / "frontend" / "e2e" / "make_group.py"
    spec = importlib.util.spec_from_file_location("make_group", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


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


@pytest.fixture
def group_files(tmp_path: Path) -> tuple[Path, Path]:
    """
    骨組みのプロジェクトの実験の中の群分けの一式と, 新しい行の表を作る
    関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    files: tuple[Path, Path]
        一式のディレクトリと, 新しい行の表 (数にできない行を 1 つ含む)
    """
    load_maker().main(tmp_path / "project", tmp_path / "tables")
    bundle = next((tmp_path / "project" / "outputs").glob("*/*/*/inference"))
    files = (bundle, tmp_path / "tables" / "customers.csv")

    return files


def test_group_bundle(
    client: TestClient, group_files: tuple[Path, Path]
) -> None:
    """
    群分けの一式を登録して推論し, 群の要約と群の安定性を結論に残し,
    行ごとの群を読めて群で絞れることを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    group_files: tuple[Path, Path]
        一式と新しい行の表

    Returns
    ----------
    None
    """
    # 版 9 の一式を登録すると, 説明に群の欄があり, 問題の設定を読まない
    path, table = group_files
    bundle = register(client, path)
    description = bundle["description"]
    assert (description["kind"], description["format_version"]) == (
        "group", 9
    )
    assert description["output"]["form"] == "cluster"
    assert description["group"]["columns"] == ["spend", "visits", "region"]
    state = client.get(f"/api/bundles/{bundle['id']}/problem").json()
    assert state["linked"] is None and state["hint"] is None

    # 結論は ML-Inference の群分けの結論のまま (読み直して数え直さない)
    response = client.post(
        f"/api/bundles/{bundle['id']}/predict", json={"source": str(table)}
    )
    assert response.status_code == 201, response.text
    result: dict[str, Any] = response.json()
    summary = result["summary"]
    assert (summary["rows"], summary["predicted"], summary["skipped"],
            summary["k"]) == (16, 15, 1, 3)
    groups = summary["groups"]
    assert [row["group"] for row in groups] == [0, 1, 2, "全体"]
    assert list(groups[0]) == ["group", "rows", "share", "stability",
                               "spend", "visits", "region"]
    assert [row["rows"] for row in groups] == [5, 5, 5, 15]
    assert groups[0]["region"] == "north (100 %)"
    assert summary["stability"]["ari"] == 0.97
    assert summary["stability"]["baseline"] == (
        "列の特徴をそのまま k-means で分けた群"
    )
    assert result["sentences"][0] == (
        "16 行のうち 15 行を 3 個の群に割り当てた (1 行は割り当てられず)"
    )
    assert "学習の群の安定性" in result["sentences"][2]
    assert result["overall_count"] == len(result["sentences"])
    assert (result["output_form"], result["column_output"]) == (
        "cluster", False
    )

    # ML-Inference の group_summary に同じ表を渡した行と同じ
    frame = ml_inference.predict(str(path), table)
    assert groups == ml_inference.group_summary(str(path), frame)

    # 結果に群を作る列と学習のときの群の要約を残す (開き直しても同じ)
    grouping = result["grouping"]
    assert grouping["k"] == 3
    assert grouping["columns"] == ["spend", "visits", "region"]
    assert [row["group"] for row in grouping["learned"]] == [
        0, 1, 2, "全体"
    ]
    assert grouping["learned"][0]["rows"] == 20
    again = client.get(f"/api/results/{result['id']}").json()
    assert again["summary"] == summary and again["grouping"] == grouping

    # 行ごとの群は整数で読み, 割り当てない行は欠損と理由
    rows = client.get(
        f"/api/results/{result['id']}/rows", params={"limit": 100}
    ).json()
    assert rows["total"] == 16
    assert rows["rows"][0]["group"] == 0
    assert isinstance(rows["rows"][0]["distance"], float)
    last = rows["rows"][-1]
    assert (last["status"], last["group"], last["reason"]) == (
        "skipped", None, NOT_NUMBER
    )

    # 群の番号で絞る
    one = client.get(
        f"/api/results/{result['id']}/rows", params={"group": 1}
    ).json()
    assert one["total"] == 5
    assert {row["group"] for row in one["rows"]} == {1}
    assert {row["region"] for row in one["rows"]} == {"south"}


def test_group_refusals(
    tmp_path: Path, client: TestClient, group_files: tuple[Path, Path]
) -> None:
    """
    群分けの一式に結論の決め方, 意思決定, 並べるの欄を渡すと断り,
    群分けでない結果を群で絞ると断ることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    client: TestClient
        クライアント
    group_files: tuple[Path, Path]
        一式と新しい行の表

    Returns
    ----------
    None
    """
    # 列の出力でない一式と同じく, 決め方と意思決定と並べるの欄は使えない
    path, table = group_files
    bundle = register(client, path)
    url = f"/api/bundles/{bundle['id']}/predict"
    for extra in ({"rules": {"min_confidence": 0.5}},
                  {"decision": {"type": "act_threshold"}},
                  {"forecast": True}, {"k": 3}):
        response = client.post(url, json={"source": str(table), **extra})
        assert response.status_code == 400, extra

    # 群を作る列の無い表は, 無い列を挙げて断る
    lacking = tmp_path / "lacking.csv"
    pd.read_csv(table).drop(columns=["region"]).to_csv(lacking, index=False)
    response = client.post(url, json={"source": str(lacking)})
    assert response.status_code == 400
    assert "region" in response.json()["detail"]

    # 群分けの結果だけを群で絞れる (結論の出し直しも断る)
    result = client.post(url, json={"source": str(table)}).json()
    response = client.post(f"/api/results/{result['id']}/conclude",
                           json={"rules": {}})
    assert response.status_code == 400
    first = client.post("/api/samples").json()[0]
    other = client.post(
        f"/api/bundles/{first['id']}/predict",
        json={"source": first["sample_input"]}
    ).json()
    response = client.get(f"/api/results/{other['id']}/rows",
                          params={"group": 0})
    assert response.status_code == 400
    assert response.json()["detail"] == "群で絞れるのは群分けの結果だけです"


def test_group_representatives(
    client: TestClient, group_files: tuple[Path, Path]
) -> None:
    """
    群分けの結論に群の代表の行があり, 件数を変えて求められ, 代表の行の
    位置で 1 行に絞れることと, 求められない場合に断ることを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    group_files: tuple[Path, Path]
        一式と新しい行の表

    Returns
    ----------
    None
    """
    # 結論の代表の行は ML-Inference の group_representatives の既定の 3 件
    path, table = group_files
    bundle = register(client, path)
    result = client.post(
        f"/api/bundles/{bundle['id']}/predict", json={"source": str(table)}
    ).json()
    representatives = result["summary"]["representatives"]
    frame = ml_inference.predict(str(path), table)
    assert representatives == ml_inference.group_representatives(
        str(path), frame
    )
    assert [item["group"] for item in representatives] == [0, 1, 2]
    assert [len(item["rows"]) for item in representatives] == [3, 3, 3]
    first = representatives[0]["rows"][0]
    assert set(first) == {"row", "distance", "values"}
    assert list(first["values"]) == ["spend", "visits", "region"]

    # 件数を変えると, 置き場の表と登録した一式で求め直す (群の行が少なければ
    # その数だけ)
    url = f"/api/results/{result['id']}/representatives"
    response = client.get(url, params={"count": 10})
    assert response.status_code == 200, response.text
    again = response.json()
    assert again["count"] == 10
    assert [len(item["rows"]) for item in again["groups"]] == [5, 5, 5]
    assert [item["rows"][:3] for item in again["groups"]] == [
        item["rows"] for item in representatives
    ]
    for item in again["groups"]:
        distances = [line["distance"] for line in item["rows"]]
        assert distances == sorted(distances)

    # 代表の行の位置で, 割り当てた行の表を 1 行に絞る
    one = client.get(
        f"/api/results/{result['id']}/rows", params={"row": first["row"]}
    ).json()
    assert one["total"] == 1
    assert one["rows"][0]["index"] == first["row"]
    assert one["rows"][0]["group"] == 0

    # 上限の 20 件 (群の行が少なければその数だけ)
    most = client.get(url, params={"count": 20}).json()
    assert most["count"] == 20
    assert most["groups"] == ml_inference.group_representatives(
        str(path), frame, 20
    )

    # 行の位置の絞りは, 状態と群の絞りと重ねる (合わなければ 0 行)
    rows_url = f"/api/results/{result['id']}/rows"
    for params, total in (
        ({"row": first["row"], "group": 0}, 1),
        ({"row": first["row"], "group": 1}, 0),
        ({"row": first["row"], "status": "ok"}, 1),
        ({"row": first["row"], "status": "skipped"}, 0),
        ({"row": 15, "status": "skipped"}, 1),
        ({"row": 16}, 0),
        ({"row": 1000}, 0),
    ):
        part = client.get(rows_url, params=params).json()
        assert part["total"] == total, params
        assert len(part["rows"]) == total, params
    assert client.get(rows_url, params={"row": -1}).status_code == 422

    # 件数の範囲の外と, 群分けでない結果と, 一式を消した結果は断る
    for count in (0, 21):
        assert client.get(url, params={"count": count}).status_code == 422
    sample = client.post("/api/samples").json()[0]
    other = client.post(
        f"/api/bundles/{sample['id']}/predict",
        json={"source": sample["sample_input"]}
    ).json()
    response = client.get(
        f"/api/results/{other['id']}/representatives", params={"count": 5}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "代表の行は群分けの結果だけで求めます"
    assert client.delete(f"/api/bundles/{bundle['id']}").status_code == 204
    response = client.get(url, params={"count": 5})
    assert response.status_code == 400
    assert "一式が登録されていない" in response.json()["detail"]
    # 結論の 3 件は残る
    kept = client.get(f"/api/results/{result['id']}").json()
    assert kept["summary"]["representatives"] == representatives


def test_group_text_values(
    tmp_path: Path, client: TestClient, group_files: tuple[Path, Path]
) -> None:
    """
    群分けの結果を読み直しても, 種類の値の "NA" と数に見える文字 "001" が
    文字のまま出て, 件数を変えた代表の行が推論した表と同じになることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    client: TestClient
        クライアント
    group_files: tuple[Path, Path]
        一式と新しい行の表

    Returns
    ----------
    None
    """
    # 文字のまま持つ表 (parquet) の, 数に見える番号と種類の値 NA
    path, table = group_files
    frame = pd.read_csv(table, dtype=str, keep_default_na=False)
    frame["customer"] = [f"{n:03d}" for n in range(len(frame))]
    frame["visits"] = pd.to_numeric(frame["visits"])
    frame.loc[1, "region"] = "NA"
    source = tmp_path / "customers.parquet"
    frame.to_parquet(source)

    bundle = register(client, path)
    result = client.post(
        f"/api/bundles/{bundle['id']}/predict", json={"source": str(source)}
    ).json()
    predicted = ml_inference.predict(str(path), source)
    assert result["summary"]["representatives"] == (
        ml_inference.group_representatives(str(path), predicted)
    )
    assert "customer" in result["grouping"]["texts"]
    assert "visits" not in result["grouping"]["texts"]

    # 割り当てた行の表は文字のまま
    rows = client.get(
        f"/api/results/{result['id']}/rows", params={"limit": 3}
    ).json()["rows"]
    assert [row["customer"] for row in rows] == ["000", "001", "002"]
    assert rows[1]["region"] == "NA"
    assert isinstance(rows[0]["visits"], int)

    # 列の型は版 3 の結果に残り, 列の型の無い古い結果 (版 1) も texts で
    # 文字のまま読む
    place = tmp_path / "home" / "results" / result["id"]
    meta = yaml.safe_load((place / "result.yaml").read_text(encoding="utf-8"))
    assert result["format_version"] == 3
    assert meta["column_types"]["customer"] == "text"
    old = {**meta, "format_version": 1}
    del old["column_types"]
    (place / "result.yaml").write_text(yaml.safe_dump(old), encoding="utf-8")
    rows = client.get(
        f"/api/results/{result['id']}/rows", params={"limit": 3}
    ).json()["rows"]
    assert [row["customer"] for row in rows] == ["000", "001", "002"]
    assert isinstance(rows[0]["group"], int)
    (place / "result.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")

    # 件数を変えた代表の行は, 推論した表から求めた代表の行と同じ
    again = client.get(
        f"/api/results/{result['id']}/representatives",
        params={"count": 5}
    ).json()
    assert again["groups"] == ml_inference.group_representatives(
        str(path), predicted, 5
    )
    values = {line["row"]: line["values"]
              for item in again["groups"] for line in item["rows"]}
    assert values[1]["region"] == "NA"
    assert again["groups"][0]["rows"][:3] == (
        result["summary"]["representatives"][0]["rows"]
    )


def test_group_compare(
    tmp_path: Path, client: TestClient, group_files: tuple[Path, Path]
) -> None:
    """
    同じ群分けの一式の結果どうしを比べ, 群分けでない結果と混ぜた組は
    理由を返すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    client: TestClient
        クライアント
    group_files: tuple[Path, Path]
        一式と新しい行の表

    Returns
    ----------
    None
    """
    # 同じ一式で, 行の違う 2 つの表 (別の時期の新しいデータの代わり)
    path, table = group_files
    bundle = register(client, path)
    later = tmp_path / "later.csv"
    pd.read_csv(table).iloc[:8].to_csv(later, index=False)
    url = f"/api/bundles/{bundle['id']}/predict"
    first = client.post(url, json={"source": str(table)}).json()
    second = client.post(url, json={"source": str(later)}).json()
    both = client.get(
        f"/api/results/compare?ids={first['id']},{second['id']}"
    ).json()
    assert both["comparable"] and both["grouping"]
    assert both["basis"] == ["bundle"] and both["reasons"] == []
    groups = [r["summary"]["groups"] for r in both["results"]]
    assert [row["group"] for row in groups[1]] == [0, 1, 2, "全体"]
    assert groups[0][-1]["rows"] == 15
    learned = both["results"][1]["grouping"]["learned"]
    assert learned == both["results"][0]["grouping"]["learned"]

    # 列の出力の結果と混ぜると, 結論の形が違う理由を返す (列の出力で
    # ない理由は出さない)
    samples = client.post("/api/samples").json()
    sample = next(b for b in samples if "解約" in b["name"])
    other = client.post(
        f"/api/bundles/{sample['id']}/predict",
        json={"source": sample["sample_input"]}
    ).json()
    mixed = client.get(
        f"/api/results/compare?ids={first['id']},{other['id']}"
    ).json()
    assert not mixed["comparable"] and not mixed["grouping"]
    assert any("群分けの結果と他の結果" in r for r in mixed["reasons"])
    assert not any("列の出力でない" in r for r in mixed["reasons"])
