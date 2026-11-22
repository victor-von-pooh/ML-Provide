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

from ml_provide import reasons
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
    # 5 つの見本 (2 回目は登録し直さない)
    added = client.post("/api/samples").json()
    assert [b["name"] for b in added][0] == (
        "見本: 図形の領域 (画像の領域の分割)"
    )
    assert len(added) == 5 and all(b["ok"] for b in added)
    assert client.post("/api/samples").json() == []

    # 見本の新しいデータの推論 (日々の売上は先の 7 日を 2 店)
    tables = [b for b in added if "図形の領域" not in b["name"]]
    for bundle in tables:
        daily = "日々の売上" in bundle["name"]
        response = client.post(
            f"/api/bundles/{bundle['id']}/predict",
            json={"source": bundle["sample_input"], "forecast": daily},
        )
        assert response.status_code == 201, response.text
        result = response.json()
        assert result["summary"]["predicted"] == (14 if daily else 50)
        assert result["forecast"] is daily

    # 図形の領域の見本は, 新しい画像のディレクトリを推論し, 1 件の領域を
    # 値ごとの名前で見せる (過去のデータは無い)
    masks = added[0]
    assert "sample_past" not in masks
    description = masks["description"]
    assert description["kind"] == "items"
    assert description["output"] == {
        "form": "mask", "classes": ["背景", "円", "四角"]
    }
    assert description["items"]["values"] == [0, 1, 2]
    response = client.post(
        f"/api/bundles/{masks['id']}/predict",
        json={"source": masks["sample_input"]},
    )
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["summary"] == {"rows": 6, "predicted": 6, "skipped": 0}
    assert result["output_form"] == "mask"
    preview = client.get(
        f"/api/results/{result['id']}/artifacts/0"
    ).json()
    assert preview["view"] == "grid" and preview["shape"] == [16, 16]
    assert {item["label"] for item in preview["labels"]} <= {0, 1, 2}


def test_explain(tmp_path: Path, client: TestClient) -> None:
    """
    結果の入力の列の効きを求めて, 結果に残すことを確かめる関数

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
    # ファイルを推論した結果の列の効き (前処理で落とす shop は 0)
    paths = write_project(tmp_path / "project")
    bundle = create(client, paths)
    result = client.post(
        f"/api/bundles/{bundle['id']}/predict",
        json={"source": str(paths["csv"])},
    ).json()
    response = client.post(f"/api/results/{result['id']}/explain")
    assert response.status_code == 200, response.text
    explanation = response.json()["explanation"]
    effects = {c["column"]: c["effect"] for c in explanation["columns"]}
    assert explanation["measure"] == "probability"
    assert effects["shop"] == 0 and set(effects) == {"shop", "a", "b"}

    # 出し直しても残る
    again = client.post(
        f"/api/results/{result['id']}/conclude", json={"rules": {}}
    ).json()
    assert again["explanation"] == explanation


def test_row_reason(
    tmp_path: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    結果の 1 行の理由を, 選んだ行と比べる行だけで求めることを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    client: TestClient
        クライアント
    monkeypatch: pytest.MonkeyPatch
        比べる行の数を小さくする

    Returns
    ----------
    None
    """
    # 選んだ行と比べる 5 行だけを渡す (前処理で落とす shop は効かない)
    monkeypatch.setattr(reasons, "COMPARE_ROWS", 5)
    passed: list[int] = []
    explain_rows = ml_inference.explain_rows

    def counted(bundle: Any, source: Any, **options: Any) -> dict:
        passed.append(len(source))
        return explain_rows(bundle, source, **options)

    monkeypatch.setattr(ml_inference, "explain_rows", counted)
    paths = write_project(tmp_path / "project")
    bundle = create(client, paths)
    result = client.post(
        f"/api/bundles/{bundle['id']}/predict",
        json={"source": str(paths["csv"])},
    ).json()
    response = client.post(f"/api/results/{result['id']}/rows/3/reason")
    assert response.status_code == 200, response.text
    reason = response.json()
    assert passed == [6] and reason["rows"] == 6
    assert reason["row"] == 3 and reason["measure"] == "probability"
    assert reason["status"] == "ok" and reason["prediction"] in {"良", "悪"}
    effects = {e["column"]: e["effect"] for e in reason["effects"]}
    assert set(effects) == {"shop", "a", "b"} and effects["shop"] == 0
    assert reason["effects"][-1]["column"] == "shop"
    assert reason["reason"] is None or "shop" not in reason["reason"]

    # 比べる行の上限より行が少なければ, 全ての行を渡す (40 行)
    monkeypatch.setattr(reasons, "COMPARE_ROWS", 200)
    whole = client.post(f"/api/results/{result['id']}/rows/0/reason").json()
    assert whole["rows"] == 40 and passed[-1] == 40

    # 負の行は 422, 無い行は 404, 入力のファイルの無い結果は 400
    negative = client.post(f"/api/results/{result['id']}/rows/-1/reason")
    assert negative.status_code == 422
    missing = client.post(f"/api/results/{result['id']}/rows/40/reason")
    assert missing.status_code == 404
    paths["csv"].unlink()
    gone = client.post(f"/api/results/{result['id']}/rows/3/reason")
    assert gone.status_code == 400 and "入力のファイル" in gone.text


def test_rows_for() -> None:
    """
    1 行の理由に渡す行の選び方 (比べる行の上限, 前の続いた行の上限,
    先頭に近い行) を確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 過去の値を使わない一式: 選んだ行と, 等しい間隔の 200 行
    rows = reasons.rows_for(0, 1000, False)
    assert len(rows) == reasons.COMPARE_ROWS + 1 and rows[0] == 0
    assert rows == sorted(set(rows)) and rows[-1] == 999
    assert reasons.rows_for(7, 40, False) == list(range(40))
    middle = reasons.rows_for(500, 1000, False)
    assert 500 in middle and len(middle) == reasons.COMPARE_ROWS + 1

    # 過去の値を使う一式: 選んだ行までの続いた行 (上限 1000 行. 先頭に
    # 近い行は先頭から)
    assert reasons.rows_for(3, 5000, True) == [0, 1, 2, 3]
    tail = reasons.rows_for(2000, 5000, True)
    assert len(tail) == reasons.LAG_ROWS and tail[-1] == 2000
    assert tail == list(range(2000 - reasons.LAG_ROWS + 1, 2001))



def test_read_input_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    結果の入力を読めない文は中の例外の文を出さずに確かめ方を添え, 無い
    場所とディレクトリの文はそのまま出すことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        ML-Inference の読み込みの差し替え

    Returns
    ----------
    None
    """
    path = tmp_path / "input.csv"

    def fail(message: str) -> None:
        def read_source(*args: Any) -> None:
            raise ValueError(message)

        monkeypatch.setattr(ml_inference, "read_source", read_source)

    # 読めない場合: 括弧の中の英語の例外は出さない
    fail("input.csv を読めません (Error tokenizing data. C error)")
    with pytest.raises(ValueError) as caught:
        reasons.read_input(None, path, None)  # type: ignore[arg-type]
    assert str(caught.value) == (
        "input.csv を読めません. 推論したときの読み込みの設定で読める"
        "ファイルかを確かめてください"
    )

    # 無い場所とディレクトリの文はそのまま
    for message in (f"{path} がありません",
                    f"{tmp_path} はディレクトリです. table の一式には"
                    "ファイルを選んでください"):
        fail(message)
        with pytest.raises(ValueError) as caught:
            reasons.read_input(None, path, None)  # type: ignore[arg-type]
        assert str(caught.value) == message

def test_row_reason_lag(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    過去の値の列を作る一式では, 選んだ行までの続いた行を渡して理由を
    求め, 何段も先の予測の結果は断ることを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント
    monkeypatch: pytest.MonkeyPatch
        前の続いた行の上限を小さくする

    Returns
    ----------
    None
    """
    # 日々の売上の過去のデータの 1 つ先の予測
    added = client.post("/api/samples").json()
    daily = next(b for b in added if "日々の売上" in b["name"])
    result = client.post(
        f"/api/bundles/{daily['id']}/predict",
        json={"source": daily["sample_past"], "decision": {}},
    ).json()
    rows = client.get(
        f"/api/results/{result['id']}/rows?status=ok&limit=500"
    ).json()["rows"]
    row = int(rows[-1]["index"])
    response = client.post(f"/api/results/{result['id']}/rows/{row}/reason")
    assert response.status_code == 200, response.text
    reason = response.json()
    assert reason["measure"] == "value" and reason["rows"] == row + 1
    names = {e["column"] for e in reason["effects"]}
    assert "sales (過去の値)" in names
    assert reason["reason"]

    # 前の続いた行の上限 (小さくして, 選んだ行までの 60 行だけ)
    monkeypatch.setattr(reasons, "LAG_ROWS", 60)
    capped = client.post(f"/api/results/{result['id']}/rows/{row}/reason")
    assert capped.status_code == 200, capped.text
    assert capped.json()["rows"] == 60

    # 何段も先の予測の結果は求めない
    ahead = client.post(
        f"/api/bundles/{daily['id']}/predict",
        json={"source": daily["sample_input"], "forecast": True},
    ).json()
    refused = client.post(f"/api/results/{ahead['id']}/rows/0/reason")
    assert refused.status_code == 400 and "何段も先" in refused.text


def test_decision_flow(client: TestClient) -> None:
    """
    見本の既定の意思決定で推論し, 決め直しと過去のデータのたどり直しが
    できることを確かめる関数

    Parameters
    ----------
    client: TestClient
        クライアント

    Returns
    ----------
    None
    """
    # 解約の見本は対象を選ぶ既定値を持つ
    added = client.post("/api/samples").json()
    churn = next(b for b in added if "解約" in b["name"])
    assert churn["description"]["decision"]["type"] == "select"

    # 過去のデータ (目的の列あり) を既定の意思決定で推論
    result = client.post(
        f"/api/bundles/{churn['id']}/predict",
        json={"source": churn["sample_past"],
              "decision": churn["description"]["decision"]},
    ).json()
    assert result["decision"]["summary"]["acting"] == 10
    assert result["has_target"] is True
    rows = client.get(f"/api/results/{result['id']}/rows?limit=5").json()
    assert "action" in rows["columns"] and "gain_if_acted" in rows["columns"]

    # たどり直しと, 決め直し (振り分けに変える. たどり直しは古くなる)
    replayed = client.post(f"/api/results/{result['id']}/backtest").json()
    assert "この方針" in replayed["backtest"]["policies"]
    triage = {"type": "triage", "positive": "解約", "miss_cost": 9,
              "false_alarm_cost": 1}
    again = client.post(f"/api/results/{result['id']}/decide",
                        json={"decision": triage}).json()
    assert again["decision"]["decision"]["type"] == "triage"
    assert again["backtest"] is None
    # 振り分けの列だけが残り, 選ぶ型の列は消える
    columns = client.get(
        f"/api/results/{result['id']}/rows?limit=5"
    ).json()["columns"]
    assert "expected_cost" in columns and "gain_if_acted" not in columns
    bad = client.post(f"/api/results/{result['id']}/decide",
                      json={"decision": {"type": "quantity"}})
    assert bad.status_code == 400

    # 意思決定を外す
    off = client.post(f"/api/results/{result['id']}/decide",
                      json={"decision": None}).json()
    assert off["decision"] is None


def test_remeasure(client: TestClient) -> None:
    """
    過去のデータの結果で外れ方を確かめ, 一式に書き戻せることを確かめる
    関数

    Parameters
    ----------
    client: TestClient
        クライアント

    Returns
    ----------
    None
    """
    # 来店数の見本の過去のデータ (目的の列あり) を推論
    added = client.post("/api/samples").json()
    shop = next(b for b in added if "来店数" in b["name"])
    result = client.post(
        f"/api/bundles/{shop['id']}/predict",
        json={"source": shop["sample_past"], "decision": {}},
    ).json()

    # 確かめるだけでは一式は変わらない
    checked = client.post(f"/api/results/{result['id']}/remeasure",
                          json={"write": False}).json()
    assert checked["written"] is False and checked["rows"] > 0
    assert "80" in checked["before"]["coverage"]
    described = client.get(f"/api/bundles/{shop['id']}").json()
    assert described["description"]["uncertainty"]["split"] != "recent"

    # 書き戻すと, 一式の説明の外れ方が実績の分割になる
    saved = client.post(f"/api/results/{result['id']}/remeasure",
                        json={"write": True}).json()
    assert saved["written"] is True
    described = client.get(f"/api/bundles/{shop['id']}").json()
    assert described["description"]["uncertainty"]["split"] == "recent"
    assert described["description"]["uncertainty"]["restorable"] is True

    # 店ごとに分けて測り, 書き戻すと一式の説明に分けた列と種類の数
    grouped = client.post(f"/api/results/{result['id']}/remeasure",
                          json={"write": True, "groups": "shop"})
    assert grouped.status_code == 200, grouped.text
    assert any("shop ごとの外れ方" in s for s in grouped.json()["sentences"])
    described = client.get(f"/api/bundles/{shop['id']}").json()
    groups = described["description"]["uncertainty"]["groups"]
    assert groups["column"] == "shop" and groups["count"] >= 2

    # 無い列と目的の列では分けられない (空は分けない)
    for column in ("nothing", "visitors", "prediction", "p:x", "status"):
        bad = client.post(f"/api/results/{result['id']}/remeasure",
                          json={"write": False, "groups": column})
        assert bad.status_code == 400
        assert "groups" in bad.json()["detail"]
    plain = client.post(f"/api/results/{result['id']}/remeasure",
                        json={"write": False, "groups": ""}).json()
    assert not any("ごとの外れ方" in s for s in plain["sentences"])

    # 作ったときの記録に戻すと, もう戻せない
    back = client.post(
        f"/api/bundles/{shop['id']}/restore-uncertainty"
    ).json()
    assert back["description"]["uncertainty"]["split"] != "recent"
    again = client.post(f"/api/bundles/{shop['id']}/restore-uncertainty")
    assert again.status_code == 400

    # 日々の売上の見本の過去のデータで, 7 段先までの外れ方
    daily = next(b for b in added if "日々の売上" in b["name"])
    past = client.post(
        f"/api/bundles/{daily['id']}/predict",
        json={"source": daily["sample_past"], "decision": {}},
    ).json()
    steps = client.post(f"/api/results/{past['id']}/remeasure",
                        json={"write": False, "horizon": 7,
                              "time": "date"}).json()
    assert any("何段も先の外れ方" in s for s in steps["sentences"])
    no_time = client.post(f"/api/results/{past['id']}/remeasure",
                          json={"write": False, "horizon": 7})
    assert no_time.status_code == 400

    # 目的の列の無い入力の結果は測り直せない
    fresh = client.post(
        f"/api/bundles/{shop['id']}/predict",
        json={"source": shop["sample_input"], "decision": {}},
    ).json()
    bad = client.post(f"/api/results/{fresh['id']}/remeasure",
                      json={"write": False})
    assert bad.status_code == 400


def test_remeasure_keeps_experiment(
    client: TestClient, tmp_path: Path
) -> None:
    """
    学習の実験の中の一式に書き戻すときは, 実験を書き換えずに置き場へ写す
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
    # 見本の一式を, 実験のディレクトリの中の inference に写して登録
    added = client.post("/api/samples").json()
    shop = next(b for b in added if "来店数" in b["name"])
    run = tmp_path / "outputs" / "shop" / "2026-11-17" / "22-00-00"
    run.mkdir(parents=True)
    (run / "run.json").write_text("{}", encoding="utf-8")
    shutil.copytree(shop["path"], run / "inference")
    original = (run / "inference" / "inference.yaml").read_text("utf-8")
    entry = client.post("/api/bundles",
                        json={"path": str(run / "inference")}).json()
    result = client.post(
        f"/api/bundles/{entry['id']}/predict",
        json={"source": shop["sample_past"], "decision": {}},
    ).json()

    # 確かめるだけなら写さず, 書き戻すと写した方への書き込み
    checked = client.post(f"/api/results/{result['id']}/remeasure",
                          json={"write": False}).json()
    assert checked["copied_to"] is None
    saved = client.post(f"/api/results/{result['id']}/remeasure",
                        json={"write": True}).json()
    assert saved["written"] is True and saved["copied_to"]
    assert "書き換えず" in saved["sentences"][0]
    assert (run / "inference" / "inference.yaml").read_text("utf-8") == (
        original
    )
    moved = client.get(f"/api/bundles/{entry['id']}").json()
    assert moved["path"] == saved["copied_to"]
    assert moved["description"]["uncertainty"]["split"] == "recent"
    assert moved["copied_from"] == str((run / "inference").resolve())

    # 2 回目は写した方に書き, 戻すのも写した方. 実験のファイルは変わらない
    before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    again = client.post(f"/api/results/{result['id']}/remeasure",
                        json={"write": True}).json()
    assert again["copied_to"] is None
    back = client.post(f"/api/bundles/{entry['id']}/restore-uncertainty")
    assert back.status_code == 200
    assert {p: p.read_bytes() for p in run.rglob("*") if p.is_file()} == (
        before
    )

    # 写した元の場所はもう一度登録できない. 消すと写しだけが消える
    twice = client.post("/api/bundles", json={"path": str(run / "inference")})
    assert twice.status_code == 400
    assert client.delete(f"/api/bundles/{entry['id']}").status_code == 204
    assert not Path(saved["copied_to"]).exists()
    assert (run / "inference" / "inference.yaml").is_file()


def test_compare_results(tmp_path: Path, client: TestClient) -> None:
    """
    同じ一式か同じ入力の結果を並べ, 比べられない組は理由を返すことを
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
    # 同じ一式と入力で決め方だけ違う 2 つ
    paths = write_project(tmp_path / "project")
    first = create(client, paths)
    second = create(client, paths)
    other = tmp_path / "other.csv"
    shutil.copy(paths["csv"], other)

    def predict(bundle: dict[str, Any], source: Path, rules: Any) -> str:
        response = client.post(
            f"/api/bundles/{bundle['id']}/predict",
            json={"source": str(source), "rules": rules},
        )
        assert response.status_code == 201, response.text
        result_id: str = response.json()["id"]
        return result_id

    held = predict(first, paths["csv"], {"min_confidence": 0.55})
    free = predict(first, paths["csv"], {})
    shared = predict(second, paths["csv"], {})
    apart = predict(second, other, {})

    both = client.get(f"/api/results/compare?ids={held},{free}").json()
    assert both["comparable"] and both["basis"] == ["bundle", "source"]
    assert [r["id"] for r in both["results"]] == [held, free]
    assert both["results"][0]["rules"] == {"min_confidence": 0.55}
    assert both["results"][1]["summary"]["rows"] == 40

    # 一式が違っても入力が同じなら比べる. 一式も入力も違えば理由
    same_source = client.get(
        f"/api/results/compare?ids={free},{shared}"
    ).json()
    assert same_source["comparable"] and same_source["basis"] == ["source"]
    same_bundle = client.get(
        f"/api/results/compare?ids={shared},{apart}"
    ).json()
    assert same_bundle["basis"] == ["bundle"]
    neither = client.get(f"/api/results/compare?ids={free},{apart}").json()
    assert not neither["comparable"] and neither["basis"] == []
    assert "一式も入力も違う" in neither["reasons"][0]
    assert len(neither["results"]) == 2

    # 選び方の誤り
    for ids, status in (
        (held, 400), (f"{held},{held}", 400),
        (",".join([held] * 7), 400), (f"{held},missing", 404),
    ):
        response = client.get(f"/api/results/compare?ids={ids}")
        assert response.status_code == status, ids


def test_comparison_reasons() -> None:
    """
    種類の違う結果と列の出力でない結果を比べない理由を確かめる関数

    Returns
    ----------
    None
    """
    from ml_provide.api import comparison

    table = {"bundle_id": "a", "task": "classification",
             "source_path": "/data.csv"}
    found = comparison([table, {**table, "task": "regression"}])
    assert found["basis"] == ["bundle", "source"]
    assert ["タスク" in r for r in found["reasons"]] == [True]
    found = comparison([table, {**table, "column_output": False}])
    assert ["列の出力でない" in r for r in found["reasons"]] == [True]
    # 書いた文 (読んだファイルの無い入力) は同じ入力としない
    texts = {"bundle_id": "a", "task": "classification"}
    found = comparison([texts, {**texts, "bundle_id": "b"}])
    assert found["basis"] == [] and len(found["reasons"]) == 1
