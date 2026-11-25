import json
from pathlib import Path
from typing import Any

import ml_inference
import pandas as pd
import pytest
import torch
import yaml
from fastapi.testclient import TestClient
from ml_modelcontrol import build_model, load_example
from test_bundle_v3 import head_spec, make_v3, register, write_unlabeled

import ml_provide.api as api_module
from ml_provide.app import create_app

# テストで使うトークン
TOKEN = "test-token"

# 番号の表の利用者と品物 (文字と数を混ぜ, 0 で始まる利用者も)
USERS = ["u0", "u1", "007", 103]
ITEMS = [f"i{n}" for n in range(6)]

# 反応の表の列の名前 (問題の設定の rank の列)
COLUMNS = {"user": "member", "item": "item_id", "reaction": "qty"}

# 番号の表に無い利用者の理由 (ML-Inference の文)
UNKNOWN = "番号の表に無い利用者です (学習用の反応に無い利用者は並べられません)"


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


def make_rank(root: Path, k: int = 3) -> Path:
    """
    並べるの骨組みと同じ実験から, ML-Inference の make_rank_bundle で
    並べるの一式を作る関数

    Parameters
    ----------
    root: Path
        作るディレクトリの親
    k: int = 3
        並べる件数の既定

    Returns
    ----------
    path: Path
        一式のディレクトリ
    """
    # ML-ModelControl の同梱の 2 塔を利用者 4 人と品物 6 つの大きさにする
    spec = load_example("two_tower_retrieval")
    spec["inputs"]["user"]["shape"] = [4, len(USERS)]
    spec["inputs"]["item"]["shape"] = [4, len(ITEMS)]
    spec["nodes"]["user_fc1"]["params"]["in_features"] = len(USERS)
    spec["nodes"]["item_fc1"]["params"]["in_features"] = len(ITEMS)

    # 重み, 定義, 番号の表, 実験の状態 (骨組みの実験と同じ 3 つのファイル)
    run = root / "run"
    run.mkdir(parents=True)
    model = build_model(spec, seed=0, load_weights=False)
    torch.save(model.state_dict(), run / "final_model.pt")
    (run / "model_spec.yaml").write_text(
        yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8"
    )
    (run / "rank_index.json").write_text(
        json.dumps({"users": USERS, "items": ITEMS}), encoding="utf-8"
    )
    (run / "run.json").write_text(
        json.dumps({"format_version": 1, "status": "completed"}),
        encoding="utf-8"
    )
    path = root / "shop"
    ml_inference.make_rank_bundle(path, run, k=k, columns=COLUMNS,
                                  name="shop")

    return path


def predict(
    client: TestClient, bundle: dict[str, Any], **body: Any
) -> dict[str, Any]:
    """
    一式で推論して結果のまとめを返す関数

    Parameters
    ----------
    client: TestClient
        クライアント
    bundle: dict[str, Any]
        登録した一式
    **body: Any
        推論の欄

    Returns
    ----------
    result: dict[str, Any]
        保存した結果のまとめ
    """
    response = client.post(f"/api/bundles/{bundle['id']}/predict", json=body)
    assert response.status_code == 201, response.text
    result: dict[str, Any] = response.json()

    return result


def test_rank_bundle(tmp_path: Path, client: TestClient) -> None:
    """
    並べるの一式を登録して書いた利用者を並べ, 利用者ごとの上位 K 件と
    並べられない利用者の理由を返すことを確かめる関数

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
    # 説明は版 8 の並べるの一式で, 件数の既定と反応の表の列を持つ
    bundle = register(client, make_rank(tmp_path))
    description = bundle["description"]
    assert description["format_version"] == 8
    assert description["kind"] == "rank"
    assert description["role"] == "rank"
    assert description["output"] == {"form": "ranking"}
    assert description["column_output"] is False
    assert description["rank"]["k"] == 3
    assert description["rank"]["columns"] == COLUMNS
    assert description["rank"]["users"] == len(USERS)
    assert description["rank"]["items"] == len(ITEMS)

    # 問題の設定は読まない (結論と意思決定を使わないため)
    state = client.get(f"/api/bundles/{bundle['id']}/problem").json()
    assert state == {"path": None, "linked": None, "error": None,
                     "hint": None, "hint_note": None}

    # 書いた利用者 (空の行は除く. 数の利用者も文字で引く) を既定の件数で
    result = predict(client, bundle, users=["u0", "", "007", "103", "nobody"])
    assert result["output_form"] == "ranking"
    assert result["column_output"] is False
    assert result["source"] == "利用者 4 人"
    assert result["summary"] == {"rows": 4, "predicted": 3, "skipped": 1,
                                 "k": 3, "excluded": False}
    assert result["sentences"] == [
        "利用者 4 人のうち 3 人に, 上位 3 件の品物を並べた (1 人は並べられず)"
    ]
    assert result["ranking"] == {
        "k": 3, "excluded": False, "columns": COLUMNS, "seen": None,
        "truth": None, "metrics": None,
    }
    # 0 で始まる利用者と, 並べられない利用者の欠けた順位の型を残す版 3
    assert result["format_version"] == 3
    assert result["artifacts"] is None

    # 利用者と順位ごとの行 (0 で始まる利用者は文字のまま, 順位は整数)
    rows = client.get(f"/api/results/{result['id']}/rows").json()
    assert rows["total"] == 10
    assert rows["columns"][:7] == [
        "index", "user", "status", "reason", "rank", "item", "score"
    ]
    table = rows["rows"]
    assert [row["user"] for row in table] == ["u0"] * 3 + ["007"] * 3 + [
        "103"
    ] * 3 + ["nobody"]
    assert [row["rank"] for row in table[:3]] == [1, 2, 3]
    assert all(row["item"] in ITEMS for row in table[:9])
    scores = [row["score"] for row in table[:3]]
    assert scores == sorted(scores, reverse=True)
    assert table[9]["status"] == "skipped"
    assert table[9]["reason"] == UNKNOWN
    assert table[9]["rank"] is None and table[9]["item"] is None

    # 一式の点数と同じ (ML-Inference の predict と同じ並び)
    expected = ml_inference.predict(
        Path(bundle["path"]), ["u0", "007", "103", "nobody"]
    )
    assert [row["item"] for row in table[:9]] == expected["item"][:9].tolist()

    # 読み直しても利用者ごとの数え方のまま
    reloaded = client.get(f"/api/results/{result['id']}").json()
    assert reloaded["summary"] == result["summary"]
    assert reloaded["ranking"] == result["ranking"]
    skipped = client.get(f"/api/results/{result['id']}/rows",
                         params={"status": "skipped"}).json()
    assert skipped["total"] == 1

    # 結論の決め方, 意思決定, たどり直しは使えない
    response = client.post(f"/api/results/{result['id']}/conclude",
                           json={"rules": {}})
    assert response.status_code == 400
    response = client.post(f"/api/results/{result['id']}/backtest")
    assert response.status_code == 400


def test_rank_files_and_metrics(tmp_path: Path, client: TestClient) -> None:
    """
    利用者の表のファイルを, 件数を変え, 前に反応した品物を除いて並べ,
    正解の反応の表から Recall@K と NDCG を数えることを確かめる関数

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
    # 利用者の表と, 前の反応 (qty が 0 の行は反応でない) と後の反応
    path = make_rank(tmp_path)
    bundle = register(client, path)
    users = tmp_path / "members.csv"
    pd.DataFrame({"member": ["007", "u1", "103"]}).to_csv(users, index=False)
    seen = tmp_path / "seen.csv"
    pd.DataFrame({
        "member": ["007", "007", "u1", "103"],
        "item_id": ["i0", "i1", "i2", "i3"],
        "qty": [1, 2, 0, 1],
    }).to_csv(seen, index=False)
    truth = tmp_path / "later.csv"
    pd.DataFrame({
        "member": ["007", "u1", "u1", "103", "nobody"],
        "item_id": ["i4", "i5", "i0", "i3", "i1"],
        "qty": [1, 1, 3, 1, 1],
    }).to_csv(truth, index=False)

    result = predict(client, bundle, source=str(users), k=2, seen=str(seen),
                     truth=str(truth))
    assert result["source"] == f"{users.resolve()} (利用者 3 人)"
    assert result["summary"]["k"] == 2
    assert result["summary"]["excluded"] is True
    assert "前に反応した品物は除いて並べた" in result["sentences"]
    ranking = result["ranking"]
    assert ranking["k"] == 2 and ranking["excluded"] is True
    assert ranking["seen"] == str(seen.resolve())
    assert ranking["truth"] == str(truth.resolve())

    # 前に反応した品物は並ばない (qty が 0 の行は除かない)
    rows = client.get(f"/api/results/{result['id']}/rows").json()["rows"]
    assert len(rows) == 6
    assert not {row["item"] for row in rows if row["user"] == "007"} & {
        "i0", "i1"
    }
    assert not {row["item"] for row in rows if row["user"] == "103"} & {"i3"}

    # Recall@K と NDCG は ML-Inference の rank_metrics と同じ (文字で読んだ表)
    read = {name: pd.read_csv(file, dtype=str) for name, file in (
        ("users", users), ("seen", seen), ("truth", truth)
    )}
    frame = ml_inference.predict(path, read["users"], k=2, seen=read["seen"])
    expected = ml_inference.rank_metrics(path, frame, read["truth"],
                                         seen=read["seen"], k=2)
    assert ranking["metrics"] == expected
    assert ranking["metrics"]["users"] == 2
    assert 0.0 <= ranking["metrics"]["recall_at_k"] <= 1.0
    assert 0.0 <= ranking["metrics"]["ndcg"] <= 1.0
    reloaded = client.get(f"/api/results/{result['id']}").json()
    assert reloaded["ranking"]["metrics"] == expected


def test_rank_refusals(tmp_path: Path, client: TestClient) -> None:
    """
    並べるの一式と他の一式の, 渡せない欄の組を理由付きで断ることを
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
    bundle = register(client, make_rank(tmp_path))
    url = f"/api/bundles/{bundle['id']}/predict"
    users = tmp_path / "members.csv"
    pd.DataFrame({"name": ["u0"]}).to_csv(users, index=False)
    cases: list[tuple[dict[str, Any], str]] = [
        ({"users": ["u0"], "k": 0}, "並べる件数は 1 から"),
        ({"users": ["u0"], "k": 1001}, "並べる件数は 1 から"),
        ({"users": ["u0"], "k": 2.5}, "並べる件数は 1 から"),
        ({"users": ["u0"], "k": "3"}, "並べる件数は 1 から"),
        ({"users": ["u0"], "k": True}, "並べる件数は 1 から"),
        ({"users": ["u0"] * 10001}, "利用者は 1 回に 10,000 人まで"),
        ({"source": str(tmp_path)}, "ディレクトリでなくファイルを指定してください"),
        ({"users": ["", " "]}, "利用者がありません"),
        ({}, "利用者を書くか"),
        ({"texts": ["u0"]}, "文, 件数, 種は使いません"),
        ({"count": 3}, "文, 件数, 種は使いません"),
        ({"users": ["u0"], "source": str(users)}, "どちらか"),
        ({"source": str(users)}, "利用者の列 member がありません"),
        ({"source": str(tmp_path / "none.csv")}, "がありません"),
        ({"users": ["u0"], "seen": str(tmp_path / "none.csv")},
         "前に反応した品物の表のファイル"),
        ({"users": ["u0"], "truth": str(tmp_path)},
         "正解の反応の表には, ディレクトリでなくファイル"),
        ({"users": ["u0"], "rules": {"min_confidence": 0.5}}, "結論の決め方"),
        ({"users": ["u0"], "options": {"sep": ";"}}, "読み込みの設定"),
    ]
    for body, message in cases:
        response = client.post(url, json=body)
        assert response.status_code == 400, (body, response.text)
        assert message in response.json()["detail"], (body, response.text)

    # 列の無い前の反応の表は ML-Inference の誤りの文で断る
    seen = tmp_path / "seen.csv"
    pd.DataFrame({"member": ["u0"]}).to_csv(seen, index=False)
    response = client.post(url, json={"users": ["u0"], "seen": str(seen)})
    assert response.status_code == 400
    assert "item_id" in response.json()["detail"]

    # 並べるの欄は他の一式に渡せない
    paths = write_unlabeled(tmp_path / "data")
    other = register(client, make_v3(
        tmp_path, paths, "embed", head_spec(5), form="vector", role="group"
    ))
    for body in ({"source": str(paths["csv"]), "k": 3},
                 {"users": ["u0"]},
                 {"source": str(paths["csv"]), "truth": str(seen)}):
        response = client.post(f"/api/bundles/{other['id']}/predict",
                               json=body)
        assert response.status_code == 400, response.text
        assert "並べるの一式だけ" in response.json()["detail"]


def test_rank_formats_and_limits(
    tmp_path: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    TSV と Parquet の表を読み, 正解の反応だけ (前の反応なし) で数え, 前の
    反応を除くと残らない利用者を理由付きで飛ばし, 行の上限で断ることを
    確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    client: TestClient
        クライアント
    monkeypatch: pytest.MonkeyPatch
        行の上限を小さくする

    Returns
    ----------
    None
    """
    # TSV の利用者 (0 で始まる値も文字のまま) と Parquet の正解の反応
    path = make_rank(tmp_path)
    bundle = register(client, path)
    users = tmp_path / "members.tsv"
    pd.DataFrame({"member": ["007", "u1"]}).to_csv(users, sep="\t",
                                                    index=False)
    truth = tmp_path / "later.parquet"
    later = pd.DataFrame({"member": ["007", "u1"], "item_id": ["i1", "i2"],
                          "qty": [1, 1]})
    later.to_parquet(truth)
    result = predict(client, bundle, source=str(users), truth=str(truth))
    assert result["summary"]["predicted"] == 2
    assert result["ranking"]["seen"] is None
    assert result["ranking"]["excluded"] is False
    frame = ml_inference.predict(path, ["007", "u1"])
    expected = ml_inference.rank_metrics(path, frame, later, k=3)
    assert result["ranking"]["metrics"] == expected
    assert result["ranking"]["metrics"]["users"] == 2
    rows = client.get(f"/api/results/{result['id']}/rows").json()["rows"]
    assert [row["user"] for row in rows] == ["007"] * 3 + ["u1"] * 3

    # 全ての品物に反応した利用者は, 除くと残らないので理由付きで飛ばす
    seen = tmp_path / "seen.parquet"
    pd.DataFrame({"member": ["u0"] * len(ITEMS), "item_id": ITEMS,
                  "qty": [1] * len(ITEMS)}).to_parquet(seen)
    result = predict(client, bundle, users=["u0", "u1"], seen=str(seen))
    assert result["summary"]["skipped"] == 1
    rows = client.get(f"/api/results/{result['id']}/rows").json()["rows"]
    assert rows[0]["user"] == "u0" and rows[0]["status"] == "skipped"
    assert rows[0]["reason"] == (
        "前に反応した品物を除くと, 並べる品物が残りません"
    )
    assert [row["user"] for row in rows[1:]] == ["u1"] * 3

    # 書く表の行 (利用者の数 × 件数) と, 前の反応と正解の反応の表の行の上限
    url = f"/api/bundles/{bundle['id']}/predict"
    monkeypatch.setattr(api_module, "MAX_RANK_ROWS", 5)
    response = client.post(url, json={"users": ["u0", "u1"], "k": 3})
    assert response.status_code == 400
    assert "5 行を超えます" in response.json()["detail"]
    monkeypatch.setattr(api_module, "MAX_REACTION_ROWS", 1)
    for key, what in (("seen", "前に反応した品物の表"), ("truth", "正解の反応の表")):
        response = client.post(url, json={"users": ["u0"], "k": 1,
                                          key: str(truth)})
        assert response.status_code == 400
        assert f"{what}は 1 行までです" in response.json()["detail"]
