import shutil
from pathlib import Path
from typing import Any

import ml_quest
import pytest
import yaml
from fastapi.testclient import TestClient
from ml_quest.examples import example_problem

from ml_provide.app import create_app

# テストで使うトークン
TOKEN = "test-token"

# 版 1 の問題の設定のファイル (総菜の見本を日々の売上の見本の列に合わせた
# もの. 版 2 の shop_problem と同じ答え)
SHOP_V1 = """
format_version: 1
name: 総菜の作る量
created_at: "2026-11-17T21:30:00+09:00"
updated_at: "2026-11-17T21:45:00+09:00"
issue:
  owner: 店長
  pain: 総菜の廃棄が多く, 夕方には品切れも出る
  current: 前の週の同じ曜日と同じ数を作る
  kpi: {name: 廃棄と欠品の損, unit: 円 / 月, current: 1200000,
        target: 960000, better: lower}
decomposition:
  - {cause: 作る量が勘で決まる, lever: 作る量を予測から決める, ml: true}
  - {cause: 値引きが遅い, lever: 値引きの時刻を決める, ml: false}
action:
  who: 店長
  when: 前の日の 17 時
  what: 品ごとの作る量
  decision: {type: quantity, under_cost: 200, over_cost: 150}
  basis: 欠品は粗利 200 円, 廃棄は原価 150 円
ml:
  use: true
  reason: 品と日による売れ方の違いが大きい
  predict: 翌日の品ごとの売れる数
  row: 店, 日ごとの 1 行
  target: sales
  task: forecasting
  time_column: date
  series_by: [shop]
  available_lag: 2
  horizon: 1
  baseline: 前の週の同じ曜日と同じ数
  consumer: rule
information:
  columns: [date, shop, sales]
  known: [shop]
evaluation:
  metric: mae
  link: 誤差が小さいほど損が減る
verification:
  method: both
  backtest: 過去 3 か月
  trial: 1 店で 4 週間
  success: 損が 2 割少ない
operation:
  review: 月に 1 回
  approver: 本部
  stop: 損が悪い月が 2 回続いたら止める
  remeasure: 月に 1 回, 前の月の実績で外れ方を確かめる
"""


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


def shop_problem() -> dict[str, Any]:
    """
    総菜の見本の問題の設定を, 日々の売上の見本の列 (日付, 店, 売上) に
    合わせる関数

    Parameters
    ----------
    None

    Returns
    ----------
    problem: dict[str, Any]
        店ごとに売上を予測し, 量を決める問題の設定
    """
    # 品と客数の無い表. 系列は店だけ
    problem = example_problem("総菜の作る量")
    for key, value in (
        ("information.columns", ["date", "shop", "sales"]),
        ("ml.parts.0.series_by", ["shop"]),
        ("information.known", ["shop"]),
        ("information.unknown", None),
    ):
        problem = ml_quest.answer(problem, key, value)

    return problem


def project_bundle(
    client: TestClient, tmp_path: Path, source: str
) -> tuple[Path, dict[str, Any]]:
    """
    骨組みのプロジェクトのように, 問題の設定を持つディレクトリの実験の
    中に一式を写して登録する関数

    Parameters
    ----------
    client: TestClient
        クライアント
    tmp_path: Path
        一時ディレクトリ
    source: str
        写す一式の場所

    Returns
    ----------
    made: tuple[Path, dict[str, Any]]
        プロジェクトのディレクトリと, 登録した一式
    """
    # プロジェクト/problem.yaml と プロジェクト/experiments/run/bundle
    project = tmp_path / "project"
    ml_quest.save_problem(shop_problem(), project)
    place = project / "experiments" / "run" / "bundle"
    shutil.copytree(source, place)
    bundle = client.post("/api/bundles", json={"path": str(place)}).json()
    made = (project, bundle)

    return made


def test_link_and_predict(client: TestClient, tmp_path: Path) -> None:
    """
    一式の場所から問題の設定の候補を見つけ, つないで読んだ損得と結論の
    決め方で推論できることを確かめる関数

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
    # 見本の一式 (置き場の中) には候補が無い. プロジェクトの中なら候補
    added = client.post("/api/samples").json()
    daily = next(b for b in added if "日々の売上" in b["name"])
    state = client.get(f"/api/bundles/{daily['id']}/problem").json()
    assert state == {"path": None, "linked": None, "error": None,
                     "hint": None}
    project, bundle = project_bundle(client, tmp_path, daily["path"])
    url = f"/api/bundles/{bundle['id']}/problem"
    state = client.get(url).json()
    assert state["hint"] == str((project / "problem.yaml").resolve())

    # つなぐと, 問題の設定の損得と, 一式の既定値に重ねた結論の決め方
    state = client.put(url, json={"path": state["hint"]}).json()
    linked = state["linked"]
    assert linked["decision"] == {"type": "quantity", "under_cost": 200,
                                  "over_cost": 150}
    assert linked["conclusion"] == {"group_by": "shop"}
    assert linked["basis"] == "欠品は粗利 200 円, 廃棄は原価 150 円"
    assert "量を決める" in linked["text"] and linked["issues"] == []
    assert client.get("/api/bundles").json()[0]["problem"] == str(
        (project / "problem.yaml").resolve()
    )

    # 読んだ値で推論できる
    result = client.post(f"/api/bundles/{bundle['id']}/predict", json={
        "source": daily["sample_past"], "decision": linked["decision"],
        "rules": linked["conclusion"],
    })
    assert result.status_code == 201, result.text
    assert result.json()["decision"]["decision"]["under_cost"] == 200

    # 問題の設定を直すと, 次に読むときから使う (中身の印も変わる)
    problem = ml_quest.answer(
        ml_quest.load_problem(project), "action.decision",
        {"type": "quantity", "under_cost": 300, "over_cost": 150},
    )
    ml_quest.save_problem(problem, project)
    again = client.get(url).json()["linked"]
    assert again["decision"]["under_cost"] == 300
    assert again["version"] != linked["version"]

    # 外す
    state = client.put(url, json={"path": None}).json()
    assert state["path"] is None and state["hint"] is not None
    assert "problem" not in client.get("/api/bundles").json()[0]


def test_link_problems(client: TestClient, tmp_path: Path) -> None:
    """
    一式と食い違う問題の設定, 推論に渡せない問題の設定, 置き換わった
    問題の設定の扱いを確かめる関数

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
    # 来店数の回帰の一式に, 売上の量の問題の設定をつなぐと目的の列が違う
    added = client.post("/api/samples").json()
    visitors = next(b for b in added if "来店数" in b["name"])
    churn = next(b for b in added if "解約" in b["name"])
    path = tmp_path / "shop"
    ml_quest.save_problem(shop_problem(), path)
    url = f"/api/bundles/{visitors['id']}/problem"
    linked = client.put(url, json={"path": str(path)}).json()["linked"]
    assert any("目的の列" in text for text in linked["issues"])

    # まとめる列 (総菜の店の列 store) が学習のデータに無い
    daily = next(b for b in added if "日々の売上" in b["name"])
    sozai = tmp_path / "sozai"
    ml_quest.save_problem(example_problem("総菜の作る量"), sozai)
    linked = client.put(f"/api/bundles/{daily['id']}/problem", json={
        "path": str(sozai)
    }).json()["linked"]
    assert any("まとめる列 store" in text for text in linked["issues"])

    # 場所として読めない文字は 422
    assert client.put(url, json={"path": "/tmp/a\x00b"}).status_code == 422

    # 分類の一式に量の意思決定は合わない
    other = f"/api/bundles/{churn['id']}/problem"
    linked = client.put(other, json={"path": str(path)}).json()["linked"]
    assert any(text.startswith("意思決定:") for text in linked["issues"])

    # 機械学習を使わない問題の設定と, 無い場所は断る
    plain = tmp_path / "plain"
    ml_quest.save_problem(
        ml_quest.answer(example_problem("総菜の作る量"), "ml.use", False),
        plain,
    )
    refused = client.put(url, json={"path": str(plain)})
    assert refused.status_code == 422
    assert "推論に渡せません" in refused.json()["detail"]
    assert client.put(url, json={
        "path": str(tmp_path / "none")
    }).status_code == 404

    # 別の問題の設定に置き換わると理由を出し, つなぎは残す (作った時刻
    # で見分ける)
    replaced = dict(shop_problem(), created_at="2020-01-01T00:00:00+09:00")
    ml_quest.save_problem(replaced, path)
    state = client.get(url).json()
    assert state["linked"] is None and "置き換わって" in state["error"]
    assert state["path"] == str((path / "problem.yaml").resolve())


def linked_values(linked: dict[str, Any]) -> dict[str, Any]:
    """
    つないだ問題の設定から読んだ値のうち, ファイルに依らない値を返す関数

    Parameters
    ----------
    linked: dict[str, Any]
        つないだ問題の設定から読んだ値

    Returns
    ----------
    values: dict[str, Any]
        場所, 中身の印, 作った時刻を除いた値
    """
    # ファイルごとに違う値の除外
    values = {
        key: value for key, value in linked.items()
        if key not in ("path", "version", "created_at")
    }

    return values


def test_version1_and_version2(client: TestClient, tmp_path: Path) -> None:
    """
    版 1 の問題の設定のファイルと, 版 2 に移したファイルが, 列の予測で
    同じ推論の値と意思決定を返し, 版 1 のファイルを書き換えないことを
    確かめる関数

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
    # 版 1 は YAML のまま, 版 2 は ML-Quest で移して保存
    added = client.post("/api/samples").json()
    daily = next(b for b in added if "日々の売上" in b["name"])
    old = tmp_path / "v1" / "problem.yaml"
    old.parent.mkdir()
    old.write_text(SHOP_V1, encoding="utf-8")
    new = tmp_path / "v2"
    ml_quest.save_problem(
        ml_quest.upgrade_problem(yaml.safe_load(SHOP_V1)), new
    )
    assert ml_quest.load_problem(new)  # 保存できた版 2
    assert yaml.safe_load(
        (new / "problem.yaml").read_text(encoding="utf-8")
    )["format_version"] == 2

    # どちらも列の予測として読める値の一致
    url = f"/api/bundles/{daily['id']}/problem"
    views = []
    decisions = []
    for place in (old, new):
        linked = client.put(url, json={"path": str(place)}).json()["linked"]
        assert linked["handled"] is True
        views.append(linked_values(linked))
        result = client.post(f"/api/bundles/{daily['id']}/predict", json={
            "source": daily["sample_past"], "decision": linked["decision"],
            "rules": linked["conclusion"],
        })
        assert result.status_code == 201, result.text
        decisions.append(result.json()["decision"])
    assert views[0] == views[1]
    assert views[0]["decision"] == {"type": "quantity", "under_cost": 200,
                                    "over_cost": 150}
    assert views[0]["conclusion"] == {"group_by": "shop"}
    assert views[0]["horizon"] == 1 and views[0]["issues"] == []
    assert views[0]["role"] == "予測する, 正解の種類は表の列 (クラスか値)"
    assert decisions[0] == decisions[1]

    # 版 1 のファイルはそのまま (読んだ後の移行だけ)
    assert old.read_text(encoding="utf-8") == SHOP_V1

    # 知っている版より新しいファイルは, 読めない理由での停止
    newer = dict(yaml.safe_load(SHOP_V1), format_version=99)
    future = tmp_path / "future" / "problem.yaml"
    future.parent.mkdir()
    future.write_text(yaml.safe_dump(newer, allow_unicode=True),
                      encoding="utf-8")
    refused = client.put(url, json={"path": str(future)})
    assert refused.status_code == 422
    assert "問題の設定を読めません" in refused.json()["detail"]


def generate_problem() -> dict[str, Any]:
    """
    総菜の見本の課題で, 受け持ちを「作る」にした問題の設定を作る関数

    Parameters
    ----------
    None

    Returns
    ----------
    problem: dict[str, Any]
        受け持ちが作る (画像) の問題の設定
    """
    # 課題は見本のまま, 受け持ちは作り直し (意思決定は問わない)
    problem = example_problem("総菜の作る量")
    problem["ml"]["parts"] = [{}]
    for key in ("decision", "basis", "current"):
        problem["action"].pop(key, None)
    for key, value in (
        ("role", "generate"), ("what", "品の見本の画像"), ("item", "1 枚"),
        ("baseline", "手で撮る"), ("unit", "pair"), ("output", "image"),
        ("target.source", "files"), ("method", "finetune"),
        ("target.kind", "image"), ("evaluation.metric", "fid"),
        ("evaluation.link", "見本が自然なほど撮る手間が減る"),
    ):
        problem = ml_quest.answer(problem, f"ml.parts.0.{key}", value)

    return problem


def test_other_role(client: TestClient, tmp_path: Path) -> None:
    """
    列の予測でない受け持ちの問題の設定は, つなげるが推論の値を持たず,
    まだ扱わないことを文で出し, 候補には出さないことを確かめる関数

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
    # つなぐと, 意思決定は無く結論の決め方は一式の既定値のまま
    added = client.post("/api/samples").json()
    daily = next(b for b in added if "日々の売上" in b["name"])
    place = tmp_path / "generate"
    ml_quest.save_problem(generate_problem(), place)
    url = f"/api/bundles/{daily['id']}/problem"
    state = client.put(url, json={"path": str(place)})
    assert state.status_code == 200, state.text
    linked = state.json()["linked"]
    assert linked["handled"] is False
    assert linked["role"] == "作る, 正解の種類は画像"
    assert linked["decision"] is None and linked["horizon"] is None
    assert linked["basis"] is None and linked["issues"] == []
    assert "推論の結論はまだ扱いません" in linked["text"]
    assert linked["conclusion"] == {"group_by": "shop"}  # 見本の既定値

    # 読み直しても同じ値. 一式の上にあっても候補には無し
    assert client.get(url).json()["linked"] == linked
    project = tmp_path / "project"
    ml_quest.save_problem(generate_problem(), project)
    copy = project / "experiments" / "run" / "bundle"
    shutil.copytree(daily["path"], copy)
    other = client.post("/api/bundles", json={"path": str(copy)}).json()
    assert client.get(
        f"/api/bundles/{other['id']}/problem"
    ).json()["hint"] is None
