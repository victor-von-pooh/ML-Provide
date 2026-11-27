import copy
import json
import math
import os
import shutil
import threading
import uuid
from itertools import islice
from pathlib import Path
from typing import Any, Optional, Union

import ml_inference
import numpy as np
import pandas as pd
import yaml
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi import Path as FastPath
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, StrictInt

from ml_provide.cache import (
    describe,
    forget,
    open_bundle,
    read_bundle,
    remember_warnings,
)
from ml_provide.previews import PREVIEW_LIMIT, array_preview, dtype_problem
from ml_provide.problem_files import linked_guard, uses_problem
from ml_provide.procedures import missing_procedure, registered
from ml_provide.ready import FetchError, make_ready, preview_ready
from ml_provide.reasons import read_input, row_reason
from ml_provide.samples import write_samples
from ml_provide.splits import (
    find_test_split,
    result_source,
    split_key,
    split_source,
)
from ml_provide.store import Store
from ml_provide.units import unit_hint

router = APIRouter()

# 推論を 1 つずつにする鍵 (大きなモデルでメモリを使い切らないように)
PREDICT_LOCK = threading.Lock()

# 結果の書き換えを 1 つずつにする鍵
RESULT_LOCK = threading.Lock()

# 見本を作って登録する処理を 1 つずつにする鍵
SAMPLES_LOCK = threading.Lock()


# 画面に書いた文として受け付ける上限
MAX_TEXTS = 10000

# 表として一度に返す行の上限
MAX_ROWS = 500

# 一度に比べる結果の数の上限 (画面の列が読める幅に収まるように)
MAX_COMPARE = 6

# 比べる結果の id を並べた文字の長さの上限
MAX_COMPARE_IDS = 1000

# 名前に使える長さ
MAX_NAME = 80

# 雑音だけの一式で一度に作る件数の上限
# (始まりの雑音を全ての件の分を一度に作り, 件ごとに配列を書くため)
MAX_SAMPLES = 1000

# 雑音だけの一式で一度に作る値 (件数 × 1 件の形の大きさ) の上限
# (float32 で 400 MB. 始まりの雑音と出力を記憶に持つため)
MAX_SAMPLE_VALUES = 100_000_000

# 画面で作る一式の手順の回数の上限 (拡散の学習の時刻の数の既定と同じ.
# 回数ごとにモデルを件数分呼ぶので, 作るときの確かめが長く止まらないように)
MAX_PROCEDURE_STEPS = 1000

# 雑音だけの一式を作るときの形の誤りに添える案内
NOISE_SHAPE_HINT = (
    "雑音だけの一式では, 出力の形の欄の「1 件の形」が, モデルの定義の"
    "手順の中身の入力の形 (先頭の件数の次元を除く) と合うかを見直して"
    "ください"
)

# 雑音だけの一式を作るときに ML-Inference が確かめで作る件数
CHECK_SAMPLES = 2

# 並べるの一式で, 利用者ごとに並べる件数の上限
MAX_RANK_K = 1000

# 並べるの一式で, 一度に書く表の行 (利用者の数 × 並べる件数) の上限
# (結果の表を 1 つの CSV に書き, 画面で読み直すため)
MAX_RANK_ROWS = 1_000_000

# 並べるの一式で読む前の反応と正解の反応の表の行の上限 (書く表の行の上限と
# 同じ. 利用者ごとの品物の集合を記憶に持つため)
MAX_REACTION_ROWS = MAX_RANK_ROWS

# 並べるの一式で読む表のファイルの拡張子と区切り
RANK_TABLES = {".csv": ",", ".tsv": "\t", ".parquet": None}

# 群分けの結果で, 群ごとの代表の行を一度に求める件数の上限 (群の数 × 件数の
# 行を 1 度に返すため)
MAX_REPRESENTATIVES = 20

# 推論の後に表を読み直して結論を出し直さない出力の形 (表の出力, 並べる,
# 群分け, 効果. infer の結論のまま残す)
KEPT_CONCLUSIONS = frozenset({"table", "ranking", "cluster", "effect"})

# 同じく出し直さない一式の種類 (異常の度合いと需要. 出力の形が列の出力と
# 同じ value と distribution なので, 種類で見分ける)
KEPT_KINDS = frozenset({"anomaly", "demand"})

# 異常の度合いの一式で, 人の確認に回す件数の上限 (確認の待ち行列を 1 度に
# 返す件数の上限)
MAX_CAPACITY = 1000

# 効果の一式で, 配る数の上限の上限 (JSON の整数を丸めずに渡せる大きさ)
MAX_BUDGET = 100_000

# 値段の一式で, 値段の候補の数の上限 (候補ごとに全ての行を予測し直すため)
MAX_CANDIDATES = 20

# 並べるの一式だけに渡せる欄を, ほかの一式に渡したときの文
RANK_ONLY = (
    "利用者の並び, 並べる件数, 前に反応した品物の表, 正解の反応の表は, "
    "並べるの一式だけに渡せます"
)

# 予測と判断の列 (外れ方を分ける列にしない. 画面の結果の表の右に寄せる列
# と同じ)
OWN_COLUMNS = frozenset({
    "index", "status", "reason", "step", "prediction", "confidence",
    "decision", "action", "level", "expected_cost", "point_expected_cost",
    "gain_if_acted", "expected_gain",
})

# 測り直す入力の無い結果の文
REMEASURE_INPUT = (
    "測り直すには, 目的の列を持つ入力のファイルの結果が要ります (書いた文"
    "と何段も先の予測の結果は除く)"
)

# 分割の結果に列の効きと 1 行の理由を求めたときの文
SPLIT_UNEXPLAINED = (
    "元のデータセットの分割の結果は, 前処理の前の列が無いので, 列の効きと"
    "行ごとの理由を求めません"
)

# 推論の誤りとして画面に返す例外
USER_ERRORS = (
    ValueError, FileNotFoundError, KeyError, TypeError, yaml.YAMLError
)


class BundleRegister(BaseModel):
    path: str
    name: Optional[str] = None


class BundleCreate(BaseModel):
    name: str = Field(min_length=1, max_length=MAX_NAME)
    spec: str
    dataset: Optional[str] = None
    weights: str
    inputs: Optional[dict[str, Any]] = None
    output: Optional[str] = None
    kind: Optional[str] = None
    target_scale: Union[dict[str, float], str, None] = None
    conclusion: Optional[dict[str, Any]] = None
    copy_weights: bool = True
    role: Optional[str] = None
    form: Optional[str] = None
    output_options: Optional[dict[str, Any]] = None
    procedure: Optional[dict[str, Any]] = None
    items: Optional[dict[str, Any]] = None


class ReadyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=MAX_NAME)
    library: str = "torchvision"
    model: str
    weights: str
    sha256: Optional[str] = None


class PredictRequest(BaseModel):
    source: Optional[str] = None
    texts: Optional[list[str]] = None
    # 件数と種は整数だけ (真偽値, 文字, 小数を整数にしないため)
    count: Optional[StrictInt] = None
    seed: Optional[StrictInt] = None
    options: Optional[dict[str, Any]] = None
    rules: Optional[dict[str, Any]] = None
    forecast: bool = False
    decision: Optional[dict[str, Any]] = None
    # 並べるの一式の利用者 (書いた値), 件数, 前に反応した品物の表と正解の
    # 反応の表のファイル
    users: Optional[list[str]] = None
    # 件数は整数だけ (型の誤りも 422 でなく理由付きの 400 で断るので, 受け
    # 取りでは型を決めない)
    k: Any = None
    seen: Optional[str] = None
    truth: Optional[str] = None
    # 一式の元のデータセットの分割の名前 (場所はサーバが一式の実験から
    # 見つけ直す)
    split: Optional[str] = Field(default=None, max_length=MAX_NAME)


class ConcludeRequest(BaseModel):
    rules: dict[str, Any] = Field(default_factory=dict)


class DecideRequest(BaseModel):
    decision: Optional[dict[str, Any]] = None


class RemeasureRequest(BaseModel):
    write: bool = False
    horizon: Optional[int] = None
    time: Optional[str] = None
    groups: Optional[str] = None


def get_store(request: Request) -> Store:
    """
    アプリの置き場を返す関数

    Parameters
    ----------
    request: Request
        リクエスト

    Returns
    ----------
    store: Store
        置き場
    """
    store: Store = request.app.state.store

    return store


def user_error(error: Exception) -> HTTPException:
    """
    推論や一式の誤りを, 画面に出す 400 の例外にする関数

    Parameters
    ----------
    error: Exception
        起きた例外

    Returns
    ----------
    exception: HTTPException
        400 の例外 (文は元の誤りの文)
    """
    # SpecError などの誤りの文をそのまま (登録されていない部品は起動の
    # 設定の案内付き)
    message = unit_hint(error) or str(error) or type(error).__name__
    exception = HTTPException(400, message)

    return exception


def join_hints(*hints: Optional[str]) -> Optional[str]:
    """
    起動の設定の案内をつなぐ関数

    Parameters
    ----------
    *hints: Optional[str]
        案内 (無いものは None)

    Returns
    ----------
    message: Optional[str]
        つないだ案内 (全て無ければ None)
    """
    # 部品と手順の案内を 1 つの文に
    parts = [hint for hint in hints if hint]
    message = " / ".join(parts) if parts else None

    return message


def bundle_error(path: str, error: Exception) -> HTTPException:
    """
    一式を読めない誤りを, 登録簿に無い手順の案内を添えて 400 にする関数

    Parameters
    ----------
    path: str
        一式のディレクトリ
    error: Exception
        起きた例外

    Returns
    ----------
    exception: HTTPException
        400 の例外
    """
    # 自作の部品と手順は起動の設定での登録 (一式のコードは読まない. 両方
    # 無ければ両方の案内)
    message = join_hints(unit_hint(error), missing_procedure(path))
    if message is None:
        exception = user_error(error)
        return exception
    exception = HTTPException(400, message)

    return exception


def find_bundle(request: Request, bundle_id: str) -> dict[str, Any]:
    """
    登録した一式を探す関数

    Parameters
    ----------
    request: Request
        リクエスト
    bundle_id: str
        一式の id

    Returns
    ----------
    entry: dict[str, Any]
        登録

    Raises
    ----------
    HTTPException
        無い場合は 404
    """
    entry = get_store(request).find(bundle_id)
    if entry is None:
        raise HTTPException(404, "一式が見つかりません")

    return entry


def bundle_view(entry: dict[str, Any]) -> dict[str, Any]:
    """
    登録に一式の説明を足して返す関数

    Parameters
    ----------
    entry: dict[str, Any]
        登録

    Returns
    ----------
    view: dict[str, Any]
        登録, ok, error, 説明 (読めない場合は None), uses_problem (問題
        の設定の欄を出すか)
    """
    # 消えた一式や壊れた一式は理由付き
    view: dict[str, Any] = {
        **entry, "ok": True, "error": None, "model_loaded": False,
        "uses_problem": True,
    }
    try:
        description, loaded = describe(str(entry["path"]))
        kind = description.get("kind")
        view.update(
            description=description, model_loaded=loaded,
            uses_problem=uses_problem(
                kind if isinstance(kind, str) else None
            ),
        )
    except Exception as error:
        message = join_hints(
            unit_hint(error), missing_procedure(str(entry["path"]))
        ) or str(error)
        view.update(ok=False, error=message, description=None)

    return view


@router.get("/info")
def get_info(request: Request) -> dict[str, Any]:
    """
    ML-Inference の版と結論の決め方の項目を返す関数

    Parameters
    ----------
    request: Request
        リクエスト

    Returns
    ----------
    info: dict[str, Any]
        version, rule_keys, result_version (読める結果の形の版まで),
        procedures (繰り返しの手順の登録簿), procedure_modules (起動の
        設定で読んだ手順のモジュール), unit_modules (起動の設定で読んだ
        部品のモジュールと登録した部品の名前), max_samples (雑音だけの
        一式で一度に作る件数の上限), max_procedure_steps (画面で作る一式の
        手順の回数の上限), ready_libraries (既製のモデルの出どころ)
    """
    info = {
        "version": ml_inference.__version__,
        "rule_keys": list(ml_inference.RULE_KEYS),
        "result_version": ml_inference.RESULT_FORMAT_VERSION,
        "procedures": registered(),
        "procedure_modules": list(
            getattr(request.app.state, "procedure_modules", [])
        ),
        "unit_modules": list(
            getattr(request.app.state, "unit_modules", [])
        ),
        "max_samples": MAX_SAMPLES,
        "max_procedure_steps": MAX_PROCEDURE_STEPS,
        "ready_libraries": list(ml_inference.READY_LIBRARIES),
    }

    return info


@router.get("/browse")
def browse(path: str = "~") -> dict[str, Any]:
    """
    ディレクトリの中身を返す関数 (ファイルとディレクトリを選ぶため)

    Parameters
    ----------
    path: str = "~"
        ディレクトリ (~ から書ける)

    Returns
    ----------
    listing: dict[str, Any]
        path, parent, home, entries (名前, 場所, ディレクトリか, 大きさ),
        truncated

    Raises
    ----------
    HTTPException
        ディレクトリでない場合は 404
    """
    # 隠しファイルを除いて, ディレクトリを先に名前の順で並べる
    target = Path(path).expanduser().resolve()
    if not target.is_dir():
        raise HTTPException(404, f"{target} はディレクトリではありません")
    try:
        children = sorted(
            target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
        )
    except PermissionError as error:
        raise HTTPException(403, "読む権限がありません") from error
    visible = [c for c in children if not c.name.startswith(".")]
    entries = []
    for child in visible[:1000]:
        try:
            size = child.stat().st_size if child.is_file() else None
        except OSError:
            continue
        entries.append({
            "name": child.name, "path": str(child),
            "is_dir": child.is_dir(), "size": size,
            "bundle": child.is_dir() and (child / "inference.yaml").is_file(),
        })
    listing = {
        "path": str(target),
        "parent": str(target.parent) if target.parent != target else None,
        "home": str(Path.home()),
        "entries": entries,
        "truncated": len(visible) > 1000,
    }

    return listing


@router.get("/bundles")
def list_bundles(request: Request) -> list[dict[str, Any]]:
    """
    登録した一式の一覧を返す関数

    Parameters
    ----------
    request: Request
        リクエスト

    Returns
    ----------
    bundles: list[dict[str, Any]]
        登録と説明 (新しい順)
    """
    entries = get_store(request).read()
    bundles = [bundle_view(entry) for entry in reversed(entries)]

    return bundles


@router.post("/bundles", status_code=201)
def register_bundle(
    request: Request, body: BundleRegister
) -> dict[str, Any]:
    """
    場所を指定して一式を登録する関数

    Parameters
    ----------
    request: Request
        リクエスト
    body: BundleRegister
        一式のディレクトリと名前

    Returns
    ----------
    view: dict[str, Any]
        登録と説明

    Raises
    ----------
    HTTPException
        読めない一式と, 既に登録した場所は 400
    """
    # 読めることを確かめてから登録
    path = Path(body.path).expanduser().resolve()
    try:
        bundle = read_bundle(str(path))
    except (OSError, *USER_ERRORS) as error:
        raise bundle_error(str(path), error) from error
    name = (body.name or str(bundle.info.get("name") or path.name))[:MAX_NAME]
    try:
        entry = get_store(request).add(path, name, managed=False)
    except ValueError as error:
        raise user_error(error) from error
    view = bundle_view(entry)

    return view


@router.post("/bundles/create", status_code=201)
def create_bundle(request: Request, body: BundleCreate) -> dict[str, Any]:
    """
    モデルの定義, データセット, 実験から置き場に一式を作って登録する関数

    Parameters
    ----------
    request: Request
        リクエスト
    body: BundleCreate
        名前, 定義, データセット (省くか空なら雑音だけの一式), 重みか実験,
        つなぎ方, 結論の既定値, 版 2 のデータセットの 1 件の扱い (items)

    Returns
    ----------
    view: dict[str, Any]
        登録と説明

    Raises
    ----------
    HTTPException
        一式を作れない場合は 400 (理由付き)
    """
    # 見本を推論できることまで ML-Inference が確かめる
    # (データセットを省くと雑音だけの一式. 要る欄は ML-Inference が確かめる)
    # 雑音だけの一式は作るときの確かめで 2 件を作るので, 推論と同じ上限を
    # 先に当てる (データセットのある一式の確かめの件数は ML-Inference 次第)
    dataset = (body.dataset or "").strip()
    shape = (body.output_options or {}).get("shape")
    if not dataset and isinstance(shape, list) and shape and all(
        isinstance(v, int) and not isinstance(v, bool) and v > 0
        for v in shape
    ) and sample_limit(shape) < CHECK_SAMPLES:
        raise HTTPException(
            400, f"1 件の形 {shape} が大きすぎます (確かめで作る "
            f"{CHECK_SAMPLES} 件が値の数の上限 {MAX_SAMPLE_VALUES:,} を"
            "超えます)"
        )
    # 手順の回数の上限 (作るときの確かめと推論が長く止まらないように)
    steps = (body.procedure or {}).get("steps")
    if isinstance(steps, int) and steps > MAX_PROCEDURE_STEPS:
        raise HTTPException(
            400, f"手順の回数は {MAX_PROCEDURE_STEPS:,} 回までです"
        )
    store = get_store(request)
    bundle_id, place = store.new_bundle_dir()
    try:
        ml_inference.make_bundle(
            place, Path(body.spec).expanduser(),
            Path(dataset).expanduser() if dataset else None,
            Path(body.weights).expanduser(),
            inputs=body.inputs, output=body.output or None,
            kind=body.kind or None, target_scale=body.target_scale,
            conclusion=body.conclusion, name=body.name,
            copy_weights=body.copy_weights, role=body.role or None,
            form=body.form or None, output_options=body.output_options,
            procedure=body.procedure, items=body.items
        )
    except Exception as error:
        exception = user_error(error)
        # 雑音だけの一式の 1 件の形の誤り (output.shape の誤りと, モデルの
        # 入力と形が合わない実行時の誤り) は, 画面の 1 件の形の欄へ案内
        # (重みが定義と合わない誤り (state_dict) は形の欄では直らないので除く)
        detail = str(exception.detail)
        shaped = "output.shape" in detail or (
            (isinstance(error, RuntimeError) or "RuntimeError" in detail)
            and "state_dict" not in detail
        )
        if not dataset and shaped:
            exception.detail = f"{exception.detail} ({NOISE_SHAPE_HINT})"
        raise exception from error
    entry = store.add(place, body.name, managed=True, bundle_id=bundle_id)
    view = bundle_view(entry)

    return view


@router.get("/ready/preview")
def get_ready_preview(
    library: str = Query(max_length=MAX_NAME),
    model: str = Query(max_length=200),
    weights: str = Query(max_length=200),
) -> dict[str, Any]:
    """
    既製のモデルの一式を作る前に, 種類の名前, 前処理, ライセンスを読む関数

    Parameters
    ----------
    library: str
        出どころ (ML-Inference の READY_LIBRARIES の名前)
    model: str
        モデルの名前
    weights: str
        重みの名前

    Returns
    ----------
    preview: dict[str, Any]
        library, model, weights, classes, input, preprocess, url, file,
        license (重みは取りに行かない)

    Raises
    ----------
    HTTPException
        出どころ, モデル, 重みの名前が無い場合は 400
    """
    # 出どころが入っていない場合 (ImportError) も理由を返す
    try:
        preview = preview_ready(library, model, weights)
    except (ImportError, *USER_ERRORS) as error:
        raise user_error(error) from error

    return preview


@router.post("/bundles/ready", status_code=201)
def create_ready_bundle(
    request: Request, body: ReadyCreate
) -> dict[str, Any]:
    """
    学習しない既製のモデルの一式を置き場に作って登録する関数

    Parameters
    ----------
    request: Request
        リクエスト
    body: ReadyCreate
        名前, 出どころ, モデル, 重み, 決めておいたハッシュ値

    Returns
    ----------
    view: dict[str, Any]
        登録と説明

    Raises
    ----------
    HTTPException
        一式を作れない場合は 400 (理由付き). 重みを出どころから取れない
        場合と, 取得が時間の上限を超えた場合は 502
    """
    # 重みを出どころから読んでハッシュ値を固定する (初回は取りに行く)
    store = get_store(request)
    bundle_id, place = store.new_bundle_dir()
    try:
        make_ready(
            place, body.library, body.model, body.weights, body.name,
            sha256=body.sha256
        )
    except FetchError as error:
        raise HTTPException(502, str(error)) from error
    except Exception as error:
        raise user_error(error) from error
    entry = store.add(place, body.name, managed=True, bundle_id=bundle_id)
    view = bundle_view(entry)

    return view


@router.get("/bundles/{bundle_id}/test-split")
def get_test_split(request: Request, bundle_id: str) -> dict[str, Any]:
    """
    学習の結果 (実験) から登録した一式の, 元のデータセットのテスト用の
    分割を返す関数

    Parameters
    ----------
    request: Request
        リクエスト
    bundle_id: str
        一式の id

    Returns
    ----------
    found: dict[str, Any]
        split (選べる分割の名前. 無ければ None), dataset, rows,
        experiment, reason (選べない理由)
    """
    # 一式から実験, config の data.dataset, dataset.yaml の順
    found = find_test_split(find_bundle(request, bundle_id))

    return found


@router.post("/samples")
def add_samples(request: Request) -> list[dict[str, Any]]:
    """
    操作を試すための見本の一式を, その場で学習して作り登録する関数

    Parameters
    ----------
    request: Request
        リクエスト

    Returns
    ----------
    bundles: list[dict[str, Any]]
        登録した見本 (登録済みの見本は登録し直さない)

    Raises
    ----------
    HTTPException
        見本を作れない場合は 500
    """
    # 2 つの要求が同時に来ても 2 重に登録しないように, 全体を 1 つずつ
    store = get_store(request)
    with SAMPLES_LOCK:
        try:
            samples = write_samples(store.home / "samples")
        except Exception as error:
            raise HTTPException(
                500, f"見本を作れませんでした: {error}"
            ) from error
        # 同じ場所か同じ名前の見本の登録は, 登録し直さない (置き場を移した
        # 場合も重ねない)
        entries = store.read()
        registered = {item.get("path") for item in entries} | {
            item.get("name") for item in entries if item.get("sample_input")
        }
        bundles = []
        # 一覧は新しいものが先なので, 最初の見本が先頭に来るように後ろから
        # の登録
        for name, (place, fresh, past) in reversed(samples.items()):
            if str(place.resolve()) in registered or name in registered:
                continue
            # 過去のデータは列の見本だけ
            extra = {"sample_input": str(fresh.resolve())}
            if past is not None:
                extra["sample_past"] = str(past.resolve())
            try:
                entry = store.add(place, name, managed=False, extra=extra)
            except ValueError:
                continue
            bundles.append(bundle_view(entry))

    return bundles


@router.get("/bundles/{bundle_id}")
def get_bundle(request: Request, bundle_id: str) -> dict[str, Any]:
    """
    一式の登録と説明を返す関数

    Parameters
    ----------
    request: Request
        リクエスト
    bundle_id: str
        一式の id

    Returns
    ----------
    view: dict[str, Any]
        登録と説明
    """
    view = bundle_view(find_bundle(request, bundle_id))

    return view


@router.post("/bundles/{bundle_id}/restore-uncertainty")
def restore_uncertainty(request: Request, bundle_id: str) -> dict[str, Any]:
    """
    測り直した一式の外れ方を, 一式を作ったときの記録に戻す関数

    Parameters
    ----------
    request: Request
        リクエスト
    bundle_id: str
        一式の id

    Returns
    ----------
    view: dict[str, Any]
        戻した後の登録と説明

    Raises
    ----------
    HTTPException
        戻せる記録が無い場合と, 書けない場合は 400
    """
    # 推論と同じ鍵の中での書き戻し (測り直しと重ならないように). 学習の
    # 実験の中の一式は, 書き戻しと同じく置き場に写した方への書き戻し
    find_bundle(request, bundle_id)
    store = get_store(request)
    with PREDICT_LOCK:
        entry = find_bundle(request, bundle_id)
        copied = None
        target = Path(entry["path"])
        try:
            if in_experiment(target):
                copied = copy_into_store(store, entry)
                target = copied
            ml_inference.restore_uncertainty(str(target))
            if copied is not None:
                entry = store.relocate(entry["id"], copied)
        except (OSError, KeyError, *USER_ERRORS) as error:
            if copied is not None:
                shutil.rmtree(copied, ignore_errors=True)
            raise user_error(error) from error
    view = bundle_view(entry)

    return view


@router.delete("/bundles/{bundle_id}", status_code=204)
def delete_bundle(request: Request, bundle_id: str) -> Response:
    """
    一式の登録を消す関数

    Parameters
    ----------
    request: Request
        リクエスト
    bundle_id: str
        一式の id

    Returns
    ----------
    response: Response
        204

    Raises
    ----------
    HTTPException
        無い場合は 404
    """
    entry = get_store(request).find(bundle_id)
    if not get_store(request).remove(bundle_id):
        raise HTTPException(404, "一式が見つかりません")
    # 覚えた一式 (組み立てたモデル, 説明, 警告) の手放し
    forget(str((entry or {}).get("path")))
    response = Response(status_code=204)

    return response


def text_frame(bundle: ml_inference.Bundle, texts: list[str]) -> pd.DataFrame:
    """
    画面に書いた文を, 文の列を持つ表にする関数

    Parameters
    ----------
    bundle: ml_inference.Bundle
        一式
    texts: list[str]
        文の並び (空の行は除く)

    Returns
    ----------
    frame: pd.DataFrame
        文の列だけの表

    Raises
    ----------
    HTTPException
        文の一式でない場合, 文が無い場合と多すぎる場合は 400
    """
    # 文の一式だけ
    column = bundle.dataset.info.get("text_column")
    if bundle.kind != "text" or not column:
        raise HTTPException(400, "文を書いて推論できるのは文の一式だけです")
    kept = [text.strip() for text in texts if text.strip()]
    if not kept:
        raise HTTPException(400, "文がありません")
    if len(kept) > MAX_TEXTS:
        raise HTTPException(400, f"文は {MAX_TEXTS} 件までです")
    frame: pd.DataFrame = pd.DataFrame({column: kept})

    return frame


def sample_limit(shape: list[int]) -> int:
    """
    1 件の形から, 一度に作れる件数の上限を求める関数

    Parameters
    ----------
    shape: list[int]
        1 件の形 (空なら 1 つの値)

    Returns
    ----------
    most: int
        件数の上限 (MAX_SAMPLES と, 値の数の上限に収まる件数の小さい方)

    Raises
    ----------
    HTTPException
        1 件だけで値の数の上限を超える場合は 400
    """
    # 大きな形でもあふれない Python の整数で数える
    size = math.prod(shape) if shape else 1
    if size > MAX_SAMPLE_VALUES:
        raise HTTPException(
            400, f"1 件の形 {shape} が大きすぎます (1 件の値の数 {size:,} "
            f"が上限の {MAX_SAMPLE_VALUES:,} を超えます)"
        )
    most = min(MAX_SAMPLES, MAX_SAMPLE_VALUES // size)

    return most


def noise_source(
    bundle: ml_inference.Bundle, body: PredictRequest
) -> dict[str, int]:
    """
    雑音だけの一式に渡す件数と種を確かめる関数

    Parameters
    ----------
    bundle: ml_inference.Bundle
        一式
    body: PredictRequest
        件数と種 (種は省くと一式の手順の種)

    Returns
    ----------
    source: dict[str, int]
        ML-Inference の predict に渡す件数と種の辞書

    Raises
    ----------
    HTTPException
        雑音だけの一式でない場合, ファイルか文も渡した場合, 件数が無い
        場合と上限を超える場合は 400
    """
    # 件数と種は雑音だけの一式だけ (データの代わりに受け取るもの)
    if not bundle.from_noise:
        raise HTTPException(
            400, "件数と種で推論できるのは, データセットを持たない"
            "雑音だけの一式だけです"
        )
    if body.source or body.texts is not None:
        raise HTTPException(
            400, "雑音だけの一式はファイルや文を読みません. "
            "件数と種を指定してください"
        )
    if body.count is None:
        raise HTTPException(400, "作る件数を指定してください")

    # 件数の上限 (始まりの雑音を全ての件の分を一度に作るため)
    shape = [int(v) for v in bundle.output_info.get("shape") or []]
    most = sample_limit(shape)
    if body.count > most:
        raise HTTPException(
            400, f"1 件の形 {shape} では, 記憶に収まるように件数は "
            f"{most} 件までです"
        )
    source = {"count": body.count}
    if body.seed is not None:
        source["seed"] = body.seed

    return source


def rank_table(
    text: str, what: str, most: Optional[int] = None
) -> tuple[Path, pd.DataFrame]:
    """
    並べるの一式に渡す表 (利用者, 前の反応, 正解の反応) のファイルを読む
    関数

    Parameters
    ----------
    text: str
        ファイルの場所 (~ から書ける)
    what: str
        誤りの文に使う表の呼び名
    most: Optional[int] = None
        行の上限 (省くと決めない)

    Returns
    ----------
    table: tuple[Path, pd.DataFrame]
        読んだ場所と表 (CSV と TSV は全ての列を文字で読み, 空の欄だけを
        欠損にする)

    Raises
    ----------
    HTTPException
        無いファイル, ディレクトリ, 読めない形式, 行が上限を超える表は 400
    """
    # 利用者と品物の値は文字のまま (番号の表と文字にして引くので, 007 を
    # 7 にしないように)
    path = Path(text).expanduser().resolve()
    if path.is_dir():
        raise HTTPException(
            400, f"{what}には, ディレクトリでなくファイルを指定してください "
            f"({path} はディレクトリです)"
        )
    if not path.is_file():
        raise HTTPException(400, f"{what}のファイル {path} がありません")
    suffix = path.suffix.lower()
    if suffix not in RANK_TABLES:
        raise HTTPException(
            400, f"{what}は CSV, TSV, Parquet のファイルにしてください "
            f"(今は {path.name})"
        )
    try:
        if suffix == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(
                path, sep=RANK_TABLES[suffix], encoding="utf-8-sig",
                dtype=str, keep_default_na=False, na_values=[""]
            )
    except Exception as error:
        raise HTTPException(
            400, f"{what}のファイル {path.name} を読めません: {error}"
        ) from error
    if most is not None and len(frame) > most:
        raise HTTPException(
            400, f"{what}は {most:,} 行までです (今は {len(frame):,} 行)"
        )
    table = (path, frame)

    return table


def rank_request(
    bundle: ml_inference.Bundle, body: PredictRequest
) -> dict[str, Any]:
    """
    並べるの一式に渡す利用者, 件数, 前に反応した品物と正解の反応を確かめる
    関数

    Parameters
    ----------
    bundle: ml_inference.Bundle
        一式
    body: PredictRequest
        書いた利用者か利用者の列を持つ表のファイル, 件数 (省くと一式の
        rank.k), 前に反応した品物の表と正解の反応の表のファイル (任意)

    Returns
    ----------
    request: dict[str, Any]
        source (ML-Inference に渡す利用者の並びか表), described (入力の
        文. 書いた利用者は None), path (表のファイル), k, seen と truth
        (読んだ表と場所. 渡さなければ None)

    Raises
    ----------
    HTTPException
        並べるの一式でないのに並べるの欄を渡した場合, 利用者が無い場合,
        件数が範囲の外の場合, 表が多すぎる場合は 400
    """
    # 並べるの欄は並べるの一式だけ (他の一式の入力と混ぜない)
    given = (body.users is not None
             or (body.k is not None and not bundle.from_search)
             or body.seen is not None or body.truth is not None)
    if not bundle.from_rank:
        if given:
            raise HTTPException(400, RANK_ONLY)
        return {}
    if body.texts is not None or body.count is not None or (
        body.seed is not None
    ):
        raise HTTPException(
            400, "並べるの一式には, 利用者を書くか, 利用者の列を持つ表の"
            "ファイルを渡してください (文, 件数, 種は使いません)"
        )
    if body.options:
        raise HTTPException(
            400, "並べるの一式では読み込みの設定を使えません"
        )

    # 書いた利用者 (空の行は除く) か, 利用者の列を持つ表
    column = str(bundle.info["rank"]["columns"]["user"])
    path: Optional[Path] = None
    if body.users is not None and body.source:
        raise HTTPException(
            400, "利用者は, 書くか表のファイルかのどちらかで渡してください"
        )
    if body.users is not None:
        source: Any = [user.strip() for user in body.users if user.strip()]
        if not source:
            raise HTTPException(400, "利用者がありません")
        count = len(source)
    elif body.source:
        path, source = rank_table(body.source, "利用者の表")
        if column not in source.columns:
            raise HTTPException(
                400, f"利用者の表に利用者の列 {column} がありません (列: "
                f"{', '.join(map(str, source.columns))})"
            )
        count = len(source)
    else:
        raise HTTPException(
            400, "利用者を書くか, 利用者の表のファイルを指定してください"
        )
    if count > MAX_TEXTS:
        raise HTTPException(400, f"利用者は 1 回に {MAX_TEXTS:,} 人までです")

    # 件数 (省くと一式の rank.k) と, 書く表の行の上限
    k = int(bundle.info["rank"]["k"]) if body.k is None else body.k
    if isinstance(k, bool) or not isinstance(k, int) or not (
        1 <= k <= MAX_RANK_K
    ):
        raise HTTPException(
            400, f"並べる件数は 1 から {MAX_RANK_K} の整数にしてください"
        )
    rows = count * min(k, int(bundle.info["rank"]["items"]))
    if rows > MAX_RANK_ROWS:
        raise HTTPException(
            400, f"利用者 {count:,} 人に {k} 件ずつでは, 表が "
            f"{MAX_RANK_ROWS:,} 行を超えます. 利用者か件数を減らしてください"
        )
    seen = rank_table(
        body.seen, "前に反応した品物の表", MAX_REACTION_ROWS
    ) if body.seen else None
    truth = rank_table(
        body.truth, "正解の反応の表", MAX_REACTION_ROWS
    ) if body.truth else None
    request = {
        "source": source, "path": path, "k": k, "seen": seen,
        "truth": truth,
        "described": None if path is None else f"{path} (利用者 {count:,} 人)",
    }

    return request


def ranking_meta(
    bundle: ml_inference.Bundle, result: dict[str, Any],
    ranked: dict[str, Any]
) -> dict[str, Any]:
    """
    並べるの結果に残す, 件数, 除いたか, 使った表と Recall@K と NDCG を
    まとめる関数

    Parameters
    ----------
    bundle: ml_inference.Bundle
        並べるの一式
    result: dict[str, Any]
        ML-Inference の infer の結果 (frame は attrs に k を持つ)
    ranked: dict[str, Any]
        rank_request の返り値

    Returns
    ----------
    ranking: dict[str, Any]
        k, excluded, columns (反応の表の列の名前), seen と truth (読んだ
        ファイルの場所), metrics (正解の反応の表を渡した場合の users, k,
        recall_at_k, ndcg. 渡さなければ None)
    """
    # 正解の反応があれば, 前に反応した品物を正解から除いて数える
    seen, truth = ranked["seen"], ranked["truth"]
    metrics = None
    if truth is not None:
        metrics = ml_inference.rank_metrics(
            bundle, result["frame"], truth[1],
            seen=None if seen is None else seen[1], k=ranked["k"]
        )
    summary = result.get("summary") or {}
    ranking = {
        "k": ranked["k"],
        "excluded": bool(summary.get("excluded")),
        "columns": dict(bundle.info["rank"]["columns"]),
        "seen": None if seen is None else str(seen[0]),
        "truth": None if truth is None else str(truth[0]),
        "metrics": metrics,
    }

    return ranking


@router.post("/bundles/{bundle_id}/predict", status_code=201)
def predict(
    request: Request, bundle_id: str, body: PredictRequest
) -> dict[str, Any]:
    """
    ファイルか文を推論し, 結論を出して結果を保存する関数

    Parameters
    ----------
    request: Request
        リクエスト
    bundle_id: str
        一式の id
    body: PredictRequest
        ファイルの場所か文の並び (雑音だけの一式は件数と種), 読み込みの
        設定, 結論の決め方

    Returns
    ----------
    view: dict[str, Any]
        保存した結果のまとめ

    Raises
    ----------
    HTTPException
        入力が無い場合と, 推論できない場合は 400
    """
    # 一式と入力
    entry = find_bundle(request, bundle_id)
    try:
        bundle = open_bundle(str(entry["path"]))
    except (OSError, *USER_ERRORS) as error:
        raise bundle_error(str(entry["path"]), error) from error
    rules, decision = body.rules, body.decision
    if bundle.from_anomaly:
        # 異常の度合いの一式は振り分けだけ (省くと一式の既定)
        decision = anomaly_decision(bundle, body)
        rules = None
    elif bundle.from_uplift:
        # 効果の一式は配る判断だけ (省くと一式の既定)
        decision = uplift_decision(bundle, body)
        rules = None
    elif bundle.from_price:
        # 値段の一式は値段の判断だけ (省くと一式の既定)
        decision = price_decision(bundle, body)
        rules = None
    elif bundle.from_demand:
        # 需要の一式は発注の判断だけ (省くと一式の既定)
        decision = demand_decision(body)
        rules = None
    elif bundle.from_search:
        # 検索の一式は返す件数の検索の判断だけ (省くと一式の search.k)
        decision = search_decision(bundle, body)
        rules = None
    elif not bundle.column_output:
        # 列の出力でない一式は件数だけ (決め方と意思決定は使えないもの)
        given = any(v is not None for v in (rules or {}).values())
        if given or decision or body.forecast:
            raise HTTPException(400, column_only(bundle.form))
        rules, decision = None, None
    # 分割の名前だけを受け, 場所は一式の実験から見つけ直す (他の入力と
    # 一緒には受けない)
    noise = body.count is not None or body.seed is not None
    others = (body.source, body.texts, body.users, body.seen, body.truth)
    if body.split is not None and (
        noise or body.k is not None or any(v is not None for v in others)
    ):
        raise HTTPException(
            400, "分割と, ファイル, 文, 件数, 利用者は一緒に渡せません"
        )
    ranked: dict[str, Any] = (
        {} if body.split is not None else rank_request(bundle, body)
    )
    if body.split is not None:
        try:
            source: Any = split_source(entry, body.split)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        described = (
            f"元のデータセットのテスト用の分割 ({source['dataset']} の "
            f"{source['split']})"
        )
    elif ranked:
        source = ranked["source"]
        described = ranked["described"]
    elif bundle.from_noise or noise:
        source = noise_source(bundle, body)
        described = None
    elif body.texts is not None:
        source = text_frame(bundle, body.texts)
        described = f"書いた文 {len(source)} 件"
    elif body.source:
        source = Path(body.source).expanduser().resolve()
        if not source.exists():
            raise HTTPException(400, f"{source} がありません")
        described = str(source)
    else:
        raise HTTPException(400, "推論するファイルか文を指定してください")

    # 推論と結論 (1 つずつ)
    with PREDICT_LOCK:
        try:
            # (並べるの一式だけに件数と前に反応した品物の表を渡す)
            extra: dict[str, Any] = {} if not ranked else {
                "k": ranked["k"],
                "seen": None if ranked["seen"] is None else ranked["seen"][1],
            }
            if bundle.from_search and decision:
                extra = {"k": decision["k"]}
            result = ml_inference.infer(
                bundle, source, rules=rules, options=body.options,
                forecast=body.forecast, decision=decision, **extra
            )
            ranking = ranking_meta(bundle, result, ranked) if ranked else None
        except Exception as error:
            raise user_error(error) from error
        warned = list(bundle.cache.get("pretrained_warnings") or [])
        remember_warnings(str(entry["path"]), bundle)
    # 雑音だけの一式の入力の文 (「件数 n, 種 s」) は ML-Inference のまま
    result.update(
        source=described or result.get("source"),
        bundle_id=bundle_id, task=bundle.task,
        classes=bundle.classes, kind=bundle.kind, forecast=body.forecast,
        has_target=bool(
            str(bundle.dataset.info.get("target")) in result["frame"].columns
            and result["frame"][str(bundle.dataset.info.get("target"))]
            .notna().any()
        ),
        options=body.options,
        source_path=(
            str(source) if isinstance(source, Path)
            else str(ranked["path"]) if ranked.get("path") else None
        ),
        role=bundle.role, output_form=bundle.form,
        column_output=bundle.column_output,
        target_info=ml_inference.target_of(bundle),
        origin_note=ml_inference.origin_sentence(
            ml_inference.target_of(bundle)
        ),
        pretrained_warnings=warned,
        table_columns=table_columns(bundle),
        ranking=ranking,
        grouping=grouping_meta(bundle, result["frame"]),
        uplift=uplift_meta(bundle),
        price=price_meta(bundle, result["frame"]),
        demand=demand_meta(bundle, result["frame"]),
        search=search_meta(bundle, result)
    )
    # 書いた表を読み直して結論を出し直す (出し直しと同じ型の値でまとめ,
    # 日時のような列のまとめの名前が, 後の出し直しで変わらないように)
    # (書き出しの途中で失敗した結果は, 壊れたまま残さずに削除)
    result_id, place = get_store(request).new_result_dir()
    meta = {k: v for k, v in result.items() if k != "frame"}
    # (表の出力, 並べる, 群分け, 異常の度合い, 効果, 値段, 需要の結果は
    # infer の結論のまま. 表の出力の件数は数え直しても同じ件数. 並べるの
    # 利用者ごとの数え方, 群分けの群の要約と群の安定性, 異常の度合いと効果
    # の分布と上位の行, 値段の弾力性と実験の評価, 需要の分位点の要約と実験
    # の評価は件数だけの結論に無い中身)
    # (出し直しは書いた result.yaml を読んで渡す. 予測の表を読む列の型
    # column_types は write_result が書くときに決めるため)
    try:
        ml_inference.write_result(result, place)
        kept = meta.get("output_form") in KEPT_CONCLUSIONS or (
            meta.get("kind") in KEPT_KINDS
        )
        if not kept:
            with RESULT_LOCK:
                conclude_into(
                    place, read_meta(place), meta.get("rules") or {}
                )
    except Exception as error:
        shutil.rmtree(place, ignore_errors=True)
        raise user_error(error) from error
    view = result_view(result_id, place)

    return view


def anomaly_decision(
    bundle: ml_inference.Bundle, body: PredictRequest
) -> Optional[dict[str, Any]]:
    """
    異常の度合いの一式に渡す振り分けを確かめる関数

    Parameters
    ----------
    bundle: ml_inference.Bundle
        異常の度合いの一式
    body: PredictRequest
        推論の要求

    Returns
    ----------
    decision: Optional[dict[str, Any]]
        渡す振り分け (None は一式の既定, 空の辞書は振り分けない. 型と
        項目は ML-Inference が確かめる)

    Raises
    ----------
    HTTPException
        結論の決め方か何段も先の予測を渡した場合と, 件数の上限が
        MAX_CAPACITY を超える場合は 400
    """
    # 結論の決め方と何段も先の予測は使えないもの
    given = any(v is not None for v in (body.rules or {}).values())
    if given or body.forecast:
        raise HTTPException(
            400, "異常の度合いの一式には, 結論の決め方と何段も先の予測を"
            "使えません. 意思決定は振り分け ({type: triage, capacity}) だけ"
            "です"
        )
    decision = body.decision
    found = bundle.info.get("decision") if decision is None else decision
    capacity = (found or {}).get("capacity") if isinstance(
        found, dict
    ) else None
    if isinstance(capacity, int) and not isinstance(capacity, bool) and (
        capacity > MAX_CAPACITY
    ):
        raise HTTPException(
            400, f"人の確認に回す件数の上限は {MAX_CAPACITY} 件までに"
            f"してください (今は {capacity} 件)"
        )

    return decision


def uplift_decision(
    bundle: ml_inference.Bundle, body: PredictRequest
) -> Optional[dict[str, Any]]:
    """
    効果の一式に渡す配る判断を確かめる関数

    Parameters
    ----------
    bundle: ml_inference.Bundle
        効果の一式
    body: PredictRequest
        推論の要求

    Returns
    ----------
    decision: Optional[dict[str, Any]]
        渡す配る判断 (None は一式の既定, 空の辞書は配らない. 型と項目は
        ML-Inference が確かめる)

    Raises
    ----------
    HTTPException
        結論の決め方か何段も先の予測を渡した場合と, 配る数の上限が
        MAX_BUDGET を超える場合は 400
    """
    # 結論の決め方と何段も先の予測は使えないもの
    given = any(v is not None for v in (body.rules or {}).values())
    if given or body.forecast:
        raise HTTPException(
            400, "効果の一式には, 結論の決め方と何段も先の予測を使えません. "
            "意思決定は効果の上位から配る判断 ({type: treat, budget}) だけ"
            "です"
        )
    decision = body.decision
    found = bundle.info.get("decision") if decision is None else decision
    budget = (found or {}).get("budget") if isinstance(found, dict) else None
    if isinstance(budget, (int, float)) and not isinstance(budget, bool) and (
        budget > MAX_BUDGET
    ):
        raise HTTPException(
            400, f"配る数の上限は {MAX_BUDGET:,} 件までにしてください (今は "
            f"{budget:,} 件)"
        )

    return decision


def price_decision(
    bundle: ml_inference.Bundle, body: PredictRequest
) -> Optional[dict[str, Any]]:
    """
    値段の一式に渡す値段の判断を確かめる関数

    Parameters
    ----------
    bundle: ml_inference.Bundle
        値段の一式
    body: PredictRequest
        推論の要求

    Returns
    ----------
    decision: Optional[dict[str, Any]]
        渡す値段の判断 (None は一式の既定, 空の辞書は値段を決めない. 型と
        項目, 候補と上限の形は ML-Inference が確かめる)

    Raises
    ----------
    HTTPException
        結論の決め方か何段も先の予測を渡した場合と, 値段の候補が
        MAX_CANDIDATES 個を超える場合は 400
    """
    # 結論の決め方と何段も先の予測は使えないもの
    given = any(v is not None for v in (body.rules or {}).values())
    if given or body.forecast:
        raise HTTPException(
            400, "値段の一式には, 結論の決め方と何段も先の予測を使えません. "
            "意思決定は上限の中で利益が最も大きい値段を選ぶ判断 ({type: "
            "price, candidates, max_change}) だけです"
        )
    decision = body.decision
    found = bundle.info.get("decision") if decision is None else decision
    candidates = (
        (found or {}).get("candidates") if isinstance(found, dict) else None
    )
    if isinstance(candidates, list) and len(candidates) > MAX_CANDIDATES:
        raise HTTPException(
            400, f"値段の候補は {MAX_CANDIDATES} 個までにしてください (今は "
            f"{len(candidates):,} 個)"
        )

    return decision


def price_meta(
    bundle: ml_inference.Bundle, frame: pd.DataFrame
) -> Optional[dict[str, Any]]:
    """
    値段の結果に残す, 予測した行の弾力性の分布と弾力性の上げ幅を返す関数

    Parameters
    ----------
    bundle: ml_inference.Bundle
        読んだ一式
    frame: pd.DataFrame
        推論の直後の予測の表

    Returns
    ----------
    price: Optional[dict[str, Any]]
        step (弾力性を数える値段の上げ幅), elasticity (予測した行の弾力性の
        counted, mean, min, p10, p50, p90, max, negative. 数えた行が無ければ
        None). 値段の一式でなければ None
    """
    # 弾力性の分布 (ML-Inference の結論は平均だけなので予測の表から数える)
    if not bundle.from_price:
        return None
    values = pd.to_numeric(
        frame.loc[frame["status"] == ml_inference.OK, "elasticity"],
        errors="coerce"
    ).to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    elasticity = None
    if len(values):
        elasticity = {
            "counted": int(len(values)),
            "mean": float(values.mean()),
            "min": float(values.min()),
            "p10": float(np.percentile(values, 10)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "max": float(values.max()),
            "negative": float((values < 0).mean()),
        }
    price = {
        "step": float(bundle.info["price"]["elasticity_step"]),
        "elasticity": elasticity,
    }

    return price


def search_decision(
    bundle: ml_inference.Bundle, body: PredictRequest
) -> dict[str, Any]:
    """
    検索の一式に渡す検索の判断 (返す件数) を確かめる関数

    Parameters
    ----------
    bundle: ml_inference.Bundle
        検索の一式
    body: PredictRequest
        推論の要求 (問いの写真のファイルかディレクトリと, 件数)

    Returns
    ----------
    decision: dict[str, Any]
        {type: search, k} (件数を省くと一式の search.k)

    Raises
    ----------
    HTTPException
        写真の他の入力, 結論の決め方, 意思決定, 何段も先の予測を渡した
        場合と, 件数が範囲の外の場合, 写真か表が多すぎる場合は 400
    """
    # 問いの写真だけ (文, 件数と種, 読み込みの設定と決め方は使えない)
    given = any(v is not None for v in (body.rules or {}).values())
    if (body.texts is not None or body.count is not None
            or body.seed is not None or body.options or given
            or body.decision or body.forecast):
        raise HTTPException(
            400, "検索の一式には, 問いの写真のファイルかディレクトリと返す"
            "件数だけを渡してください (文, 件数と種, 読み込みの設定, 結論の"
            "決め方, 意思決定, 何段も先の予測は使いません)"
        )
    if not body.source:
        raise HTTPException(
            400, "問いの写真のファイルかディレクトリを指定してください"
        )

    # 件数 (省くと一式の search.k) と, 書く表の行の上限
    search = bundle.info["search"]
    k = int(search["k"]) if body.k is None else body.k
    if isinstance(k, bool) or not isinstance(k, int) or not (
        1 <= k <= MAX_RANK_K
    ):
        raise HTTPException(
            400, f"返す件数は 1 から {MAX_RANK_K} の整数にしてください"
        )
    # (ディレクトリのファイルは上限を 1 つ超えたところで数えるのをやめる)
    place = Path(body.source).expanduser()
    count = sum(1 for _ in islice(
        (item for item in place.rglob("*") if item.is_file()), MAX_TEXTS + 1
    )) if place.is_dir() else 1
    if count > MAX_TEXTS:
        raise HTTPException(
            400, f"問いの写真は 1 回に {MAX_TEXTS:,} 枚までです (今は "
            f"{count:,} 個のファイル)"
        )
    if count * min(k, int(search["items"])) > MAX_RANK_ROWS:
        raise HTTPException(
            400, f"問いの写真 {count:,} 枚に {k} 件ずつでは, 表が "
            f"{MAX_RANK_ROWS:,} 行を超えます. 写真か件数を減らしてください"
        )
    decision = {"type": "search", "k": k}

    return decision


def search_meta(
    bundle: ml_inference.Bundle, result: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """
    検索の結果に残す, 件数, 索引, 待ち時間, 評価をまとめる関数

    Parameters
    ----------
    bundle: ml_inference.Bundle
        一式
    result: dict[str, Any]
        ML-Inference の infer の結果

    Returns
    ----------
    search: Optional[dict[str, Any]]
        k, items, max_latency_ms, labeled, evaluation, latency (median と
        max), over_latency (検索の一式でなければ None)
    """
    # 一式の search の欄と, 結論の待ち時間の要約
    if not bundle.from_search:
        return None
    info = bundle.info["search"]
    summary = result.get("summary") or {}
    search = {
        "k": int(summary.get("k") or info["k"]),
        "items": int(info["items"]),
        "max_latency_ms": info["max_latency_ms"],
        "labeled": bool(info["labeled"]),
        "evaluation": copy.deepcopy(info["evaluation"]),
        "latency": summary.get("latency"),
        "over_latency": summary.get("over_latency"),
    }

    return search


def demand_decision(body: PredictRequest) -> Optional[dict[str, Any]]:
    """
    需要の一式に渡す発注の判断を確かめる関数

    Parameters
    ----------
    body: PredictRequest
        推論の要求

    Returns
    ----------
    decision: Optional[dict[str, Any]]
        渡す発注の判断 (None は一式の既定, 空の辞書は発注の数を決めない.
        型と項目, 損と上限の形は ML-Inference が確かめる)

    Raises
    ----------
    HTTPException
        結論の決め方か何段も先の予測を渡した場合は 400
    """
    # 結論の決め方と何段も先の予測は使えないもの (先の期間は一式の
    # horizon で決まる)
    given = any(v is not None for v in (body.rules or {}).values())
    if given or body.forecast:
        raise HTTPException(
            400, "需要の一式には, 結論の決め方と何段も先の予測を使えません. "
            "意思決定は水準 τ の分位点から在庫を引いて発注する判断 ({type: "
            "order, shortage_cost, holding_cost, max_order}) だけです"
        )

    return body.decision


def demand_meta(
    bundle: ml_inference.Bundle, frame: pd.DataFrame
) -> Optional[dict[str, Any]]:
    """
    需要の結果に残す, 予測した行の水準ごとの分位点の平均と, 最も小さい
    水準から最も大きい水準までの分位点の幅の分布を返す関数

    Parameters
    ----------
    bundle: ml_inference.Bundle
        読んだ一式
    frame: pd.DataFrame
        推論の直後の予測の表

    Returns
    ----------
    demand: Optional[dict[str, Any]]
        horizon (先の期間の日数), levels (学んだ水準), quantiles (水準ごと
        の予測した行の分位点の平均. 予測した行が無ければ None), low と
        high (幅を数える水準), width (幅の counted, mean, min, p10, p50,
        p90, max. 予測した行が無ければ None). 需要の一式でなければ None
    """
    # 分位点の要約 (ML-Inference の結論は 0.5 の点の平均と和だけなので,
    # 予測の表の q:<水準> の列から数える)
    if not bundle.from_demand:
        return None
    levels = [float(level) for level in bundle.info["output"]["quantiles"]]
    ok = frame[frame["status"] == ml_inference.OK]
    values = np.column_stack([
        pd.to_numeric(ok[f"q:{level:g}"], errors="coerce").to_numpy(
            dtype=float
        )
        for level in levels
    ]) if len(ok) else np.zeros((0, len(levels)))
    values = values[np.isfinite(values).all(axis=1)]
    quantiles, width = None, None
    if len(values):
        quantiles = [float(value) for value in values.mean(axis=0)]
        spread = values[:, -1] - values[:, 0]
        width = {
            "counted": int(len(spread)),
            "mean": float(spread.mean()),
            "min": float(np.min(spread)),
            "p10": float(np.percentile(spread, 10)),
            "p50": float(np.percentile(spread, 50)),
            "p90": float(np.percentile(spread, 90)),
            "max": float(np.max(spread)),
        }
    demand = {
        "horizon": int(bundle.info["demand"]["horizon"]),
        "levels": levels,
        "quantiles": quantiles,
        "low": levels[0],
        "high": levels[-1],
        "width": width,
    }

    return demand


def uplift_meta(bundle: Any) -> Optional[dict[str, Any]]:
    """
    効果の結果に残す, 実験の Qini の曲線, 結果の種類, K を返す関数

    Parameters
    ----------
    bundle: Any
        読んだ一式 (ML-Inference の Bundle)

    Returns
    ----------
    uplift: Optional[dict[str, Any]]
        curves (分割ごとの rows, fraction, qini, random の並び), outcome
        (binary か amount), k. 効果の一式でなければ None
    """
    # 一式の登録を消した後も結果の画面に曲線を出すための写し
    if not bundle.from_uplift:
        return None
    found = bundle.info["uplift"]
    uplift = {
        "curves": copy.deepcopy(found.get("curves") or {}),
        "outcome": str(found["outcome"]),
        "k": float(found["k"]),
    }

    return uplift


def grouping_meta(
    bundle: Any, frame: pd.DataFrame
) -> Optional[dict[str, Any]]:
    """
    群分けの結果に残す, 群を作る列と学習のときの群の要約を返す関数

    Parameters
    ----------
    bundle: Any
        読んだ一式 (ML-Inference の Bundle)
    frame: pd.DataFrame
        推論した予測の表

    Returns
    ----------
    grouping: Optional[dict[str, Any]]
        k, columns (群を作る列), learned (学習のときの群の要約の行. 最後は
        全体), texts (値が全て文字の元の列. 読み直すときに文字で読む) と,
        群の中心から遠い行の上限を持つ一式 (版 16 と 17) は far (上限の
        写し), 既製の埋め込みの群分けの一式 (版 10 と 17) は photos (写真
        の縮小画像を読める印. true). 群分けの一式でなければ None
    """
    # 一式を消した後も学習のときの群の大きさと比べるための写し
    if not bundle.from_group:
        return None
    group = bundle.info["group"]
    grouping = {
        "k": int(group["k"]),
        "columns": [str(name) for name in group["columns"]],
        "learned": copy.deepcopy(list(group.get("summary") or [])),
        "texts": text_columns(frame),
    }
    # 上限を持たない一式 (版 9 と 10) の結果は今までの形のまま
    if group.get("far") is not None:
        grouping["far"] = copy.deepcopy(group["far"])
    # 写真を群に分けた結果だけが, 推論した写真の縮小画像を読める
    if bundle.from_ready:
        grouping["photos"] = True

    return grouping


def text_columns(frame: pd.DataFrame) -> list[str]:
    """
    予測の表の, 値が全て文字の元の列の名前を返す関数

    Parameters
    ----------
    frame: pd.DataFrame
        推論した予測の表

    Returns
    ----------
    names: list[str]
        欠けを除いた値が全て文字の列 (予測と判断の列は除く)
    """
    # 書き出した CSV を読み直すと, 数に見える文字 ("001") が数になるため
    # (数と文字の混ざる列は, 数を文字にしないように除く)
    names = []
    for name in frame.columns:
        if name in OWN_COLUMNS or name in ("group", "distance"):
            continue
        values = frame[name].dropna()
        if len(values) and all(isinstance(v, str) for v in values):
            names.append(str(name))

    return names


def table_columns(bundle: Any) -> Optional[list[dict[str, Any]]]:
    """
    表の出力の一式の列の名前と型を返す関数

    Parameters
    ----------
    bundle: Any
        読んだ一式 (ML-Inference の Bundle)

    Returns
    ----------
    columns: Optional[list[dict[str, Any]]]
        列の符号化の順の name, type (number か category), integer (数の
        列の整数か), categories (種類の列の値の種類の数). 表の出力でない
        一式は None
    """
    # 結果に残し, 一式を消しても表の列の順と型が分かるように
    if bundle.form != "table":
        return None
    columns: list[dict[str, Any]] = []
    for entry in bundle.output_info.get("columns") or []:
        column: dict[str, Any] = {
            "name": str(entry["name"]), "type": str(entry["type"])
        }
        if column["type"] == "category":
            column["categories"] = len(entry.get("categories") or [])
        else:
            column["integer"] = bool(entry.get("integer"))
        columns.append(column)

    return columns


def conclude_into(
    place: Path, meta: dict[str, Any], rules: dict[str, Any]
) -> None:
    """
    保存した予測から結論を出し直し, 結果を書き換える関数

    Parameters
    ----------
    place: Path
        結果のディレクトリ
    meta: dict[str, Any]
        result.yaml の中身
    rules: dict[str, Any]
        結論の決め方

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        決め方に誤りがある場合
    """
    # 列の出力でない結果は件数だけ (conclude をかけず, 配列は持ち越し)
    if not is_column(meta):
        count_into(place, meta)
        return

    # 判断の列を除いた予測からの出し直し (正解の出どころの文は結論の文に
    # 混ぜず origin_note に. まとめる列ごとの文と分けるため)
    frame = read_frame(place, meta).drop(columns=["decision"], errors="ignore")
    conclusion = ml_inference.conclude(
        frame, str(meta.get("task")), list(meta.get("classes") or []), rules
    )
    meta = {
        **meta, "summary": conclusion["summary"],
        "sentences": conclusion["sentences"], "rules": conclusion["rules"],
        "frame": conclusion["frame"],
    }

    # 隣に書いてから置き換え (読んでいる途中で壊れた表を見せないように)
    temporary = place / f".writing-{uuid.uuid4().hex[:8]}"
    try:
        ml_inference.write_result(meta, temporary)
        for name in ("predictions.csv", "result.yaml"):
            os.replace(temporary / name, place / name)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def count_into(place: Path, meta: dict[str, Any]) -> None:
    """
    列の出力でない結果の件数だけの結論を, 書いた表から数え直す関数

    Parameters
    ----------
    place: Path
        結果のディレクトリ
    meta: dict[str, Any]
        result.yaml の中身 (正解の出どころ)

    Returns
    ----------
    None
    """
    # read_result の結果をそのまま write_result へ (同じ場所なので
    # artifacts/ はそのまま)
    result = ml_inference.read_result(place)
    frame = result["frame"].drop(columns=["decision"], errors="ignore")
    conclusion = ml_inference.conclude_outputs(frame)
    result.update(conclusion)
    ml_inference.write_result(result, place)


def is_column(meta: dict[str, Any]) -> bool:
    """
    結果が列の出力の一式の結果かを返す関数

    Parameters
    ----------
    meta: dict[str, Any]
        result.yaml の中身

    Returns
    ----------
    column: bool
        列の出力か (記録の無い前の結果は列の出力)
    """
    column = meta.get("column_output") is not False

    return column


def column_only(form: Any) -> str:
    """
    列の出力でない一式に, 列の出力だけの機能を断る文を返す関数

    Parameters
    ----------
    form: Any
        出力の形

    Returns
    ----------
    message: str
        断る文
    """
    message = (
        f"出力の形が {form} の一式 (列の出力でない一式) の結果には, 結論の"
        "決め方, 意思決定, 何段も先の予測, 列の効き, たどり直し, 外れ方の"
        "測り直しを使えません. 結論は件数だけです"
    )

    return message


def require_column(meta: dict[str, Any]) -> None:
    """
    列の出力の結果でなければ 400 で断る関数

    Parameters
    ----------
    meta: dict[str, Any]
        result.yaml の中身

    Returns
    ----------
    None

    Raises
    ----------
    HTTPException
        列の出力でない一式の結果の場合は 400
    """
    if not is_column(meta):
        raise HTTPException(400, column_only(meta.get("output_form")))


# 意思決定の型ごとに, 予測の表に足す列 (決め直すときに, この列だけを除く)
TYPE_COLUMNS = {
    "quantity": ["action", "level", "expected_cost", "point_expected_cost"],
    "select": ["action", "gain_if_acted", "expected_gain"],
    "triage": ["action", "expected_cost"],
}


def decision_columns(meta: dict[str, Any]) -> list[str]:
    """
    結果の意思決定が予測の表に足した列を返す関数

    Parameters
    ----------
    meta: dict[str, Any]
        result.yaml の中身

    Returns
    ----------
    columns: list[str]
        足した列 (意思決定の無い結果は空. 入力の同じ名前の列は残す)
    """
    # 前の決め方の型の列だけ
    decided = (meta.get("decision") or {}).get("decision") or {}
    columns = list(TYPE_COLUMNS.get(str(decided.get("type")), []))

    return columns


def write_meta(place: Path, meta: dict[str, Any]) -> None:
    """
    結果の result.yaml を, 隣に書いてから置き換える関数

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
    # 読んでいる途中で壊れた中身を見せないように
    temporary = place / f".result-{uuid.uuid4().hex[:8]}.yaml"
    temporary.write_text(
        yaml.safe_dump(meta, allow_unicode=True, sort_keys=False),
        encoding="utf-8"
    )
    os.replace(temporary, place / "result.yaml")


def decide_into(
    place: Path, meta: dict[str, Any], bundle: ml_inference.Bundle,
    decision: Optional[dict[str, Any]]
) -> None:
    """
    保存した予測から意思決定をし直し, 結果を書き換える関数

    Parameters
    ----------
    place: Path
        結果のディレクトリ
    meta: dict[str, Any]
        result.yaml の中身
    bundle: ml_inference.Bundle
        結果の一式 (外れ方を使う)
    decision: Optional[dict[str, Any]]
        意思決定の決め方 (None で意思決定を外す)

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        決め方に誤りがある場合
    """
    # 前の意思決定が足した列を除いた予測から決め直す (たどり直しの結果は
    # 古くなる)
    frame = read_frame(place, meta).drop(
        columns=decision_columns(meta), errors="ignore"
    )
    meta = {**meta, "backtest": None}
    if decision:
        decided = ml_inference.decide(bundle, frame, decision)
        frame = decided.pop("frame")
        meta["decision"] = decided
    else:
        meta["decision"] = None
    temporary = place / f".writing-{uuid.uuid4().hex[:8]}"
    try:
        ml_inference.write_result({**meta, "frame": frame}, temporary)
        for name in ("predictions.csv", "result.yaml"):
            os.replace(temporary / name, place / name)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


class NewerResult(ValueError):
    """
    知らない新しい版の結果を読んだ場合の誤り
    """


def read_meta(place: Path) -> dict[str, Any]:
    """
    結果の result.yaml を読む関数

    Parameters
    ----------
    place: Path
        結果のディレクトリ

    Returns
    ----------
    meta: dict[str, Any]
        result.yaml の中身
    """
    meta = yaml.safe_load((place / "result.yaml").read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"{place.name} の result.yaml が壊れています")
    # 知らない新しい版の結果は読み違えずに停止
    version = meta.get("format_version")
    version = 1 if version is None else version
    newest = ml_inference.RESULT_FORMAT_VERSION
    if not isinstance(version, int) or version > newest:
        raise NewerResult(
            f"format_version {version} の結果は, このアプリの ML-Inference "
            f"(結果の形の版 {newest} まで) では読めません"
        )

    return meta


def read_frame(place: Path, meta: dict[str, Any]) -> pd.DataFrame:
    """
    結果の予測の表を読む関数

    Parameters
    ----------
    place: Path
        結果のディレクトリ
    meta: dict[str, Any]
        result.yaml の中身 (タスク, 出力の形, 残した列の型で, 型を渡す列を
        決める)

    Returns
    ----------
    frame: pd.DataFrame
        予測と判断の表 (状態, 理由と判断の空は空の文字. 文の出力の予測
        した行の空の文と番号も空の文字)
    """
    # column_types より強く当てる型. 画面の都合: 状態, 理由, 判断は文字
    # (空を空の文字で出すため), 表の出力の種類の列は文字 (欠けた行の有無に
    # 依らず種類の値を文字で出すため), 群分けの文字の元の列は文字 (推論した
    # ときの値のまま出すため). 引く都合: 並べるの利用者と品物は文字 (番号の
    # 表と同じく文字にして引くため). 結論の都合: 分類の予測は文字 (数字の
    # クラスも, 決め方のクラスの名前と合わせるため. 文の出力は下)
    text = meta.get("output_form") == "text"
    override: dict[str, str] = {
        "status": "text", "reason": "text", "decision": "text"
    }
    if meta.get("task") == "classification" and not text:
        override["prediction"] = "text"
    if meta.get("output_form") == "ranking":
        override.update(user="text", item="text")
    columns = meta.get("table_columns") or []
    for column in columns:
        if column.get("type") == "category":
            override[str(column["name"])] = "text"
    if meta.get("output_form") == "cluster":
        for name in (meta.get("grouping") or {}).get("texts") or []:
            override[str(name)] = "text"

    # 版 1 と 2 の補い: column_types に無い列だけに, このアプリが結果に
    # 残した型を当てる (版 1 と 2 の結果は列の型を持たないため)
    fallback: dict[str, str] = {}
    for column in columns:
        if column.get("type") != "category" and column.get("integer"):
            fallback[str(column["name"])] = "integer"
    if meta.get("output_form") == "ranking":
        fallback["rank"] = "integer"
    if meta.get("output_form") == "cluster":
        fallback["group"] = "integer"

    # 版の補い: 文の出力の予測は文字 (古い文の結果は tokens の列を持たない
    # ので, 出力の形で補う. 全ての行が番号 1 つでも数にしないように)
    if text:
        override["prediction"] = "text"

    # 表は read_result で 1 回だけ読む (読み方の決まりを 2 か所に持たない
    # ため)
    frame: pd.DataFrame = ml_inference.read_result(
        place, override_types=override, fallback_types=fallback
    )["frame"]

    # 版の補い: 古い文の結果の予測した行の空の文は, 欠損と分けて空の文字に
    if (text and "tokens" not in frame.columns
            and "prediction" in frame.columns and "status" in frame.columns):
        frame["prediction"] = frame["prediction"].astype(object)
        done = frame["status"] == ml_inference.OK
        frame.loc[done & frame["prediction"].isna(), "prediction"] = ""
    for name in ("status", "reason", "decision"):
        if name in frame.columns:
            frame[name] = frame[name].astype(object).fillna("").astype(str)

    return frame


def result_view(
    result_id: str, place: Path, meta: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """
    結果のまとめを返す関数

    Parameters
    ----------
    result_id: str
        結果の id
    place: Path
        結果のディレクトリ
    meta: Optional[dict[str, Any]] = None
        読んだ result.yaml の中身 (省くと読む)

    Returns
    ----------
    view: dict[str, Any]
        id, 一式, 入力, 時刻, タスク, クラス, 決め方, まとめ, 文, 列
    """
    # 表の列は先頭の行だけ読んで決める
    if meta is None:
        meta = read_meta(place)
    columns = list(
        pd.read_csv(place / "predictions.csv", encoding="utf-8-sig", nrows=0)
    )
    view = {
        "id": result_id,
        "bundle": meta.get("bundle"),
        "bundle_id": meta.get("bundle_id"),
        "source": meta.get("source"),
        "split": meta.get("split"),
        "created_at": meta.get("created_at"),
        "task": meta.get("task"),
        "classes": meta.get("classes") or [],
        "rules": meta.get("rules") or {},
        "summary": meta.get("summary"),
        "sentences": meta.get("sentences") or [],
        "overall_count": overall_count(meta),
        "origin_note": (
            meta.get("origin_note")
            or ml_inference.origin_sentence(meta.get("target_info"))
        ),
        "columns": columns,
        "kind": meta.get("kind"),
        "forecast": bool(meta.get("forecast")),
        "decision": meta.get("decision"),
        "backtest": meta.get("backtest"),
        "has_target": bool(meta.get("has_target")),
        "explanation": meta.get("explanation"),
        "format_version": meta.get("format_version", 1),
        "role": meta.get("role", "predict"),
        "output_form": meta.get("output_form"),
        "column_output": is_column(meta),
        "target_info": meta.get("target_info"),
        "pretrained_warnings": meta.get("pretrained_warnings") or [],
        "artifacts": meta.get("artifacts"),
        "table_columns": meta.get("table_columns"),
        "ranking": meta.get("ranking"),
        "grouping": meta.get("grouping"),
        "uplift": meta.get("uplift"),
        "price": meta.get("price"),
        "demand": meta.get("demand"),
        "search": meta.get("search"),
    }

    return view


def overall_count(meta: dict[str, Any]) -> int:
    """
    結論の文のうち, 全体の文 (まとめる列ごとの文の前まで) の数を返す関数

    Parameters
    ----------
    meta: dict[str, Any]
        result.yaml の中身

    Returns
    ----------
    count: int
        全体の文の数 (まとめる列の無い結果は全ての文)
    """
    # まとめる列を除いたまとめから作り直した文の数 (文の数での推し量りはしないこと)
    sentences = list(meta.get("sentences") or [])
    summary = meta.get("summary") or {}
    count = len(sentences)
    if not summary.get("groups"):
        return count
    try:
        head = ml_inference.summary_sentences(
            {**summary, "groups": []}, str(meta.get("task"))
        )
    except (KeyError, TypeError, ValueError):
        return count
    if sentences[:len(head)] == head:
        count = len(head)

    return count


def find_result(request: Request, result_id: str) -> Path:
    """
    結果のディレクトリを探す関数

    Parameters
    ----------
    request: Request
        リクエスト
    result_id: str
        結果の id

    Returns
    ----------
    place: Path
        結果のディレクトリ

    Raises
    ----------
    HTTPException
        無い場合は 404
    """
    place, _ = find_result_meta(request, result_id)

    return place


def find_result_meta(
    request: Request, result_id: str
) -> tuple[Path, dict[str, Any]]:
    """
    結果のディレクトリを探し, result.yaml の中身と一緒に返す関数

    Parameters
    ----------
    request: Request
        リクエスト
    result_id: str
        結果の id

    Returns
    ----------
    place: Path
        結果のディレクトリ
    meta: dict[str, Any]
        result.yaml の中身

    Raises
    ----------
    HTTPException
        無い場合は 404, 新しい版の結果は 400, 読めない場合は 500
    """
    place = get_store(request).result_dir(result_id)
    if place is None:
        raise HTTPException(404, "結果が見つかりません")
    try:
        meta = read_meta(place)
    except NewerResult as error:
        raise HTTPException(400, str(error)) from error
    except (ValueError, yaml.YAMLError) as error:
        raise HTTPException(500, str(error)) from error

    return place, meta


@router.get("/results")
def list_results(
    request: Request, bundle: Optional[str] = None
) -> list[dict[str, Any]]:
    """
    結果の一覧を新しい順に返す関数

    Parameters
    ----------
    request: Request
        リクエスト
    bundle: Optional[str] = None
        一式の id で絞る

    Returns
    ----------
    results: list[dict[str, Any]]
        結果のまとめ (表の列を除く)
    """
    # 読めない結果は飛ばす
    store = get_store(request)
    results = []
    for result_id in store.result_ids():
        place = store.results_dir / result_id
        try:
            meta = read_meta(place)
        except (OSError, ValueError, yaml.YAMLError):
            continue
        if bundle is not None and meta.get("bundle_id") != bundle:
            continue
        results.append({
            "id": result_id,
            "bundle": meta.get("bundle"),
            "bundle_id": meta.get("bundle_id"),
            "source": meta.get("source"),
            "created_at": meta.get("created_at"),
            "task": meta.get("task"),
            "summary": meta.get("summary"),
            "sentences": meta.get("sentences") or [],
        })

    return results


def same_value(values: list[Any]) -> bool:
    """
    全ての値が同じで, 空でないかを返す関数

    Parameters
    ----------
    values: list[Any]
        値の並び

    Returns
    ----------
    same: bool
        全てが同じ空でない値か
    """
    same = values[0] is not None and all(v == values[0] for v in values)

    return same


def is_cluster(meta: dict[str, Any]) -> bool:
    """
    結果が群分けの一式の結果かを返す関数

    Parameters
    ----------
    meta: dict[str, Any]
        result.yaml の中身

    Returns
    ----------
    cluster: bool
        出力の形が群 (cluster) か
    """
    cluster = meta.get("output_form") == "cluster"

    return cluster


def has_grouping(meta: dict[str, Any]) -> bool:
    """
    群分けの結果が群の情報 (群の数) を残しているかを返す関数

    Parameters
    ----------
    meta: dict[str, Any]
        result.yaml の中身

    Returns
    ----------
    kept: bool
        grouping に群の数があるか
    """
    kept = (meta.get("grouping") or {}).get("k") is not None

    return kept


def has_far(meta: dict[str, Any]) -> bool:
    """
    結果が群の中心から遠い行の印を持つかを返す関数

    Parameters
    ----------
    meta: dict[str, Any]
        result.yaml の中身

    Returns
    ----------
    found: bool
        群分けの結果で, 結論に遠い行の件数 (summary.far) があるか (版 16
        と 17 の一式の結果)
    """
    # 予測の表の far の列は, 上限を持つ一式の結果だけが印の列
    found = is_cluster(meta) and (
        (meta.get("summary") or {}).get("far") is not None
    )

    return found


def same_grouping(metas: list[dict[str, Any]]) -> bool:
    """
    群分けの結果の群の数, 群を作る列, 学習のときの群の要約が全て同じかを
    返す関数

    Parameters
    ----------
    metas: list[dict[str, Any]]
        result.yaml の中身の並び

    Returns
    ----------
    same: bool
        全ての結果の grouping (k, columns, learned) が同じか (群の情報を
        持たない結果は has_grouping で先に除く)
    """
    # 推論した表で文字だった列 (texts) は入力で変わるので見ない
    keys = [
        (
            (m.get("grouping") or {}).get("k"),
            (m.get("grouping") or {}).get("columns"),
            (m.get("grouping") or {}).get("learned"),
        )
        for m in metas
    ]
    same = all(key == keys[0] for key in keys)

    return same


def comparison(metas: list[dict[str, Any]]) -> dict[str, Any]:
    """
    結果どうしを比べられるかと, 何が同じかを返す関数

    Parameters
    ----------
    metas: list[dict[str, Any]]
        result.yaml の中身の並び

    Returns
    ----------
    found: dict[str, Any]
        同じもの (`basis`. `bundle` は一式, `source` は入力のファイル),
        比べられない理由 (`reasons`), 群分けの結果どうしか (`grouping`)
    """
    # 一式は id (一式の id が無い古い結果は一式の場所), 入力は読んだ
    # ファイルの場所 (書いた文は同じ件数でも中身が違いうるので同じ入力と
    # しない. ファイルの中身が後から変わっても同じ場所なら同じ入力)
    bundles = [
        m.get("bundle_id") or (m.get("bundle") or {}).get("path")
        for m in metas
    ]
    sources = [m.get("source_path") or split_key(m) for m in metas]
    basis = [
        name for name, values in (("bundle", bundles), ("source", sources))
        if same_value(values)
    ]
    clusters = [is_cluster(m) for m in metas]
    grouping = all(clusters)
    reasons = []
    if any(clusters) and not grouping:
        reasons.append(
            "群分けの結果と他の結果は比べられません (群の要約と判断の内訳で"
            "結論の形が違うため)"
        )
    if not all(is_column(m) or is_cluster(m) for m in metas):
        reasons.append(
            "列の出力でない一式の結果は比べられません (判断の内訳と意思決定を"
            "持たないため)"
        )
    if len({str(m.get("task")) for m in metas}) > 1:
        reasons.append(
            "タスクの違う結果は比べられません (判断の内訳と得の意味が違うため)"
        )
    if grouping:
        # 群の情報 (grouping) を残す前の結果は, 群の数, 群を作る列,
        # 学習のときの群の要約を持たないので比べない. 群の番号の意味は
        # 一式ごとに違うので, 同じ入力でも違う一式は比べない (同じ場所に
        # 登録し直した一式は残した群で見分ける)
        if not all(has_grouping(m) for m in metas):
            reasons.append(
                "群の情報を持たない古い群分けの結果は比べられません "
                "(推論し直すと比べられます)"
            )
        elif "bundle" not in basis or not same_grouping(metas):
            reasons.append(
                "違う群分けの一式の結果は比べられません (群の番号と群を作る"
                "列の意味が一式ごとに違うため). 同じ群分けの一式の結果を"
                "選んでください"
            )
    elif not basis:
        reasons.append(
            "一式も入力も違う結果は比べられません (違いが一式によるものか"
            "入力によるものかを分けられないため). 同じ一式か同じ入力の"
            "ファイルの結果を選んでください"
        )
    found = {"basis": basis, "reasons": reasons, "grouping": grouping}

    return found


@router.get("/results/compare")
def compare_results(
    request: Request, ids: str
) -> dict[str, Any]:
    """
    選んだ結果のまとめを並べ, 比べられるかを返す関数

    Parameters
    ----------
    request: Request
        リクエスト
    ids: str
        カンマで区切った結果の id

    Returns
    ----------
    view: dict[str, Any]
        結果のまとめの並び (`results`), 比べられるか (`comparable`),
        同じもの (`basis`), 比べられない理由 (`reasons`), 群分けの
        結果どうしか (`grouping`)

    Raises
    ----------
    HTTPException
        ids が長すぎるか, 選んだ数が範囲の外か, 同じ結果を 2 度選んだ
        場合は 400, 無い結果は 404
    """
    # 結果の読み込みは 1 つの結果の画面と同じ (表は読まない. result.yaml
    # は 1 つにつき 1 回)
    if len(ids) > MAX_COMPARE_IDS:
        raise HTTPException(400, "選んだ結果の id が長すぎます")
    chosen = [part.strip() for part in ids.split(",") if part.strip()]
    if len(chosen) < 2:
        raise HTTPException(400, "比べる結果を 2 つ以上選んでください")
    if len(chosen) > MAX_COMPARE:
        raise HTTPException(
            400, f"一度に比べられる結果は {MAX_COMPARE} つまでです"
        )
    if len(set(chosen)) < len(chosen):
        raise HTTPException(400, "同じ結果を 2 度選んでいます")
    found_metas = [
        find_result_meta(request, result_id) for result_id in chosen
    ]
    found = comparison([meta for _, meta in found_metas])
    view = {
        "results": [
            result_view(result_id, place, meta)
            for result_id, (place, meta) in zip(
                chosen, found_metas, strict=True
            )
        ],
        "comparable": not found["reasons"],
        **found,
    }

    return view


@router.get("/results/{result_id}")
def get_result(request: Request, result_id: str) -> dict[str, Any]:
    """
    結果のまとめを返す関数

    Parameters
    ----------
    request: Request
        リクエスト
    result_id: str
        結果の id

    Returns
    ----------
    view: dict[str, Any]
        結果のまとめ
    """
    view = result_view(result_id, find_result(request, result_id))

    return view


@router.get("/results/{result_id}/rows")
def get_rows(
    request: Request, result_id: str,
    offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=MAX_ROWS),
    status: Optional[str] = None, decision: Optional[str] = None,
    group: Optional[int] = Query(None, ge=0),
    far: Optional[bool] = None,
    row: Optional[int] = Query(None, ge=0)
) -> dict[str, Any]:
    """
    予測の表の一部を返す関数

    Parameters
    ----------
    request: Request
        リクエスト
    result_id: str
        結果の id
    offset: int = Query(0, ge=0)
        先頭から飛ばす行の数
    limit: int = Query(50, ge=1, le=MAX_ROWS)
        返す行の数
    status: Optional[str] = None
        ok か skipped で絞る
    decision: Optional[str] = None
        判断で絞る
    group: Optional[int] = Query(None, ge=0)
        群の番号で絞る (群分けの結果だけ)
    far: Optional[bool] = None
        群の中心から遠い行 (True) か遠くない行 (False) に絞る (群の中心
        から遠い行の上限を持つ群分けの結果だけ)
    row: Optional[int] = Query(None, ge=0)
        行の位置 (index) で 1 行だけに絞る

    Returns
    ----------
    rows: dict[str, Any]
        total (絞った後の行の数), offset, columns, rows (行の位置 index
        付き)

    Raises
    ----------
    HTTPException
        群分けでない結果に群の番号を渡した場合と, 遠い行の上限を持たない
        結果に遠さを渡した場合は 400
    """
    # 絞ってから切り出す
    # 行の位置の列 (同じ名前の元の列は名前を変えて残す)
    place = find_result(request, result_id)
    meta = read_meta(place)
    if group is not None and meta.get("output_form") != "cluster":
        raise HTTPException(400, "群で絞れるのは群分けの結果だけです")
    if far is not None and not has_far(meta):
        raise HTTPException(
            400, "遠さで絞れるのは, 群の中心から遠い行の上限を持つ群分けの"
            "結果だけです"
        )
    frame = read_frame(place, meta)
    if "index" in frame.columns:
        frame = frame.rename(columns={"index": "index (入力)"})
    # (列の多い表 (種類ごとの確率) は, 読んだ表が断片なのでまとめてから)
    frame = frame.copy()
    frame.insert(0, "index", range(len(frame)))
    if status:
        frame = frame[frame["status"] == status]
    if decision is not None and "decision" in frame.columns:
        frame = frame[frame["decision"] == decision]
    if group is not None:
        frame = frame[frame["group"] == group]
    if far is not None:
        frame = frame[frame["far"].fillna(False).astype(bool) == far]
    if row is not None:
        frame = frame[frame["index"] == row]
    part = frame.iloc[offset:offset + limit]
    records = json.loads(part.to_json(orient="records", force_ascii=False))
    rows = {
        "total": int(len(frame)),
        "offset": offset,
        "columns": list(frame.columns),
        "rows": records,
    }

    return rows


@router.get("/results/{result_id}/representatives")
def get_representatives(
    request: Request, result_id: str,
    count: int = Query(ge=1, le=MAX_REPRESENTATIVES)
) -> dict[str, Any]:
    """
    群分けの結果の, 群ごとの代表の行を件数を変えて求める関数 (結果には
    残さない)

    Parameters
    ----------
    request: Request
        リクエスト
    result_id: str
        結果の id
    count: int = Query(ge=1, le=MAX_REPRESENTATIVES)
        群ごとの代表の行の件数の上限

    Returns
    ----------
    representatives: dict[str, Any]
        count と groups (ML-Inference の group_representatives の並び)

    Raises
    ----------
    HTTPException
        群分けでない結果の場合と, 一式が登録されていないか読めない場合は
        400
    """
    # 群を作る列は一式から読むので, 登録した一式で求める
    place = find_result(request, result_id)
    meta = read_meta(place)
    if meta.get("output_form") != "cluster":
        raise HTTPException(400, "代表の行は群分けの結果だけで求めます")
    entry = get_store(request).find(str(meta.get("bundle_id")))
    if entry is None:
        raise HTTPException(
            400, "一式が登録されていないので, 代表の行の件数を変えられません"
        )
    try:
        bundle = open_bundle(str(entry["path"]))
    except (OSError, *USER_ERRORS) as error:
        raise bundle_error(str(entry["path"]), error) from error
    frame = read_frame(place, meta)
    try:
        groups = ml_inference.group_representatives(bundle, frame, count)
    except USER_ERRORS as error:
        raise user_error(error) from error
    representatives = {"count": count, "groups": groups}

    return representatives


# 配列の見本を同時に作る数 (大きな配列でメモリを使い切らないための上限)
PREVIEW_SLOTS = threading.Semaphore(2)


def artifact_file(place: Path, row: int) -> Path:
    """
    結果の行の位置の配列のファイルを返す関数

    Parameters
    ----------
    place: Path
        結果のディレクトリ
    row: int
        行の位置

    Returns
    ----------
    path: Path
        配列のファイル (結果のディレクトリの中)

    Raises
    ----------
    HTTPException
        配列の出力の結果でない場合, 行が無い場合, 推論しなかった行の
        場合は 404
    """
    # 配列の記録の無い結果 (版 1) は読まずに断り, 他は read_result で
    # 版を確かめ, artifact_path で結果の中に解決
    if not (read_meta(place).get("artifacts") or {}).get("form"):
        raise HTTPException(404, "この結果は配列の出力を持ちません")
    try:
        result = ml_inference.read_result(place)
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise HTTPException(400, str(error)) from error
    frame = result["frame"]
    if "artifact" not in frame.columns:
        raise HTTPException(404, "この結果は配列の出力を持ちません")
    if row >= len(frame):
        raise HTTPException(404, "その行はありません")
    relative = str(frame["artifact"].iloc[row] or "")
    if not relative:
        raise HTTPException(404, "この行は推論していないので配列がありません")
    try:
        path: Path = ml_inference.artifact_path(place, relative)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    if not path.is_file():
        raise HTTPException(404, f"{relative} がありません")

    return path


@router.get("/results/{result_id}/artifacts/{row}")
def get_artifact(
    request: Request, result_id: str, row: int = FastPath(ge=0)
) -> dict[str, Any]:
    """
    配列の出力の結果の, 行の位置の配列の見本を返す関数

    Parameters
    ----------
    request: Request
        リクエスト
    result_id: str
        結果の id
    row: int = FastPath(ge=0)
        行の位置

    Returns
    ----------
    preview: dict[str, Any]
        row, file (配列のファイルの名前) と配列の見本
    """
    # 配列は pickle を読まずに mmap で開き, 間引いた分だけの読み込み
    # (大きすぎる配列と数でない型は断り. ファイルは落とせるもの)
    place = find_result(request, result_id)
    path = artifact_file(place, row)
    form = str((read_meta(place).get("artifacts") or {}).get("form"))
    with PREVIEW_SLOTS:
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise HTTPException(400, f"配列を読めません: {error}") from error
        problem = dtype_problem(array)
        if problem is not None:
            raise HTTPException(400, problem)
        if array.size > PREVIEW_LIMIT:
            raise HTTPException(
                400, f"配列が大きい ({array.size:,} 個の値) ので見せません. "
                ".npy を落として見てください"
            )
        try:
            shown = array_preview(array, form)
        except (ValueError, TypeError) as error:
            raise HTTPException(
                400, f"配列を見せる形にできません: {error}"
            ) from error
    preview = {"row": row, "file": path.name, **shown}

    return preview


@router.get("/results/{result_id}/artifacts/{row}/file")
def get_artifact_file(
    request: Request, result_id: str, row: int = FastPath(ge=0)
) -> FileResponse:
    """
    配列の出力の結果の, 行の位置の配列のファイル (.npy) を返す関数

    Parameters
    ----------
    request: Request
        リクエスト
    result_id: str
        結果の id
    row: int = FastPath(ge=0)
        行の位置

    Returns
    ----------
    response: FileResponse
        .npy のファイル
    """
    place = find_result(request, result_id)
    path = artifact_file(place, row)
    response = FileResponse(
        path, media_type="application/octet-stream",
        filename=f"{result_id}-{row:06d}.npy"
    )

    return response


@router.post("/results/{result_id}/conclude")
def conclude_again(
    request: Request, result_id: str, body: ConcludeRequest
) -> dict[str, Any]:
    """
    決め方を変えて結論だけを出し直す関数 (結果を書き換える)

    Parameters
    ----------
    request: Request
        リクエスト
    result_id: str
        結果の id
    body: ConcludeRequest
        結論の決め方

    Returns
    ----------
    view: dict[str, Any]
        書き換えた結果のまとめ

    Raises
    ----------
    HTTPException
        決め方に誤りがある場合は 400
    """
    # 消した項目を除く (陽性のクラスを消したら, しきい値も消す)
    place = find_result(request, result_id)
    require_column(read_meta(place))
    rules = {k: v for k, v in body.rules.items() if v is not None}
    if "positive" not in rules:
        rules.pop("threshold", None)
    with RESULT_LOCK:
        try:
            conclude_into(place, read_meta(place), rules)
        except ValueError as error:
            raise user_error(error) from error
    view = result_view(result_id, place)

    return view


@router.post("/results/{result_id}/explain")
def explain_result(request: Request, result_id: str) -> dict[str, Any]:
    """
    結果の入力の列ごとの効きを求めて結果に残す関数

    Parameters
    ----------
    request: Request
        リクエスト
    result_id: str
        結果の id

    Returns
    ----------
    view: dict[str, Any]
        列の効き (explanation) を足した結果のまとめ

    Raises
    ----------
    HTTPException
        一式か入力のファイルが無い場合と, 求められない場合は 400
    """
    # 結果の一式と入力のファイル
    place = find_result(request, result_id)
    meta = read_meta(place)
    require_column(meta)
    if meta.get("forecast"):
        raise HTTPException(
            400, "何段も先の予測の結果は, 列の効きを求めません (並べ替えて"
            "推論し直すのは 1 つ先の予測のため)"
        )
    if meta.get("split"):
        raise HTTPException(400, SPLIT_UNEXPLAINED)
    entry = get_store(request).find(str(meta.get("bundle_id")))
    if entry is None:
        raise HTTPException(400, "一式が登録されていないので求められません")
    source = meta.get("source_path")
    if not source or not Path(source).exists():
        raise HTTPException(
            400, "入力のファイルが無いので求められません (書いた文の結果は"
            "求めない)"
        )

    # 推論を繰り返すので, 推論と同じく 1 つずつ
    with PREDICT_LOCK:
        try:
            bundle = open_bundle(str(entry["path"]))
        except (OSError, *USER_ERRORS) as error:
            raise bundle_error(str(entry["path"]), error) from error
        try:
            explanation = ml_inference.explain(bundle, Path(source))
        except (OSError, *USER_ERRORS) as error:
            raise user_error(error) from error
    with RESULT_LOCK:
        meta = read_meta(place)
        meta["explanation"] = explanation
        write_meta(place, meta)
    view = result_view(result_id, place)

    return view


@router.post("/results/{result_id}/rows/{row}/reason")
def explain_row(
    request: Request, result_id: str, row: int = FastPath(ge=0)
) -> dict[str, Any]:
    """
    結果の 1 行の, 列ごとの予測の効きと理由を求める関数 (結果には残さない)

    Parameters
    ----------
    request: Request
        リクエスト
    result_id: str
        結果の id
    row: int = FastPath(ge=0)
        行の位置 (予測の表の index)

    Returns
    ----------
    reason: dict[str, Any]
        行の理由 (reasons.row_reason の形)

    Raises
    ----------
    HTTPException
        求められない結果と行の場合は 400, 行が無い場合は 404
    """
    # 表と時系列の, ファイルを 1 つ先まで推論した結果だけ
    place = find_result(request, result_id)
    meta = read_meta(place)
    require_column(meta)
    if meta.get("kind") not in ("table", "timeseries"):
        raise HTTPException(
            400, "行ごとの理由は, 表と時系列の一式の結果だけで求めます"
        )
    if meta.get("forecast"):
        raise HTTPException(
            400, "何段も先の予測の結果は, 行ごとの理由を求めません (並べ替えて"
            "推論し直すのは 1 つ先の予測のため)"
        )
    if meta.get("split"):
        raise HTTPException(400, SPLIT_UNEXPLAINED)
    source = meta.get("source_path")
    if not source or not Path(source).exists():
        raise HTTPException(
            400, "入力のファイルが無いので求められません (書いた文の結果は"
            "求めない)"
        )
    frame = read_frame(place, meta)
    if row >= len(frame):
        raise HTTPException(404, f"{row} 行目はありません")
    if frame["status"].iloc[row] != ml_inference.OK:
        raise HTTPException(400, "予測しなかった行の理由は求めません")
    bundle = result_bundle(request, meta)

    # 推論を繰り返すので, 推論と同じく 1 つずつ
    with PREDICT_LOCK:
        try:
            given = read_input(bundle, Path(source), meta.get("options"))
            if len(given) != len(frame):
                raise ValueError(
                    "入力のファイルの行の数が推論したときと違うので, 行を"
                    "合わせられません. 推論し直してください"
                )
            reason = row_reason(bundle, given, row)
        except (OSError, *USER_ERRORS) as error:
            raise user_error(error) from error

    return reason


def result_bundle(
    request: Request, meta: dict[str, Any]
) -> ml_inference.Bundle:
    """
    結果の一式を読む関数

    Parameters
    ----------
    request: Request
        リクエスト
    meta: dict[str, Any]
        result.yaml の中身

    Returns
    ----------
    bundle: ml_inference.Bundle
        一式

    Raises
    ----------
    HTTPException
        一式が登録されていない場合と読めない場合は 400
    """
    entry = get_store(request).find(str(meta.get("bundle_id")))
    if entry is None:
        raise HTTPException(400, "一式が登録されていないので決められません")
    try:
        bundle = open_bundle(str(entry["path"]))
    except (OSError, *USER_ERRORS) as error:
        raise bundle_error(str(entry["path"]), error) from error

    return bundle


@router.post("/results/{result_id}/decide")
def decide_again(
    request: Request, result_id: str, body: DecideRequest
) -> dict[str, Any]:
    """
    決め方を変えて意思決定だけをし直す関数 (推論はし直さない)

    Parameters
    ----------
    request: Request
        リクエスト
    result_id: str
        結果の id
    body: DecideRequest
        意思決定の決め方 (None で外す)

    Returns
    ----------
    view: dict[str, Any]
        書き換えた結果のまとめ

    Raises
    ----------
    HTTPException
        決め方に誤りがある場合は 400
    """
    # 一式の外れ方を使うので一式も読む
    place = find_result(request, result_id)
    require_column(read_meta(place))
    bundle = result_bundle(request, read_meta(place))
    with RESULT_LOCK:
        try:
            decide_into(place, read_meta(place), bundle, body.decision)
        except ValueError as error:
            raise user_error(error) from error
    view = result_view(result_id, place)

    return view


def check_guard_kind(
    guard: Optional[dict[str, Any]], decision: dict[str, Any]
) -> None:
    """
    問題の設定の守る指標が数える量を, 結果の意思決定の型で数えられるかを
    確かめる関数

    結果の画面で意思決定の型を変えると, 問題の設定の型の量を数えられない

    Parameters
    ----------
    guard: Optional[dict[str, Any]]
        問題の設定の守る指標 (無ければ None)
    decision: dict[str, Any]
        結果の意思決定の決め方

    Returns
    ----------
    None

    Raises
    ----------
    HTTPException
        数える量が結果の意思決定の型の一覧に無い場合は 400
    """
    # 量の名前ではなく説明で知らせる (名前は画面に出さない)
    counted = (guard or {}).get("counted")
    counts = ml_inference.GUARD_COUNTS
    if not isinstance(counted, str) or counted in counts.get(
        str(decision.get("type")), {}
    ):
        return
    described = next(
        (items[counted]["description"] for items in counts.values()
         if counted in items), counted
    )
    raise HTTPException(
        400, f"問題の設定の守る指標「{(guard or {}).get('name')}」が数える量"
        f" ({described}) は, この結果の意思決定の型では数えられません. "
        "意思決定の型を問題の設定に合わせて決め直すか, 問題の設定の守る"
        "指標を選び直してください"
    )


@router.post("/results/{result_id}/backtest")
def backtest_result(request: Request, result_id: str) -> dict[str, Any]:
    """
    結果の入力 (目的の列を持つ) で, 意思決定の方針をたどり直す関数

    Parameters
    ----------
    request: Request
        リクエスト
    result_id: str
        結果の id

    Returns
    ----------
    view: dict[str, Any]
        たどり直し (backtest) を足した結果のまとめ

    Raises
    ----------
    HTTPException
        意思決定の無い結果, 入力のファイルが無い場合, 目的の列が無い
        場合, 守る指標を答えた問題の設定を推論に渡せない場合, 守る指標を
        数えられない場合は 400 (守る指標の有無を見られない問題の設定は,
        数えずに知らせの文を添える)
    """
    # 入力のファイルを読み直して, 方針と素朴なやり方を比べる. 一式に
    # 問題の設定をつないでいれば, その守る指標もこの方針で数える
    place = find_result(request, result_id)
    meta = read_meta(place)
    require_column(meta)
    decided = meta.get("decision") or {}
    if not decided.get("decision"):
        raise HTTPException(400, "先に意思決定の決め方を選んでください")
    try:
        source = result_source(
            meta, get_store(request).find(str(meta.get("bundle_id")))
        )
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    if source is None or meta.get("forecast"):
        raise HTTPException(
            400, "たどり直すには, 目的の列を持つ入力のファイルの結果が"
            "要ります (書いた文と何段も先の予測の結果は除く)"
        )
    bundle = result_bundle(request, meta)
    entry = get_store(request).find(str(meta.get("bundle_id"))) or {}
    guard, guard_note = linked_guard(entry)
    check_guard_kind(guard, decided["decision"])
    with PREDICT_LOCK:
        try:
            replay = ml_inference.backtest(
                bundle, source, decided["decision"],
                options=meta.get("options"), guard=guard
            )
        except (OSError, *USER_ERRORS) as error:
            raise user_error(error) from error
    replay.pop("decision", None)
    if guard_note:
        replay["sentences"] = [*replay.get("sentences", []), guard_note]
    # たどり直している間に決め直していたら, 古い方針の結果は書かない
    with RESULT_LOCK:
        meta = read_meta(place)
        now = (meta.get("decision") or {}).get("decision")
        if now != decided["decision"]:
            raise HTTPException(
                409, "たどり直している間に意思決定が変わりました. もう一度"
                "たどり直してください"
            )
        meta["backtest"] = replay
        write_meta(place, meta)
    view = result_view(result_id, place)

    return view


@router.post("/results/{result_id}/remeasure")
def remeasure_result(
    request: Request, result_id: str, body: RemeasureRequest
) -> dict[str, Any]:
    """
    結果の入力 (目的の列を持つ) で, 一式の予測の外れ方を測り直す関数

    Parameters
    ----------
    request: Request
        リクエスト
    result_id: str
        結果の id
    body: RemeasureRequest
        write (一式に書き戻すか), horizon (何段も先の外れ方も測る段の
        数), time (予測を始める時刻の列), groups (外れ方を種類ごとに
        分けて測る列. 回帰と予測の一式だけ)

    Returns
    ----------
    view: dict[str, Any]
        rows, before (今の記録の当たり具合), sentences, written

    Raises
    ----------
    HTTPException
        入力のファイルが無い場合, 目的の列が無い場合, 行が少ない場合,
        分ける列が使えない場合は 400
    """
    # 予測と判断の列と種類ごとの確率の列では分けない (画面と同じ決まり.
    # 入力の列ではないので, 種類にならない)
    if body.groups and (
        body.groups in OWN_COLUMNS or body.groups.startswith("p:")
    ):
        raise HTTPException(
            400, f"groups の列 {body.groups} は予測か判断か確率の列なので, "
            "外れ方を分けられません (入力の列を選んでください)"
        )

    # 入力のファイルと一式の場所 (覚えた一式は書き換えず, 読み直させる)
    place = find_result(request, result_id)
    meta = read_meta(place)
    require_column(meta)
    if (not meta.get("split") and not meta.get("source_path")) or meta.get(
        "forecast"
    ):
        raise HTTPException(400, REMEASURE_INPUT)
    store = get_store(request)
    with PREDICT_LOCK:
        # 登録は鍵の中で読み直し (同時の書き戻しで二重に写さないように)
        entry = store.find(str(meta.get("bundle_id")))
        if entry is None:
            raise HTTPException(400, "一式が登録されていないので測り直せません")
        # 分割は一式の実験から見つけ直し, 記録と同じときだけ読む
        try:
            source = result_source(meta, entry)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        if source is None:
            raise HTTPException(400, REMEASURE_INPUT)

        # 学習の実験の中の一式は書き換えず, 置き場に写した方への書き戻し
        # (実験は書き換えないという ML-Found の約束). 登録を移すのは
        # 書き戻せた後だけで, 失敗したら写しを消去
        copied = None
        target = Path(entry["path"])
        try:
            if body.write and in_experiment(target):
                copied = copy_into_store(store, entry)
                target = copied
            measured = ml_inference.remeasure(
                str(target), source,
                options=meta.get("options"), write=body.write,
                horizon=body.horizon, time=body.time or None,
                groups=body.groups or None
            )
            if copied is not None:
                store.relocate(entry["id"], copied)
        except (OSError, KeyError, *USER_ERRORS) as error:
            if copied is not None:
                shutil.rmtree(copied, ignore_errors=True)
            raise user_error(error) from error
    view = {key: measured[key]
            for key in ("rows", "before", "sentences", "written")}
    view["copied_to"] = str(copied) if copied else None
    if copied:
        view["sentences"] = [
            "学習の実験の中の一式は書き換えず, ML-Provide の置き場に写して"
            "から書き戻しました (この一式の登録は写した方を指します)",
            *view["sentences"],
        ]

    return view


def in_experiment(path: Path) -> bool:
    """
    一式が ML-Found の実験のディレクトリの中にあるかを返す関数

    Parameters
    ----------
    path: Path
        一式のディレクトリ

    Returns
    ----------
    inside: bool
        一式か, その上のディレクトリが run.json を持つか (学習の
        スクリプトは実験の中の inference に一式を作る)
    """
    # 一式自身, その親, そのまた親の run.json の有無
    resolved = path.resolve()
    inside = any((parent / "run.json").is_file()
                 for parent in (resolved, *resolved.parents[:2]))

    return inside


def copy_into_store(store: Store, entry: dict[str, Any]) -> Path:
    """
    登録した一式を置き場に写す関数

    Parameters
    ----------
    store: Store
        置き場
    entry: dict[str, Any]
        一式の登録

    Returns
    ----------
    place: Path
        写した一式のディレクトリ (置き場の bundles の下)
    """
    # 一式の中の重みごとの写し (外の重みは絶対パスのまま)
    _, place = store.new_bundle_dir()
    place.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(entry["path"], place)

    return place


@router.get("/results/{result_id}/download")
def download(request: Request, result_id: str) -> FileResponse:
    """
    予測と判断の CSV を返す関数

    Parameters
    ----------
    request: Request
        リクエスト
    result_id: str
        結果の id

    Returns
    ----------
    response: FileResponse
        predictions.csv (結果の id の名前で保存させる)
    """
    place = find_result(request, result_id)
    response = FileResponse(
        place / "predictions.csv", media_type="text/csv",
        filename=f"predictions-{result_id}.csv"
    )

    return response


@router.delete("/results/{result_id}", status_code=204)
def delete_result(request: Request, result_id: str) -> Response:
    """
    結果を消す関数

    Parameters
    ----------
    request: Request
        リクエスト
    result_id: str
        結果の id

    Returns
    ----------
    response: Response
        204
    """
    place = find_result(request, result_id)
    with RESULT_LOCK:
        shutil.rmtree(place)
    response = Response(status_code=204)

    return response
