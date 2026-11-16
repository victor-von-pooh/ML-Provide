import json
import os
import shutil
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, Union

import ml_inference
import pandas as pd
import yaml
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from ml_inference.conclude import RULE_KEYS
from pydantic import BaseModel, Field

from ml_provide.samples import write_samples
from ml_provide.store import Store

router = APIRouter(prefix="/api")

# 推論を 1 つずつにする鍵 (大きなモデルでメモリを使い切らないように)
PREDICT_LOCK = threading.Lock()

# 結果の書き換えを 1 つずつにする鍵
RESULT_LOCK = threading.Lock()

# 見本を作って登録する処理を 1 つずつにする鍵
SAMPLES_LOCK = threading.Lock()

# 読み込んだ一式を覚えておく数と, 場所ごとの (更新時刻, 一式)
BUNDLE_CACHE_SIZE = 64
BUNDLES: OrderedDict[str, tuple[tuple[float, ...], ml_inference.Bundle]] = (
    OrderedDict()
)
BUNDLES_LOCK = threading.Lock()

# 画面に書いた文として受け付ける上限
MAX_TEXTS = 10000

# 表として一度に返す行の上限
MAX_ROWS = 500

# 名前に使える長さ
MAX_NAME = 80

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
    dataset: str
    weights: str
    inputs: Optional[dict[str, Any]] = None
    output: Optional[str] = None
    kind: Optional[str] = None
    target_scale: Union[dict[str, float], str, None] = None
    conclusion: Optional[dict[str, Any]] = None
    copy_weights: bool = True


class PredictRequest(BaseModel):
    source: Optional[str] = None
    texts: Optional[list[str]] = None
    options: Optional[dict[str, Any]] = None
    rules: Optional[dict[str, Any]] = None
    forecast: bool = False
    decision: Optional[dict[str, Any]] = None


class ConcludeRequest(BaseModel):
    rules: dict[str, Any] = Field(default_factory=dict)


class DecideRequest(BaseModel):
    decision: Optional[dict[str, Any]] = None


class RemeasureRequest(BaseModel):
    write: bool = False
    horizon: Optional[int] = None
    time: Optional[str] = None


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
    # SpecError などの誤りの文をそのまま
    message = str(error) or type(error).__name__
    exception = HTTPException(400, message)

    return exception


def open_bundle(path: str) -> ml_inference.Bundle:
    """
    一式を読む関数 (inference.yaml が変わっていなければ覚えたものを使う)

    Parameters
    ----------
    path: str
        一式のディレクトリ

    Returns
    ----------
    bundle: ml_inference.Bundle
        一式
    """
    # inference.yaml と重みの更新時刻と大きさで, 覚えたものが古くないか
    # を確かめる (写さない重みは, 学習し直すと実験の中で変わる)
    stamp = file_stamp(Path(path) / "inference.yaml")
    with BUNDLES_LOCK:
        cached = BUNDLES.get(path)
        if cached is not None and cached[0] == stamp:
            BUNDLES.move_to_end(path)
            return cached[1]
    bundle = ml_inference.load_bundle(path)
    stamp = stamp + file_stamp(bundle.weights_path)
    with BUNDLES_LOCK:
        BUNDLES[path] = (stamp, bundle)
        BUNDLES.move_to_end(path)
        while len(BUNDLES) > BUNDLE_CACHE_SIZE:
            BUNDLES.popitem(last=False)

    return bundle


def file_stamp(path: Path) -> tuple[float, ...]:
    """
    ファイルの更新時刻と大きさを返す関数

    Parameters
    ----------
    path: Path
        ファイル

    Returns
    ----------
    stamp: tuple[float, ...]
        更新時刻と大きさ (無ければ空)
    """
    # 消えたファイルは空 (読み込みで誤りになる)
    try:
        status = path.stat()
    except OSError:
        return ()
    stamp = (status.st_mtime, float(status.st_size))

    return stamp


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
        登録, ok, error, 説明 (読めない場合は None)
    """
    # 消えた一式や壊れた一式は理由付き
    view: dict[str, Any] = {**entry, "ok": True, "error": None}
    try:
        view["description"] = ml_inference.describe_bundle(
            open_bundle(str(entry["path"]))
        )
    except Exception as error:
        view.update(ok=False, error=str(error), description=None)

    return view


@router.get("/info")
def get_info() -> dict[str, Any]:
    """
    ML-Inference の版と結論の決め方の項目を返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    info: dict[str, Any]
        version, rule_keys
    """
    info = {
        "version": ml_inference.__version__,
        "rule_keys": list(RULE_KEYS),
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
        bundle = open_bundle(str(path))
    except (OSError, *USER_ERRORS) as error:
        raise user_error(error) from error
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
        名前, 定義, データセット, 重みか実験, つなぎ方, 結論の既定値

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
    store = get_store(request)
    bundle_id, place = store.new_bundle_dir()
    try:
        ml_inference.make_bundle(
            place, Path(body.spec).expanduser(),
            Path(body.dataset).expanduser(), Path(body.weights).expanduser(),
            inputs=body.inputs, output=body.output or None,
            kind=body.kind or None, target_scale=body.target_scale,
            conclusion=body.conclusion, name=body.name,
            copy_weights=body.copy_weights
        )
    except Exception as error:
        raise user_error(error) from error
    entry = store.add(place, body.name, managed=True, bundle_id=bundle_id)
    view = bundle_view(entry)

    return view


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
            try:
                entry = store.add(
                    place, name, managed=False,
                    extra={"sample_input": str(fresh.resolve()),
                           "sample_past": str(past.resolve())}
                )
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
    # 推論と同じ鍵の中での書き戻し (測り直しと重ならないように)
    entry = find_bundle(request, bundle_id)
    with PREDICT_LOCK:
        try:
            ml_inference.restore_uncertainty(str(entry["path"]))
        except (OSError, *USER_ERRORS) as error:
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
    if not get_store(request).remove(bundle_id):
        raise HTTPException(404, "一式が見つかりません")
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
        ファイルの場所か文の並び, 読み込みの設定, 結論の決め方

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
        raise user_error(error) from error
    if body.texts is not None:
        source: Any = text_frame(bundle, body.texts)
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
            result = ml_inference.infer(
                bundle, source, rules=body.rules, options=body.options,
                forecast=body.forecast, decision=body.decision
            )
        except Exception as error:
            raise user_error(error) from error
    result.update(
        source=described, bundle_id=bundle_id, task=bundle.task,
        classes=bundle.classes, kind=bundle.kind, forecast=body.forecast,
        has_target=bool(
            str(bundle.dataset.info.get("target")) in result["frame"].columns
            and result["frame"][str(bundle.dataset.info.get("target"))]
            .notna().any()
        ),
        options=body.options,
        source_path=str(source) if isinstance(source, Path) else None
    )
    # 書いた表を読み直して結論を出し直す (出し直しと同じ型の値でまとめ,
    # 日時のような列のまとめの名前が, 後の出し直しで変わらないように)
    result_id, place = get_store(request).new_result_dir()
    ml_inference.write_result(result, place)
    meta = {k: v for k, v in result.items() if k != "frame"}
    with RESULT_LOCK:
        conclude_into(place, meta, meta.get("rules") or {})
    view = result_view(result_id, place)

    return view


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
    # 判断の列を除いた予測から出し直す
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

    return meta


def read_frame(place: Path, meta: dict[str, Any]) -> pd.DataFrame:
    """
    結果の予測の表を読む関数

    Parameters
    ----------
    place: Path
        結果のディレクトリ
    meta: dict[str, Any]
        result.yaml の中身 (タスクで予測の列の型を決める)

    Returns
    ----------
    frame: pd.DataFrame
        予測と判断の表 (理由と判断の空は空の文字)
    """
    # クラスの名前と判断は文字のまま (数字のクラスも)
    texts = {"status": str, "reason": str, "decision": str}
    if meta.get("task") == "classification":
        texts["prediction"] = str
    frame = pd.read_csv(
        place / "predictions.csv", encoding="utf-8-sig", dtype=texts
    )
    for name in ("reason", "decision"):
        if name in frame.columns:
            frame[name] = frame[name].fillna("")

    return frame


def result_view(result_id: str, place: Path) -> dict[str, Any]:
    """
    結果のまとめを返す関数

    Parameters
    ----------
    result_id: str
        結果の id
    place: Path
        結果のディレクトリ

    Returns
    ----------
    view: dict[str, Any]
        id, 一式, 入力, 時刻, タスク, クラス, 決め方, まとめ, 文, 列
    """
    # 表の列は先頭の行だけ読んで決める
    meta = read_meta(place)
    columns = list(
        pd.read_csv(place / "predictions.csv", encoding="utf-8-sig", nrows=0)
    )
    view = {
        "id": result_id,
        "bundle": meta.get("bundle"),
        "bundle_id": meta.get("bundle_id"),
        "source": meta.get("source"),
        "created_at": meta.get("created_at"),
        "task": meta.get("task"),
        "classes": meta.get("classes") or [],
        "rules": meta.get("rules") or {},
        "summary": meta.get("summary"),
        "sentences": meta.get("sentences") or [],
        "columns": columns,
        "kind": meta.get("kind"),
        "forecast": bool(meta.get("forecast")),
        "decision": meta.get("decision"),
        "backtest": meta.get("backtest"),
        "has_target": bool(meta.get("has_target")),
        "explanation": meta.get("explanation"),
    }

    return view


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
    place = get_store(request).result_dir(result_id)
    if place is None:
        raise HTTPException(404, "結果が見つかりません")
    try:
        read_meta(place)
    except (ValueError, yaml.YAMLError) as error:
        raise HTTPException(500, str(error)) from error

    return place


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
    status: Optional[str] = None, decision: Optional[str] = None
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

    Returns
    ----------
    rows: dict[str, Any]
        total (絞った後の行の数), offset, columns, rows (行の位置 index
        付き)
    """
    # 絞ってから切り出す
    # 行の位置の列 (同じ名前の元の列は名前を変えて残す)
    place = find_result(request, result_id)
    frame = read_frame(place, read_meta(place))
    if "index" in frame.columns:
        frame = frame.rename(columns={"index": "index (入力)"})
    frame.insert(0, "index", range(len(frame)))
    if status:
        frame = frame[frame["status"] == status]
    if decision is not None and "decision" in frame.columns:
        frame = frame[frame["decision"] == decision]
    part = frame.iloc[offset:offset + limit]
    records = json.loads(part.to_json(orient="records", force_ascii=False))
    rows = {
        "total": int(len(frame)),
        "offset": offset,
        "columns": list(frame.columns),
        "rows": records,
    }

    return rows


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
    if meta.get("forecast"):
        raise HTTPException(
            400, "何段も先の予測の結果は, 列の効きを求めません (並べ替えて"
            "推論し直すのは 1 つ先の予測のため)"
        )
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
            explanation = ml_inference.explain(bundle, Path(source))
        except (OSError, *USER_ERRORS) as error:
            raise user_error(error) from error
    with RESULT_LOCK:
        meta = read_meta(place)
        meta["explanation"] = explanation
        write_meta(place, meta)
    view = result_view(result_id, place)

    return view


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
        raise user_error(error) from error

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
    bundle = result_bundle(request, read_meta(place))
    with RESULT_LOCK:
        try:
            decide_into(place, read_meta(place), bundle, body.decision)
        except ValueError as error:
            raise user_error(error) from error
    view = result_view(result_id, place)

    return view


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
        場合は 400
    """
    # 入力のファイルを読み直して, 方針と素朴なやり方を比べる
    place = find_result(request, result_id)
    meta = read_meta(place)
    decided = meta.get("decision") or {}
    source = meta.get("source_path")
    if not decided.get("decision"):
        raise HTTPException(400, "先に意思決定の決め方を選んでください")
    if not source or not Path(source).exists() or meta.get("forecast"):
        raise HTTPException(
            400, "たどり直すには, 目的の列を持つ入力のファイルの結果が"
            "要ります (書いた文と何段も先の予測の結果は除く)"
        )
    bundle = result_bundle(request, meta)
    with PREDICT_LOCK:
        try:
            replay = ml_inference.backtest(
                bundle, Path(source), decided["decision"],
                options=meta.get("options")
            )
        except (OSError, *USER_ERRORS) as error:
            raise user_error(error) from error
    replay.pop("decision", None)
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
        数), time (予測を始める時刻の列)

    Returns
    ----------
    view: dict[str, Any]
        rows, before (今の記録の当たり具合), sentences, written

    Raises
    ----------
    HTTPException
        入力のファイルが無い場合, 目的の列が無い場合, 行が少ない場合は
        400
    """
    # 入力のファイルと一式の場所 (覚えた一式は書き換えず, 読み直させる)
    place = find_result(request, result_id)
    meta = read_meta(place)
    source = meta.get("source_path")
    if not source or not Path(source).exists() or meta.get("forecast"):
        raise HTTPException(
            400, "測り直すには, 目的の列を持つ入力のファイルの結果が要ります"
            " (書いた文と何段も先の予測の結果は除く)"
        )
    entry = get_store(request).find(str(meta.get("bundle_id")))
    if entry is None:
        raise HTTPException(400, "一式が登録されていないので測り直せません")
    with PREDICT_LOCK:
        try:
            measured = ml_inference.remeasure(
                str(entry["path"]), Path(source),
                options=meta.get("options"), write=body.write,
                horizon=body.horizon, time=body.time or None
            )
        except (OSError, *USER_ERRORS) as error:
            raise user_error(error) from error
    view = {key: measured[key]
            for key in ("rows", "before", "sentences", "written")}

    return view


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
