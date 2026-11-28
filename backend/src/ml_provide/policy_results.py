import csv
import io
import json
import math
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel

from ml_provide.api import MAX_ROWS, get_store, user_error
from ml_provide.policies import (
    ACT_LOCK,
    MAX_OBSERVATION,
    READ_ERRORS,
    count_items,
    find_policy,
    open_policy,
    to_json,
)
from ml_provide.store import Store

router = APIRouter()

# 方策の結果を置くディレクトリ (置き場の中) と, 中のファイル
RESULTS_DIR = "policy_results"
META_FILE = "policy_result.yaml"
ACTIONS_FILE = "actions.csv"

# 方策の結果の形の版 (このアプリの置き場の中だけで読む)
POLICY_RESULT_VERSION = 1

# 1 つのファイルで決める観測の数の上限と, 値の数の上限 (1 件ずつ方策を
# 動かすので, 長く止まらないように)
MAX_BATCH_ROWS = 10_000
MAX_BATCH_VALUES = 1_000_000

# 読む前に止めるファイルの大きさの上限 (CSV は 1 つの値に 32 バイトまで
# で約 32 MB. .npy は 1 つの値に 8 バイトまでとヘッダの分)
MAX_CSV_BYTES = MAX_BATCH_VALUES * 32
MAX_NPY_BYTES = MAX_BATCH_VALUES * 8 + 65_536

# .npy で読む値の型 (小数, 整数, 真偽値. 文字や複素数や構造は観測にしない)
NPY_KINDS = {"f": "小数", "i": "整数", "u": "整数", "b": "真偽値"}

# CSV の持ち出しで, 表計算のソフトが式として読む欄の頭
FORMULA_HEADS = ("=", "+", "-", "@")

# 守りの結果の並び (件数を出す順. ML-Inference の GuardedPolicy の status)
STATUSES = ("ok", "clipped", "fallback", "stop", "handover")

# 表の列 (行の位置, 観測と行動は JSON の文字, 理由は「 / 」で並べる)
ACTION_COLUMNS = ["index", "observation", "action", "status", "reasons"]


class BatchRequest(BaseModel):
    source: str


def results_root(store: Store) -> Path:
    """
    方策の結果を置くディレクトリを返す関数

    Parameters
    ----------
    store: Store
        置き場

    Returns
    ----------
    path: Path
        置き場の policy_results
    """
    path = store.home / RESULTS_DIR

    return path


def result_place(store: Store, result_id: str) -> Optional[Path]:
    """
    方策の結果の id からディレクトリを返す関数

    Parameters
    ----------
    store: Store
        置き場
    result_id: str
        方策の結果の id

    Returns
    ----------
    place: Optional[Path]
        結果のディレクトリ (無い, id の形が違う場合は None)
    """
    # id は英数字とハイフンだけ (置き場の外を指せないように)
    if not result_id or not all(c.isalnum() or c == "-" for c in result_id):
        return None
    place = results_root(store) / result_id
    if not (place / META_FILE).is_file():
        return None

    return place


def read_meta(place: Path) -> dict[str, Any]:
    """
    方策の結果の policy_result.yaml を読む関数

    Parameters
    ----------
    place: Path
        結果のディレクトリ

    Returns
    ----------
    meta: dict[str, Any]
        policy_result.yaml の中身

    Raises
    ----------
    ValueError
        壊れている場合と, 知らない新しい版の場合
    """
    meta = yaml.safe_load((place / META_FILE).read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"{place.name} の {META_FILE} が壊れています")
    version = meta.get("format_version")
    if not isinstance(version, int) or version > POLICY_RESULT_VERSION:
        raise ValueError(
            "この方策の結果は新しい版のため, 推論の段階では読めません"
        )

    return meta


def cell_value(text: str) -> Any:
    """
    CSV の 1 つの欄を, 整数, 小数, 文字のどれかにする関数

    Parameters
    ----------
    text: str
        欄の文字

    Returns
    ----------
    value: Any
        整数か小数 (inf と nan も) にできれば数, できなければ文字のまま
        (方策の守りが外れたときの動きにするため)
    """
    # 整数, 小数の順
    stripped = text.strip()
    try:
        value: Any = int(stripped)
        return value
    except ValueError:
        pass
    try:
        value = float(stripped)
    except ValueError:
        value = text

    return value


def csv_rows(text: str) -> tuple[list[list[Any]], Optional[list[str]]]:
    """
    CSV の文を, 1 行に 1 つの観測の並びにする関数

    Parameters
    ----------
    text: str
        CSV の中身

    Returns
    ----------
    rows: tuple[list[list[Any]], Optional[list[str]]]
        行ごとの欄の値と, 列の名前 (1 行目に空でない数にできない欄があれ
        ば列の名前として読む. 無ければ None)
    """
    # 空の行は飛ばす
    lines = [row for row in csv.reader(io.StringIO(text)) if row]
    header: Optional[list[str]] = None
    # (空の欄は列の名前にならないので見ない. 欠けた値のある観測の行を
    # 列の名前と読まないため)
    if lines and any(
        cell.strip() and isinstance(cell_value(cell), str)
        for cell in lines[0]
    ):
        header = [cell.strip() for cell in lines[0]]
        lines = lines[1:]
    rows = [[cell_value(cell) for cell in row] for row in lines]
    found = (rows, header)

    return found


def shaped(row: list[Any], space: dict[str, Any]) -> Any:
    """
    CSV の 1 行を, 方策の観測の空間の形にする関数

    Parameters
    ----------
    row: list[Any]
        欄の値
    space: dict[str, Any]
        policy.json の observation_space

    Returns
    ----------
    observation: Any
        discrete は 1 つの値, box は形に並べ直した入れ子の並び (数でない
        欄があるか数が合わなければ, 並びのまま渡して守りに任せる)
    """
    # 1 つの値の観測は値そのまま
    if space.get("type") == "discrete" and len(row) == 1:
        return row[0]
    shape = space.get("shape")
    numbers = all(isinstance(v, (int, float)) for v in row)
    if (
        space.get("type") == "box" and isinstance(shape, list)
        and len(shape) > 1 and numbers
        and math.prod(int(n) for n in shape) == len(row)
    ):
        observation = np.asarray(row, dtype=float).reshape(shape).tolist()
        return observation

    return row


def npy_header(path: Path) -> tuple[tuple[int, ...], np.dtype]:
    """
    .npy のヘッダだけを読み, 配列の形と型を返す関数 (中身は読まない)

    Parameters
    ----------
    path: Path
        .npy のファイル

    Returns
    ----------
    header: tuple[tuple[int, ...], np.dtype]
        形と型

    Raises
    ----------
    ValueError
        .npy の形式でない場合と, 読めない版の場合
    """
    # 版 1 と 2 のヘッダ (版 3 は文字の名前の構造の型にだけ使う)
    readers = {(1, 0): np.lib.format.read_array_header_1_0,
               (2, 0): np.lib.format.read_array_header_2_0}
    try:
        with open(path, "rb") as handle:
            version = np.lib.format.read_magic(handle)
            reader = readers.get(version)
            if reader is not None:
                shape, _, dtype = reader(handle)
    except Exception as error:
        raise ValueError(f"{path.name} は .npy の形式ではありません") from error
    if reader is None:
        raise ValueError(
            f"{path.name} は読めない版 ({version[0]}.{version[1]}) の .npy "
            "です"
        )
    header = (tuple(int(n) for n in shape), np.dtype(dtype))

    return header


def read_npy(path: Path) -> list[Any]:
    """
    .npy の並びを, 読む前に大きさと型を確かめてから 1 件ずつの観測にする
    関数

    Parameters
    ----------
    path: Path
        .npy のファイル

    Returns
    ----------
    observations: list[Any]
        先頭の次元の並び

    Raises
    ----------
    ValueError
        大きすぎる場合, 型が数でない場合, 1 つの値だけの配列の場合
    """
    # ファイルの大きさ, ヘッダの形の値の数, 型の順に, 中身を読む前に止める
    if path.stat().st_size > MAX_NPY_BYTES:
        raise ValueError(
            f"{path.name} が大きすぎます ({MAX_NPY_BYTES:,} バイトまで)"
        )
    shape, dtype = npy_header(path)
    if not shape:
        raise ValueError(
            ".npy は先頭の次元を観測の並びにします. 1 つの値だけの配列は"
            "読めません"
        )
    if math.prod(shape) > MAX_BATCH_VALUES:
        raise ValueError(
            f"観測の値は全てで {MAX_BATCH_VALUES:,} 個までにしてください "
            f"(形 {shape})"
        )
    if dtype.kind not in NPY_KINDS or dtype.fields is not None:
        raise ValueError(
            f"{path.name} の値の型 {dtype} は観測にできません (小数, 整数, "
            "真偽値の配列にしてください)"
        )
    try:
        array = np.load(path, allow_pickle=False)
    except MemoryError as error:
        raise ValueError(f"{path.name} を読む記憶が足りません") from error
    observations: list[Any] = [item.tolist() for item in array]

    return observations


def read_observations(
    path: Path, space: dict[str, Any]
) -> list[Any]:
    """
    観測のファイル (CSV の行か .npy の並び) を, 1 件ずつの観測にする関数

    Parameters
    ----------
    path: Path
        観測のファイル
    space: dict[str, Any]
        policy.json の observation_space (CSV の行を形に並べ直すため)

    Returns
    ----------
    observations: list[Any]
        観測の並び

    Raises
    ----------
    ValueError
        読めない場合, 形式が違う場合, 観測が無いか多すぎる場合
    """
    # .npy は先頭の次元を件の並びに (pickle は読まない)
    suffix = path.suffix.lower()
    observations: list[Any]
    if suffix == ".npy":
        observations = read_npy(path)
    elif suffix == ".csv":
        if path.stat().st_size > MAX_CSV_BYTES:
            raise ValueError(
                f"{path.name} が大きすぎます ({MAX_CSV_BYTES:,} バイトまで)"
            )
        rows, _ = csv_rows(path.read_text(encoding="utf-8-sig"))
        if sum(len(row) for row in rows) > MAX_BATCH_VALUES:
            raise ValueError(
                f"観測の値は全てで {MAX_BATCH_VALUES:,} 個までにしてください"
            )
        observations = [shaped(row, space) for row in rows]
    else:
        raise ValueError(
            f"{path.name} は観測のファイル (.csv か .npy) ではありません"
        )
    if not observations:
        raise ValueError(f"{path.name} に観測がありません")
    if len(observations) > MAX_BATCH_ROWS:
        raise ValueError(
            f"観測は 1 つのファイルで {MAX_BATCH_ROWS:,} 件までにしてください"
        )

    return observations


def json_text(value: Any) -> str:
    """
    観測と行動を, 表に入れる JSON の文字にする関数

    Parameters
    ----------
    value: Any
        観測か行動

    Returns
    ----------
    text: str
        JSON の文字 (有限でない値は NaN, Infinity のまま)
    """
    text = json.dumps(to_json(value), ensure_ascii=False)

    return text


def result_view(result_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    """
    方策の結果のまとめを返す関数

    Parameters
    ----------
    result_id: str
        方策の結果の id
    meta: dict[str, Any]
        policy_result.yaml の中身

    Returns
    ----------
    view: dict[str, Any]
        id, policy_id, policy (名前と場所), source, created_at, rows,
        counts (守りの結果ごとの件数. 決まった順)
    """
    counts = meta.get("counts") or {}
    view = {
        "id": result_id,
        "policy_id": meta.get("policy_id"),
        "policy": meta.get("policy"),
        "source": meta.get("source"),
        "created_at": meta.get("created_at"),
        "rows": meta.get("rows", 0),
        "counts": {name: int(counts.get(name, 0)) for name in STATUSES},
    }

    return view


def find_place(request: Request, result_id: str) -> Path:
    """
    方策の結果のディレクトリを探す関数

    Parameters
    ----------
    request: Request
        リクエスト
    result_id: str
        方策の結果の id

    Returns
    ----------
    place: Path
        結果のディレクトリ

    Raises
    ----------
    HTTPException
        無い場合は 404
    """
    place = result_place(get_store(request), result_id)
    if place is None:
        raise HTTPException(404, "方策の結果が見つかりません")

    return place


@router.post("/policies/{policy_id}/batch", status_code=201)
def act_batch(
    request: Request, policy_id: str, body: BatchRequest
) -> dict[str, Any]:
    """
    観測のファイルの 1 件ずつに守りを当てた行動を決め, 結果を置き場に残す
    関数 (環境や機械には触れない)

    Parameters
    ----------
    request: Request
        リクエスト
    policy_id: str
        方策の一式の id
    body: BatchRequest
        観測のファイル (.csv の行か .npy の並び)

    Returns
    ----------
    view: dict[str, Any]
        残した結果のまとめ

    Raises
    ----------
    HTTPException
        無い一式は 404, 読めない一式とファイルは 400
    """
    # 一式と観測のファイル
    entry = find_policy(request, policy_id)
    try:
        policy = open_policy(str(entry.get("path")))
    except READ_ERRORS as error:
        raise user_error(error) from error
    source = Path(body.source).expanduser().resolve()
    if not source.is_file():
        raise HTTPException(400, f"{source} がありません")
    try:
        observations = read_observations(
            source, policy.spec.get("observation_space") or {}
        )
    except (OSError, ValueError, UnicodeDecodeError) as error:
        raise user_error(ValueError(str(error))) from error

    # 1 件ずつ守りを当てる (観測の誤りは守りの外れたときの動き)
    rows = []
    counts = dict.fromkeys(STATUSES, 0)
    # (鍵は件ごとに取り直す. 長いファイルの間も 1 件ずつの行動を待たせ
    # ないように)
    for index, observation in enumerate(observations):
        if count_items(observation, MAX_OBSERVATION) > MAX_OBSERVATION:
            outcome: dict[str, Any] = {
                "action": None, "status": "stop",
                "reasons": [f"観測の要素が {MAX_OBSERVATION} 個を超えます"],
            }
        else:
            with ACT_LOCK:
                outcome = policy.act(observation)
        status = str(outcome["status"])
        counts[status] = counts.get(status, 0) + 1
        rows.append({
            "index": index,
            "observation": json_text(observation),
            "action": json_text(outcome["action"]),
            "status": status,
            "reasons": " / ".join(str(r) for r in outcome["reasons"]),
        })

    # 隣に書いてから置き換える (書き出しの途中の結果を残さない)
    store = get_store(request)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_id = f"{stamp}-{uuid.uuid4().hex[:6]}"
    place = results_root(store) / result_id
    writing = place.with_name(f".{result_id}.writing")
    meta = {
        "format_version": POLICY_RESULT_VERSION,
        "policy_id": policy_id,
        "policy": {"name": entry.get("name"), "path": entry.get("path")},
        "source": str(source),
        "created_at": datetime.now().astimezone().isoformat(),
        "rows": len(rows),
        "counts": counts,
    }
    try:
        writing.mkdir(parents=True)
        with open(writing / ACTIONS_FILE, "w", encoding="utf-8-sig",
                  newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ACTION_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        (writing / META_FILE).write_text(
            yaml.safe_dump(meta, allow_unicode=True, sort_keys=False),
            encoding="utf-8"
        )
        writing.rename(place)
    except OSError as error:
        shutil.rmtree(writing, ignore_errors=True)
        raise user_error(error) from error
    view = result_view(result_id, meta)

    return view


@router.get("/policy-results")
def list_results(
    request: Request, policy: Optional[str] = None
) -> list[dict[str, Any]]:
    """
    方策の結果の一覧を新しい順に返す関数

    Parameters
    ----------
    request: Request
        リクエスト
    policy: Optional[str] = None
        方策の一式の id で絞る

    Returns
    ----------
    results: list[dict[str, Any]]
        結果のまとめ (読めない結果は飛ばす)
    """
    root = results_root(get_store(request))
    if not root.is_dir():
        return []
    results = []
    for place in sorted(root.iterdir(), key=lambda p: p.name, reverse=True):
        if place.name.startswith(".") or not (place / META_FILE).is_file():
            continue
        try:
            meta = read_meta(place)
        except (OSError, ValueError, yaml.YAMLError):
            continue
        if policy is not None and meta.get("policy_id") != policy:
            continue
        results.append(result_view(place.name, meta))

    return results


@router.get("/policy-results/{result_id}")
def get_result(request: Request, result_id: str) -> dict[str, Any]:
    """
    方策の結果のまとめを返す関数

    Parameters
    ----------
    request: Request
        リクエスト
    result_id: str
        方策の結果の id

    Returns
    ----------
    view: dict[str, Any]
        結果のまとめ

    Raises
    ----------
    HTTPException
        無い場合は 404, 読めない場合は 400
    """
    place = find_place(request, result_id)
    try:
        meta = read_meta(place)
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise HTTPException(400, str(error)) from error
    view = result_view(result_id, meta)

    return view


@router.get("/policy-results/{result_id}/rows")
def get_rows(
    request: Request, result_id: str,
    offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=MAX_ROWS),
    status: Optional[str] = None
) -> dict[str, Any]:
    """
    方策の結果の表の一部を返す関数

    Parameters
    ----------
    request: Request
        リクエスト
    result_id: str
        方策の結果の id
    offset: int = Query(0, ge=0)
        先頭から飛ばす行の数
    limit: int = Query(50, ge=1, le=MAX_ROWS)
        返す行の数
    status: Optional[str] = None
        守りの結果で絞る

    Returns
    ----------
    rows: dict[str, Any]
        total (絞った後の行の数), offset, rows (index, observation, action,
        status, reasons. 観測と行動は JSON の文字)
    """
    # 絞ってから切り出す
    # (消している途中で無くなった結果は 404)
    place = find_place(request, result_id)
    try:
        with open(place / ACTIONS_FILE, encoding="utf-8-sig",
                  newline="") as handle:
            records = [
                {**row, "index": int(row["index"])}
                for row in csv.DictReader(handle)
                if not status or row.get("status") == status
            ]
    except FileNotFoundError as error:
        raise HTTPException(404, "方策の結果が見つかりません") from error
    rows = {
        "total": len(records),
        "offset": offset,
        "rows": records[offset:offset + limit],
    }

    return rows


def safe_cell(text: str) -> str:
    """
    表計算のソフトが式として読む文字の欄の頭に ' を付ける関数

    Parameters
    ----------
    text: str
        欄の文字

    Returns
    ----------
    cell: str
        数ならそのまま, = + - @ で始まる文字なら頭に ' を付けたもの
    """
    # 数 (-1 や -0.5 の行動) は式でないのでそのまま
    if not text.startswith(FORMULA_HEADS):
        return text
    try:
        float(text)
    except ValueError:
        cell = f"'{text}"
        return cell

    return text


@router.get("/policy-results/{result_id}/download")
def download(request: Request, result_id: str) -> Response:
    """
    方策の結果の表の CSV を返す関数

    Parameters
    ----------
    request: Request
        リクエスト
    result_id: str
        方策の結果の id

    Returns
    ----------
    response: Response
        actions.csv の中身 (式として読まれる文字の欄は頭に ' を付ける.
        結果の id の名前で保存させる)

    Raises
    ----------
    HTTPException
        無い場合と, 消している途中で無くなった場合は 404
    """
    # 欄ごとに式の頭を外して書き直す
    place = find_place(request, result_id)
    try:
        with open(place / ACTIONS_FILE, encoding="utf-8-sig",
                  newline="") as handle:
            lines = list(csv.reader(handle))
    except FileNotFoundError as error:
        raise HTTPException(404, "方策の結果が見つかりません") from error
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerows([[safe_cell(cell) for cell in line] for line in lines])
    response = Response(
        content="\ufeff" + buffer.getvalue(), media_type="text/csv",
        headers={"Content-Disposition":
                 f'attachment; filename="actions-{result_id}.csv"'}
    )

    return response


@router.delete("/policy-results/{result_id}", status_code=204)
def delete_result(request: Request, result_id: str) -> Response:
    """
    方策の結果を消す関数

    Parameters
    ----------
    request: Request
        リクエスト
    result_id: str
        方策の結果の id

    Returns
    ----------
    response: Response
        204
    """
    place = find_place(request, result_id)
    try:
        shutil.rmtree(place)
    except FileNotFoundError as error:
        raise HTTPException(404, "方策の結果が見つかりません") from error
    response = Response(status_code=204)

    return response
