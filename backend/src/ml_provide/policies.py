import json
import pickle
import threading
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import ml_inference
import numpy as np
import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from ml_inference.policy import fallback_result
from pydantic import BaseModel

from ml_provide.api import MAX_NAME, PREDICT_LOCK, get_store, user_error
from ml_provide.store import Store

router = APIRouter(prefix="/api")

# 登録した方策の一式の一覧のファイル
POLICIES_FILE = "policies.json"

# 方策の一式の中身を書くファイルと, 重みのファイルの既定
SPEC_FILE = "policy.json"
WEIGHTS_FILE = "policy.pt"

# 読み込んだ方策を覚える数 (方策は小さいので, 推論の一式の既定と同じ 4 つ)
POLICY_CACHE = 4

# 観測の要素の数の上限 (大きな配列で止まらないように. 空間の形と違えば
# 守りが外れたときの動きにする)
MAX_OBSERVATION = 100_000

# 方策の一式を読めないときの例外
READ_ERRORS = (
    OSError, ValueError, KeyError, TypeError, RuntimeError, yaml.YAMLError,
    pickle.UnpicklingError
)

# 登録の一覧を書き換える鍵と, 覚えた方策の鍵
POLICIES_LOCK = threading.Lock()
CACHE_LOCK = threading.Lock()

# ファイルごとの (更新時刻, 大きさ) の組. 無いファイルは None
Marks = tuple[Optional[tuple[int, int]], ...]
CACHE: "OrderedDict[str, tuple[Marks, ml_inference.GuardedPolicy]]" = (
    OrderedDict()
)


class PolicyRegister(BaseModel):
    path: str
    name: Optional[str] = None


class ActRequest(BaseModel):
    observation: Any


def policies_path(store: Store) -> Path:
    """
    登録した方策の一式の一覧のファイルの場所を返す関数

    Parameters
    ----------
    store: Store
        置き場

    Returns
    ----------
    path: Path
        policies.json の場所
    """
    path = store.home / POLICIES_FILE

    return path


def read_policies(store: Store) -> list[dict[str, Any]]:
    """
    登録した方策の一式の一覧を読む関数

    Parameters
    ----------
    store: Store
        置き場

    Returns
    ----------
    policies: list[dict[str, Any]]
        登録 (無い, 壊れている場合は空)
    """
    # 壊れたファイルは, 登録を失わないように .bak に退避してから空
    path = policies_path(store)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError):
        try:
            path.replace(path.with_suffix(".bak"))
        except OSError:
            pass
        return []
    policies = [
        item for item in loaded if isinstance(item, dict)
    ] if isinstance(loaded, list) else []

    return policies


def write_policies(store: Store, policies: list[dict[str, Any]]) -> None:
    """
    登録した方策の一式の一覧を書く関数

    Parameters
    ----------
    store: Store
        置き場
    policies: list[dict[str, Any]]
        登録

    Returns
    ----------
    None
    """
    # 一時ファイルに書いてから置き換え
    store.home.mkdir(parents=True, exist_ok=True)
    path = policies_path(store)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(policies, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def find_policy(request: Request, policy_id: str) -> dict[str, Any]:
    """
    登録した方策の一式を探す関数

    Parameters
    ----------
    request: Request
        リクエスト
    policy_id: str
        方策の一式の id

    Returns
    ----------
    entry: dict[str, Any]
        登録

    Raises
    ----------
    HTTPException
        無い場合は 404
    """
    entry = next(
        (item for item in read_policies(get_store(request))
         if item.get("id") == policy_id),
        None
    )
    if entry is None:
        raise HTTPException(404, "方策の一式が見つかりません")

    return entry


def stamp(path: str) -> Marks:
    """
    方策の一式のファイルの更新時刻と大きさの組を返す関数

    Parameters
    ----------
    path: str
        方策の一式のディレクトリ

    Returns
    ----------
    marks: Marks
        policy.json, 重み, 写したモデルの定義の (更新時刻, 大きさ). 無い
        ファイルは None
    """
    # 学習し直して書き出し直した一式を読み直すための印. 重みと定義の
    # 場所は policy.json から読む (確かめは読み込みの ML-Inference)
    directory = Path(path)
    names: list[Any] = [SPEC_FILE, WEIGHTS_FILE, None]
    try:
        spec = json.loads((directory / SPEC_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        spec = None
    if isinstance(spec, dict):
        names[1] = spec.get("weights")
        metadata = spec.get("metadata")
        how = metadata.get("model") if isinstance(metadata, dict) else None
        names[2] = how.get("spec") if isinstance(how, dict) else None
    marks: list[Optional[tuple[int, int]]] = []
    for name in names:
        try:
            if not isinstance(name, str) or not name:
                raise OSError(name)
            info = (directory / name).stat()
            marks.append((info.st_mtime_ns, info.st_size))
        except (OSError, ValueError):
            marks.append(None)

    return tuple(marks)


def open_policy(path: str) -> ml_inference.GuardedPolicy:
    """
    方策の一式を読む関数 (変わっていなければ覚えたものを使う)

    Parameters
    ----------
    path: str
        方策の一式のディレクトリ

    Returns
    ----------
    policy: ml_inference.GuardedPolicy
        守りを当てて行動を返す方策
    """
    # ファイルが変わっていなければ使い回し, 古いものから手放す
    marks = stamp(path)
    with CACHE_LOCK:
        cached = CACHE.get(path)
        if cached is not None and cached[0] == marks:
            CACHE.move_to_end(path)
            return cached[1]
    policy = ml_inference.load_policy(path)
    with CACHE_LOCK:
        CACHE[path] = (marks, policy)
        CACHE.move_to_end(path)
        while len(CACHE) > POLICY_CACHE:
            CACHE.popitem(last=False)

    return policy


def forget_policy(path: str) -> None:
    """
    覚えた方策を手放す関数

    Parameters
    ----------
    path: str
        方策の一式のディレクトリ

    Returns
    ----------
    None
    """
    with CACHE_LOCK:
        CACHE.pop(path, None)


def policy_view(
    entry: dict[str, Any], load: bool = False
) -> dict[str, Any]:
    """
    登録に方策の一式の中身を足して返す関数

    Parameters
    ----------
    entry: dict[str, Any]
        登録
    load: bool = False
        重みも読むか (読めば読まなかった重みのキーも返す)

    Returns
    ----------
    view: dict[str, Any]
        登録と, ok, error, spec (policy.json の中身), unused_keys
    """
    # 読めない一式は理由付き (一覧を止めない)
    view = {**entry, "ok": True, "error": None, "spec": None,
            "unused_keys": None}
    path = str(entry.get("path"))
    try:
        if load:
            policy = open_policy(path)
            view["spec"] = policy.spec
            view["unused_keys"] = policy.unused_keys
        else:
            view["spec"] = ml_inference.read_policy_spec(path)
    except READ_ERRORS as error:
        view["ok"] = False
        view["error"] = str(error) or type(error).__name__

    return view


def count_leaves(value: Any, limit: int) -> int:
    """
    入れ子のリストの葉 (リストでない要素) の数を数える関数

    Parameters
    ----------
    value: Any
        観測 (JSON から読んだ値)
    limit: int
        上限 (超えたら数えるのをやめる)

    Returns
    ----------
    count: int
        葉の数 (上限を超えたら上限より 1 大きい数)
    """
    # 深い入れ子でも止まらないように, 再帰でなく積んで数える
    count = 0
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
            continue
        count += 1
        if count > limit:
            break

    return count


def to_json(value: Any) -> Any:
    """
    行動を JSON の値にする関数

    Parameters
    ----------
    value: Any
        行動 (整数, 配列, None)

    Returns
    ----------
    result: Any
        整数, 数の入れ子のリスト, None
    """
    # box の行動は float32 の配列なので, リストに
    if isinstance(value, np.ndarray):
        result = value.tolist()
        return result
    if isinstance(value, np.generic):
        result = value.item()
        return result

    return value


@router.get("/policies")
def list_policies(request: Request) -> list[dict[str, Any]]:
    """
    登録した方策の一式の一覧を返す関数

    Parameters
    ----------
    request: Request
        リクエスト

    Returns
    ----------
    policies: list[dict[str, Any]]
        登録と中身 (新しい順)
    """
    entries = read_policies(get_store(request))
    policies = [policy_view(entry) for entry in reversed(entries)]

    return policies


@router.post("/policies", status_code=201)
def register_policy(
    request: Request, body: PolicyRegister
) -> dict[str, Any]:
    """
    場所を指定して方策の一式を登録する関数

    Parameters
    ----------
    request: Request
        リクエスト
    body: PolicyRegister
        方策の一式のディレクトリと名前

    Returns
    ----------
    view: dict[str, Any]
        登録と中身

    Raises
    ----------
    HTTPException
        読めない一式と, 既に登録した場所は 400
    """
    # 重みまで読めることを確かめてから登録
    path = Path(body.path).expanduser().resolve()
    try:
        open_policy(str(path))
    except READ_ERRORS as error:
        raise user_error(error) from error
    store = get_store(request)
    name = (body.name or path.name)[:MAX_NAME]
    with POLICIES_LOCK:
        policies = read_policies(store)
        if any(item.get("path") == str(path) for item in policies):
            raise HTTPException(400, f"{path} は既に登録されています")
        entry = {
            "id": uuid.uuid4().hex[:12],
            "name": name,
            "path": str(path),
            "added_at": datetime.now().astimezone().isoformat(),
        }
        policies.append(entry)
        write_policies(store, policies)
    view = policy_view(entry, load=True)

    return view


@router.get("/policies/{policy_id}")
def get_policy(request: Request, policy_id: str) -> dict[str, Any]:
    """
    方策の一式の中身を返す関数

    Parameters
    ----------
    request: Request
        リクエスト
    policy_id: str
        方策の一式の id

    Returns
    ----------
    view: dict[str, Any]
        登録, policy.json の中身, 読まなかった重みのキー
    """
    view = policy_view(find_policy(request, policy_id), load=True)

    return view


@router.delete("/policies/{policy_id}", status_code=204)
def delete_policy(request: Request, policy_id: str) -> Response:
    """
    方策の一式の登録を消す関数 (一式のファイルは消さない)

    Parameters
    ----------
    request: Request
        リクエスト
    policy_id: str
        方策の一式の id

    Returns
    ----------
    response: Response
        204

    Raises
    ----------
    HTTPException
        無い場合は 404
    """
    store = get_store(request)
    with POLICIES_LOCK:
        policies = read_policies(store)
        target = next(
            (item for item in policies if item.get("id") == policy_id), None
        )
        if target is None:
            raise HTTPException(404, "方策の一式が見つかりません")
        write_policies(store, [item for item in policies
                               if item is not target])
    forget_policy(str(target.get("path")))
    response = Response(status_code=204)

    return response


@router.post("/policies/{policy_id}/act")
def act(
    request: Request, policy_id: str, body: ActRequest
) -> dict[str, Any]:
    """
    観測から, 守りを当てた行動を返す関数 (環境や機械には触れない)

    Parameters
    ----------
    request: Request
        リクエスト
    policy_id: str
        方策の一式の id
    body: ActRequest
        観測 (数の配列)

    Returns
    ----------
    result: dict[str, Any]
        action (行動. 止めるか元の制御に戻す場合は None), status (ok,
        clipped, fallback, stop, handover), reasons (守りが働いた理由)

    Raises
    ----------
    HTTPException
        読めない一式と, 観測が大きすぎる場合は 400
    """
    # 観測の大きさの上限 (形や値の誤りは守りが外れたときの動きにする)
    if count_leaves(body.observation, MAX_OBSERVATION) > MAX_OBSERVATION:
        raise HTTPException(
            400, f"観測の要素は {MAX_OBSERVATION} 個までにしてください"
        )
    entry = find_policy(request, policy_id)
    try:
        policy = open_policy(str(entry.get("path")))
    except READ_ERRORS as error:
        raise user_error(error) from error

    # 推論の一式と同じく, モデルを動かすのは 1 つずつ
    # 桁の大きな整数は float にできず OverflowError になるので, 数の配列に
    # できない観測と同じ外れたときの動き
    with PREDICT_LOCK:
        try:
            outcome = policy.act(body.observation)
        except OverflowError as exc:
            outcome = fallback_result(
                policy.spec, [f"観測を数の配列にできない ({exc})"]
            )
    result = {
        "action": to_json(outcome["action"]),
        "status": outcome["status"],
        "reasons": [str(reason) for reason in outcome["reasons"]],
    }

    return result
