import json
import math
from pathlib import Path
from typing import Any, Literal, Union

import ml_inference
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, StrictInt

from ml_provide.api import (
    RESULT_LOCK,
    USER_ERRORS,
    find_result_meta,
    read_frame,
    user_error,
)

router = APIRouter(prefix="/api")

# 確かめた人の名前の長さの上限 (記録の 1 行に収まる名前)
MAX_REVIEWER = 80


class ReviewRequest(BaseModel):
    # 行の id の文字 (id の列の無い結果は行の位置の整数)
    row: Union[StrictInt, str]
    judgment: Literal["anomaly", "normal"]
    reviewer: str


def require_anomaly(meta: dict[str, Any]) -> None:
    """
    異常の度合いの結果でなければ 400 で断る関数

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
        異常の度合いの一式の結果でない場合は 400
    """
    if meta.get("kind") != "anomaly":
        raise HTTPException(
            400, "人の確認は異常の度合いの一式の結果だけで使えます"
        )


def plain_value(value: Any) -> Any:
    """
    表の 1 つの値を JSON に書ける値にする関数

    Parameters
    ----------
    value: Any
        表の値

    Returns
    ----------
    plain: Any
        Python の値 (欠損と数でない小数は None)
    """
    # NumPy と pandas の値は Python の値に
    plain = value.item() if hasattr(value, "item") else value
    if plain is None or plain is pd.NA or (
        isinstance(plain, float) and not math.isfinite(plain)
    ):
        plain = None

    return plain


def acted_rows(frame: pd.DataFrame) -> pd.Series:
    """
    予測の表の, 人の確認に回した行の印を返す関数

    Parameters
    ----------
    frame: pd.DataFrame
        異常の度合いの結果の予測の表

    Returns
    ----------
    acted: pd.Series
        行ごとの真偽 (action の列の無い, 振り分けていない結果は全て偽)
    """
    # 読み直した action は真偽か欠損 (予測しなかった行)
    # (Python と NumPy の真偽と, 文字で読んだ真偽を同じに)
    acted: pd.Series = pd.Series(False, index=frame.index)
    if "action" in frame.columns:
        acted = frame["action"].astype(str) == "True"

    return acted


def latest_reviews(place: Path) -> dict[str, dict[str, str]]:
    """
    行の id ごとの最後の確認の記録を返す関数

    Parameters
    ----------
    place: Path
        結果のディレクトリ

    Returns
    ----------
    latest: dict[str, dict[str, str]]
        行の id の文字から, 最後の記録 (id, judgment, reviewer,
        reviewed_at) への対応 (同じ行の最後の記録が今の判断)
    """
    reviews = ml_inference.read_reviews(place)
    latest = {str(record["id"]): {k: str(v) for k, v in record.items()}
              for record in reviews.to_dict("records")}

    return latest


@router.get("/results/{result_id}/queue")
def get_queue(request: Request, result_id: str) -> dict[str, Any]:
    """
    異常の度合いの結果の, 人の確認に回した行を度合いの高い順に返す関数

    Parameters
    ----------
    request: Request
        リクエスト
    result_id: str
        結果の id

    Returns
    ----------
    queue: dict[str, Any]
        id (行の id の列. 無ければ None), capacity (人の確認に回す件数の
        上限. 振り分けていなければ None), rows (行の位置 row, 記録に使う
        行の id の key, 行の id の値 id, score, rank, column, column_share,
        今の判断 judgment, reviewer, reviewed_at)

    Raises
    ----------
    HTTPException
        異常の度合いの結果でない場合と, 記録が読めない場合は 400
    """
    # 回した行を順位の順に (順位の欠けた行は後ろ)
    place, meta = find_result_meta(request, result_id)
    require_anomaly(meta)
    column = (meta.get("summary") or {}).get("id")
    frame = read_frame(place, meta)
    try:
        latest = latest_reviews(place)
    except USER_ERRORS as error:
        raise user_error(error) from error
    chosen = frame[acted_rows(frame).to_numpy()]
    if "rank" in chosen.columns:
        order = pd.to_numeric(chosen["rank"], errors="coerce")
        chosen = chosen.assign(_order=order.to_numpy()).sort_values(
            "_order", kind="stable", na_position="last"
        )
    rows = []
    for position, record in zip(
        chosen.index, chosen.to_dict("records"), strict=True
    ):
        value = plain_value(record.get(column)) if column else None
        key: Union[int, str] = (
            str(value) if column and value is not None else int(position)
        )
        found = latest.get(str(key)) or {}
        rows.append({
            "row": int(position), "key": key, "id": value,
            "score": plain_value(record.get("score")),
            "rank": plain_value(record.get("rank")),
            "column": plain_value(record.get("column")),
            "column_share": plain_value(record.get("column_share")),
            "judgment": found.get("judgment"),
            "reviewer": found.get("reviewer"),
            "reviewed_at": found.get("reviewed_at"),
        })
    decided = (meta.get("decision") or {}).get("decision") or {}
    capacity = decided.get("capacity")
    # (JSON に書けない値を残さないように, 書いて読み直す)
    queue: dict[str, Any] = json.loads(json.dumps(
        {"id": column, "capacity": capacity, "rows": rows},
        ensure_ascii=False, default=str
    ))

    return queue


@router.get("/results/{result_id}/reviews")
def get_reviews(request: Request, result_id: str) -> dict[str, Any]:
    """
    異常の度合いの結果の人の確認の記録を書いた順に返す関数

    Parameters
    ----------
    request: Request
        リクエスト
    result_id: str
        結果の id

    Returns
    ----------
    reviews: dict[str, Any]
        records (id, judgment, reviewer, reviewed_at の並び. 全て文字)

    Raises
    ----------
    HTTPException
        異常の度合いの結果でない場合と, 記録が読めない場合は 400
    """
    # ML-Inference の read_reviews のまま
    place, meta = find_result_meta(request, result_id)
    require_anomaly(meta)
    try:
        found = ml_inference.read_reviews(place)
    except USER_ERRORS as error:
        raise user_error(error) from error
    records = [{k: str(v) for k, v in record.items()}
               for record in found.to_dict("records")]
    reviews = {"records": records}

    return reviews


@router.post("/results/{result_id}/reviews", status_code=201)
def add_review(
    request: Request, result_id: str, body: ReviewRequest
) -> dict[str, Any]:
    """
    異常の度合いの結果に, 人の確認の記録を 1 件足す関数

    Parameters
    ----------
    request: Request
        リクエスト
    result_id: str
        結果の id
    body: ReviewRequest
        行, 判断 (anomaly か normal), 確かめた人の名前

    Returns
    ----------
    record: dict[str, Any]
        足した記録 (id, judgment, reviewer, reviewed_at. 時刻は
        ML-Inference の既定の今の時刻)

    Raises
    ----------
    HTTPException
        異常の度合いの結果でない場合, 名前が長すぎる場合, 結果に無い行と
        形の誤りは 400
    """
    # 時刻は渡さず ML-Inference の既定 (今の時刻)
    place, meta = find_result_meta(request, result_id)
    require_anomaly(meta)
    if len(body.reviewer.strip()) > MAX_REVIEWER:
        raise HTTPException(
            400, f"確かめた人の名前は {MAX_REVIEWER} 文字までにしてください"
        )
    with RESULT_LOCK:
        try:
            record: dict[str, Any] = ml_inference.record_review(
                place, body.row, body.judgment, body.reviewer
            )
        except USER_ERRORS as error:
            raise user_error(error) from error

    return record
