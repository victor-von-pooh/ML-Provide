import json
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request

from ml_provide.api import MAX_ROWS, find_result_meta, read_frame
from ml_provide.reviews import acted_rows, plain_value

router = APIRouter(prefix="/api")


@router.get("/results/{result_id}/treated")
def get_treated(
    request: Request, result_id: str,
    offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=MAX_ROWS)
) -> dict[str, Any]:
    """
    効果の結果の, 配る行を効果の高い順に返す関数

    Parameters
    ----------
    request: Request
        リクエスト
    result_id: str
        結果の id
    offset: int = Query(0, ge=0)
        効果の高い順で先頭から飛ばす行の数
    limit: int = Query(100, ge=1, le=MAX_ROWS)
        返す行の数

    Returns
    ----------
    treated: dict[str, Any]
        id (行の id の列. 無ければ None), budget (配る数の上限. 配って
        いなければ None), total (配る行の数), offset, rows (行の位置 row,
        行の id の値 id, rank, effect, treated, control)

    Raises
    ----------
    HTTPException
        効果の結果でない場合は 400
    """
    # 配る行を順位の順に (順位の欠けた行は後ろ)
    place, meta = find_result_meta(request, result_id)
    if meta.get("kind") != "uplift":
        raise HTTPException(400, "配る行は効果の一式の結果だけで使えます")
    column = (meta.get("summary") or {}).get("id")
    decided = (meta.get("decision") or {}).get("decision") or {}
    frame = read_frame(place, meta)
    chosen = frame[acted_rows(frame).to_numpy()]
    if "rank" in chosen.columns:
        order = pd.to_numeric(chosen["rank"], errors="coerce")
        chosen = chosen.assign(_order=order.to_numpy()).sort_values(
            "_order", kind="stable", na_position="last"
        )
    part = chosen.iloc[offset:offset + limit]
    rows = [
        {
            "row": int(position),
            "id": plain_value(record.get(column)) if column else None,
            "rank": plain_value(record.get("rank")),
            "effect": plain_value(record.get("effect")),
            "treated": plain_value(record.get("treated")),
            "control": plain_value(record.get("control")),
        }
        for position, record in zip(
            part.index, part.to_dict("records"), strict=True
        )
    ]
    # (JSON に書けない値を残さないための, 書いての読み直し)
    treated: dict[str, Any] = json.loads(json.dumps(
        {"id": column, "budget": decided.get("budget"),
         "total": int(len(chosen)), "offset": offset, "rows": rows},
        ensure_ascii=False, default=str
    ))

    return treated
