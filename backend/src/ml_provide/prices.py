import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request

from ml_provide.api import MAX_ROWS, find_result_meta, read_frame
from ml_provide.reviews import plain_value

router = APIRouter()

# 選んだ値段の 1 行に返す予測の表の列 (返す名前と表の列の名前)
PRICE_COLUMNS = {
    "current_price": "current_price",
    "cost": "cost",
    "change": "change",
    "price": "price",
    "units": "units",
    "price_units": "price_units",
    "current_profit": "current_profit",
    "profit": "profit",
    "elasticity": "elasticity",
}


@router.get("/results/{result_id}/prices")
def get_prices(
    request: Request, result_id: str,
    offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=MAX_ROWS)
) -> dict[str, Any]:
    """
    値段の結果の, 値段を決めた行を見込みの利益の差の大きい順に返す関数

    Parameters
    ----------
    request: Request
        リクエスト
    result_id: str
        結果の id
    offset: int = Query(0, ge=0)
        利益の差の大きい順で先頭から飛ばす行の数
    limit: int = Query(100, ge=1, le=MAX_ROWS)
        返す行の数

    Returns
    ----------
    prices: dict[str, Any]
        id (行の id の列. 無ければ None), product (商品の id の列. 無ければ
        None), candidates と max_change (値段の候補と変える幅の上限. 決めて
        いなければ None), total (値段を決めた行の数), offset, rows (行の
        位置 row, 行の id の値 id, 商品の id の値 product_id と
        PRICE_COLUMNS の値)

    Raises
    ----------
    HTTPException
        値段の結果でない場合は 400
    """
    # 選んだ割合のある行 (決めない行と予測しなかった行は空)
    place, meta = find_result_meta(request, result_id)
    if meta.get("kind") != "price":
        raise HTTPException(400, "選んだ値段は値段の一式の結果だけで使えます")
    summary = meta.get("summary") or {}
    column, product = summary.get("id"), summary.get("product")
    decided = (meta.get("decision") or {}).get("decision") or {}
    frame = read_frame(place, meta)
    chosen = frame.iloc[0:0]
    if "change" in frame.columns:
        change = pd.to_numeric(frame["change"], errors="coerce")
        chosen = frame[change.notna().to_numpy()]

    # 見込みの利益の差の大きい順 (同じなら表の順)
    if len(chosen):
        gain = (
            pd.to_numeric(chosen["profit"], errors="coerce")
            - pd.to_numeric(chosen["current_profit"], errors="coerce")
        ).to_numpy(dtype=float)
        order = np.argsort(-np.nan_to_num(gain, nan=-np.inf), kind="stable")
        chosen = chosen.iloc[order]
    part = chosen.iloc[offset:offset + limit]
    rows = [
        {
            "row": int(position),
            "id": plain_value(record.get(column)) if column else None,
            "product_id": (
                plain_value(record.get(product)) if product else None
            ),
            **{key: plain_value(record.get(name))
               for key, name in PRICE_COLUMNS.items()},
        }
        for position, record in zip(
            part.index, part.to_dict("records"), strict=True
        )
    ]
    # JSON に書ける値にそろえるための読み直し
    prices: dict[str, Any] = json.loads(json.dumps(
        {"id": column, "product": product,
         "candidates": decided.get("candidates"),
         "max_change": decided.get("max_change"),
         "total": int(len(chosen)), "offset": offset, "rows": rows},
        ensure_ascii=False, default=str
    ))

    return prices
