import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request

from ml_provide.api import MAX_ROWS, find_result_meta, read_frame
from ml_provide.reviews import plain_value

router = APIRouter(prefix="/api")

# 発注の数を決めた 1 行に返す予測の表の列 (返す名前と表の列の名前)
ORDER_COLUMNS = {
    "order_date": "order_date",
    "on_hand": "on_hand",
    "previous_demand": "previous_demand",
    "prediction": "prediction",
    "quantile": "quantile",
    "order": "order",
    "capped": "capped",
    "expected_cost": "expected_cost",
    "baseline_order": "baseline_order",
    "baseline_cost": "baseline_cost",
}


@router.get("/results/{result_id}/orders")
def get_orders(
    request: Request, result_id: str,
    offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=MAX_ROWS)
) -> dict[str, Any]:
    """
    需要の結果の, 発注の数を決めた商品を発注の数の大きい順に返す関数

    Parameters
    ----------
    request: Request
        リクエスト
    result_id: str
        結果の id
    offset: int = Query(0, ge=0)
        発注の数の大きい順で先頭から飛ばす行の数
    limit: int = Query(100, ge=1, le=MAX_ROWS)
        返す行の数

    Returns
    ----------
    orders: dict[str, Any]
        id (行の id の列. 無ければ None), product (商品の id の列. 無ければ
        None), levels (学んだ水準), shortage_cost, holding_cost, max_order
        (発注の判断の損と上限. 決めていなければ None), total (発注の数を
        決めた行の数), offset, rows (行の位置
        row, 行の id の値 id, 商品の id の値 product_id, 水準の小さい順の
        分位点 quantiles と ORDER_COLUMNS の値)

    Raises
    ----------
    HTTPException
        需要の結果でない場合は 400
    """
    # 発注の数のある行 (決めない行と予測しなかった行は空)
    place, meta = find_result_meta(request, result_id)
    if meta.get("kind") != "demand":
        raise HTTPException(400, "発注の数は需要の一式の結果だけで使えます")
    summary = meta.get("summary") or {}
    column, product = summary.get("id"), summary.get("product")
    levels = [float(level) for level in summary.get("levels") or []]
    decided = (meta.get("decision") or {}).get("decision") or {}
    frame = read_frame(place, meta)
    chosen = frame.iloc[0:0]
    if "order" in frame.columns:
        order = pd.to_numeric(frame["order"], errors="coerce")
        chosen = frame[order.notna().to_numpy()]

    # 発注の数の大きい順 (同じなら表の順. 上限で切った商品が上に来る)
    if len(chosen):
        values = pd.to_numeric(chosen["order"], errors="coerce").to_numpy(
            dtype=float
        )
        chosen = chosen.iloc[np.argsort(-values, kind="stable")]
    names = [f"q:{level:g}" for level in levels]
    part = chosen.iloc[offset:offset + limit]
    rows = [
        {
            "row": int(position),
            "id": plain_value(record.get(column)) if column else None,
            "product_id": (
                plain_value(record.get(product)) if product else None
            ),
            "quantiles": [plain_value(record.get(name)) for name in names],
            **{key: plain_value(record.get(name))
               for key, name in ORDER_COLUMNS.items()},
        }
        for position, record in zip(
            part.index, part.to_dict("records"), strict=True
        )
    ]
    # JSON に書ける値にそろえるための読み直し
    orders: dict[str, Any] = json.loads(json.dumps(
        {"id": column, "product": product, "levels": levels,
         "shortage_cost": decided.get("shortage_cost"),
         "holding_cost": decided.get("holding_cost"),
         "max_order": decided.get("max_order"),
         "total": int(len(chosen)), "offset": offset, "rows": rows},
        ensure_ascii=False, default=str
    ))

    return orders
