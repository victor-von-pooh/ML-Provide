import math
from pathlib import Path
from typing import Any, Optional

import ml_inference
import numpy as np
import pandas as pd

# 1 行の理由を求めるときに, 選んだ行と一緒に渡す比べる行の数 (行ごとの
# 効きは「その列の値が他の行の値に替わったら」なので, 他の行が要る.
# 推論の回数は列の数 × 回数で, 1 回の推論の行の数をこの数に抑える)
COMPARE_ROWS = 200

# 過去の値の列を作る一式で, 選んだ行の前から渡す続いた行の数 (過去の値
# に前の行が要るため. 系列が混ざった表でも, 選んだ行の系列の前の行が
# 入るよう多めにする)
LAG_ROWS = 1000

# 列ごとに並べ替える回数 (ML-Inference の explain_rows の既定と同じ)
ROW_REPEATS = 10


def uses_lag(bundle: ml_inference.Bundle) -> bool:
    """
    一式のレシピが過去の値の列を作るかを返す関数

    Parameters
    ----------
    bundle: ml_inference.Bundle
        推論の一式

    Returns
    ----------
    lag: bool
        過去の値の手順があるか
    """
    steps = bundle.dataset.recipe.get("steps") or []
    lag = any(step.get("op") == "lag" for step in steps)

    return lag


def read_input(
    bundle: ml_inference.Bundle, path: Path,
    options: Optional[dict[str, Any]]
) -> pd.DataFrame:
    """
    結果の入力のファイルを, 推論のときと同じ読み込みで読む関数

    ML-Inference の read_source で, 学習のときの読み込みと設定に結果の
    設定を上書きして読む (読めなければ形式を判定し直す)

    Parameters
    ----------
    bundle: ml_inference.Bundle
        推論の一式
    path: Path
        入力のファイル
    options: Optional[dict[str, Any]]
        推論のときの読み込みの設定

    Returns
    ----------
    frame: pd.DataFrame
        読んだ表 (行の位置は推論の結果の行と同じ)

    Raises
    ----------
    ValueError
        読めない場合
    """
    try:
        data = ml_inference.read_source(bundle, path, options)
    except ValueError as error:
        raise ValueError(
            f"{error}. 推論したときの読み込みの設定で読めるファイルかを"
            "確かめてください"
        ) from error
    frame: pd.DataFrame = data.frame.reset_index(drop=True)

    return frame


def rows_for(row: int, total: int, lag: bool) -> list[int]:
    """
    1 行の理由を求めるのに渡す行の位置を返す関数

    Parameters
    ----------
    row: int
        理由を見せる行の位置
    total: int
        入力の行の数
    lag: bool
        過去の値の列を作る一式か

    Returns
    ----------
    rows: list[int]
        渡す行の位置 (元の並び. 選んだ行を含む)
    """
    # 過去の値の列は前の行が要るので, 選んだ行までの続いた行
    if lag:
        rows = list(range(max(0, row - LAG_ROWS + 1), row + 1))
        return rows

    # 他は選んだ行と, 全体から等しい間隔で選んだ比べる行
    others = [i for i in range(total) if i != row]
    if len(others) > COMPARE_ROWS:
        picks = np.linspace(0, len(others) - 1, COMPARE_ROWS).round()
        others = [others[int(i)] for i in np.unique(picks)]
    rows = sorted([row, *others])

    return rows


def number_or_none(value: Any) -> Optional[float]:
    """
    有限の数を float に, 欠損と無限を None にする関数

    Parameters
    ----------
    value: Any
        値

    Returns
    ----------
    number: Optional[float]
        数か None
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None

    return number


def row_reason(
    bundle: ml_inference.Bundle, frame: pd.DataFrame, row: int
) -> dict[str, Any]:
    """
    入力の表の 1 行の, 列ごとの予測の効きと理由を求める関数

    Parameters
    ----------
    bundle: ml_inference.Bundle
        推論の一式 (表か時系列)
    frame: pd.DataFrame
        入力の表
    row: int
        理由を見せる行の位置

    Returns
    ----------
    reason: dict[str, Any]
        row, measure (分類は probability, 回帰と予測は value), status,
        prediction, reason (効きの大きい 3 列の文. 無ければ None), effects
        (列と効き. 効きの絶対値の大きい順で, 求められない列は最後),
        errors (列の名前から求められない理由), rows (渡した行の数),
        repeats (並べ替えた回数)

    Raises
    ----------
    ValueError
        ML-Inference が求められない場合
    """
    # 選んだ行と比べる行だけを渡して求める
    rows = rows_for(row, len(frame), uses_lag(bundle))
    part = frame.iloc[rows].reset_index(drop=True)
    found = ml_inference.explain_rows(bundle, part, repeats=ROW_REPEATS)
    table = found["frame"]
    line = table.iloc[rows.index(row)]

    # 効きの大きい順 (求められない列は最後)
    prefix = "effect:"
    effects = [
        {"column": name[len(prefix):], "effect": number_or_none(line[name])}
        for name in found["columns"]
    ]
    effects.sort(key=lambda item: (
        item["effect"] is None, -abs(item["effect"] or 0.0)
    ))
    prediction = line.get("prediction")
    text = line.get("reason")
    reason = {
        "row": row,
        "measure": found["measure"],
        "status": str(line.get("status")),
        "prediction": None if pd.isna(prediction) else (
            number_or_none(prediction) if bundle.task != "classification"
            else str(prediction)
        ),
        "reason": text if isinstance(text, str) and text else None,
        "effects": effects,
        "errors": dict(found["errors"]),
        "rows": len(rows),
        "repeats": ROW_REPEATS,
    }

    return reason
