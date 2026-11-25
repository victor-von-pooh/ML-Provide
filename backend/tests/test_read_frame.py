from pathlib import Path
from typing import Any

import ml_inference
import numpy as np
import pandas as pd
import yaml

from ml_provide.api import (
    ARTIFACT_COLUMN,
    COLUMN_DTYPES,
    TOKENS_COLUMN,
    read_frame,
)


def values(column: pd.Series) -> list[Any]:
    """
    列の値を, 欠けを None にした並びで返す関数

    Parameters
    ----------
    column: pd.Series
        表の列

    Returns
    ----------
    values: list[Any]
        値の並び (欠けは None)
    """
    # 欠けの種類 (NaN, None, NA) の違いを除いて比べるため
    found = [None if pd.isna(v) else v for v in column.astype(object)]

    return found


def write(place: Path, frame: pd.DataFrame, **meta: Any) -> dict[str, Any]:
    """
    ML-Inference の write_result で結果を書き, result.yaml の中身を返す
    関数

    Parameters
    ----------
    place: Path
        結果のディレクトリ
    frame: pd.DataFrame
        予測の表
    **meta: Any
        result.yaml に残す項目

    Returns
    ----------
    meta: dict[str, Any]
        書いた result.yaml の中身
    """
    ml_inference.write_result({**meta, "frame": frame}, place)
    written: dict[str, Any] = yaml.safe_load(
        (place / "result.yaml").read_text(encoding="utf-8")
    )

    return written


def write_old(
    place: Path, csv: str, meta: dict[str, Any]
) -> dict[str, Any]:
    """
    列の型を持たない古い結果 (版 1 と 2) をそのまま書く関数

    Parameters
    ----------
    place: Path
        結果のディレクトリ
    csv: str
        predictions.csv の中身
    meta: dict[str, Any]
        result.yaml の中身

    Returns
    ----------
    meta: dict[str, Any]
        書いた result.yaml の中身
    """
    place.mkdir(parents=True)
    (place / "predictions.csv").write_text(csv, encoding="utf-8-sig")
    (place / "result.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")

    return meta


def test_number_categories(tmp_path: Path) -> None:
    """
    数の種類の値の列は, 欠けた行の有無に依らず文字で読むことを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 欠けた行があると ML-Inference は種類の列の型を integer で残す
    columns = [{"name": "shop", "type": "category", "categories": 2}]
    frame = pd.DataFrame({
        "sample": [0, 1, 2], "status": ["ok", "skipped", "ok"],
        "reason": [None, "予測の値を表に戻せません", None],
        "shop": pd.array([1, None, 2], dtype="Int64"),
    })
    meta = write(tmp_path / "gap", frame, output_form="table",
                 table_columns=columns)
    assert meta["column_types"]["shop"] == "integer"
    assert values(read_frame(tmp_path / "gap", meta)["shop"]) == [
        "1", None, "2"
    ]

    # 欠けの無い同じ列も文字
    full = frame.assign(status="ok", reason=None,
                        shop=pd.array([1, 2, 2], dtype="Int64"))
    meta = write(tmp_path / "full", full, output_form="table",
                 table_columns=columns)
    assert values(read_frame(tmp_path / "full", meta)["shop"]) == [
        "1", "2", "2"
    ]


def test_missing_like_labels(tmp_path: Path) -> None:
    """
    分類の結果のラベルの NA, None, null を欠損にせず文字のまま読み,
    空の欄だけを欠損にすることを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    frame = pd.DataFrame({
        "status": ["ok"] * 4 + ["skipped"],
        "reason": [None] * 4 + ["列 a を数にできません"],
        "label": ["NA", "None", "null", "良", None],
        "prediction": ["NA", "None", "null", "1", None],
        "decision": [None] * 4 + ["予測なし"],
    })
    meta = write(tmp_path / "labels", frame, task="classification",
                 classes=["1", "NA", "None", "null", "良"])
    read = read_frame(tmp_path / "labels", meta)
    assert values(read["label"]) == ["NA", "None", "null", "良", None]
    assert values(read["prediction"]) == ["NA", "None", "null", "1", None]
    assert list(read["reason"]) == ["", "", "", "", "列 a を数にできません"]
    assert list(read["decision"]) == ["", "", "", "", "予測なし"]


def test_regression_predictions(tmp_path: Path) -> None:
    """
    回帰の結果の予測は小数で, 予測しなかった行は欠損で読むことを確かめる
    関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    frame = pd.DataFrame({
        "status": ["ok", "skipped", "ok"],
        "reason": [None, "列 a を数にできません", None],
        "prediction": [1.5, np.nan, 2.0],
        "decision": [None, "予測なし", None],
    })
    meta = write(tmp_path / "values", frame, task="regression")
    read = read_frame(tmp_path / "values", meta)
    assert values(read["prediction"]) == [1.5, None, 2.0]
    assert read["prediction"].dtype == np.float64
    assert list(read["status"]) == ["ok", "skipped", "ok"]


def test_text_predictions(tmp_path: Path) -> None:
    """
    文の出力の予測と番号を文字のまま読み, tokens の列の無い古い結果も
    予測を文字のまま読むことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # tokens の有る結果 (予測した行の空の文と番号は空の文字)
    csv = "status,reason,prediction,tokens\nok,,5,4\nok,,NA,3 2\nok,,,\n"
    meta = {"format_version": 1, "output_form": "text",
            "task": "classification"}
    write_old(tmp_path / "text", csv, meta)
    read = read_frame(tmp_path / "text", meta)
    assert list(read["prediction"]) == ["5", "NA", ""]
    assert list(read["tokens"]) == ["4", "3 2", ""]

    # tokens の列の無い古い結果 (全ての行が番号 1 つでも数にしない)
    csv = "status,reason,prediction\nok,,5\nok,,07\nok,,\n"
    write_old(tmp_path / "old", csv, meta)
    read = read_frame(tmp_path / "old", meta)
    assert list(read["prediction"]) == ["5", "07", ""]


def test_old_ranking(tmp_path: Path) -> None:
    """
    列の型を持たない版 1 の並べるの結果で, 利用者と品物を文字, 順位を
    欠損を持てる整数で読むことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    csv = (
        "user,status,reason,rank,item,score,decision\n"
        "001,ok,,1,010,0.9,\n001,ok,,2,020,0.5,\n"
        "002,skipped,知らない利用者,,,,予測なし\n"
    )
    meta = write_old(tmp_path / "rank", csv,
                     {"format_version": 1, "output_form": "ranking"})
    read = read_frame(tmp_path / "rank", meta)
    assert list(read["user"]) == ["001", "001", "002"]
    assert values(read["item"]) == ["010", "020", None]
    assert str(read["rank"].dtype) == "Int64"
    assert values(read["rank"]) == [1, 2, None]
    assert list(read["reason"]) == ["", "", "知らない利用者"]


def test_old_artifacts(tmp_path: Path) -> None:
    """
    列の型を持たない版 2 の結果 (配列を artifacts/ に置く) で, 配列の
    相対パスと状態を読むことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    csv = (
        "status,reason,artifact,decision\n"
        "ok,,artifacts/000000.npy,\nskipped,読めません,,予測なし\n"
    )
    meta = write_old(tmp_path / "arrays", csv, {
        "format_version": 2, "output_form": "vector", "column_output": False,
        "artifacts": {"form": "vector", "directory": "artifacts"},
    })
    read = read_frame(tmp_path / "arrays", meta)
    assert list(read["artifact"]) == ["artifacts/000000.npy", ""]
    assert list(read["status"]) == ["ok", "skipped"]
    assert list(read["reason"]) == ["", "読めません"]
    assert list(read["decision"]) == ["", "予測なし"]


def test_same_as_read_result(tmp_path: Path) -> None:
    """
    画面の都合の列の他は, ML-Inference の read_result と同じ値と型で
    読むことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ

    Returns
    ----------
    None
    """
    # 型と欠損の決まりの写しが ML-Inference と食い違わないこと
    from ml_inference import result as inference_result

    assert COLUMN_DTYPES == inference_result.COLUMN_DTYPES
    assert TOKENS_COLUMN == inference_result.TOKENS_COLUMN
    assert ARTIFACT_COLUMN == inference_result.ARTIFACT_COLUMN

    # 数に見える文字, 欠けた整数, NA に見える文字, 小数, 文の出力の表
    plain = pd.DataFrame({
        "sample": [0, 1, 2], "status": ["ok", "skipped", "ok"],
        "reason": [None, "列 a を数にできません", None],
        "code": ["001", "007", None], "label": ["NA", "null", None],
        "count": pd.array([3, None, 2 ** 60], dtype="Int64"),
        "prediction": [1.5, np.nan, 2.0],
        "decision": [None, "予測なし", None],
    })
    text = pd.DataFrame({
        "status": ["ok", "ok", "skipped"], "reason": [None, None, "長い"],
        "prediction": ["5", "", None], "tokens": ["4", "", None],
    })
    cases = [
        (plain, {"task": "regression"}),
        (text, {"task": "classification", "output_form": "text"}),
    ]
    for number, (frame, extra) in enumerate(cases):
        place = tmp_path / str(number)
        meta = write(place, frame, **extra)
        if number == 0:
            assert meta["column_types"] == {"code": "text",
                                            "count": "integer"}
        read = read_frame(place, meta)
        expected = ml_inference.read_result(place)["frame"]
        assert list(read.columns) == list(expected.columns)
        for name in expected.columns:
            if name in ("status", "reason", "decision"):
                continue
            assert read[name].dtype == expected[name].dtype, name
            assert values(read[name]) == values(expected[name]), name


def test_reads_once(tmp_path: Path, monkeypatch: Any) -> None:
    """
    予測の表を 1 回だけ読み, result.yaml を読まないことを確かめる関数

    Parameters
    ----------
    tmp_path: Path
        pytest が用意する一時ディレクトリ
    monkeypatch: Any
        pytest の差し替え

    Returns
    ----------
    None
    """
    # 読み直す列 (状態, 分類の予測, 種類の列, 残した型の列) のある結果
    columns = [
        {"name": "shop", "type": "category", "categories": 2},
        {"name": "units", "type": "number", "integer": True},
    ]
    frame = pd.DataFrame({
        "status": ["ok", "skipped"], "reason": [None, "読めません"],
        "shop": pd.array([1, None], dtype="Int64"),
        "units": pd.array([3, None], dtype="Int64"),
        "prediction": ["1", None], "decision": [None, "予測なし"],
    })
    meta = write(tmp_path / "once", frame, task="classification",
                 output_form="table", table_columns=columns)

    # pandas の CSV の読みと yaml の読みを数える
    reads: list[Any] = []
    loads: list[Any] = []
    original = pd.read_csv
    safe_load = yaml.safe_load

    def counted(*args: Any, **kwargs: Any) -> Any:
        reads.append(args[0] if args else kwargs.get("filepath_or_buffer"))
        return original(*args, **kwargs)

    def counted_load(*args: Any, **kwargs: Any) -> Any:
        loads.append(args)
        return safe_load(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", counted)
    monkeypatch.setattr(yaml, "safe_load", counted_load)
    read = read_frame(tmp_path / "once", meta)
    assert reads == [tmp_path / "once" / "predictions.csv"]
    assert loads == []
    assert values(read["shop"]) == ["1", None]
    assert str(read["units"].dtype) == "Int64"
    assert values(read["prediction"]) == ["1", None]
