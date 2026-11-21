"""
繰り返しの手順の登録簿と, 起動の設定で明示したモジュールの読み込み

研究の自作の手順は, 起動の設定で明示したモジュールだけを読み込む (ML_tester
の決めたこと 14). 一式やファイルに書かれたコードは自動では読まない
"""

import importlib
import inspect
import math
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from ml_inference import PROCEDURES

# 自作の手順を登録するモジュールを書く環境変数 (, で区切る)
PROCEDURES_ENV = "ML_PROVIDE_PROCEDURES"

# 起動の設定の書き方 (誤りの文に添える案内)
SETTING_HINT = (
    "自作の手順なら, 登録するモジュールを起動の設定 (ml-provide の "
    f"--procedures か環境変数 {PROCEDURES_ENV}) に足してから起動し直して"
    "ください"
)


def module_names(value: Optional[str]) -> list[str]:
    """
    , で区切ったモジュールの名前を並びにする関数

    Parameters
    ----------
    value: Optional[str]
        設定の文字 (無ければ None)

    Returns
    ----------
    names: list[str]
        モジュールの名前 (空と重なりを除く)
    """
    # 空白と空の名前の除去
    names: list[str] = []
    for part in (value or "").split(","):
        name = part.strip()
        if name and name not in names:
            names.append(name)

    return names


def configured_modules(given: Optional[list[str]] = None) -> list[str]:
    """
    起動の設定の手順のモジュールを返す関数

    Parameters
    ----------
    given: Optional[list[str]] = None
        起動の引数で渡した名前 (None なら環境変数から)

    Returns
    ----------
    names: list[str]
        モジュールの名前
    """
    # 引数を優先し, 無ければ環境変数
    if given is not None:
        names = module_names(",".join(given))
        return names
    names = module_names(os.environ.get(PROCEDURES_ENV))

    return names


def load_modules(names: list[str]) -> list[str]:
    """
    起動の設定で明示したモジュールを import する関数

    モジュールは import のときに ml_inference.register_procedure で手順を
    登録する

    Parameters
    ----------
    names: list[str]
        モジュールの名前

    Returns
    ----------
    loaded: list[str]
        読んだモジュールの名前

    Raises
    ----------
    RuntimeError
        モジュールを読めない場合
    """
    # 読めないモジュールでの起動の停止 (気付かずに手順の無いまま動かないため)
    loaded: list[str] = []
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as error:
            raise RuntimeError(
                f"起動の設定の手順のモジュール {name} を読めません: {error}"
            ) from error
        loaded.append(name)

    return loaded


def option_names(function: Any) -> Optional[list[str]]:
    """
    手順の関数の設定 (5 つ目からの引数) の名前を返す関数

    Parameters
    ----------
    function: Any
        登録簿の手順の関数 (model, state, steps, generator, 設定...)

    Returns
    ----------
    names: Optional[list[str]]
        設定の名前 (設定の無い手順は空). **kwargs を持つ関数と, 引数を
        読めない関数は None (画面は名前=値の行で受け取る)
    """
    # 引数を読めない呼べるもの (組み込みの関数など) も一覧を止めない
    try:
        params = list(inspect.signature(function).parameters.values())[4:]
    except (TypeError, ValueError):
        return None
    names: Optional[list[str]] = [p.name for p in params]
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        names = None

    return names


def option_defaults(function: Any) -> dict[str, Any]:
    """
    手順の関数の設定 (5 つ目からの引数) の既定値を返す関数

    Parameters
    ----------
    function: Any
        登録簿の手順の関数

    Returns
    ----------
    defaults: dict[str, Any]
        設定の名前ごとの既定値 (有限の数, 真偽値, 文字の既定値だけ. 引数を
        読めない関数は空)
    """
    # 画面に出せる値の既定値だけ (inf と NaN は JSON にできないので除く)
    try:
        params = list(inspect.signature(function).parameters.values())[4:]
    except (TypeError, ValueError):
        return {}
    defaults = {
        p.name: p.default for p in params
        if isinstance(p.default, (int, float, bool, str))
        and not (isinstance(p.default, float) and not math.isfinite(p.default))
    }

    return defaults


def registered() -> list[dict[str, Any]]:
    """
    登録簿の手順の一覧を返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    procedures: list[dict[str, Any]]
        name, builtin, description, start (始まり. noise は雑音, given は
        渡した番号の並び), time (時刻を渡すか), options (設定の名前. 何でも
        受け取る自作の手順は None), defaults (設定の既定値) (名前の順)
    """
    # 画面で一式の手順と見比べ, 一式を作る欄で選ぶため
    procedures = [
        {
            "name": name,
            "builtin": bool(item.builtin),
            "description": item.description,
            "start": item.start,
            "time": bool(item.time),
            "options": option_names(item.function),
            "defaults": option_defaults(item.function),
        }
        for name, item in sorted(PROCEDURES.items())
    ]

    return procedures


def missing_procedure(path: str) -> Optional[str]:
    """
    一式が登録簿に無い手順を使うかを確かめ, 案内の文を返す関数

    Parameters
    ----------
    path: str
        一式のディレクトリ

    Returns
    ----------
    message: Optional[str]
        登録簿に無い手順の案内 (無ければ None)
    """
    # inference.yaml の procedure の名前だけの読み込み (コードは読まないこと)
    message: Optional[str] = None
    try:
        text = (Path(path) / "inference.yaml").read_text(encoding="utf-8")
        info = yaml.safe_load(text)
    except (OSError, yaml.YAMLError):
        return message
    procedure = info.get("procedure") if isinstance(info, dict) else None
    if not isinstance(procedure, dict) or "name" not in procedure:
        return message
    name = str(procedure["name"])
    if name in PROCEDURES:
        return message
    message = f"手順 {name} は登録簿にありません. {SETTING_HINT}"

    return message
