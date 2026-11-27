"""
繰り返しの手順の登録簿と, 起動の設定で明示したモジュールの読み込み

研究の自作の手順は, 起動の設定で明示したモジュールだけを読み込む (手元で
任意のコードを読み込まないため). 一式やファイルに書かれたコードは自動では読まない
"""

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Optional

import yaml
from ml_inference import PROCEDURES, procedure_options

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
    登録する. 登録簿はプロセスで 1 つなので, 読み込み済みのモジュールは
    import し直さない (何度呼んでも, 同じプロセスの他のアプリが先に読み
    込んでいても重ねて登録しない)

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
    ValueError
        モジュールを読めない場合
    """
    # 読めないモジュールでの起動の停止 (気付かずに手順の無いまま動かないため)
    loaded: list[str] = []
    for name in names:
        load_module(name)
        loaded.append(name)

    return loaded


def load_module(name: str) -> None:
    """
    1 つのモジュールを import して手順を登録する関数

    Parameters
    ----------
    name: str
        モジュールの名前

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        モジュールを読めない場合 (途中で止まったモジュールが登録した手順は
        登録簿から外す)
    """
    # 読み込み済みのモジュールは sys.modules の前の結果 (登録し直さない)
    before = set(PROCEDURES)
    try:
        importlib.import_module(name)
    except Exception as error:
        # 途中で止まったモジュールの手順の除去 (同じプロセスで直して読み
        # 直したときに, 既にある名前の誤りにならないように). 止まった
        # モジュールは sys.modules から外れる. そのモジュールが読んで読み
        # 終えた別のモジュールは sys.modules に残り, 読み直しで import
        # し直さないので, その手順は外さない
        for added in set(PROCEDURES) - before:
            if stopped(PROCEDURES[added].function):
                PROCEDURES.pop(added, None)
        raise ValueError(
            f"起動の設定の手順のモジュール {name} を読めません: {error}"
        ) from error


def stopped(function: Any) -> bool:
    """
    手順の関数を定義したモジュールが, 読み込みの途中で止まったかを返す関数

    Parameters
    ----------
    function: Any
        登録した手順の関数

    Returns
    ----------
    result: bool
        定義したモジュール (関数の __module__) が sys.modules に無いか
    """
    # import が例外で止まったモジュールは sys.modules から外れるため
    module = getattr(function, "__module__", None)
    result = not isinstance(module, str) or module not in sys.modules

    return result


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
    # 画面で一式の手順と見比べ, 一式を作る欄で選ぶため (設定の名前と既定値
    # の決まりは ML-Inference の procedure_options)
    procedures = []
    for name, item in sorted(PROCEDURES.items()):
        options = procedure_options(name)
        procedures.append({
            "name": name,
            "builtin": bool(item.builtin),
            "description": item.description,
            "start": item.start,
            "time": bool(item.time),
            "options": options["names"],
            "defaults": options["defaults"],
        })

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
