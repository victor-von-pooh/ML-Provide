"""
繰り返しの手順の登録簿と, 起動の設定で明示したモジュールの読み込み

研究の自作の手順は, 起動の設定で明示したモジュールだけを読み込む (ML_tester
の決めたこと 14). 一式やファイルに書かれたコードは自動では読まない
"""

import importlib
import os
from pathlib import Path
from typing import Any, Optional

import yaml
from ml_inference import PROCEDURES

# 自作の手順を登録するモジュールを書く環境変数 (, で区切る)
PROCEDURES_ENV = "ML_PROVIDE_PROCEDURES"

# 起動の設定の書き方 (誤りの文に添える)
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
    # 読めないモジュールは起動を止める (気付かずに手順の無いまま動かない)
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


def registered() -> list[dict[str, Any]]:
    """
    登録簿の手順の一覧を返す関数

    Parameters
    ----------
    None

    Returns
    ----------
    procedures: list[dict[str, Any]]
        name, builtin, description (名前の順)
    """
    # 画面で一式の手順と見比べるため
    procedures = [
        {
            "name": name,
            "builtin": bool(item.builtin),
            "description": item.description,
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
    # inference.yaml の procedure の名前だけを読む (コードは読まない)
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
