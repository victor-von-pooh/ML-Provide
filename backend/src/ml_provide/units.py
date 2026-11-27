"""
自作の部品の, 起動の設定で明示したモジュールからの登録

研究の自作の部品は, 起動の設定で明示したモジュールだけを読み込む (手元で
任意のコードを読み込まないため). 一式のモデルの定義に書かれた名前からは何も読み込まない
"""

import os
from typing import Any, Optional

import ml_inference

from ml_provide.procedures import drop_stopped, module_names

# 自作の部品を宣言するモジュールを書く環境変数 (, で区切る)
UNITS_ENV = "ML_PROVIDE_UNITS"

# 起動の設定の書き方 (誤りの文に添える案内)
UNIT_SETTING_HINT = (
    "自作の部品なら, 部品を宣言したモジュールを起動の設定 (ml-provide の "
    f"--units か環境変数 {UNITS_ENV}) に足してから起動し直してください "
    "(一式に書かれた名前からは何も読み込みません)"
)


def configured_units(given: Optional[list[str]] = None) -> list[str]:
    """
    起動の設定の部品のモジュールを返す関数

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
    names = module_names(os.environ.get(UNITS_ENV))

    return names


def load_unit_modules(names: list[str]) -> list[dict[str, Any]]:
    """
    起動の設定で明示したモジュールから自作の部品を登録する関数

    Parameters
    ----------
    names: list[str]
        モジュールの名前

    Returns
    ----------
    modules: list[dict[str, Any]]
        name (モジュールの名前), units (登録した部品の名前)

    Raises
    ----------
    ValueError
        モジュールを読めない, 部品を宣言していない場合
    """
    # 読めないモジュールでの起動の停止 (気付かずに部品の無いまま動かない
    # ため)
    modules: list[dict[str, Any]] = []
    for name in names:
        before = set(ml_inference.PROCEDURES)
        try:
            loaded = ml_inference.register_units([name])
        except Exception as error:
            # 手順も登録するモジュールが途中で止まった場合の, そのモジュール
            # の手順の除去 (手順の読み込みと同じ外し方. 読み直したときに
            # 既にある名前の誤りにならないように)
            drop_stopped(before)
            raise ValueError(
                f"起動の設定の部品のモジュール {name} を読めません: {error}"
            ) from error
        modules.append({"name": name, "units": list(loaded.get(name, []))})

    return modules


def unregistered_of(
    error: BaseException,
) -> Optional[ml_inference.UnregisteredUnitError]:
    """
    例外とその元をたどり, 登録されていない部品の誤りを探す関数

    Parameters
    ----------
    error: BaseException
        起きた例外

    Returns
    ----------
    found: Optional[UnregisteredUnitError]
        見つけた誤り (無ければ None)
    """
    # 包み直した例外の元 (__cause__, __context__) までのたどり
    found: Optional[ml_inference.UnregisteredUnitError] = None
    current: Optional[BaseException] = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, ml_inference.UnregisteredUnitError):
            found = current
            return found
        seen.add(id(current))
        current = current.__cause__ or current.__context__

    return found


def unit_hint(error: BaseException) -> Optional[str]:
    """
    登録されていない部品の誤りに, 起動の設定の案内を添えた文を返す関数

    Parameters
    ----------
    error: BaseException
        起きた例外

    Returns
    ----------
    message: Optional[str]
        部品の名前, ノード, 起動の設定の案内 (他の誤りなら None)
    """
    # 登録されていない部品の誤りだけへの案内
    message: Optional[str] = None
    found = unregistered_of(error)
    if found is None:
        return message
    names = ", ".join(repr(unit) for unit in found.units)
    message = (
        f"この部品 ({names}) は登録されていません (モデルの定義のノード "
        f"{', '.join(found.nodes)}). {UNIT_SETTING_HINT}"
    )

    return message
