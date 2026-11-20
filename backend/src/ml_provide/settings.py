import os
from pathlib import Path

# アプリのデータの置き場を変える環境変数
HOME_ENV = "ML_PROVIDE_HOME"

# 問題の設定の段階のアプリ (ML-Pathfinder) の置き場の環境変数
PATHFINDER_ENV = "ML_PATHFINDER_HOME"


def get_data_dir() -> Path:
    """
    アプリ自身のデータを置くディレクトリを取得する関数

    Parameters
    ----------
    None

    Returns
    ----------
    data_dir: Path
        一式と推論の結果の置き場 (書き込むときに作成)
    """
    # 環境変数があればそれを使い, 無ければホームディレクトリの下に設定
    value = os.environ.get(HOME_ENV)
    if value:
        data_dir = Path(value).expanduser().resolve()
    else:
        data_dir = Path.home() / ".ml-provide"

    return data_dir


def get_problems_dir() -> Path:
    """
    ML-Pathfinder の問題の設定の置き場を取得する関数

    Parameters
    ----------
    None

    Returns
    ----------
    problems_dir: Path
        問題の設定のディレクトリ (ML-Pathfinder の置き場の problems)
    """
    # ML-Pathfinder と同じ環境変数と既定の場所
    value = os.environ.get(PATHFINDER_ENV)
    if value:
        home = Path(value).expanduser().resolve()
    else:
        home = Path.home() / ".ml-pathfinder"
    problems_dir = home / "problems"

    return problems_dir
