import os
from pathlib import Path

# アプリのデータの置き場を変える環境変数
HOME_ENV = "ML_PROVIDE_HOME"


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
