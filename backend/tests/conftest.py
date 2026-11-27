from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def pathfinder_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    ML-Pathfinder の置き場を一時ディレクトリにする関数

    アプリは作ったときに問題の設定の置き場を決めるので, アプリを作る
    fixture より先に環境変数を置く (手元の置き場を読まないためにも)

    Parameters
    ----------
    tmp_path: Path
        一時ディレクトリ
    monkeypatch: pytest.MonkeyPatch
        環境変数の差し替え

    Returns
    ----------
    home: Path
        一時ディレクトリの下の pathfinder (無いこともある)
    """
    home = tmp_path / "pathfinder"
    monkeypatch.setenv("ML_PATHFINDER_HOME", str(home))

    return home
