"""
学習の結果 (実験) から登録した一式の, 元のデータセットのテスト用の分割を
見つける

一式から実験 (run.json を持つディレクトリ), 実験の config.yaml の
data.dataset, その dataset.yaml の splits.test の順にたどる. 画面からは
場所を受け取らず, 推論でも同じ道で見つけ直す
"""

from pathlib import Path
from typing import Any, Optional, Union

import yaml

from ml_provide.cache import read_bundle

# 推論するデータに選べる分割の名前 (学習に使わなかった分割)
TEST_SPLIT = "test"

# 一式から実験までのディレクトリの数 (一式自身, 親, その親. 学習の
# スクリプトは実験の中の inference に一式を作る)
RUN_LEVELS = 3

# 実験から上にプロジェクト (configs/default/train.yaml を持つ) を探す数
# の上限 (出力先からの名前/日付/時刻と少し深い置き方, 出力先の 1 つ)
PROJECT_LEVELS = 8

# 読む config.yaml, task.yaml, dataset.yaml の大きさの上限
MAX_YAML_BYTES = 1024 * 1024

# プロジェクトを見分けるファイル (ML-Found のプロジェクトの形)
PROJECT_MARKER = Path("configs") / "default" / "train.yaml"

# 推論できる一式の種類 (表と時系列の版 1 のデータセットの一式)
TABLE_KINDS = ("table", "timeseries")


def bundle_source(entry: dict[str, Any]) -> Path:
    """
    実験をたどる一式の場所を返す関数

    Parameters
    ----------
    entry: dict[str, Any]
        一式の登録

    Returns
    ----------
    place: Path
        置き場に写した一式は写した元 (copied_from), 他は登録した場所
    """
    # 写しの元の実験の場所
    copied = entry.get("copied_from")
    place = Path(copied if isinstance(copied, str) else str(entry["path"]))

    return place


def find_run(bundle: Path) -> Optional[Path]:
    """
    一式を持つ実験のディレクトリを返す関数

    Parameters
    ----------
    bundle: Path
        一式のディレクトリ

    Returns
    ----------
    run: Optional[Path]
        一式自身か上の 2 つのうち, 普通のファイルの run.json を持つ最も
        近いディレクトリ (無ければ None)
    """
    # リンクの run.json の除外 (書き戻しの保護の in_experiment は広く見る)
    resolved = bundle.resolve()
    for place in (resolved, *resolved.parents[:RUN_LEVELS - 1]):
        marker = place / "run.json"
        if marker.is_file() and not marker.is_symlink():
            return place

    return None


def find_project(run: Path) -> Optional[Path]:
    """
    実験を持つプロジェクトのディレクトリを返す関数

    Parameters
    ----------
    run: Path
        実験のディレクトリ

    Returns
    ----------
    project: Optional[Path]
        上にたどって最初の, configs/default/train.yaml を持つディレクトリ
        (無ければ None)
    """
    # 上限の階層までの探索 (リンクの印は除外)
    for place in run.parents[:PROJECT_LEVELS]:
        marker = place / PROJECT_MARKER
        if marker.is_file() and not marker.is_symlink():
            return place

    return None


def read_yaml(file: Path) -> Any:
    """
    大きさの上限の中の, 普通のファイルの YAML を読む関数

    Parameters
    ----------
    file: Path
        YAML のファイル

    Returns
    ----------
    data: Any
        読んだ中身 (無い, リンク, 普通でない, 大きすぎる, 読めない場合は
        None)
    """
    # リンクと大きなファイルの除外
    try:
        if file.is_symlink() or not file.is_file():
            return None
        if file.stat().st_size > MAX_YAML_BYTES:
            return None
        data = yaml.safe_load(file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return None

    return data


def recorded_dataset(run: Path) -> Optional[str]:
    """
    実験に記録した学習のデータセットの場所を返す関数

    Parameters
    ----------
    run: Path
        実験のディレクトリ

    Returns
    ----------
    place: Optional[str]
        config.yaml の data.dataset (無ければ task.yaml の dataset.path.
        書いた文字のまま. どちらも無ければ None)
    """
    # 学習に使った config の値の優先
    config = read_yaml(run / "config.yaml")
    data = config.get("data") if isinstance(config, dict) else None
    place = data.get("dataset") if isinstance(data, dict) else None
    if isinstance(place, str) and place.strip():
        return place
    task = read_yaml(run / "task.yaml")
    dataset = task.get("dataset") if isinstance(task, dict) else None
    place = dataset.get("path") if isinstance(dataset, dict) else None
    if isinstance(place, str) and place.strip():
        return place

    return None


def inside(place: Path, root: Path) -> bool:
    """
    リンクをたどった後の場所が, ディレクトリの中かを返す関数

    Parameters
    ----------
    place: Path
        確かめる場所
    root: Path
        ディレクトリ (たどった後の場所)

    Returns
    ----------
    result: bool
        中なら True
    """
    # たどった後の場所での比べ
    result = place.resolve().is_relative_to(root)

    return result


def split_bundle(path: Path) -> Optional[str]:
    """
    一式が分割を推論できる一式かを確かめる関数

    Parameters
    ----------
    path: Path
        一式のディレクトリ

    Returns
    ----------
    reason: Optional[str]
        推論できなければ理由 (できれば None)
    """
    # 表と時系列の版 1 のデータセットの列の推論の一式だけ (推論のときは
    # ML-Inference も確かめる)
    try:
        bundle = read_bundle(str(path))
    except Exception:
        return "一式を読めません"
    special = (
        bundle.from_rank or bundle.from_search or bundle.from_group
        or bundle.from_anomaly or bundle.from_uplift or bundle.from_price
        or bundle.from_demand or bundle.from_items or bundle.from_ready
        or bundle.from_noise
    )
    if special or bundle.kind not in TABLE_KINDS:
        return "テスト用の分割を選べるのは, 表と時系列の一式だけです"

    return None


def search_test_split(entry: dict[str, Any]) -> dict[str, Any]:
    """
    一式の元のデータセットのテスト用の分割を見つける関数

    Parameters
    ----------
    entry: dict[str, Any]
        一式の登録

    Returns
    ----------
    found: dict[str, Any]
        split (分割の名前. 見つからなければ None), dataset (データセット
        のたどった後の場所), rows (分割の行の数), experiment (実験の
        場所), reason (見つからない理由. 見つかれば None)
    """
    # 一式, 実験, プロジェクトの順
    found: dict[str, Any] = {
        "split": None, "dataset": None, "rows": None, "experiment": None,
        "reason": None,
    }
    reason = split_bundle(Path(str(entry["path"])))
    if reason:
        found["reason"] = reason
        return found
    run = find_run(bundle_source(entry))
    if run is None:
        found["reason"] = "学習の結果 (実験) の中の一式ではありません"
        return found
    found["experiment"] = str(run)
    written = recorded_dataset(run)
    if written is None:
        found["reason"] = (
            "実験の config.yaml に学習のデータセット (data.dataset) が"
            "ありません"
        )
        return found
    project = find_project(run)
    if project is None:
        found["reason"] = "実験を持つ学習のプロジェクトが見つかりません"
        return found

    # データセットはプロジェクトの中だけ (書いた場所の .. やリンクで外を
    # 指すものは選ばない)
    project = project.resolve()
    place = Path(written).expanduser()
    root = (place if place.is_absolute() else project / place).resolve()
    if not root.is_relative_to(project):
        found["reason"] = (
            "学習のデータセットがプロジェクトの外を指すので選べません"
        )
        return found
    info = read_yaml(root / "dataset.yaml")
    if not isinstance(info, dict):
        found["reason"] = (
            "学習のデータセットの dataset.yaml を読めません"
        )
        return found
    if info.get("format_version") != 1:
        found["reason"] = (
            "テスト用の分割を選べるのは, 版 1 のデータセットだけです"
        )
        return found

    # 分割のファイルはデータセットの中の普通のファイル
    splits = info.get("splits")
    test = splits.get(TEST_SPLIT) if isinstance(splits, dict) else None
    if not isinstance(test, dict):
        test = {}
    file = test.get("file")
    if not isinstance(file, str) or not file:
        found["reason"] = "学習のデータセットにテスト用の分割がありません"
        return found
    relative = Path(file)
    if relative.is_absolute() or ".." in relative.parts or not inside(
        root / relative, root
    ) or not (root / relative).is_file():
        found["reason"] = "テスト用の分割のファイルを読めません"
        return found
    rows = test.get("rows")
    counted = isinstance(rows, int) and not isinstance(rows, bool)
    found.update(split=TEST_SPLIT, dataset=str(root),
                 rows=rows if counted else None)

    return found


def split_source(entry: dict[str, Any], name: str) -> dict[str, str]:
    """
    推論に渡す分割の辞書を, 一式から見つけ直して返す関数

    Parameters
    ----------
    entry: dict[str, Any]
        一式の登録
    name: str
        画面で選んだ分割の名前

    Returns
    ----------
    source: dict[str, str]
        ML-Inference の分割の入力 ({"dataset": 場所, "split": 名前})

    Raises
    ----------
    ValueError
        選べない名前と, 見つからない場合 (理由付き)
    """
    # 画面からは名前だけ (場所は受け取らない)
    if name != TEST_SPLIT:
        raise ValueError(f"選べる分割は {TEST_SPLIT} だけです")
    found = find_test_split(entry)
    if found["split"] is None:
        raise ValueError(str(found["reason"]))
    source = {"dataset": str(found["dataset"]), "split": TEST_SPLIT}

    return source


def result_source(
    meta: dict[str, Any]
) -> Optional[Union[Path, dict[str, str]]]:
    """
    結果を読み直すときの入力を返す関数

    Parameters
    ----------
    meta: dict[str, Any]
        結果の result.yaml

    Returns
    ----------
    source: Optional[Any]
        分割の結果は分割の辞書 (データセットが無ければ None), 他は入力の
        ファイルの場所 (無ければ None)
    """
    # 分割は記録した場所と名前からの作り直し (結果はサーバだけが書くので,
    # 記録した場所をそのまま使う. 説明とレシピは ML-Inference が照合)
    split = meta.get("split")
    if isinstance(split, dict):
        place, name = split.get("dataset"), split.get("name")
        if isinstance(place, str) and isinstance(name, str) and (
            Path(place) / "dataset.yaml"
        ).is_file():
            return {"dataset": place, "split": name}
        return None
    path = meta.get("source_path")
    if isinstance(path, str) and Path(path).exists():
        return Path(path)

    return None


def find_test_split(entry: dict[str, Any]) -> dict[str, Any]:
    """
    一式の元のデータセットのテスト用の分割を見つけ, たどれない場所を理由に
    する関数

    Parameters
    ----------
    entry: dict[str, Any]
        一式の登録

    Returns
    ----------
    found: dict[str, Any]
        search_test_split の値. 場所をたどれない場合 (~ の利用者が無い,
        リンクの輪, NUL の字, 読む権限が無いなど) は split を空にし理由
    """
    # 記録の値 1 つで一式の画面が止まらないように, 理由にして返す
    try:
        found = search_test_split(entry)
    except (OSError, RuntimeError, ValueError) as error:
        found = {
            "split": None, "dataset": None, "rows": None,
            "experiment": None,
            "reason": f"学習のデータセットの場所をたどれません ({error})",
        }

    return found


def split_key(meta: dict[str, Any]) -> Optional[str]:
    """
    結果の入力の分割を, 比べるための文字にする関数

    Parameters
    ----------
    meta: dict[str, Any]
        結果の result.yaml

    Returns
    ----------
    key: Optional[str]
        分割の結果は「データセットの場所 の 名前」, 他と記録の欠けた結果は
        None
    """
    # 同じデータセットの同じ分割なら同じ入力
    split = meta.get("split")
    if not isinstance(split, dict):
        return None
    place, name = split.get("dataset"), split.get("name")
    if not isinstance(place, str) or not isinstance(name, str):
        return None
    key = f"{place} の {name}"

    return key
