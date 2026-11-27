import os
from datetime import datetime
from pathlib import Path
from typing import Any

import ml_quest
import yaml
from fastapi import APIRouter, Request

from ml_provide.api import get_store
from ml_provide.policies import SPEC_FILE, read_policies

router = APIRouter()

# 問題の設定のファイルを読めないときの例外
READ_ERRORS = (OSError, ValueError, TypeError, AttributeError,
               yaml.YAMLError)

# ML-Found の実験の出力先 (プロジェクトのディレクトリから)
OUTPUTS = "outputs"

# 出力先から一式までのディレクトリの数の上限 (<名前>/<日付>/<時刻>/
# inference が 4. 少し深い置き方も拾う)
RUN_DEPTH = 5

# 一覧に出す実験の数の上限 (大きなディレクトリで止まらないように)
MAX_EXPERIMENTS = 500

# 学習の実験の中の一式のディレクトリと, 一式を見分けるファイルの名前
# (骨組みの学習のスクリプト. 推論の一式と方策の一式)
BUNDLE_DIR = "inference"
BUNDLE_FILE = "inference.yaml"
POLICY_DIR = "policy"
POLICY_FILE = SPEC_FILE


def known_projects(problems_dir: Path) -> list[dict[str, Any]]:
    """
    問題の設定の置き場から, 書き出したプロジェクトの記録を集める関数

    Parameters
    ----------
    problems_dir: Path
        ML-Pathfinder の問題の設定の置き場 (problems)

    Returns
    ----------
    projects: list[dict[str, Any]]
        path (プロジェクトのディレクトリ), problem (問題の設定の名前),
        problem_path (problem.yaml), problem_ids (そのプロジェクトを記録
        した問題の設定の識別子. 置き場のディレクトリの名前) の並び (同じ
        プロジェクトは 1 つ)
    """
    # 読めない問題の設定と, 記録の無いものと, 無いディレクトリは飛ばす
    projects: list[dict[str, Any]] = []
    if not problems_dir.is_dir():
        return projects
    seen: dict[str, dict[str, Any]] = {}
    for child in sorted(problems_dir.iterdir()):
        file = child / "problem.yaml"
        try:
            if not file.is_file():
                continue
            problem = ml_quest.load_problem(file)
            upgraded = ml_quest.upgrade_problem(problem)
            place = ml_quest.project_of(upgraded)["path"]
        except READ_ERRORS:
            continue
        if not isinstance(place, str) or not Path(place).is_dir():
            continue
        # 同じプロジェクトを記録した別の問題の設定は, 識別子だけを足す
        # (殻で選んだどちらの問題の設定でも, その実験に絞れるように)
        if place in seen:
            seen[place]["problem_ids"].append(child.name)
            continue
        name = problem.get("name")
        seen[place] = {
            "path": place,
            "problem": name if isinstance(name, str) and name.strip()
            else file.parent.name,
            "problem_path": str(file.resolve()),
            "problem_ids": [child.name]
        }
        projects.append(seen[place])

    return projects


def project_bundles(
    project: Path, limit: int, folder: str = BUNDLE_DIR,
    marker: str = BUNDLE_FILE
) -> list[Path]:
    """
    プロジェクトの実験の出力先から, 一式を持つ実験を探す関数

    Parameters
    ----------
    project: Path
        プロジェクトのディレクトリ
    limit: int
        探す数の上限
    folder: str = BUNDLE_DIR
        実験の中の一式のディレクトリの名前
    marker: str = BUNDLE_FILE
        一式を見分けるファイルの名前

    Returns
    ----------
    bundles: list[Path]
        一式のディレクトリ (marker を持つ folder. 既定は inference.yaml を
        持つ inference)
    """
    # 出力先の下を決まった深さまで, 名前の新しい順 (日付と時刻の
    # ディレクトリ) に. 上限で打ち切っても新しい実験が残るように. リンク
    # はたどらず, リンクの一式も拾わない
    bundles: list[Path] = []
    root = project / OUTPUTS
    if not root.is_dir():
        return bundles
    for current, dirs, _ in os.walk(root):
        place = Path(current)
        depth = len(place.relative_to(root).parts)
        bundle = place / folder
        # 一式と判定した所だけ下りない (実験の名前が folder と同じでも
        # 下りられるように)
        skip = ""
        if folder in dirs and not bundle.is_symlink() and (
            bundle / marker
        ).is_file():
            bundles.append(bundle)
            skip = folder
            if len(bundles) >= limit:
                break
        dirs[:] = sorted(
            (name for name in dirs
             if name != skip and depth + 1 < RUN_DEPTH),
            reverse=True
        )

    return bundles


def experiment_row(
    bundle: Path, project: dict[str, Any], registered: dict[str, str],
    marker: str = BUNDLE_FILE, key: str = "bundle_id"
) -> dict[str, Any]:
    """
    一覧の 1 行 (実験の一式) を作る関数

    Parameters
    ----------
    bundle: Path
        一式のディレクトリ
    project: dict[str, Any]
        プロジェクトの記録 (known_projects の 1 つ)
    registered: dict[str, str]
        登録した一式の場所 (と写した元の場所) から id
    marker: str = BUNDLE_FILE
        一式を見分けるファイルの名前 (更新の時刻を読む)
    key: str = "bundle_id"
        登録済みの id の項目の名前

    Returns
    ----------
    row: dict[str, Any]
        path, project, problem, problem_ids (プロジェクトを記録した問題
        の設定の識別子), experiment (出力先からの実験の場所), modified
        (一式の更新の時刻), key (登録済みならその id)
    """
    # 実験の場所は出力先からの相対 (名前/日付/時刻)
    resolved = bundle.resolve()
    run = bundle.parent.relative_to(Path(project["path"]) / OUTPUTS)
    stamp = (bundle / marker).stat().st_mtime
    row = {
        "path": str(resolved),
        "project": project["path"],
        "problem": project["problem"],
        "problem_ids": list(project["problem_ids"]),
        "experiment": run.as_posix(),
        "modified": datetime.fromtimestamp(stamp).astimezone().isoformat(),
        key: registered.get(str(resolved)),
    }

    return row


def collect_rows(
    problems_dir: Path, registered: dict[str, str], folder: str,
    marker: str, key: str
) -> dict[str, Any]:
    """
    問題の設定に記録したプロジェクトの, 一式を持つ実験の一覧を作る関数

    Parameters
    ----------
    problems_dir: Path
        ML-Pathfinder の問題の設定の置き場
    registered: dict[str, str]
        登録した一式の場所から id
    folder: str
        実験の中の一式のディレクトリの名前
    marker: str
        一式を見分けるファイルの名前
    key: str
        登録済みの id の項目の名前

    Returns
    ----------
    listing: dict[str, Any]
        problems (探した問題の設定の置き場), projects (見たプロジェクトの
        数), experiments (新しい順の行), truncated (上限で打ち切ったか)
    """
    # プロジェクトごとの実験 (読めなくなったものは飛ばす)
    projects = known_projects(problems_dir)
    rows: list[dict[str, Any]] = []
    truncated = False
    for project in projects:
        limit = MAX_EXPERIMENTS - len(rows)
        if limit <= 0:
            truncated = True
            break
        # 1 つ多く探して, 上限を超えたかを見分ける
        found = project_bundles(
            Path(project["path"]), limit + 1, folder, marker
        )
        if len(found) > limit:
            truncated = True
            found = found[:limit]
        for bundle in found:
            try:
                rows.append(experiment_row(
                    bundle, project, registered, marker, key
                ))
            except (OSError, ValueError):
                continue
    rows.sort(key=row_time, reverse=True)
    listing = {"problems": str(problems_dir), "projects": len(projects),
               "experiments": rows, "truncated": truncated}

    return listing


@router.get("/experiments")
def list_experiments(request: Request) -> dict[str, Any]:
    """
    問題の設定に記録したプロジェクトの, 推論の一式を持つ実験の一覧を返す
    関数

    Parameters
    ----------
    request: Request
        リクエスト (request.app は置き場と問題の設定の置き場を持つ)

    Returns
    ----------
    listing: dict[str, Any]
        problems (探した問題の設定の置き場), projects (見たプロジェクトの
        数), experiments (新しい順の行. bundle_id), truncated (上限で
        打ち切ったか)
    """
    # 登録済みの一式は, 場所と写した元の場所の両方で見分ける
    registered: dict[str, str] = {}
    for entry in get_store(request).read():
        for key in ("path", "copied_from"):
            if isinstance(entry.get(key), str):
                registered[entry[key]] = str(entry["id"])
    problems_dir: Path = request.app.state.problems_dir
    listing = collect_rows(
        problems_dir, registered, BUNDLE_DIR, BUNDLE_FILE, "bundle_id"
    )

    return listing


@router.get("/experiments/policies")
def list_policy_experiments(request: Request) -> dict[str, Any]:
    """
    問題の設定に記録したプロジェクトの, 方策の一式 (policy.json を持つ
    policy) を持つ実験の一覧を返す関数

    Parameters
    ----------
    request: Request
        リクエスト (request.app は置き場と問題の設定の置き場を持つ)

    Returns
    ----------
    listing: dict[str, Any]
        problems (探した問題の設定の置き場), projects (見たプロジェクトの
        数), experiments (新しい順の行. policy_id), truncated (上限で
        打ち切ったか)
    """
    # 登録済みの方策の一式は場所で見分ける
    registered = {
        str(entry["path"]): str(entry["id"])
        for entry in read_policies(get_store(request))
        if isinstance(entry.get("path"), str)
    }
    problems_dir: Path = request.app.state.problems_dir
    listing = collect_rows(
        problems_dir, registered, POLICY_DIR, POLICY_FILE, "policy_id"
    )

    return listing


def row_time(row: dict[str, Any]) -> datetime:
    """
    一覧の行を新しい順に並べるための時刻を返す関数

    Parameters
    ----------
    row: dict[str, Any]
        一覧の行

    Returns
    ----------
    stamp: datetime
        一式の更新の時刻 (タイムゾーン付き)
    """
    # 時差の違う行も時刻で比べる
    stamp = datetime.fromisoformat(str(row["modified"]))

    return stamp
