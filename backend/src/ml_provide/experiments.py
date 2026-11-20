import os
from datetime import datetime
from pathlib import Path
from typing import Any

import ml_quest
import yaml
from fastapi import APIRouter, Request
from ml_quest.handoff import project_of

from ml_provide.api import get_store
from ml_provide.settings import get_problems_dir

router = APIRouter(prefix="/api")

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

# 学習の実験の中の一式のディレクトリの名前 (骨組みの学習のスクリプト)
BUNDLE_DIR = "inference"


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
        problem_path (problem.yaml) の並び (同じプロジェクトは 1 つ)
    """
    # 読めない問題の設定と, 記録の無いものと, 無いディレクトリは飛ばす
    projects: list[dict[str, Any]] = []
    if not problems_dir.is_dir():
        return projects
    seen: set[str] = set()
    for child in sorted(problems_dir.iterdir()):
        file = child / "problem.yaml"
        try:
            if not file.is_file():
                continue
            problem = ml_quest.load_problem(file)
            place = project_of(ml_quest.upgrade_problem(problem))["path"]
        except READ_ERRORS:
            continue
        if not isinstance(place, str) or place in seen or not (
            Path(place).is_dir()
        ):
            continue
        seen.add(place)
        projects.append({"path": place, "problem": str(problem.get("name")),
                         "problem_path": str(file.resolve())})

    return projects


def project_bundles(project: Path, limit: int) -> list[Path]:
    """
    プロジェクトの実験の出力先から, 推論の一式を持つ実験を探す関数

    Parameters
    ----------
    project: Path
        プロジェクトのディレクトリ
    limit: int
        探す数の上限

    Returns
    ----------
    bundles: list[Path]
        一式のディレクトリ (inference.yaml を持つ inference)
    """
    # 出力先の下を決まった深さまで (リンクはたどらない)
    bundles: list[Path] = []
    root = project / OUTPUTS
    if not root.is_dir():
        return bundles
    for current, dirs, _ in os.walk(root):
        place = Path(current)
        depth = len(place.relative_to(root).parts)
        if BUNDLE_DIR in dirs and (
            place / BUNDLE_DIR / "inference.yaml"
        ).is_file():
            bundles.append(place / BUNDLE_DIR)
            if len(bundles) >= limit:
                break
        dirs[:] = sorted(
            name for name in dirs
            if name != BUNDLE_DIR and depth + 1 < RUN_DEPTH
        )

    return bundles


def experiment_row(
    bundle: Path, project: dict[str, Any], registered: dict[str, str]
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

    Returns
    ----------
    row: dict[str, Any]
        path, project, problem, experiment (出力先からの実験の場所),
        modified (一式の更新の時刻), bundle_id (登録済みならその id)
    """
    # 実験の場所は出力先からの相対 (名前/日付/時刻)
    resolved = bundle.resolve()
    run = bundle.parent.relative_to(Path(project["path"]) / OUTPUTS)
    stamp = (bundle / "inference.yaml").stat().st_mtime
    row = {
        "path": str(resolved),
        "project": project["path"],
        "problem": project["problem"],
        "experiment": run.as_posix(),
        "modified": datetime.fromtimestamp(stamp).astimezone().isoformat(),
        "bundle_id": registered.get(str(resolved)),
    }

    return row


@router.get("/experiments")
def list_experiments(request: Request) -> dict[str, Any]:
    """
    問題の設定に記録したプロジェクトの, 推論の一式を持つ実験の一覧を返す
    関数

    Parameters
    ----------
    request: Request
        リクエスト

    Returns
    ----------
    listing: dict[str, Any]
        problems (探した問題の設定の置き場), projects (見たプロジェクトの
        数), experiments (新しい順の行), truncated (上限で打ち切ったか)
    """
    # 登録済みの一式は, 場所と写した元の場所の両方で見分ける
    problems_dir = get_problems_dir()
    registered: dict[str, str] = {}
    for entry in get_store(request).read():
        for key in ("path", "copied_from"):
            if isinstance(entry.get(key), str):
                registered[entry[key]] = str(entry["id"])

    # プロジェクトごとの実験 (読めなくなったものは飛ばす)
    projects = known_projects(problems_dir)
    rows: list[dict[str, Any]] = []
    truncated = False
    for project in projects:
        limit = MAX_EXPERIMENTS - len(rows)
        if limit <= 0:
            truncated = True
            break
        for bundle in project_bundles(Path(project["path"]), limit):
            try:
                rows.append(experiment_row(bundle, project, registered))
            except (OSError, ValueError):
                continue
    truncated = truncated or len(rows) >= MAX_EXPERIMENTS
    rows.sort(key=row_time, reverse=True)
    listing = {"problems": str(problems_dir), "projects": len(projects),
               "experiments": rows, "truncated": truncated}

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
