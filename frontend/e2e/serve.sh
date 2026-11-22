#!/usr/bin/env bash
# 画面の通しのテスト (Playwright) 用に, 作業用の置き場でアプリを起動する.
# 画面は pnpm build で作っておく. README の画面の例を撮るときは e2e/screens.sh から,
# 作業用の置き場の場所 (E2E_WORK) を変え, そこをホーム (E2E_AS_HOME) にして起動する
set -euo pipefail

cd "$(dirname "$0")"
WORK="${E2E_WORK:-$PWD/.work}"
E2E_DIR="$PWD"
PORT="${E2E_PORT:-8794}"

# 前回の結果を消し, 置き場を作業用のディレクトリにする
rm -rf "$WORK"
mkdir -p "$WORK"
cd ../../backend

# 問題の設定の見本 (総菜の作る量) を, 日々の売上の見本の列 (日付, 店, 売上) に合わせて保存する.
# 今のやり方は前の週の同じ曜日と同じ量 (7 期前). 受け持ちを作るにした問題の設定も並べて保存する
uv run python - "$WORK/problem" <<'PY'
import sys

import ml_quest
from ml_quest.examples import example_problem

problem = example_problem("総菜の作る量")
for key, value in (
    ("information.columns", ["date", "shop", "sales"]),
    ("ml.parts.0.series_by", ["shop"]),
    ("information.known", ["shop"]),
    ("information.unknown", None),
    ("action.current", {"type": "lag", "periods": 7}),
):
    problem = ml_quest.answer(problem, key, value)
ml_quest.save_problem(problem, sys.argv[1])

# 受け持ちを作る (画像) にした問題の設定 (推論の結論はまだ扱わない)
other = example_problem("総菜の作る量")
other["ml"]["parts"] = [{}]
for key in ("decision", "basis", "current"):
    other["action"].pop(key, None)
for key, value in (
    ("role", "generate"), ("what", "品の見本の画像"), ("item", "1 枚"),
    ("baseline", "手で撮る"), ("unit", "pair"), ("output", "image"),
    ("target.source", "files"), ("method", "finetune"),
    ("target.kind", "image"), ("evaluation.metric", "fid"),
    ("evaluation.link", "見本が自然なほど撮る手間が減る"),
):
    other = ml_quest.answer(other, f"ml.parts.0.{key}", value)
ml_quest.save_problem(other, sys.argv[1] + "-generate")
PY

# ML-Pathfinder の置き場の問題の設定と, そこから書き出した骨組みのプロジェクト.
# 実験 (outputs/<名前>/<日付>/<時刻>) は, 学習のスクリプトのように問題の設定の写しと
# 推論の一式 (inference. 日々の売上の見本) を持つ. 写しは元の場所 (origin) を持つ
uv run python - "$WORK" <<'PY'
import shutil
import sys
from pathlib import Path

import ml_quest
from ml_provide.samples import SAMPLES, write_sample

work = Path(sys.argv[1])
problem = ml_quest.load_problem(work / "problem")
store = work / "pathfinder" / "problems" / "p1"
project = work / "project"
ml_quest.save_problem(problem, store)
ml_quest.make_project(problem, project, origin=store)
ml_quest.save_problem(ml_quest.record_project(problem, project), store)
name = next(key for key in SAMPLES if "日々の売上" in key)
bundle, _, _ = write_sample(work / "experiment-source", name, 3, 0)
run = project / "outputs" / "sales" / "2026-01-02" / "09-00-00"
shutil.copytree(bundle, run / "inference")
shutil.copy(project / "problem.yaml", run / "problem.yaml")
PY

# 推論の一式の版 3 (列の出力でない一式. 領域とベクトル)
uv run python ../frontend/e2e/make_v3.py "$WORK/v3"

# 版 2 のデータセット (画像と領域, 撮影と点群) と, 見本の定義を 2 エポック学習した実験.
# 骨組みのプロジェクトの実験の中にも, 学習のスクリプトのように推論の一式 (版 4) を作る
uv run python ../frontend/e2e/make_items.py "$WORK/items" "$WORK/project"

# 方策の問題の設定 (棒を立てる制御) とそこから書き出した骨組みのプロジェクトを別に作り,
# その実験の中に, 方策の骨組みの学習のように方策の一式 (policy) を置く.
# 実験の一覧に出る名前のためのもので, 問題の設定の中身は総菜の見本の名前だけを変えたもの
uv run python - "$WORK" <<'PY'
import sys
from pathlib import Path

import ml_quest

work = Path(sys.argv[1])
problem = ml_quest.load_problem(work / "problem")
problem["name"] = "棒を立てる制御"
store = work / "pathfinder" / "problems" / "p2"
project = work / "policy-project"
ml_quest.save_problem(problem, store)
ml_quest.make_project(problem, project, origin=store)
ml_quest.save_problem(ml_quest.record_project(problem, project), store)
PY
uv run python ../frontend/e2e/make_policy.py "$WORK/policy-project"

# 拡散のモデルの雑音の予測器の定義と, 少しだけ学習した実験 (雑音だけの一式の元)
uv run python ../frontend/e2e/make_noise.py "$WORK/noise"

# 決まったトークンで起動. 自作の部品は起動の設定 (環境変数) で明示した
# 見本のモジュール (e2e/e2e_units.py) だけを読む
export PYTHONPATH="$E2E_DIR${PYTHONPATH:+:$PYTHONPATH}" ML_PROVIDE_UNITS=e2e_units \
  ML_PATHFINDER_HOME="$WORK/pathfinder" ML_PROVIDE_TOKEN=e2e-token
if [ -n "${E2E_AS_HOME:-}" ]; then
  # 作業用の置き場をホームにし, 一式の置き場を既定の ~/.ml-provide にする
  # (画面の例に手元の場所が写らないように. uv の置き場が変わらないよう venv を直に呼ぶ)
  unset ML_PROVIDE_HOME
  HOME="$WORK" exec .venv/bin/ml-provide --port "$PORT"
fi
ML_PROVIDE_HOME="$WORK/home" exec uv run ml-provide --port "$PORT"
