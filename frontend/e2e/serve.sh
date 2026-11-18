#!/usr/bin/env bash
# 画面の通しのテスト (Playwright) 用に, 作業用の置き場でアプリを起動する.
# 画面は pnpm build で作っておく
set -euo pipefail

cd "$(dirname "$0")"
WORK="$PWD/.work"
PORT="${E2E_PORT:-8794}"

# 前回の結果を消し, 置き場を作業用のディレクトリにする
rm -rf "$WORK"
mkdir -p "$WORK"
cd ../../backend

# 問題の設定の見本 (総菜の作る量) を, 日々の売上の見本の列 (日付, 店, 売上) に合わせて保存する
uv run python - "$WORK/problem" <<'PY'
import sys

import ml_quest
from ml_quest.examples import example_problem

problem = example_problem("総菜の作る量")
for key, value in (
    ("information.columns", ["date", "shop", "sales"]),
    ("ml.series_by", ["shop"]),
    ("information.known", ["shop"]),
    ("information.unknown", None),
):
    problem = ml_quest.answer(problem, key, value)
ml_quest.save_problem(problem, sys.argv[1])
PY

# 決まったトークンで起動
ML_PROVIDE_HOME="$WORK/home" ML_PROVIDE_TOKEN=e2e-token \
  exec uv run ml-provide --port "$PORT"
