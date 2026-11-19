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

# 問題の設定の見本 (総菜の作る量) を, 日々の売上の見本の列 (日付, 店, 売上) に合わせて保存する.
# 受け持ちを作るにした問題の設定も並べて保存する
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

# 推論の一式の版 3 (列の出力でない一式. 領域とベクトル)
uv run python ../frontend/e2e/make_v3.py "$WORK/v3"

# 決まったトークンで起動
ML_PROVIDE_HOME="$WORK/home" ML_PROVIDE_TOKEN=e2e-token \
  exec uv run ml-provide --port "$PORT"
