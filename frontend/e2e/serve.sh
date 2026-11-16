#!/usr/bin/env bash
# 画面の通しのテスト (Playwright) 用に, 作業用の置き場でアプリを起動する.
# 画面は pnpm build で作っておく
set -euo pipefail

cd "$(dirname "$0")"
WORK="$PWD/.work"
PORT="${E2E_PORT:-8794}"

# 前回の結果を消し, 置き場を作業用のディレクトリにして決まったトークンで起動
rm -rf "$WORK"
mkdir -p "$WORK"
cd ../../backend
ML_PROVIDE_HOME="$WORK/home" ML_PROVIDE_TOKEN=e2e-token \
  exec uv run ml-provide --port "$PORT"
