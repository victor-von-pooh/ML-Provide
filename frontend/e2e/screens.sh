#!/usr/bin/env bash
# README の画面の例を撮るために, 画面の通しのテストと同じ見本でアプリを起動する (CI では使わない).
# 作業用の置き場 (e2e/.screens) をホームにし, 一式の置き場を既定の ~/.ml-provide にする.
# 画面は pnpm build で作っておく. 撮るのは e2e/screens.ts
set -euo pipefail

cd "$(dirname "$0")"
E2E_WORK="$PWD/.screens" E2E_AS_HOME=1 E2E_PORT="${SCREENS_PORT:-8797}" exec bash serve.sh
