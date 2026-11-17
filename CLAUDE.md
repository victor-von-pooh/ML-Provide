# CLAUDE.md

ML-Provide で作業するときの方針をまとめる.

## このリポジトリの目的

ML-Inference を使って, 学習したモデルで新しいデータを推論し, 予測から結論を出すローカルの Web アプリ. 設計は [docs/DESIGN.md](docs/DESIGN.md) にまとめる.

- ML-Inference は submodule. 推論, 結論, 推論の一式の形など, GUI 以外でも使う仕組みは ML-Inference に足す. 画面と API に固有のものはこのリポジトリに置く
- 構成と道具は ML-Dashboard, ML-CreateArchitecture, ML-Fathom にそろえる (将来の統合のため)

## コードの書き方

- Python: [ML-Inference/CODING_STYLE.md](ML-Inference/CODING_STYLE.md) に従う (日本語の NumPy 形式の docstring, コメントは体言止め, 79 文字以内など). Ruff と mypy (strict) を通す
- TypeScript: Prettier と oxlint に従い, `strict` で型検査を通す. コメントは日本語で書く
- ドキュメントとコードのコメントでは, 「機械学習を全て GUI 上で行う」という目的のために開発しているリポジトリ (今は ML-Found, ML-Dashboard, ML-ModelControl, ML-CreateArchitecture, ML-EDA, ML-Fathom, ML-Inference, ML-Provide, ML-Quest, ML-Pathfinder, ML_tester) 以外に触れない. ローカルのディレクトリ構成やパスも書かない

## 進める流れ

1. 方針を変える場合や新しい機能を加える場合は, 先に docs/DESIGN.md を直す
2. `make check` を通し, 実際に起動して画面を確かめる
3. **セルフレビュー**: 複数のエージェントを並行して動かし, バックエンド (バグ, パスの扱いとセキュリティ), 画面 (バグ, 状態の扱い, アクセシビリティ, 型), README の手順だけで起動できるかを確認する
4. 指摘を直し, 直した内容と残した指摘をユーザーに報告する

## git

- 意味のある変更ごとにブランチを切って PR を作る. `main` に直接プッシュしない
- CI が通ったことを確かめてからマージする. 自分の PR は GitHub の仕様で Approve できないので, 確かめた内容をレビューのコメントに残してからマージする
- コミットとプッシュ, PR のマージは基本的にユーザーが行う. ユーザーが `[Commit] <日時>までなら...` の prefix を付けて指示した場合は, その日時まで Claude の判断で行ってよい
- コミットメッセージと PR のタイトルは日本語の 1 行
