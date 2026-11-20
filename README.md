# ML-Provide

学習したモデルで新しいデータを推論し, 予測から結論を出すローカルの Web アプリ.

[ML-Inference](https://github.com/victor-von-pooh/ML-Inference) を使い, 推論の一式 (モデルの定義, 学習した重み, 前処理) を選んで, 新しいデータのファイルか画面に書いた文を推論する. 予測の表と, 判断の内訳, 列ごとのまとめ, 結論の文を見られ, 結論の決め方 (確信の低いものを保留にする, 陽性とする確率, 値の帯, まとめる列) を変えて推論し直さずに出し直せる. 結果は保存して CSV で持ち出せる.

推論の一式は, [ML-Fathom](https://github.com/victor-von-pooh/ML-Fathom) で書き出したデータセット, [ML-CreateArchitecture](https://github.com/victor-von-pooh/ML-CreateArchitecture) で組んだモデルの定義, [ML-Found](https://github.com/victor-von-pooh/ML-Found) の実験から, 画面で作れる.

設計の方針は [docs/DESIGN.md](docs/DESIGN.md) を参照.

## 必要なもの

- [uv](https://docs.astral.sh/uv/) (Python は uv が用意する. 3.10 以上)
- Node.js 24 以上と [pnpm](https://pnpm.io/) 11 以上 (`corepack enable` で `package.json` の版が入る)

## 導入と起動

```bash
git clone --recursive https://github.com/victor-von-pooh/ML-Provide.git
cd ML-Provide
make setup   # submodule, Python と Node.js の依存 (PyTorch を含む. 初めては数分かかる)
make serve   # 画面をビルドして http://127.0.0.1:8769 で起動 (使う間は端末を開いたままにする. Ctrl+C で止める)
```

起動すると `ML-Provide を開く: http://127.0.0.1:8769/api/auth?token=...` の行が出るので, その URL をブラウザで開く (トークンは起動ごとに変わる). 毎回同じトークンにする場合は環境変数 `ML_PROVIDE_TOKEN` で指定する. ポートは `make serve PORT=8800` のように変えられる.

## 使い方

画面ごとの詳しい手順は [操作ガイド](docs/GUIDE.md) (見本の一式で試せる).

1. 「推論の一式」で一式を用意する. 試すだけなら「見本の一式を入れる」で, 見本のデータで学習した 4 つの一式 (解約の予測, 来店数の予測, 日々の売上の予測, レビューの分類) が入る. 一式の画面の「見本の新しいデータを使う」で, すぐに推論できる
    - **学習の結果から作る**: 名前, モデルの定義 (YAML か JSON), データセット (`dataset.yaml` を持つディレクトリ), 実験のディレクトリ (か重みのファイル) を選んで「作る」. 学習用の見本を推論できることを確かめてから作る. 学習のスクリプトが前処理の外で並びの先頭に番号 ([CLS] など) を足した場合や, 目的の値を `(y - 中心) / 幅` にした場合は, 畳んだ欄に書く
    - 前提: モデルは, そのデータセットの前処理の後の値で, その定義から ML-ModelControl で組み立てて学習したもの (ML-Found で学習した実験をそのまま選べばよい). 重みは `state_dict`. 入力が 1 つのモデルは, 表なら学習に使う列, 画像, 文, 点群なら中身の配列を受け取る. 出力が複数のモデルは, 使う出力の名前を畳んだ欄に書く (入力が複数のモデルは ML-Inference の `make_bundle` で作って, 場所で登録する)
    - **場所で登録**: 学習のスクリプトが ML-Inference の `make_bundle` で作った一式 (`inference.yaml` を持つディレクトリ) を選ぶ
2. 一式の画面の「推論する」で, 推論するデータ (学習のときと同じ列のファイルか, 画像などのディレクトリ. 文の一式は文を書いてもよい) と結論の決め方を選んで「推論する」. 目的の列は有っても無くてもよい (有れば結果の表に残るので, 予測と見比べられる). 目的の列の過去の値を使う時系列の一式は「何段も先まで予測する」で, 目的の列が空の先の行を順に予測できる (結果の表の目的の列には予測した値が入る)
3. 推論の画面の「意思決定」で, 量を決める, 対象を選ぶ, 振り分けるの型を選び, 損得と制約を書くと, 結果に行ごとの行動 (足す量, 選ぶか, 回すか) と前提が出る. 目的の列のある過去のデータの結果なら「過去のデータでたどり直す」で, 方針と素朴なやり方の実際の得と損を比べられる. 「外れ方の確かめ」で, 一式の外れ方の記録がその実績に当たっているかを確かめ, 測り直して一式に書き戻せる. 一式の画面の「問題の設定」で, [ML-Pathfinder](https://github.com/victor-von-pooh/ML-Pathfinder) の問題の設定 (problem.yaml) をつなぐと, 意思決定と結論の決め方の既定がその損得になる
4. 列の予測でない一式 (ML-Inference の一式の版 3 で, 出力の形が領域, 点群, ベクトル, 画像, 文の一式) は, 結論が推論した件数だけで, 結論の決め方と意思決定の欄は出ない. 結果の画面で 1 件ごとの出力 (領域と画像は絵, 点群は散布図, ベクトルは形と最初の値) を見て, `.npy` で持ち出せる
5. 結果の画面で, 結論の文, 判断の内訳, まとめる列ごとの表, 予測の表 (予測しなかった行と判断で絞れる) を見る. 決め方を変えて「出し直す」と, 推論し直さずに結論だけが変わる. 「CSV」で予測と判断の表を保存する

止めるときは, 起動した端末で Ctrl+C を押す. 画面は API を使って動くが, API は画面のためのもので, スクリプトから推論する場合は ML-Inference を直接使う.

拡散のサンプリングのような繰り返しの手順を使う一式で, 研究の自作の手順を使う場合は, 手順を `ml_inference.register_procedure` で登録するモジュールを起動の設定に書く (`uv run ml-provide --procedures my_package.procedures` か, 環境変数 `ML_PROVIDE_PROCEDURES=my_package.procedures`. `,` で区切って複数). 書いたモジュールだけを起動のときに読み, 一式やファイルに書かれたコードは読まない. 登録簿に無い手順の一式は, 一式の画面でそう出る.

研究の自作の部品 (ML-ModelControl の `register_unit` で登録する部品) を使う一式も同じく, 部品を宣言したモジュール (`UNITS` か `register_units(register)`. 書き方は ML-ModelControl の README の「自作の部品」) を起動の設定に書く (`uv run ml-provide --units my_package.units` か, 環境変数 `ML_PROVIDE_UNITS=my_package.units`. `,` で区切って複数). 読めないモジュールがあれば起動しない. 一式のモデルの定義に書かれた名前からは何も読み込まず, 登録されていない部品を使う一式は, 部品の名前と起動の設定への足し方が一式の画面, 登録, 画面で作る一式, 推論の誤りに出る. 読んだモジュールと部品は `/api/info` の `unit_modules` で分かる.

推論に使う一式は組み立てたモデルごと 4 つまで覚えて使い回す. 大きなモデルでメモリが足りない場合は環境変数 `ML_PROVIDE_MODEL_CACHE=1` で減らす.

置き場は `~/.ml-provide/` (環境変数 `ML_PROVIDE_HOME` で変えられる). 画面で作った一式と結果を置く. 元のデータ, データセット, 実験のディレクトリは書き換えない.

## 開発

```bash
make dev-backend    # API (変更を自動で読み直す. 開く URL を出す)
make dev-frontend   # 画面 (別の端末で)
make check          # lint, 型検査, テスト
cd frontend && pnpm e2e   # 画面の通しのテスト
```

| ディレクトリ | 内容 |
| --- | --- |
| `backend/` | API (Python, FastAPI). ML-Inference を呼び, 登録した一式と結果を保存する |
| `frontend/` | 画面 (TypeScript, React, Vite) |
| `ML-Inference/` | submodule |
| `docs/DESIGN.md` | 設計の方針と次の段階 |

## ライセンス

MIT
