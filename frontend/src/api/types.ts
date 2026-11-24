/** ディレクトリの中身の 1 つ */
export interface Entry {
  name: string
  path: string
  is_dir: boolean
  size: number | null
  bundle: boolean
}

export interface Listing {
  path: string
  parent: string | null
  home: string
  entries: Entry[]
  truncated: boolean
}

/** 値の帯 (最後の帯は上限無し) */
export interface Band {
  label: string
  upper?: number | null
}

/** 結論の決め方 */
export interface Rules {
  min_confidence?: number | null
  positive?: string | null
  threshold?: number | null
  bands?: Band[] | null
  group_by?: string | null
}

export interface InputPlan {
  source: 'features' | 'items'
  prepend: number[]
}

/** 推論を取った実験の状態 */
export interface RunInfo {
  path: string
  status?: string | null
  monitor?: string | null
  best_value?: number | null
  best_epoch?: number | null
  final_weights?: string | null
  finished_at?: string | null
}

/** ML-Inference の describe_bundle */
export interface Description {
  path: string
  name: string | null
  created_at: string | null
  kind: string
  task: string
  target: string | null
  classes: string[]
  features: string[]
  text_column: string | null
  reading: { loader: string | null; options: Record<string, unknown> }
  steps: string[]
  item_steps: string[]
  model: {
    name: string | null
    inputs: Record<string, InputPlan>
    output: string | null
    kind: string
  }
  run: RunInfo | null
  conclusion: Rules
  /** 意思決定の決め方の既定値 */
  decision?: Decision | null
  /** 目的の列の過去の値を使う回帰と予測の一式か (何段も先まで予測できるか) */
  forecastable?: boolean
  check: RunCheck | null
  /** 外れ方の記録の要約 (意思決定の前提) */
  uncertainty?: UncertaintySummary | null
  notes: string[]
  /** 一式の形の版 (版 3 から受け持ちと出力の形を持つ) */
  format_version?: number | null
  /** 受け持ち (predict, generate, rank, group, effect. 版 2 までは predict) */
  role?: string
  /** 出力の形と形ごとの欄 */
  output?: OutputInfo
  /** 列の出力 (class, value, distribution) か. 偽なら結論の決め方と意思決定を使えない */
  column_output?: boolean
  /** 正解の種類と出どころ (target は前と同じく目的の列の名前) */
  target_info?: TargetInfo | null
  /** 繰り返しの手順 */
  procedure?: Procedure | null
  /** 学習済みの重みの置き方 (ノードごと) */
  pretrained?: PretrainedRecord[]
  /** 出どころから読んだ重みの警告 (モデルを組み立てた後に出る) */
  pretrained_warnings?: string[]
  /** 版 2 のデータセットの一式 (版 4) の 1 件の扱い */
  items?: ItemsInfo | null
  /** 既製のモデルの一式 (版 6) の出どころ, 前処理, ライセンスと重みの置き場 */
  ready?: ReadyInfo | null
  /** 並べるの一式 (版 8) の番号の表と索引の大きさ, 件数の既定, 反応の表の列 */
  rank?: RankInfo | null
}

/** 反応の表 (前に反応した品物と正解の反応) の列の名前 */
export interface RankColumns {
  user: string
  item: string
  reaction?: string | null
}

/** 並べるの一式の rank の欄 */
export interface RankInfo {
  users: number
  items: number
  width: number
  scale: number
  /** 並べる件数の既定 (問題の設定の rank.k) */
  k: number
  columns: RankColumns
}

/** 並べるの結果の Recall@K と NDCG (正解の反応の表を渡した場合) */
export interface RankMetrics {
  /** 数えた利用者 (正解のある利用者) の数 */
  users: number
  k: number
  recall_at_k: number
  ndcg: number
}

/** 並べるの結果に残した, 件数, 除いたか, 使った表と評価 */
export interface RankingInfo {
  k: number
  excluded: boolean
  columns: RankColumns
  seen: string | null
  truth: string | null
  metrics: RankMetrics | null
}

/** 既製のモデルの重みのライセンス (画面は alert が真なら message と source で警告する) */
export interface ReadyLicense {
  license: string | null
  status: string | null
  redistributable: boolean | null
  gated: boolean | null
  alert: boolean | null
  message: string | null
  source: string | null
  url: string | null
}

/** 既製のモデルの前処理 (短い辺をそろえ, 真ん中を切り出し, 正規化する) */
export interface ReadyPreprocess {
  resize: number
  crop: number
  interpolation: string
  mean: number[]
  std: number[]
}

/** 既製のモデルの一式の ready の欄 */
export interface ReadyInfo {
  library: string
  model: string
  weights: string
  input: string
  preprocess: ReadyPreprocess
  url: string | null
  file: string | null
  license: ReadyLicense | null
  /** 出どころの重みのファイルを置く場所と, そこにあるか (無ければ初回に取りに行く) */
  weights_file: string
  weights_cached: boolean
}

/** 既製のモデルの一式を作る前の下見 (重みは取りに行かない) */
export interface ReadyPreview {
  library: string
  model: string
  /** 版の決まった重みの名前 (DEFAULT は直した名前) */
  weights: string
  classes: string[]
  input: string | null
  preprocess: ReadyPreprocess
  url: string | null
  file: string | null
  license: ReadyLicense
}

export interface ReadyCreateBody {
  name: string
  library: string
  model: string
  weights: string
  /** 決めておいた重みのハッシュ値 (読んだ重みと違えば作らない) */
  sha256?: string | null
}

/** 版 4 の一式の 1 件の扱い (学習のスクリプトと同じ扱いを推論でかける) */
export interface ItemsInfo {
  /** モデルに渡す入力の項目の名前と種類 (image, capture, point_cloud) */
  input: string
  input_kind?: string | null
  /** 正解の項目 (一式の確かめで使う) */
  target: string | null
  /** 扱い方 (image, capture_points, points) */
  prepare: string
  /** 扱い方の設定 (点を選ぶ数 points, 正解から選ぶ数 target_points, 種 seed) */
  options: Record<string, unknown>
  /** 領域だけ: 出力のクラスの番号ごとの領域の値 */
  values: number[] | null
  /** 点群だけ: shift なら正規化の平行移動を戻して元の座標にする */
  restore: string | null
}

/** 書き出しでかけた中身の手順 (ML-EDA の版 3 の項目の steps. op と引数) */
export interface ItemStep {
  op: string
  [key: string]: unknown
}

/** データセットの説明 (一式を作る画面で, ファイルの組 (版 2 と版 3) の項目を見せるため) */
export interface DatasetDescription {
  path: string
  format_version: number
  name?: string | null
  role?: string | null
  target?: { kind: string | null; entries: string[]; origin: TargetOrigin | null }
  /** 項目 (points は点を選ぶ数が要る入力の種類か, steps は版 3 の中身の手順) */
  entries?: {
    name: string
    kind: string
    role: string
    shape: number[] | null
    points: boolean
    steps?: ItemStep[]
  }[]
  /** 分割ごとの件数 */
  splits?: Record<string, number | null>
  /** 領域の値と名前 */
  classes?: { value: number; name: string | null }[]
  /** 撮影の正規化があるか */
  normalized?: boolean
}

/** 出力の形 (class, value, distribution, mask, point_cloud, vector, text, image) と形ごとの欄 */
export interface OutputInfo {
  form: string
  /** 分布の分位点の水準 */
  quantiles?: number[]
  /** 領域の番号ごとの名前 */
  classes?: string[]
  /** 雑音から始まる手順の 1 件の形 (表は符号化した行の幅) */
  shape?: number[]
  /** 表の出力の列の符号化 (列の順. ML-Inference の output.columns) */
  columns?: TableColumnSpec[]
  [key: string]: unknown
}

/** 表の出力の 1 列の符号化 (数の列は mean, std, integer. 種類の列は categories) */
export interface TableColumnSpec {
  name: string
  type: 'number' | 'category' | string
  mean?: number
  std?: number
  integer?: boolean
  categories?: unknown[]
}

/** 結果に残した表の出力の 1 列 (名前と型. 種類の列は種類の数) */
export interface TableColumn {
  name: string
  type: 'number' | 'category' | string
  integer?: boolean
  categories?: number
}

/** 正解の出どころ (ML-EDA の版 2 の target.origin) */
export interface TargetOrigin {
  type: 'process' | 'annotation' | 'measurement' | 'synthetic' | 'unknown' | string
  name?: string | null
  settings?: unknown
}

export interface TargetInfo {
  kind: string
  origin?: TargetOrigin | null
}

/** 繰り返しの手順 (拡散のサンプリング, 文の生成) */
export interface Procedure {
  name: string
  steps: number
  seed: number
  options?: Record<string, unknown> | null
}

/** 学習済みの重みを読むノードの置き方 */
export interface PretrainedRecord {
  node: string
  unit?: string | null
  model?: string | null
  hub_id?: string | null
  revision?: string | null
  pinned_revision?: string | null
  license?: string | null
  status?: string | null
  redistributable?: boolean | null
  source?: string | null
  /** bundle (一式の重みに写した) か source (出どころから読む) */
  place: 'bundle' | 'source' | string
  sha256?: string | null
  keys?: number | null
}

/** 外れ方の記録の要約 (split は valid, train, recent (渡した実績)) */
export interface UncertaintySummary {
  split: string | null
  rows: number | null
  kind: string | null
  measured_at: string | null
  coverage: Record<'80' | '90', number> | null
  ece: number | null
  /** 一式を作ったときの記録に戻せるか */
  restorable?: boolean
  /** 段ごとの外れ方を測った段の数と, 測れた最も先の段 (0 は測れていない) */
  horizon?: number | null
  steps_measured?: number
  /** 外れ方を種類ごとに測った列と種類の数 (測っていなければ null) */
  groups?: { column: string | null; count: number } | null
}

/** 検証用の分割を一式で推論した評価と, 実験の記録との比べ */
export interface RunCheck {
  split: string
  rows: number
  /** 正解率, 平均絶対誤差, 領域の IoU, 点群の Chamfer 距離 */
  metric: 'acc' | 'mae' | 'iou' | 'chamfer'
  value: number
  recorded: number | null
  matches: boolean | null
}

/** 登録した一式 */
export interface Bundle {
  id: string
  name: string
  path: string
  managed: boolean
  added_at: string
  /** 学習の実験の中から置き場に写した一式の, 写した元 */
  copied_from?: string
  /** 見本の一式の, 推論を試す新しいデータ */
  sample_input?: string
  /** 見本の一式の, たどり直しを試す目的の列のある過去のデータ */
  sample_past?: string
  /** つないだ問題の設定 (problem.yaml) の場所 */
  problem?: string
  ok: boolean
  error: string | null
  description: Description | null
  /** モデルを組み立てたか (学習済みの重みの警告は組み立てた後に分かる) */
  model_loaded?: boolean
}

export interface DecisionCount {
  decision: string
  count: number
  share: number
}

export interface Counts {
  rows: number
  predicted: number
  skipped: number
  /** 判断ごとの件数 (列の出力でない一式の件数だけの結論には無い) */
  decisions?: DecisionCount[]
  /** 並べるの結果の, 利用者ごとに並べた件数と前に反応した品物を除いたか (rows は利用者の数) */
  k?: number
  excluded?: boolean
  values?: { sum: number; mean: number | null; min: number | null; max: number | null }
}

export interface Summary extends Counts {
  groups?: (Counts & { group: string })[]
}

/** 推論の結果のまとめ */
export interface Result {
  id: string
  bundle: { name: string | null; path: string } | null
  bundle_id: string | null
  source: string | null
  created_at: string | null
  task: string | null
  classes?: string[]
  rules?: Rules
  summary: Summary | null
  sentences: string[]
  columns?: string[]
  kind?: string | null
  /** 何段も先の予測の結果か (表は先の行だけ) */
  forecast?: boolean
  /** 意思決定 (決め方を選んだ場合) */
  decision?: DecisionResult | null
  /** 過去のデータでのたどり直し */
  backtest?: Backtest | null
  /** 入力に目的の列があったか (たどり直せるか) */
  has_target?: boolean
  explanation?: Explanation | null
  /** 結果の形の版 (表に収まらない中身があれば 2) */
  format_version?: number
  role?: string
  output_form?: string | null
  /** 列の出力の一式の結果か (前の結果は真) */
  column_output?: boolean
  target_info?: TargetInfo | null
  /** 正解が今の処理の出力の場合の注意 (結論の文とは分ける) */
  origin_note?: string | null
  /** 結論の文のうち全体の文の数 (残りはまとめる列ごとの文) */
  overall_count?: number
  pretrained_warnings?: string[]
  /** 表に収まらない中身 (artifacts/) の記録 */
  artifacts?: {
    directory: string
    column: string
    format: string
    form: string
    count: number
  } | null
  /** 表の出力の結果の列 (列の符号化の順. 他の結果は無いか null) */
  table_columns?: TableColumn[] | null
  /** 並べるの結果の件数, 使った表, Recall@K と NDCG */
  ranking?: RankingInfo | null
}

/** 結果どうしの比べ (GET /api/results/compare) */
export interface Comparison {
  results: Result[]
  /** 比べられるか (理由が無いか) */
  comparable: boolean
  /** 全ての結果で同じもの (bundle は一式, source は入力のファイル) */
  basis: ('bundle' | 'source')[]
  /** 比べられない理由 */
  reasons: string[]
}

/** 配列の出力の 1 件の見本 */
export interface ArtifactPreview {
  row: number
  file: string
  form: string
  shape: number[]
  dtype: string
  /** 見せ方 (領域の格子, 画像, 点群, 値だけ) */
  view: 'grid' | 'image' | 'points' | 'values'
  /** 最初の値 */
  values: number[]
  size: number
  /** 長さ (L2. 数で表せない大きさは null) */
  norm: number | null
  grid?: number[][]
  labels?: { label: number; count: number }[]
  /** (高さ, 幅) か (高さ, 幅, 3) の 0 から 255 */
  image?: (number | number[])[][]
  /** 間引いた点 (最初の 3 つの次元まで) */
  points?: number[][]
  total?: number
}

/** 意思決定の決め方 (値は数か, 予測の表の列の名前) */
export interface Decision {
  type: 'quantity' | 'select' | 'triage'
  [key: string]: unknown
}

/** 意思決定の結果 */
export interface DecisionResult {
  decision: Decision
  summary: Record<string, unknown> & {
    rows: number
    acting: number
    baselines?: Record<string, number>
  }
  sentences: string[]
  notes: string[]
}

/** たどり直し (方針ごとの実際の得 (正) と損 (負)) */
export interface Backtest {
  rows: number
  policies: Record<string, number>
  /** 問題の設定の守る指標をこの方針で数えた結果 (つないでいなければ null) */
  guard?: BacktestGuard | null
  sentences: string[]
}

/** たどり直しで数えた守る指標 (ML-Inference の backtest の guard) */
export interface BacktestGuard {
  name: string
  unit: string | null
  /** 数えた量の名前 (指していなければ null. 画面には説明を出す) */
  counted: string | null
  description: string | null
  /** この方針で数えた値 (割合は 0 から 1. 数えなければ null) */
  value: number | null
  line: number | null
  better: 'lower' | 'higher' | null
  /** 守る線を守れたか (線が無いか数えなければ null) */
  kept: boolean | null
  note: string | null
}

/** 列ごとの効き (並べ替えの重要度) */
export interface Explanation {
  measure: 'probability' | 'value'
  rows: number
  columns: {
    column: string
    /** column (列), series (系列を分ける列. 名前を入れ替える), past (目的の列の過去の値), time (求めない時刻の列) */
    kind: 'column' | 'series' | 'past' | 'time'
    effect: number | null
    spread: number | null
    rows: number
    lost: number
    error: string | null
  }[]
}

/** 1 行の列ごとの効きと理由 (ML-Inference の explain_rows) */
export interface RowReason {
  row: number
  measure: 'probability' | 'value'
  status: string
  prediction: string | number | null
  /** 効きの大きい 3 列の文 (無ければ null) */
  reason: string | null
  /** 効きの絶対値の大きい順 (求められない列は最後) */
  effects: { column: string; effect: number | null }[]
  /** 列の名前から求められない理由 */
  errors: Record<string, string>
  /** 選んだ行と一緒に渡した行の数 */
  rows: number
  repeats: number
}

export interface Rows {
  total: number
  offset: number
  columns: string[]
  rows: Record<string, unknown>[]
}

/** 繰り返しの手順の登録簿の 1 つ (GET /api/info) */
export interface ProcedureEntry {
  name: string
  builtin: boolean
  description: string
  /** 始まり (noise は雑音, given は渡した番号の並び) */
  start: string
  /** 時刻をモデルに渡すか */
  time: boolean
  /** 設定の名前 (何でも受け取る自作の手順は null) */
  options: string[] | null
  /** 設定の既定値 (数, 真偽値, 文字の既定値だけ) */
  defaults?: Record<string, unknown>
}

/** ML-Inference の版と, 画面で使う登録簿と上限 */
export interface AppInfo {
  version: string
  rule_keys: string[]
  result_version: number
  procedures: ProcedureEntry[]
  procedure_modules: string[]
  unit_modules: unknown[]
  /** 雑音だけの一式で一度に作る件数の上限 */
  max_samples: number
  /** 画面で作る一式の手順の回数の上限 */
  max_procedure_steps: number
  /** 既製のモデルの出どころ (ML-Inference の READY_LIBRARIES) */
  ready_libraries?: string[]
}

/** モデルの入力ごとの出どころ (features, items, state, time) と先頭に足す番号 */
export interface InputSource {
  source: string
  prepend?: number[]
}

export interface BundleCreateBody {
  name: string
  spec: string
  /** 省くと雑音だけの一式 (データセットを持たない) */
  dataset?: string | null
  weights: string
  inputs?: Record<string, InputSource> | null
  output?: string | null
  kind?: string | null
  /** 中心と幅か, 学習用の分割の平均と標準偏差 (train) */
  target_scale?: { center: number; scale: number } | 'train' | null
  conclusion?: Rules | null
  copy_weights: boolean
  role?: string | null
  form?: string | null
  /** 出力の形ごとの欄 (分布の水準 quantiles, 領域のクラスの名前 classes, 1 件の形 shape) */
  output_options?: Record<string, unknown> | null
  procedure?: Procedure | null
  /** 版 2 のデータセットの 1 件の扱い (省いた欄はデータセットから) */
  items?: { input?: string; options?: Record<string, number> } | null
}

/** 外れ方の測り直し (確かめるだけか, 一式に書き戻したか) */
export interface Remeasure {
  rows: number
  before: { coverage?: Record<'80' | '90', number>; ece?: number }
  sentences: string[]
  written: boolean
  /** 学習の実験の中の一式を書き戻すために写した場所 (写さなければ null) */
  copied_to: string | null
}

/** 一式につないだ問題の設定から読んだ推論の値 */
export interface LinkedProblem {
  name: string
  path: string
  /** 問題の設定のファイルの中身の印 (変わったら既定を読み直す) */
  version: string
  created_at: string
  /** 機械学習の受け持ちの名前 (作る, 正解の種類は画像 など) */
  role: string
  /**
   * 推論の結論を扱う受け持ちか (列の予測だけ). 扱わない受け持ちは意思決定が null で,
   * 結論の決め方は一式の既定値のまま
   */
  handled: boolean
  /** 意思決定 (意思決定しない問題の設定と, 扱わない受け持ちは null) */
  decision: Decision | null
  /** 一式の既定値に問題の設定の値を重ねた結論の決め方 */
  conclusion: Rules
  /** 何期先まで予測するか */
  horizon: number | null
  /** 損得の値の根拠 */
  basis: string | null
  /** 意思決定の文 */
  text: string
  /** 一式との食い違い */
  issues: string[]
}

/** 一式と問題の設定のつなぎの状態 */
export interface ProblemState {
  /** つないだ場所 (つないでいなければ null) */
  path: string | null
  linked: LinkedProblem | null
  /** つないだ問題の設定を読めない理由 */
  error: string | null
  /** つないでいない一式の, 上のディレクトリで見つけた候補 (元の場所を持つ写しなら元) */
  hint: string | null
  /** 候補が元の問題の設定か, 学習の時点の写しかを伝える文 */
  hint_note: string | null
}

/** 問題の設定に記録したプロジェクトの, 推論の一式を持つ実験 */
export interface ExperimentRow {
  /** 一式のディレクトリ (実験の inference) */
  path: string
  /** プロジェクトのディレクトリ */
  project: string
  /** プロジェクトを記録した問題の設定の名前 */
  problem: string
  /** 出力先からの実験の場所 (名前/日付/時刻) */
  experiment: string
  /** 一式の更新の時刻 */
  modified: string
  /** 登録済みならその id */
  bundle_id: string | null
}

/** 問題の設定に記録したプロジェクトの, 方策の一式 (実験の policy) を持つ実験 */
export interface PolicyExperimentRow extends Omit<ExperimentRow, 'bundle_id' | 'path'> {
  /** 方策の一式のディレクトリ (実験の policy) */
  path: string
  /** 登録済みならその方策の一式の id */
  policy_id: string | null
}

export interface ExperimentList<Row = ExperimentRow> {
  /** 探した問題の設定の置き場 (ML-Pathfinder) */
  problems: string
  /** 見たプロジェクトの数 */
  projects: number
  experiments: Row[]
  /** 上限で打ち切ったか */
  truncated: boolean
}

/** 方策の一式の行動か観測の空間 (policy.json の observation_space, action_space) */
export type PolicySpace =
  | { type: 'discrete'; n: number; start: number }
  | { type: 'box'; shape: number[]; low: (number | null)[]; high: (number | null)[] }

/** 方策の一式の守り (policy.json の guards. null は制限なし) */
export interface PolicyGuards {
  action_low?: (number | null)[] | null
  action_high?: (number | null)[] | null
  allowed_actions?: number[] | null
  observation_low?: (number | null)[] | null
  observation_high?: (number | null)[] | null
  fallback: { kind: 'stop' | 'handover' | 'action'; action?: unknown }
}

/** 方策の一式の中身 (policy.json) */
export interface PolicySpec {
  format_version: number
  kind: 'policy'
  head: 'categorical' | 'deterministic'
  observation_space: PolicySpace
  action_space: PolicySpace
  guards: PolicyGuards
  metadata?: Record<string, unknown>
}

/** 登録した方策の一式 */
export interface Policy {
  id: string
  name: string
  path: string
  added_at: string
  ok: boolean
  error: string | null
  spec: PolicySpec | null
  /** 一式の重みのうちモデルに無く読まなかったキー (中身を読んだときだけ) */
  unused_keys: string[] | null
}

/** 観測のファイルでまとめて決めた方策の結果 (置き場の policy_results) */
export interface PolicyResult {
  id: string
  policy_id: string
  policy: { name: string; path: string } | null
  /** 観測のファイル */
  source: string
  created_at: string | null
  rows: number
  /** 守りの結果ごとの件数 (ok, clipped, fallback, stop, handover の順) */
  counts: Record<PolicyAct['status'], number>
}

/** 方策の結果の表の一部 (観測と行動は JSON の文字) */
export interface PolicyResultRows {
  total: number
  offset: number
  rows: {
    index: number
    observation: string
    action: string
    status: PolicyAct['status']
    /** 守りが働いた理由 (「 / 」で並べる) */
    reasons: string
  }[]
}

/** 守りを当てた行動の結果 */
export interface PolicyAct {
  /** 行動 (番号か数の入れ子の並び). 止めるか元の制御に戻す場合は null */
  action: number | unknown[] | null
  status: 'ok' | 'clipped' | 'fallback' | 'stop' | 'handover'
  reasons: string[]
}
