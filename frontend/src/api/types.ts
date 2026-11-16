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
  /** 目的の列の過去の値を使う回帰と予測の一式か (何段も先まで予測できるか) */
  forecastable?: boolean
  check: RunCheck | null
  notes: string[]
}

/** 検証用の分割を一式で推論した評価と, 実験の記録との比べ */
export interface RunCheck {
  split: string
  rows: number
  metric: 'acc' | 'mae'
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
  /** 見本の一式の, 推論を試す新しいデータ */
  sample_input?: string
  ok: boolean
  error: string | null
  description: Description | null
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
  decisions: DecisionCount[]
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
  explanation?: Explanation | null
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

export interface Rows {
  total: number
  offset: number
  columns: string[]
  rows: Record<string, unknown>[]
}

export interface BundleCreateBody {
  name: string
  spec: string
  dataset: string
  weights: string
  inputs?: Record<string, { source: string; prepend: number[] }> | null
  output?: string | null
  kind?: string | null
  target_scale?: { center: number; scale: number } | null
  conclusion?: Rules | null
  copy_weights: boolean
}
