import type {
  ClusterFar,
  ClusterRepresentatives,
  ClusterRow,
  ClusterStability,
  Result,
} from '../api/types'

/** 群分けの一式で使えない機能 (出力が行ごとの群のため) */
export const CLUSTER_UNSUPPORTED =
  'この一式は行ごとに最も近い群を出すので, 結論は群ごとの件数と特徴です. 結論の決め方と意思決定は, 数や種類を予測する一式で使うため, ここでは使えません'

/** 群分けの一式の画面に問題の設定の欄が無い理由 (学習した符号化器の一式) */
export const CLUSTER_NO_PROBLEM =
  '問題の設定の欄はありません. 群の数と群を作る列は, 一式を作るときに学習の実験から写してあります'

/** 既製の埋め込みの群分けの一式の画面に問題の設定の欄が無い理由 */
export const READY_CLUSTER_NO_PROBLEM =
  '問題の設定の欄はありません. 群の数と群の中心は, 一式を作るときに骨組みの成果物から写してあります'

/** 群の要約の最後の行の名前 (ML-EDA と ML-Inference と同じ) */
export const ALL_ROWS = '全体'

/** 群の安定性の作り直し方の日本語の名前 (ML-Inference の文と同じ言葉) */
export const STABILITY_METHODS: Record<string, string> = {
  seed: '種を変えて作り直す',
  sample: '標本を変えて作り直す',
}

/** 群に割り当てた行の表の見出しの日本語の名前 (CSV の見出しは英字のまま) */
export const CLUSTER_HEADINGS: Record<string, string> = {
  index: '#',
  group: '群',
  distance: '距離',
  far: '中心から遠い',
  status: '状態',
  reason: '理由',
}

/** 群に割り当てた行の表の状態の日本語の名前 */
export const CLUSTER_STATUS_LABELS: Record<string, string> = {
  ok: '割り当てた',
  skipped: '割り当てられず',
}

/** 群に割り当てた行の表で, 先に並べる列 (行の位置, 群, 距離, 中心から遠いか, 状態, 理由) */
const CLUSTER_LEAD = ['index', 'group', 'distance', 'far', 'status', 'reason']

/**
 * 群に割り当てた行の表の列の並び. 行の位置, 群, 距離, 中心から遠いか (遠さの上限を持つ一式
 * だけ), 状態, 理由, 元の列の順.
 * 判断の列 (列の出力でないので割り当てた行は空) は出さない
 */
export function clusterRowColumns(columns: string[]): string[] {
  const present = new Set(columns)
  const ordered = CLUSTER_LEAD.filter((name) => present.has(name))
  const placed = new Set([...ordered, 'decision'])
  return [...ordered, ...columns.filter((name) => !placed.has(name))]
}

/**
 * 群分けの結果の群の要約 (summary.groups). 群分けの結論では ML-Inference の
 * group_summary の行の並びで, まとめる列ごとの件数 (列の出力の結論) とは形が違う
 */
export function clusterRows(result: Result): ClusterRow[] {
  if (result.output_form !== 'cluster') return []
  return (result.summary?.groups ?? []) as unknown as ClusterRow[]
}

/**
 * 群を作る列の特徴の見出し. 数の列は平均, 種類の列は最も多い値と割合 (ML-EDA の群の要約の決まり).
 * 値の型は群の要約の行から見る (行の無い群は null なので, 値のある行で決める)
 */
export function featureHeading(name: string, rows: ClusterRow[]): string {
  const value = rows.map((row) => row[name]).find((item) => item !== null && item !== undefined)
  if (typeof value === 'number') return `${name} (平均)`
  if (typeof value === 'string') return `${name} (最も多い値)`
  return name
}

/** 群の名前 (番号は「群 0」, 全体はそのまま) */
export function clusterName(group: number | string): string {
  return typeof group === 'number' ? `群 ${group}` : String(group)
}

/** 群の安定性の表の 1 行 */
export interface StabilityLine {
  key: 'ari' | 'ari_min' | 'silhouette'
  label: string
  note: string
  model: number
  baseline: number
  diff: number
}

/** 群の安定性の 3 つの指標の, モデルの値, 比べる相手の値, 差 (モデル − 比べる相手) */
export function stabilityLines(stability: ClusterStability): StabilityLine[] {
  return [
    {
      key: 'ari',
      label: '調整ランド指数 (平均)',
      note: '作り直しても同じ群になるか. 1 に近いほど安定',
      model: stability.ari,
      baseline: stability.ari_baseline,
      diff: stability.ari_diff,
    },
    {
      key: 'ari_min',
      label: '調整ランド指数 (最も低い)',
      note: '作り直したうち最も違った群との一致',
      model: stability.ari_min,
      baseline: stability.ari_min_baseline,
      diff: stability.ari_min_diff,
    },
    {
      key: 'silhouette',
      label: 'シルエット係数',
      note: '群どうしの離れ方. -1 から 1 で, 高いほど分かれている',
      model: stability.silhouette,
      baseline: stability.silhouette_baseline,
      diff: stability.silhouette_diff,
    },
  ]
}

/** 安定性の値を小数 3 桁で (無い値は –) */
export function scoreText(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '–'
  return value.toFixed(3)
}

/** 差を符号付きの小数 3 桁で (+0.070, -0.020) */
export function diffText(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '–'
  const text = Math.abs(value).toFixed(3)
  if (Number(text) === 0) return '±0.000'
  return `${value > 0 ? '+' : '-'}${text}`
}

/** 作り直し方と回数の文 (種を変えて作り直す, 5 回) */
export function stabilityMethodText(stability: ClusterStability): string {
  const method = STABILITY_METHODS[stability.method] ?? stability.method
  return `${method}, ${stability.repeats} 回`
}

/** 群の代表の行の既定の件数 (ML-Inference の結論の件数と同じ) */
export const DEFAULT_REPRESENTATIVES = 3

/** 群の代表の行の件数の選択肢 (API の上限は 20) */
export const REPRESENTATIVE_COUNTS = [1, 3, 5, 10, 20]

/**
 * 群分けの結論の群の代表の行 (summary.representatives. 既定の 3 件).
 * 結論に無い結果 (代表の行を持つ前に作った結果) は undefined
 */
export function clusterRepresentatives(result: Result): ClusterRepresentatives[] | undefined {
  if (result.output_form !== 'cluster') return undefined
  return result.summary?.representatives
}

/**
 * 代表の行の, 群を作る列の値の見せ方. 数の列 (群の要約が平均の列) の値は, 文字でも数に
 * 読めれば数として見せる (数にできない値を含む列は, 予測の表で文字の列になるため)
 */
export function representativeValue(value: unknown, numeric: boolean): unknown {
  if (numeric && typeof value === 'string' && value.trim() !== '') {
    const number = Number(value)
    if (Number.isFinite(number)) return number
  }
  return value
}

/**
 * 群分けの結論の群の中心から遠い行の件数 (summary.far). 遠さの上限を持たない一式
 * (版 9 と 10) の結果と群分けでない結果は undefined
 */
export function clusterFar(result: Result): ClusterFar | undefined {
  if (result.output_form !== 'cluster') return undefined
  return result.summary?.far ?? undefined
}

/** 遠さの分位点の文 (0.99 は「99 % 点」. ML-Inference の文と同じ書き方) */
export function quantileText(quantile: number): string {
  return `${Number((quantile * 100).toPrecision(6))} % 点`
}

/** 群に割り当てた行の表の, 中心から遠い列の値の見せ方 (遠い行は「遠い」, 他は空) */
export function farMark(value: unknown): string {
  return value === true ? '遠い' : ''
}

/** 遠さで絞る欄の選択肢 (値は API の far. 空は絞らない) */
export const FAR_FILTERS: { value: '' | 'true' | 'false'; label: string }[] = [
  { value: '', label: '遠さを問わない' },
  { value: 'true', label: '中心から遠い行' },
  { value: 'false', label: '遠くない行' },
]
