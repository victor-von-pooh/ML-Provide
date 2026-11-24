import type { RankColumns } from '../api/types'

/** 利用者ごとに並べる件数の上限 (API の MAX_RANK_K と同じ) */
export const MAX_RANK_K = 1000

/** 並べるの一式で使えない機能 (出力が利用者ごとの品物の一覧のため) */
export const RANK_UNSUPPORTED =
  'この一式は利用者ごとの品物の一覧を出すので, 結論は並べた利用者の人数だけです. 結論の決め方と意思決定は, 数や種類を予測する一式で使うため, ここでは使えません'

/** 並べるの一式の画面に問題の設定の欄が無い理由 */
export const RANK_NO_PROBLEM =
  '問題の設定の欄はありません. 並べる件数と反応の表の列は, 一式を作るときに問題の設定から写してあります'

/** 並べた品物の表の見出しの日本語の名前 (CSV の見出しは英字のまま) */
export const RANK_HEADINGS: Record<string, string> = {
  index: '#',
  user: '利用者',
  status: '状態',
  reason: '理由',
  rank: '順位',
  item: '品物',
  score: '点数',
}

/** 並べた品物の表の状態の日本語の名前 */
export const RANK_STATUS_LABELS: Record<string, string> = {
  ok: '並べた',
  skipped: '並べられず',
}

/** 並べた品物の表で, 先に並べる列 (行の位置, 利用者, 順位, 品物, 点数, 状態, 理由) */
const RANK_LEAD = ['index', 'user', 'rank', 'item', 'score', 'status', 'reason']

/**
 * 並べた品物の表の列の並び. 行の位置, 利用者, 順位, 品物, 点数, 状態, 理由, 残りの列の順.
 * 判断の列 (列の出力でないので並べた行は空) は出さない
 */
export function rankRowColumns(columns: string[]): string[] {
  const present = new Set(columns)
  const ordered = RANK_LEAD.filter((name) => present.has(name))
  const placed = new Set([...ordered, 'decision'])
  return [...ordered, ...columns.filter((name) => !placed.has(name))]
}

/** 書いた利用者 (1 行に 1 人. 前後の空白を除き, 空の行は飛ばす) */
export function rankUsers(text: string): string[] {
  return text
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line !== '')
}

/** 並べる件数の欄の誤り (空は一式の既定) */
export function rankKProblem(text: string): string | null {
  const value = text.trim()
  if (!value) return null
  const number = Number(value)
  if (!/^\d+$/.test(value) || number < 1 || number > MAX_RANK_K)
    return `並べる件数は 1 から ${MAX_RANK_K.toLocaleString()} の整数で書いてください`
  return null
}

/** 反応の表の列の文 (利用者 member, 品物 item_id, 反応の強さ qty) */
export function rankColumnsText(columns: RankColumns): string {
  const parts = [`利用者 ${columns.user}`, `品物 ${columns.item}`]
  if (columns.reaction) parts.push(`反応の強さ ${columns.reaction}`)
  return parts.join(', ')
}

/** 反応の表のどの行を反応とするかの文 */
export function reactionRuleText(columns: RankColumns): string {
  return columns.reaction
    ? `${columns.reaction} が 0 より大きい行を反応とします`
    : '全ての行を反応とします'
}

/** 0 から 1 の評価の値を小数 3 桁で */
export function metricText(value: number): string {
  return value.toFixed(3)
}
