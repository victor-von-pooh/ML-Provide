import type { Description, PretrainedRecord, TargetInfo } from '../api/types'

/** 受け持ちの日本語の名前 (ML-Quest と同じ言葉) */
export const ROLE_LABELS: Record<string, string> = {
  predict: '予測する',
  generate: '作る',
  rank: '並べる',
  group: 'まとめる',
  effect: '効果を見積もる',
}

/** 出力の形の日本語の名前 */
export const FORM_LABELS: Record<string, string> = {
  class: 'クラス',
  value: '値',
  distribution: '分布 (分位点)',
  mask: '領域',
  point_cloud: '点群',
  vector: 'ベクトル',
  text: '文',
  image: '画像',
}

/** 列の出力の形 (結論の決め方と意思決定を使える) */
export const COLUMN_FORMS = new Set(['class', 'value', 'distribution'])

/** 正解の種類の日本語の名前 */
export const TARGET_KIND_LABELS: Record<string, string> = {
  column: '表の列',
  mask: '領域',
  point_cloud: '点群',
  image: '画像',
  text: '文',
  vector: 'ベクトル',
}

/** 正解の出どころの日本語の名前 */
export const ORIGIN_LABELS: Record<string, string> = {
  process: '今の処理の出力',
  annotation: '人の付けた正解',
  measurement: '測った値',
  synthetic: '合成',
  unknown: '不明',
}

/** 列の出力の一式か (版 2 までの一式と項目の無い説明は列の出力) */
export function isColumnOutput(description: Pick<Description, 'column_output'>): boolean {
  return description.column_output !== false
}

/**
 * 正解が今の処理の出力の場合の注意 (ML-Inference の結論の文と同じ).
 * 他の出どころは null
 */
export function originNote(target: TargetInfo | null | undefined): string | null {
  const origin = target?.origin
  if (!origin || origin.type !== 'process') return null
  const name = origin.name || '今の処理'
  return `正解は ${name} の出力なので, 予測はその処理との一致までしか分からず, その処理を超えられません`
}

/** 学習済みの重みの置き場の文 */
export function placeText(record: PretrainedRecord): string {
  if (record.place === 'source') {
    return `出どころ (${record.hub_id ?? record.model ?? record.unit ?? '–'}${
      record.pinned_revision ? ` @ ${record.pinned_revision}` : ''
    }) から読む`
  }
  return '一式の重みに写した'
}

/** ライセンスの文 (再配布できないものと不明なものは注意) */
export function licenseText(record: PretrainedRecord): { text: string; warn: boolean } {
  const license = record.license || '不明'
  if (record.redistributable === true) return { text: `${license} (再配布できる)`, warn: false }
  if (record.redistributable === false) return { text: `${license} (再配布できない)`, warn: true }
  return { text: `${license} (再配布できるか不明)`, warn: true }
}

/** 出力の形ごとの欄の文 (分布の水準, 領域のクラス, 1 件の形) */
export function outputDetails(output: Description['output']): string[] {
  if (!output) return []
  const details: string[] = []
  if (output.quantiles && output.quantiles.length > 0) {
    details.push(`水準 ${output.quantiles.join(', ')}`)
  }
  if (output.classes && output.classes.length > 0) {
    details.push(`番号ごとの名前 ${output.classes.map((name, i) => `${i}: ${name}`).join(', ')}`)
  }
  if (output.shape && output.shape.length > 0) {
    details.push(`1 件の形 (${output.shape.join(', ')})`)
  }
  return details
}
