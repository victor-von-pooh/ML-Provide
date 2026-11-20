import type { Description, ItemStep, PretrainedRecord, TargetInfo } from '../api/types'

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

/**
 * 出力の形ごとの欄の文 (分布の水準, 領域のクラス, 1 件の形).
 * 版 2 のデータセットの一式の領域は, 番号でなく領域の値ごとの名前
 */
export function outputDetails(output: Description['output'], values?: number[] | null): string[] {
  if (!output) return []
  const details: string[] = []
  if (output.quantiles && output.quantiles.length > 0) {
    details.push(`水準 ${output.quantiles.join(', ')}`)
  }
  if (output.classes && output.classes.length > 0) {
    const pairs = output.classes.map((name, i) => `${values?.[i] ?? i}: ${name}`)
    details.push(`${values ? '値' : '番号'}ごとの名前 ${pairs.join(', ')}`)
  }
  if (output.shape && output.shape.length > 0) {
    details.push(`1 件の形 (${output.shape.join(', ')})`)
  }
  return details
}

/**
 * 領域の格子の値から名前 (版 2 のデータセットの一式は領域の値, 他はクラスの番号).
 * 凡例で使う
 */
export function maskNames(description: Description | null | undefined): Record<number, string> {
  const names: Record<number, string> = {}
  const classes = description?.output?.classes ?? []
  const values = description?.items?.values
  classes.forEach((name, i) => {
    names[values?.[i] ?? i] = name
  })
  return names
}

/** 一式での評価の指標の名前 */
export const METRIC_LABELS: Record<string, string> = {
  acc: '正解率',
  mae: '平均絶対誤差',
  iou: 'IoU (背景を除くクラスの平均)',
  chamfer: 'Chamfer 距離',
}

/** 版 2 のデータセットの項目の種類の名前 */
export const ENTRY_KIND_LABELS: Record<string, string> = {
  image: '画像',
  mask: '領域',
  capture: '撮影',
  point_cloud: '点群',
}

/** 1 件の扱い方の文 (ML-Inference の items.prepare) */
export const PREPARE_LABELS: Record<string, string> = {
  image: '画素を 0 から 1 にする',
  capture_points: '深度から点に戻して決まった数を選ぶ',
  points: '決まった数の点を選ぶ',
}

/** 推論するデータの欄の名前 (版 4 の一式は入力の項目の種類から) */
export function sourceLabel(inputKind: string | null | undefined): string {
  if (inputKind === 'image')
    return '推論するデータ (画像のファイルか, 画像を並べたディレクトリ. 版 2 のデータセットも)'
  if (inputKind === 'capture')
    return '推論するデータ (撮影のディレクトリか, 撮影を並べたディレクトリ. 版 2 のデータセットも)'
  if (inputKind === 'point_cloud')
    return '推論するデータ (点群のファイルかディレクトリ. 版 2 のデータセットも)'
  return '推論するデータ (学習のときと同じ列のファイルか, 画像などのディレクトリ)'
}

/** 中身の手順の並びを, 名前と引数の文にする (resize (width 64, height 64), grayscale) */
export function stepsText(steps: ItemStep[]): string {
  return steps
    .map((step) => {
      const values = Object.entries(step)
        .filter(([key]) => key !== 'op')
        .map(([key, value]) => `${key} ${String(value)}`)
        .join(', ')
      return values ? `${step.op} (${values})` : step.op
    })
    .join(', ')
}
