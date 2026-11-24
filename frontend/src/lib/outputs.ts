import type {
  Description,
  ItemStep,
  PretrainedRecord,
  TableColumn,
  TableColumnSpec,
  TargetInfo,
} from '../api/types'
import { formatValue } from './format'

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
  table: '表',
  ranking: '順位 (利用者ごとの上位の品物)',
  cluster: '群 (行ごとに最も近い群)',
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
  if (output.form === 'table' && output.columns) {
    // 表は 1 件の形の代わりに列の数と 1 件の幅 (符号化した行の幅. 列ごとは tableColumnText)
    details.push(
      `列 ${output.columns.length}${output.shape?.length ? `, 1 件の幅 ${output.shape.join(', ')}` : ''}`,
    )
  } else if (output.shape && output.shape.length > 0) {
    details.push(`1 件の形 (${output.shape.join(', ')})`)
  }
  return details
}

/** 作った表の見出しの日本語の名前 (CSV の見出しは英字のまま) */
export const TABLE_HEADINGS: Record<string, string> = {
  index: '#',
  sample: '件の番号',
  seed: '種',
  status: '状態',
  reason: '理由',
}

/** 表の出力の結果で, 列の符号化の列より前に並べる列 (行の位置, 件の番号, 種, 状態, 理由) */
const TABLE_LEAD = ['index', 'sample', 'seed', 'status', 'reason']

/**
 * 表の出力の結果の予測の表の列の並び. 行の位置, 件の番号, 種, 状態, 理由, 列の符号化の順の列,
 * 残りの列の順. 判断の列 (列の出力でないので空) は出さない. 表の index の列は, 行の位置と
 * 重ならないよう API が「index (入力)」にする
 */
export function tableRowColumns(columns: string[], table: TableColumn[]): string[] {
  const present = new Set(columns)
  const names = table.map((entry) => (entry.name === 'index' ? 'index (入力)' : entry.name))
  const ordered = [...TABLE_LEAD, ...names].filter((name) => present.has(name))
  const placed = new Set([...ordered, 'decision'])
  return [...ordered, ...columns.filter((name) => !placed.has(name))]
}

/**
 * 表の出力の 1 列の型の文 (数 (平均 10, 標準偏差 2), 整数 (…), 種類 (3 種類: blue, green, red)).
 * 一式の説明の列の符号化 (値の種類も持つ) と, 結果に残した列 (種類の数だけ) のどちらも受け取る
 */
export function tableColumnText(column: TableColumnSpec | TableColumn): string {
  if (column.type === 'category') {
    const values = Array.isArray(column.categories) ? column.categories : null
    const count = values ? values.length : Number(column.categories ?? 0)
    if (!values) return `種類 (${count} 種類)`
    // 種類が多い列は先頭の 10 個と残りの数
    const head = values.slice(0, 10).map(String).join(', ')
    const rest = values.length > 10 ? ` ほか ${values.length - 10}` : ''
    return `種類 (${count} 種類: ${head}${rest})`
  }
  if (column.type !== 'number') return column.type
  const kind = column.integer ? '整数' : '数'
  const spec = column as TableColumnSpec
  if (typeof spec.mean !== 'number' || typeof spec.std !== 'number') return kind
  return `${kind} (平均 ${formatValue(spec.mean)}, 標準偏差 ${formatValue(spec.std)})`
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

/** 推論するデータの欄の名前 (版 4 の一式は入力の項目の種類から, 他は一式の種類から) */
export function sourceLabel(inputKind: string | null | undefined, kind?: string): string {
  if (inputKind === 'image')
    return '推論するデータ (画像のファイルか, 画像を並べたディレクトリ. 版 2 のデータセットも)'
  if (inputKind === 'capture')
    return '推論するデータ (撮影のディレクトリか, 撮影を並べたディレクトリ. 版 2 のデータセットも)'
  if (inputKind === 'point_cloud')
    return '推論するデータ (点群のファイルかディレクトリ. 版 2 のデータセットも)'
  if (kind === 'audio')
    return '推論するデータ (音声のファイルか, 音声を並べたディレクトリ. WAVE, FLAC など)'
  return '推論するデータ (学習のときと同じ列のファイルか, 画像などのディレクトリ)'
}

/**
 * 数, 真偽, null に見える文字か (大文字と小文字を区別しない).
 * 設定の欄 (10 進の数と true, false) と YAML 1.1 (yes, on, ~, 0x10, 1_000, 1:30, .inf, 日付など) で
 * 文字でない値に読まれるものと, JavaScript と Python の数の書き方 (Infinity, nan, 0b1, 0o7) を含む
 */
function looksTyped(text: string): boolean {
  const lower = text.toLowerCase()
  if (/^(true|false|yes|no|on|off|y|n|null|none|~)$/.test(lower)) return true
  // 10 進 (桁の _ を含む), 0x, 0o, 0b, 60 進 (1:30), 無限と非数, 日付と日時
  return (
    /^[+-]?(\d[\d_]*\.?[\d_]*|\.\d[\d_]*)(e[+-]?\d+)?$/.test(lower) ||
    /^[+-]?0(x[\da-f_]+|o[0-7_]+|b[01_]+)$/.test(lower) ||
    /^[+-]?\d[\d_]*(:[0-5]?\d)+(\.[\d_]*)?$/.test(lower) ||
    /^[+-]?\.?(inf|infinity|nan)$/.test(lower) ||
    // YAML 1.1 の日付と日時 (2026-11-22, 2026-11-22T10:00:00Z)
    /^\d{4}-\d\d?-\d\d?([Tt ]\d\d?:\d\d:\d\d(\.\d*)?\s*(Z|[+-]\d\d?(:\d\d)?)?)?$/.test(text)
  )
}

/**
 * 設定の辞書を「名前 値」の並びの文にする (eta 0, train_steps 1000).
 * 入れ子の値と, 区切りや別の型と紛れる文字 (空, 空白, カンマ, 括弧を含む. 引用符や [ { で始まる.
 * 数, 真偽, null に見える) は JSON で引用符を付ける (文字の "0" と数の 0 を見分けるため)
 */
export function optionsText(options: Record<string, unknown>): string {
  return Object.entries(options)
    .map(([key, value]) => {
      const quoted =
        (value !== null && typeof value === 'object') ||
        (typeof value === 'string' &&
          (value === '' || /[\s,()]/.test(value) || /^["'[{]/.test(value) || looksTyped(value)))
      return `${key} ${quoted ? JSON.stringify(value) : String(value)}`
    })
    .join(', ')
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
