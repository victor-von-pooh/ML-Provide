/** 大きな数を 1.2K のように短くする */
export function formatCount(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '–'
  const abs = Math.abs(value)
  if (abs >= 1e9) return `${(value / 1e9).toFixed(1)}B`
  if (abs >= 1e6) return `${(value / 1e6).toFixed(1)}M`
  if (abs >= 1e4) return `${(value / 1e3).toFixed(1)}K`
  return value.toLocaleString()
}

/** 数を見やすい桁で表す */
export function formatNumber(value: unknown): string {
  if (value === null || value === undefined) return '–'
  if (typeof value !== 'number') return String(value)
  if (Number.isInteger(value)) return value.toLocaleString()
  const abs = Math.abs(value)
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e6)) return value.toExponential(3)
  return value.toLocaleString(undefined, { maximumFractionDigits: 4 })
}

/** バイト数を KB, MB にする */
export function formatBytes(value: number | null | undefined): string {
  if (value === null || value === undefined) return '–'
  if (value >= 1 << 30) return `${(value / (1 << 30)).toFixed(1)} GB`
  if (value >= 1 << 20) return `${(value / (1 << 20)).toFixed(1)} MB`
  if (value >= 1 << 10) return `${(value / (1 << 10)).toFixed(1)} KB`
  return `${value} B`
}

/** 割合を % にする */
export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined) return '–'
  return `${(value * 100).toFixed(digits)}%`
}

/** 秒の間隔を, 日, 時間, 分, 秒の分かりやすい形にする */
export function formatInterval(seconds: number | null | undefined): string {
  if (!seconds) return '–'
  const units: [number, string][] = [
    [86400 * 365, '年'],
    [86400 * 30, 'か月'],
    [86400 * 7, '週'],
    [86400, '日'],
    [3600, '時間'],
    [60, '分'],
    [1, '秒'],
  ]
  for (const [size, name] of units) {
    if (seconds >= size && Math.abs(seconds / size - Math.round(seconds / size)) < 0.01) {
      return `${Math.round(seconds / size)} ${name}`
    }
  }
  return `${seconds} 秒`
}

/** 列の型の日本語の名前 */
export const TYPE_LABELS: Record<string, string> = {
  numeric: '数値',
  categorical: 'カテゴリ',
  boolean: '2 値',
  datetime: '日時',
  text: '文',
  id: 'ID',
  constant: '定数',
  empty: '空',
  other: 'その他',
}

/** データの種類の日本語の名前 */
export const KIND_LABELS: Record<string, string> = {
  table: '表',
  timeseries: '時系列',
  image: '画像',
  text: '文',
  pointcloud: '点群',
  audio: '音声',
  items: 'ファイルの組',
  noise: '雑音だけ',
  ready: '既製のモデル',
  rank: '並べる (利用者と品物)',
}

/** タスクの日本語の名前 */
export const TASK_LABELS: Record<string, string> = {
  classification: '分類',
  regression: '回帰',
  forecasting: '予測 (時系列)',
  none: '決めない',
}

/** 予測の値を見やすくする (大きな値は整数, 確率は小数 3 桁) */
export function formatValue(value: unknown, probability = false): string {
  if (typeof value !== 'number') return formatNumber(value)
  if (probability) return value.toFixed(3)
  if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 })
  return formatNumber(value)
}
