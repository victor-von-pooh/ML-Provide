import type { BacktestGuard } from '../api/types'

/** 量を ML-Inference の文と同じ桁にする (1,000 以上は整数, 他は有効数字 3 桁) */
function amountText(value: number): string {
  if (Math.abs(value) >= 1000) return Math.round(value).toLocaleString('en-US')
  return String(Number(value.toPrecision(3)))
}

/** 割合の量か (値と線は 0 から 1 なので % で出す) */
function isRate(guard: BacktestGuard): boolean {
  return !!guard.counted?.endsWith('_rate')
}

/** 守る指標の値か線を, 割合は %, 他は単位付きの文にする */
export function guardAmount(guard: BacktestGuard, value: number | null): string {
  if (value === null) return '–'
  if (isRate(guard)) return `${amountText(value * 100)} %`
  return guard.unit ? `${amountText(value)} ${guard.unit}` : amountText(value)
}

/** 守る線の文 (向きの以下か以上を添える. 線が無ければ「無い」) */
export function guardLine(guard: BacktestGuard): string {
  if (guard.line === null) return '無い'
  const side = guard.better === 'higher' ? '以上' : '以下'
  return `${guardAmount(guard, guard.line)} ${side}`
}

/** 守れたかの文 (数えなかった, 線が無い場合はそう書く) */
export function guardVerdict(guard: BacktestGuard): string {
  if (guard.kept === true) return '守れた'
  if (guard.kept === false) return '守れなかった'
  if (guard.value === null) return '数えていない'
  return '守る線が無いので決めない'
}
