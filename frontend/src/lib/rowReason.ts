import type { Result } from '../api/types'

/** 行ごとの理由を求められる結果か (表と時系列の, 1 つ先まで予測した列の出力の結果) */
export function rowReasonable(result: Result, bundleAlive: boolean): boolean {
  if (result.column_output === false || result.forecast || !bundleAlive) return false
  return !result.kind || ['table', 'timeseries'].includes(result.kind)
}
