import type { Result } from '../api/types'

/** 行ごとの理由を求められる結果か (表と時系列の, 1 つ先まで予測した列の出力の結果) */
export function rowReasonable(result: Result, bundleAlive: boolean): boolean {
  // 元のデータセットの分割の結果は, 前処理の前の列が無いので求めない
  if (result.column_output === false || result.forecast || result.split || !bundleAlive)
    return false
  return !result.kind || ['table', 'timeseries'].includes(result.kind)
}
