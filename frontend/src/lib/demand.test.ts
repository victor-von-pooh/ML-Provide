import type { Result } from '../api/types'
import {
  baselineText,
  demandSummary,
  demandText,
  levelsText,
  levelText,
  orderDecisionSummary,
  outsideLevel,
  parseCost,
  parseMaxOrder,
  stockText,
  tauOf,
} from './demand'

describe('需要の一式の欄と言葉', () => {
  it('損の欄を正の数で読む', () => {
    expect(parseCost('3', '品切れの損')).toEqual({ value: 3, problem: null })
    expect(parseCost(' 0.5 ', '品切れの損')).toEqual({ value: 0.5, problem: null })
    expect(parseCost('1e2', '品切れの損')).toEqual({ value: 100, problem: null })
    expect(parseCost('', '品切れの損').problem).toBe('品切れの損を書いてください (省けません)')
    expect(parseCost('abc', '売れ残りの損').problem).toBe('売れ残りの損は数で書いてください')
    expect(parseCost('0', '売れ残りの損').problem).toMatch(/0 より大きい数/)
    expect(parseCost('-1', '売れ残りの損').problem).toMatch(/0 より大きい数/)
    expect(parseCost('1e400', '売れ残りの損').problem).toMatch(/0 より大きい数/)
  })

  it('発注の数の上限を 0 以上の整数で読む', () => {
    expect(parseMaxOrder('60')).toEqual({ value: 60, problem: null })
    expect(parseMaxOrder('0')).toEqual({ value: 0, problem: null })
    expect(parseMaxOrder('').problem).toMatch(/省けません/)
    expect(parseMaxOrder('2.5').problem).toMatch(/0 以上の整数/)
    expect(parseMaxOrder('-1').problem).toMatch(/0 以上の整数/)
    expect(parseMaxOrder('99999999999999999999').problem).toMatch(/大きすぎます/)
  })

  it('水準 τ と, 学んだ水準の範囲の外かを求める', () => {
    expect(tauOf(3, 1)).toBe(0.75)
    expect(tauOf(null, 1)).toBeNull()
    expect(levelText(tauOf(1, 2))).toBe('0.3333')
    expect(levelText(null)).toBe('–')
    const levels = [0.1, 0.25, 0.5, 0.75, 0.9]
    expect(outsideLevel(0.75, levels)).toBeNull()
    expect(outsideLevel(0.9, levels)).toBeNull()
    expect(outsideLevel(0.95, levels)).toBe(0.9)
    expect(outsideLevel(0.05, levels)).toBe(0.1)
    expect(outsideLevel(null, levels)).toBeNull()
    expect(levelsText(levels)).toBe('0.1, 0.25, 0.5, 0.75, 0.9')
  })

  it('先の期間と在庫と比べる決め方の言葉', () => {
    expect(demandText(7)).toMatch(/先の 7 日/)
    expect(baselineText(7)).toMatch(/前の 7 日に売れた数の合計/)
    expect(stockText('stock')).toMatch(/在庫の数の列 stock/)
    expect(stockText(null)).toMatch(/在庫を 0 として/)
  })

  it('需要の結果と発注の判断のまとめだけを読む', () => {
    const demand = {
      kind: 'demand',
      summary: { rows: 1, predicted: 1, skipped: 0, levels: [0.5], horizon: 7 },
      decision: {
        decision: { type: 'order', shortage_cost: 3, holding_cost: 1, max_order: 60 },
        summary: { rows: 1, capped: 0 },
        sentences: [],
        notes: [],
      },
    } as unknown as Result
    expect(demandSummary(demand)?.horizon).toBe(7)
    expect(orderDecisionSummary(demand)?.capped).toBe(0)
    const other = { ...demand, kind: 'price', decision: null } as unknown as Result
    expect(demandSummary(other)).toBeNull()
    expect(orderDecisionSummary(other)).toBeNull()
  })
})
