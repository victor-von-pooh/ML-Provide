import type { Result } from '../api/types'
import {
  candidatesText,
  changeText,
  costText,
  elasticityText,
  limitText,
  outsideCandidates,
  parseCandidates,
  parseMaxChange,
  priceDecisionSummary,
  priceSummary,
} from './price'

describe('値段の一式の欄と言葉', () => {
  it('割合を % で書く', () => {
    expect(changeText(0.1)).toBe('+10 %')
    expect(changeText(-0.05)).toBe('-5 %')
    expect(changeText(0)).toBe('0 %')
    expect(changeText(0.07)).toBe('+7 %')
    expect(changeText(null)).toBe('–')
    expect(limitText(0.1)).toBe('±10 %')
    expect(candidatesText([-0.1, 0, 0.2])).toBe('-10 %, 0 %, +20 %')
  })

  it('値段の候補の欄をカンマか空白で区切った割合の小数で読む', () => {
    expect(parseCandidates('-0.1, 0, 0.1')).toEqual({ values: [-0.1, 0, 0.1], problem: null })
    expect(parseCandidates('-0.1 0 .05')).toEqual({ values: [-0.1, 0, 0.05], problem: null })
    expect(parseCandidates('').problem).toMatch(/割合の小数で書いて/)
    expect(parseCandidates('0, 10%').problem).toMatch(/「10%」は数ではありません/)
    expect(parseCandidates('0, -1').problem).toMatch(/-1 より大きい/)
    expect(parseCandidates('0, 0.1, 0.10').problem).toMatch(/重なって/)
    expect(parseCandidates('0.1, 0.2').problem).toMatch(/今の値段 \(0\) を含めて/)
    const many = Array.from({ length: 21 }, (_, index) => (index / 100).toString()).join(',')
    expect(parseCandidates(many).problem).toMatch(/20 個まで/)
  })

  it('変える幅の上限は省けず, 0 以上の割合', () => {
    expect(parseMaxChange('0.1')).toEqual({ value: 0.1, problem: null })
    expect(parseMaxChange('0')).toEqual({ value: 0, problem: null })
    expect(parseMaxChange(' ').problem).toMatch(/省けません/)
    expect(parseMaxChange('-0.1').problem).toMatch(/0 以上/)
    expect(parseMaxChange('abc').problem).toMatch(/数で書いて/)
  })

  it('上限を超える候補を返す (上限と同じ割合は選べる)', () => {
    expect(outsideCandidates([-0.1, 0, 0.1, 0.2], 0.1)).toEqual([0.2])
    expect(outsideCandidates([-0.1, 0, 0.1], null)).toEqual([])
  })

  it('原価の列と弾力性の上げ幅に合わせて説明を書く', () => {
    expect(costText('unit_cost')).toContain('原価の列 unit_cost')
    expect(costText(null)).toContain('利益を売上')
    expect(elasticityText(0.01)).toContain('値段を 1 % 上げたとき')
  })

  it('値段の結果と値段の判断のまとめだけを読む', () => {
    const result = {
      kind: 'price',
      summary: { rows: 1, predicted: 1, skipped: 0, evaluations: {} },
      decision: {
        decision: { type: 'price', candidates: [0], max_change: 0 },
        summary: { rows: 1, limited: 0 },
        sentences: [],
        notes: [],
      },
    } as unknown as Result
    expect(priceSummary(result)).not.toBeNull()
    expect(priceDecisionSummary(result)?.limited).toBe(0)
    expect(priceSummary({ ...result, kind: 'uplift' })).toBeNull()
    expect(priceDecisionSummary({ ...result, decision: null } as unknown as Result)).toBeNull()
  })
})
