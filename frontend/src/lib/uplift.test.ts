import type { Result } from '../api/types'
import { budgetProblem, distributionText, topText, treatSummary, upliftSummary } from './uplift'

describe('効果の欄と結果', () => {
  it('上位 K 割を % で書く (小数の誤差を出さない)', () => {
    expect(topText(0.3)).toBe('上位 30 %')
    expect(topText(0.25)).toBe('上位 25 %')
    expect(topText(0.125)).toBe('上位 12.5 %')
    expect(topText(1)).toBe('上位 100 %')
  })

  it('配る数の上限は 1 以上の整数だけ', () => {
    expect(budgetProblem('8')).toBeNull()
    expect(budgetProblem(' 5000 ')).toBeNull()
    for (const text of ['', '0', '2.5', '-1', 'a'])
      expect(budgetProblem(text)).toMatch(/1 以上の整数/)
  })

  it('効果の結果と配る判断のまとめだけを読む', () => {
    const summary = { rows: 1, predicted: 1, skipped: 0, top: [] }
    const result = {
      kind: 'uplift',
      summary,
      decision: {
        decision: { type: 'treat', budget: 3 },
        summary: { rows: 1, acted: 1, budget: 3 },
        sentences: [],
        notes: [],
      },
    } as unknown as Result
    expect(upliftSummary(result)).toBe(summary)
    expect(treatSummary(result)?.budget).toBe(3)
    expect(upliftSummary({ ...result, kind: 'anomaly' })).toBeNull()
    expect(treatSummary({ ...result, decision: null })).toBeNull()
  })

  it('学習用の行の効果の分布の文', () => {
    expect(
      distributionText({
        rows: 200,
        mean: 0.36,
        min: -0.3,
        p10: -0.02,
        p50: 0.32,
        p90: 0.83,
        max: 0.97,
        positive: 0.89,
      }),
    ).toBe(
      '行 200, 平均 0.36, 最小 -0.3, 10 % 点 -0.02, 中央 0.32, 90 % 点 0.83, 最大 0.97, 効果が正の行 89.0 %',
    )
  })
})
