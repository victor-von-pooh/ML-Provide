import type { Result, UpliftInfo } from '../api/types'
import {
  budgetProblem,
  distributionText,
  learnerText,
  modelText,
  resultK,
  topText,
  treatSummary,
  upliftSummary,
} from './uplift'

describe('効果の欄と結果', () => {
  it('上位 K 割を % で書く (小数の誤差を出さない)', () => {
    expect(topText(0.3)).toBe('上位 30 %')
    expect(topText(0.25)).toBe('上位 25 %')
    expect(topText(0.125)).toBe('上位 12.5 %')
    expect(topText(1)).toBe('上位 100 %')
    expect(topText(null)).toBe('上位 – %')
  })

  it('配る数の上限は 1 から 100,000 の整数だけ', () => {
    expect(budgetProblem('8')).toBeNull()
    expect(budgetProblem(' 100000 ')).toBeNull()
    for (const text of ['', '0', '2.5', '-1', 'a', '100001', '1e21'])
      expect(budgetProblem(text)).toMatch(/1 から 100,000 の整数/)
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
    const blank = { ...result.decision!, summary: null } as unknown as Result['decision']
    expect(treatSummary({ ...result, decision: blank })).toBeNull()
  })

  it('K は一式から写した K, 無ければ結論の K, どちらも無ければ null', () => {
    const summary = { top_k: { k: 0.2, rows: 1, mean: 0, lowest: 0 } } as never
    const base = { kind: 'uplift' } as Result
    expect(resultK({ ...base, uplift: { k: 0.25, outcome: 'binary', curves: {} } }, summary)).toBe(
      0.25,
    )
    expect(resultK(base, summary)).toBe(0.2)
    expect(resultK(base, { top_k: null } as never)).toBeNull()
  })

  it('効果の決め方の名前と, 入力の幅と出力の文 (S-learner は処置の列を足した幅と出力 1 つ)', () => {
    expect(learnerText('s_learner')).toBe(
      '処置の列を入力の終わりに足した 1 つの予測の差 (S-learner). 処置したときの予測は処置の列を 1 に, しないときの予測は 0 にして入力の終わりに足した 1 つの予測です',
    )
    expect(learnerText('t_learner')).toMatch(
      /^処置の群と対照の群の 2 つの予測の差 \(T-learner\)\. /,
    )
    expect(learnerText('x_learner')).toBe('x_learner')
    const base = { features: ['age', 'visits'], treatment: 'coupon', input: 'x' }
    const single = { ...base, learner: 's_learner', output: 'outcome' } as UpliftInfo
    expect(modelText(single)).toBe(
      '入力 x の幅 3 (見積もりに使う列 2 と処置の列 coupon の 1. 処置の列には一式が 1 と 0 を入れます). 出力は結果の予測 outcome の 1 つで, 処置の列を 1 と 0 にして 2 回予測します',
    )
    const pair = { ...base, learner: 't_learner', treated: 'treated', control: 'control' }
    expect(modelText(pair as UpliftInfo)).toBe(
      '入力 x の幅 2 (見積もりに使う列. 処置の列は入れません). 出力は処置したときの予測 treated としないときの予測 control の 2 つです',
    )
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
