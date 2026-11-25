import type { Result } from '../api/types'
import {
  anomalySummary,
  capacityProblem,
  normalText,
  reviewerProblem,
  triageSummary,
} from './anomaly'

describe('異常の度合いの欄と結果', () => {
  it('件数の上限は 1 から 1000 の整数だけ', () => {
    expect(capacityProblem('3')).toBeNull()
    expect(capacityProblem(' 1000 ')).toBeNull()
    for (const text of ['', '0', '1001', '2.5', '-1', 'a'])
      expect(capacityProblem(text)).toMatch(/1 から 1000 の整数/)
  })

  it('確かめた人の名前は空でなく 80 文字まで', () => {
    expect(reviewerProblem('佐藤')).toBeNull()
    expect(reviewerProblem('  ')).toMatch(/名前を書いて/)
    expect(reviewerProblem('a'.repeat(81))).toMatch(/80 文字まで/)
  })

  it('異常の度合いの結果のまとめと振り分けだけを読む', () => {
    const summary = { rows: 2, predicted: 2, skipped: 0, top: [], evaluations: {} }
    const result = {
      kind: 'anomaly',
      summary,
      decision: {
        decision: { type: 'triage', capacity: 3 },
        summary: { rows: 2, acted: 2, capacity: 3, lowest: 1, normal_p99: 2, left_beyond: 0 },
        sentences: [],
        notes: [],
      },
    } as unknown as Result
    expect(anomalySummary(result)).toBe(summary)
    expect(triageSummary(result)?.acted).toBe(2)
    expect(anomalySummary({ ...result, kind: 'table' })).toBeNull()
    expect(triageSummary({ ...result, decision: null })).toBeNull()
  })

  it('正常の行の分布を 1 行の文に', () => {
    expect(
      normalText({ rows: 36, mean: 0.9, p50: 0.8, p90: 1.8, p95: 2.1, p99: 3.4, max: 3.8 }),
    ).toBe('行 36, 平均 0.9, 中央 0.8, 90 % 点 1.8, 95 % 点 2.1, 99 % 点 3.4, 最大 3.8')
  })
})
