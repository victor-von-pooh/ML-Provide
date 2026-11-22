import type { Result } from '../api/types'
import { compareSections } from './compare'

// 解約の見本の同じ入力の結果 (決め方と意思決定だけが違う)
const base: Result = {
  id: 'a',
  bundle: { name: '解約', path: '/bundles/churn' },
  bundle_id: 'b1',
  source: 'past.csv',
  created_at: '2026-01-01T00:00:00Z',
  task: 'classification',
  rules: { min_confidence: 0.6 },
  summary: {
    rows: 10,
    predicted: 10,
    skipped: 0,
    decisions: [
      { decision: '解約', count: 3, share: 0.3 },
      { decision: '保留', count: 7, share: 0.7 },
    ],
  },
  sentences: ['全体の文', 'まとめる列の文'],
  overall_count: 1,
  columns: ['index', 'prediction', 'decision'],
  decision: {
    decision: { type: 'select', value: 100, cost: 10, effect: 0.3, positive: '解約' },
    summary: { rows: 10, acting: 2, expected_gain: 40, baselines: { 全てに行動する: -5 } },
    sentences: [],
    notes: [],
  },
}

const other: Result = {
  ...base,
  id: 'b',
  created_at: '2026-01-02T00:00:00Z',
  rules: {},
  summary: {
    rows: 10,
    predicted: 10,
    skipped: 0,
    decisions: [{ decision: '解約', count: 4, share: 0.4 }],
  },
  columns: ['index', 'prediction', 'decision', 'action'],
  decision: {
    ...base.decision!,
    decision: { ...base.decision!.decision, cost: 20 },
    summary: { rows: 10, acting: 1, expected_gain: 25, baselines: { 全てに行動する: -5 } },
  },
}

/** まとまりの見出しから行の名前と違うかを引く */
function find(sections: ReturnType<typeof compareSections>, title: string, label: string) {
  return sections.find((s) => s.title === title)?.rows.find((r) => r.label === label)
}

describe('結果どうしの比べ', () => {
  const sections = compareSections([base, other])

  it('違う決め方と意思決定の値を目立たせ, 同じ値と時刻は目立たせない', () => {
    expect(find(sections, '結論の決め方', '保留にする確信の下限')).toMatchObject({
      values: ['0.6', '–'],
      differs: true,
    })
    expect(find(sections, '何を比べたか', '一式')?.differs).toBe(false)
    expect(find(sections, '何を比べたか', '時刻')?.differs).toBe(false)
    expect(find(sections, '意思決定', '行動 1 件の費用')).toMatchObject({
      values: ['10', '20'],
      differs: true,
    })
    expect(find(sections, '意思決定', '見込みの得')?.values).toEqual(['40', '25'])
    expect(find(sections, '意思決定', '全てに行動するの見込み')?.differs).toBe(false)
  })

  it('判断の内訳は全ての結果の名前の和で, 無い結果は 0 件', () => {
    expect(find(sections, '判断の内訳', '保留')?.values).toEqual(['7 (70.0%)', '0'])
    expect(find(sections, '結論', '結論の文')?.values).toEqual(['全体の文', '全体の文'])
  })

  it('値の無い行と, 全ての結果にある列は出さない', () => {
    expect(find(sections, '結論の決め方', 'まとめる列')).toBeUndefined()
    expect(find(sections, '結論', '予測の平均')).toBeUndefined()
    expect(find(sections, '予測の表の列', '他の結果に無い列')?.values).toEqual(['–', 'action'])
  })

  it('意思決定の無い結果だけならまとまりを出さない', () => {
    const plain = compareSections([
      { ...base, decision: null },
      { ...other, decision: null, columns: base.columns },
    ])
    expect(plain.map((s) => s.title)).toEqual([
      '何を比べたか',
      '結論の決め方',
      '結論',
      '判断の内訳',
    ])
  })
})
