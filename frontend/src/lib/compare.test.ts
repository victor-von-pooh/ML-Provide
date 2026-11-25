import type { Result } from '../api/types'
import { compareSections, shareDiffText } from './compare'

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
    expect(find(sections, '意思決定', '基準 (全てに行動する) の見込みの得')).toMatchObject({
      values: ['-5', '-5'],
      differs: false,
    })
    expect(find(sections, '意思決定', '基準 (全てに行動する) の見込みの損')).toBeUndefined()
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

// 同じ群分けの一式の 2 つの結果 (結果 2 は群 0 が増え, 群 2 に行が無い)
const learned = [
  { group: 0, rows: 50, share: 0.5, stability: 0.9, spend: 10, region: 'north (80 %)' },
  { group: 1, rows: 30, share: 0.3, stability: 0.8, spend: 20, region: 'south (70 %)' },
  { group: 2, rows: 20, share: 0.2, stability: 0.7, spend: 30, region: 'east (90 %)' },
  { group: '全体', rows: 100, share: 1, stability: null, spend: 17, region: 'north (40 %)' },
]
const clusterBase: Result = {
  id: 'g1',
  bundle: { name: '顧客の群', path: '/bundles/group' },
  bundle_id: 'gb',
  source: 'january.csv',
  created_at: '2026-01-01T00:00:00Z',
  task: null,
  output_form: 'cluster',
  column_output: false,
  summary: {
    rows: 10,
    predicted: 10,
    skipped: 0,
    k: 3,
    groups: [
      { group: 0, rows: 5, share: 0.5, stability: 0.9, spend: 11, region: 'north (80 %)' },
      { group: 1, rows: 3, share: 0.3, stability: 0.8, spend: 21, region: 'south (60 %)' },
      { group: 2, rows: 2, share: 0.2, stability: 0.7, spend: 31, region: 'east (50 %)' },
      { group: '全体', rows: 10, share: 1, stability: null, spend: 18, region: 'north (40 %)' },
    ] as unknown as NonNullable<Result['summary']>['groups'],
  },
  sentences: ['10 行のうち 10 行を 3 個の群に割り当てた'],
  overall_count: 1,
  columns: ['index', 'group', 'distance', 'spend', 'region'],
  grouping: { k: 3, columns: ['spend', 'region'], learned },
}
const clusterLater: Result = {
  ...clusterBase,
  id: 'g2',
  source: 'february.csv',
  summary: {
    rows: 8,
    predicted: 8,
    skipped: 0,
    k: 3,
    groups: [
      { group: 0, rows: 6, share: 0.75, stability: 0.9, spend: 12, region: 'north (90 %)' },
      { group: 1, rows: 2, share: 0.25, stability: 0.8, spend: 22, region: 'south (50 %)' },
      { group: 2, rows: 0, share: 0, stability: 0.7, spend: null, region: null },
      { group: '全体', rows: 8, share: 1, stability: null, spend: 14, region: 'north (75 %)' },
    ] as unknown as NonNullable<Result['summary']>['groups'],
  },
}

describe('群分けの結果どうしの比べ', () => {
  const sections = compareSections([clusterBase, clusterLater])
  const titles = sections.map((s) => s.title)

  it('結論の決め方, 判断の内訳, 意思決定の代わりに群の要約を並べる', () => {
    expect(titles).not.toContain('結論の決め方')
    expect(titles).not.toContain('判断の内訳')
    expect(titles).toContain('群の大きさ (件数, 割合, 学習のときの割合との差)')
    expect(titles).toContain('結果 1 からの割合の差')
    expect(titles).toContain('群 0 の列の特徴')
    expect(titles).toContain('全体の列の特徴')
    expect(find(sections, '結論', '割り当てた行')?.values).toEqual(['10', '8'])
    expect(find(sections, '結論', '群の数')?.values).toEqual(['3', '3'])
  })

  it('群ごとの件数, 割合と学習のときとの差を並べ, 行の無い群も 0 件で出す', () => {
    const title = '群の大きさ (件数, 割合, 学習のときの割合との差)'
    const zero = find(sections, title, '群 0 (学習のとき 50.0%)')
    expect(zero?.values).toEqual([
      '5 件 (50.0%, 学習から ±0.0 ポイント)',
      '6 件 (75.0%, 学習から +25.0 ポイント (増えた))',
    ])
    expect(zero?.differs).toBe(true)
    expect(find(sections, title, '群 2 (学習のとき 20.0%)')?.values[1]).toBe(
      '0 件 (0.0%, 学習から -20.0 ポイント (減った))',
    )
  })

  it('結果 1 からの差で増えた群と減った群が分かる', () => {
    const title = '結果 1 からの割合の差'
    expect(find(sections, title, '群 0')?.values).toEqual(['–', '+25.0 ポイント (増えた)'])
    expect(find(sections, title, '群 1')?.values).toEqual(['–', '-5.0 ポイント (減った)'])
    // 差が無い群は違う印を付けない
    const same = compareSections([clusterBase, { ...clusterBase, id: 'g3' }])
    expect(find(same, title, '群 0')?.differs).toBe(false)
  })

  it('列の特徴は数の列は平均, 種類の列は最も多い値で, 学習のときの値を添える', () => {
    expect(find(sections, '群 0 の列の特徴', 'spend (平均) (学習のとき 10)')?.values).toEqual([
      '11',
      '12',
    ])
    const region = find(sections, '群 2 の列の特徴', 'region (最も多い値) (学習のとき east (90 %))')
    expect(region?.values).toEqual(['east (50 %)', '–'])
  })

  it('差をポイントで書き, 表示の桁で 0 なら ±0.0 にする', () => {
    expect(shareDiffText(0.3, 0.25)).toBe('+5.0 ポイント (増えた)')
    expect(shareDiffText(0.2, 0.25)).toBe('-5.0 ポイント (減った)')
    expect(shareDiffText(0.25001, 0.25)).toBe('±0.0 ポイント')
  })
})
