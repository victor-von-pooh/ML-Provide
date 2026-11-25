import type { ClusterRow, Result } from '../api/types'
import {
  clusterName,
  clusterRepresentatives,
  clusterRowColumns,
  clusterRows,
  representativeValue,
  diffText,
  featureHeading,
  scoreText,
  stabilityLines,
  stabilityMethodText,
} from './clusters'

const stability = {
  k: 3,
  method: 'seed',
  repeats: 5,
  ari: 0.97,
  ari_baseline: 0.99,
  ari_diff: -0.02,
  ari_min: 0.95,
  ari_min_baseline: 0.98,
  ari_min_diff: -0.03,
  silhouette: 0.62,
  silhouette_baseline: 0.55,
  silhouette_diff: 0.07,
}

describe('群分けの結果と欄', () => {
  it('表の列は行の位置, 群, 距離, 状態, 理由, 元の列の順で, 判断の列は出さない', () => {
    expect(
      clusterRowColumns([
        'index',
        'customer',
        'spend',
        'status',
        'reason',
        'group',
        'distance',
        'decision',
      ]),
    ).toEqual(['index', 'group', 'distance', 'status', 'reason', 'customer', 'spend'])
  })

  it('列の特徴の見出しは, 数の列は平均, 種類の列は最も多い値 (行の無い群の null は見ない)', () => {
    const rows = [
      { group: 0, rows: 0, share: 0, stability: 0.9, spend: null, region: null },
      { group: 1, rows: 2, share: 1, stability: 0.9, spend: 10.5, region: 'north (100 %)' },
    ] as ClusterRow[]
    expect(featureHeading('spend', rows)).toBe('spend (平均)')
    expect(featureHeading('region', rows)).toBe('region (最も多い値)')
    expect(featureHeading('missing', rows)).toBe('missing')
  })

  it('群の要約は群分けの結果だけで読む', () => {
    const groups = [{ group: 0, rows: 1, share: 1, stability: null }]
    const base = { summary: { rows: 1, predicted: 1, skipped: 0, groups } } as unknown as Result
    expect(clusterRows({ ...base, output_form: 'cluster' })).toEqual(groups)
    expect(clusterRows({ ...base, output_form: 'class' })).toEqual([])
  })

  it('群の安定性の 3 つの指標と, 値と差と作り直し方の文', () => {
    const lines = stabilityLines(stability)
    expect(lines.map((line) => [line.key, line.model, line.baseline, line.diff])).toEqual([
      ['ari', 0.97, 0.99, -0.02],
      ['ari_min', 0.95, 0.98, -0.03],
      ['silhouette', 0.62, 0.55, 0.07],
    ])
    expect(scoreText(0.62)).toBe('0.620')
    expect(scoreText(null)).toBe('–')
    expect(diffText(0.07)).toBe('+0.070')
    expect(diffText(-0.02)).toBe('-0.020')
    expect(diffText(0.00001)).toBe('±0.000')
    expect(stabilityMethodText(stability)).toBe('種を変えて作り直す, 5 回')
    expect(stabilityMethodText({ ...stability, method: 'other' })).toBe('other, 5 回')
    expect(clusterName(2)).toBe('群 2')
    expect(clusterName('全体')).toBe('全体')
  })

  it('結論の群の代表の行は群分けの結果だけで, 結論に無ければ undefined', () => {
    const representatives = [{ group: 0, rows: [{ row: 4, distance: 0.1, values: { a: 1 } }] }]
    const grouped = {
      output_form: 'cluster',
      summary: { rows: 1, predicted: 1, skipped: 0, representatives },
    } as unknown as Result
    expect(clusterRepresentatives(grouped)).toEqual(representatives)
    expect(
      clusterRepresentatives({ ...grouped, summary: { rows: 1, predicted: 1, skipped: 0 } }),
    ).toBeUndefined()
    expect(clusterRepresentatives({ ...grouped, output_form: 'class' })).toBeUndefined()
  })

  it('代表の行の数の列の値は, 文字でも数に読めれば数にする', () => {
    expect(representativeValue('1045.25', true)).toBe(1045.25)
    expect(representativeValue('001', false)).toBe('001')
    expect(representativeValue('?', true)).toBe('?')
    expect(representativeValue('', true)).toBe('')
    expect(representativeValue(null, true)).toBeNull()
    expect(representativeValue(3, true)).toBe(3)
  })
})
