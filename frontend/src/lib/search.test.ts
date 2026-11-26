import { describe, expect, it } from 'vitest'

import { groupQueries } from './search'

describe('groupQueries', () => {
  it('問いごとに順位の順でまとめ, 並べられない問いは写真無しの 1 件にする', () => {
    const rows = [
      {
        path: 'a.png',
        status: 'ok',
        reason: '',
        rank: 1,
        item: 'red/1.png',
        label: 'red',
        similarity: 0.9,
        latency_ms: 3,
      },
      {
        path: 'a.png',
        status: 'ok',
        reason: '',
        rank: 2,
        item: 'red/2.png',
        label: 'red',
        similarity: 0.8,
        latency_ms: 3,
      },
      {
        path: 'b.png',
        status: 'skipped',
        reason: '画像を読めません',
        rank: null,
        item: null,
        label: null,
        similarity: null,
        latency_ms: null,
      },
      {
        path: 'c.png',
        status: 'ok',
        reason: '',
        rank: 1,
        item: 'x.png',
        label: null,
        similarity: 0.5,
        latency_ms: 2,
      },
    ]
    const queries = groupQueries(rows)
    expect(queries.map((q) => q.path)).toEqual(['a.png', 'b.png', 'c.png'])
    expect(queries[0]?.items.map((item) => item.rank)).toEqual([1, 2])
    expect(queries[0]?.latency).toBe(3)
    expect(queries[1]?.items).toEqual([])
    expect(queries[1]?.reason).toBe('画像を読めません')
    expect(queries[2]?.items[0]?.label).toBeNull()
  })

  it('同じ写真を 2 度問うと, 1 位から別の問いにする', () => {
    const row = {
      path: 'a.png',
      status: 'ok',
      reason: '',
      item: 'x.png',
      label: null,
      similarity: 1,
      latency_ms: 1,
    }
    const queries = groupQueries([
      { ...row, rank: 1 },
      { ...row, rank: 1 },
    ])
    expect(queries).toHaveLength(2)
  })
})
