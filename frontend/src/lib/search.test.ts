import { describe, expect, it } from 'vitest'

import { groupQueries, searchPage } from './search'

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

describe('searchPage', () => {
  const ok = (path: string, rank: number) => ({ path, status: 'ok', rank, item: `${path}-${rank}` })
  const skipped = (path: string) => ({ path, status: 'skipped', reason: '読めません', rank: null })

  it('後ろに行が残るときは最後の問いを次のページに回し, 次の位置を問いの境目にする', () => {
    // 問い a (2 件), 読めない b (1 行), c の 1 件目までを読んだ (c は途中)
    const rows = [ok('a', 1), ok('a', 2), skipped('b'), ok('c', 1)]
    const page = searchPage(rows, 0, 10)
    expect(page.queries.map((q) => q.path)).toEqual(['a', 'b'])
    expect(page.next).toBe(3)
  })

  it('最後のページは全ての問いを出し, 次の位置は無い', () => {
    const page = searchPage([ok('c', 1), ok('c', 2)], 3, 5)
    expect(page.queries.map((q) => q.path)).toEqual(['c'])
    expect(page.next).toBeNull()
  })

  it('問いが 1 つだけのページは切れていても出して先へ進む', () => {
    const page = searchPage([ok('a', 1), ok('a', 2)], 0, 10)
    expect(page.queries).toHaveLength(1)
    expect(page.next).toBe(2)
  })
})
