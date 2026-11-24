import {
  metricText,
  rankColumnsText,
  rankKProblem,
  rankRowColumns,
  rankUsers,
  reactionRuleText,
} from './ranking'

describe('並べるの結果と欄', () => {
  it('表の列は利用者, 順位, 品物, 点数, 状態, 理由の順で, 判断の列は出さない', () => {
    expect(
      rankRowColumns(['index', 'user', 'status', 'reason', 'rank', 'item', 'score', 'decision']),
    ).toEqual(['index', 'user', 'rank', 'item', 'score', 'status', 'reason'])
    expect(rankRowColumns(['index', 'user', 'extra', 'rank'])).toEqual([
      'index',
      'user',
      'rank',
      'extra',
    ])
  })

  it('書いた利用者は 1 行に 1 人で, 空の行を飛ばす', () => {
    expect(rankUsers(' u0 \n\n007\n  \n103')).toEqual(['u0', '007', '103'])
    expect(rankUsers('')).toEqual([])
  })

  it('並べる件数は空か 1 から 1000 の整数', () => {
    expect(rankKProblem('')).toBeNull()
    expect(rankKProblem('5')).toBeNull()
    expect(rankKProblem('1000')).toBeNull()
    for (const text of ['0', '1001', '2.5', '-1', 'a'])
      expect(rankKProblem(text)).toBe('並べる件数は 1 から 1000 の整数で書いてください')
  })

  it('反応の表の列と反応の決まりの文', () => {
    const columns = { user: 'member', item: 'item_id', reaction: 'qty' }
    expect(rankColumnsText(columns)).toBe('利用者 member, 品物 item_id, 反応の強さ qty')
    expect(reactionRuleText(columns)).toBe('qty が 0 より大きい行を反応とします')
    expect(rankColumnsText({ user: 'u', item: 'i', reaction: null })).toBe('利用者 u, 品物 i')
    expect(reactionRuleText({ user: 'u', item: 'i' })).toBe('全ての行を反応とします')
    expect(metricText(0.5)).toBe('0.500')
  })
})
