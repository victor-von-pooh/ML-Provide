import { groupColumns } from './remeasure'

describe('外れ方を分ける列', () => {
  const own = new Set(['index', 'status', 'prediction'])

  it('予測と判断の列, 確率の列, 目的の列を除く', () => {
    expect(
      groupColumns('regression', ['shop', 'visitors', 'prediction', 'p:x', 'a'], 'visitors', own),
    ).toEqual(['shop', 'a'])
  })

  it('分類の一式では分けない', () => {
    expect(groupColumns('classification', ['shop', 'label'], 'label', own)).toEqual([])
  })
})
