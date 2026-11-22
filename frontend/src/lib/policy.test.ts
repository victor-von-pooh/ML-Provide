import { actionLabel, parseObservation, spaceLabel, spaceSize } from './policy'

describe('方策の観測の欄', () => {
  it('区切りの数の並びと JSON の配列を読む', () => {
    expect(parseObservation('0.1, -0.2 3')).toEqual({ value: [0.1, -0.2, 3] })
    expect(parseObservation('[[1, 2], [3, 4]]')).toEqual({
      value: [
        [1, 2],
        [3, 4],
      ],
    })
  })

  it('空の欄と数でない値を誤りにする', () => {
    expect(parseObservation('  ')).toEqual({ error: '観測を入れてください' })
    expect(parseObservation('1, a')).toEqual({ error: 'a は数として読めません' })
    expect(parseObservation('[1,')).toEqual({ error: 'JSON の配列として読めません' })
  })
})

describe('方策の空間と行動の見せ方', () => {
  it('空間の要素の数と説明', () => {
    expect(spaceSize({ type: 'box', shape: [2, 3], low: [], high: [] })).toBe(6)
    expect(spaceSize({ type: 'discrete', n: 2, start: 0 })).toBe(1)
    expect(spaceLabel({ type: 'discrete', n: 2, start: 0 })).toBe('番号 0 から 1 (2 通り)')
  })

  it('行動なしと数の並び', () => {
    expect(actionLabel(null)).toBe('行動なし')
    expect(actionLabel(1)).toBe('1')
    expect(actionLabel([0.1])).toBe('[0.1]')
  })
})
