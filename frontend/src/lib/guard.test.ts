import type { BacktestGuard } from '../api/types'
import { guardAmount, guardLine, guardVerdict } from './guard'

const base: BacktestGuard = {
  name: '欠品の率',
  unit: null,
  counted: 'shortage_rate',
  description: '足りなかった行の割合',
  value: 0.1234,
  line: 0.1,
  better: 'lower',
  kept: false,
  note: null,
}

describe('守る指標の出し方', () => {
  it('割合は値も線も % で出す', () => {
    expect(guardAmount(base, base.value)).toBe('12.3 %')
    expect(guardLine(base)).toBe('10 % 以下')
    expect(guardVerdict(base)).toBe('守れなかった')
  })

  it('割合でない量は単位を添え, 大きな値は整数にする', () => {
    const guard = { ...base, counted: 'shortage', unit: '個', value: 12345.6, line: 20000 }
    expect(guardAmount(guard, guard.value)).toBe('12,346 個')
    expect(guardLine({ ...guard, better: 'higher' })).toBe('20,000 個 以上')
  })

  it('線が無いか数えなければそう書く', () => {
    expect(guardLine({ ...base, line: null })).toBe('無い')
    expect(guardVerdict({ ...base, line: null, kept: null })).toBe('守る線が無いので決めない')
    const uncounted = { ...base, counted: null, value: null, kept: null, note: '課題の場で数える' }
    expect(guardAmount(uncounted, uncounted.value)).toBe('–')
    expect(guardVerdict(uncounted)).toBe('数えていない')
  })
})
