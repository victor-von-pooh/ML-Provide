import { fromDecisionForm, toDecisionForm } from './decision'

describe('意思決定の編集', () => {
  it('数と列の名前を分けて渡す', () => {
    const form = {
      type: 'quantity' as const,
      positive: '',
      values: { under_cost: '3', over_cost: '1', stock: '在庫' },
    }
    expect(fromDecisionForm(form, [])).toEqual({
      decision: { type: 'quantity', under_cost: 3, over_cost: 1, stock: '在庫' },
      problems: [],
    })
  })

  it('省けない欄, 当たりのクラス, 数だけの欄を確かめる', () => {
    const form = { type: 'select' as const, positive: '', values: { value: '100', budget: 'x' } }
    const { problems } = fromDecisionForm(form, ['いる', 'いない'])
    expect(problems).toContain('当たりのクラスを選んでください')
    expect(problems.some((p) => p.includes('費用'))).toBe(true)
    expect(problems.some((p) => p.includes('数で書いて'))).toBe(true)
  })

  it('行き来できる', () => {
    const decision = { type: 'triage' as const, positive: 'yes', miss_cost: 9, false_alarm_cost: 1 }
    expect(fromDecisionForm(toDecisionForm(decision), ['no', 'yes']).decision).toEqual(decision)
  })

  it('全角の数字とカンマをそろえ, 範囲と整数を確かめる', () => {
    const form = {
      type: 'select' as const,
      positive: 'いる',
      values: { value: '１２,０００', cost: '3000', effect: '1.5', budget: '2.5' },
    }
    const { decision, problems } = fromDecisionForm(form, ['いる', 'いない'])
    expect(decision?.value).toBe(12000)
    expect(problems.some((p) => p.includes('0 から 1'))).toBe(true)
    expect(problems.some((p) => p.includes('整数'))).toBe(true)
  })

  it('選ばなければ意思決定しない', () => {
    expect(fromDecisionForm(toDecisionForm(null), [])).toEqual({ decision: null, problems: [] })
  })
})
