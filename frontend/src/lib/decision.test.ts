import { baselineText, fromDecisionForm, toDecisionForm, visibleFields } from './decision'

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

  it('問題の設定の今のやり方の決まりは, 列を書かなければそのまま渡す', () => {
    const rule = { type: 'lag' as const, periods: 7, time: 'date', by: ['shop'] }
    const decision = { type: 'quantity' as const, under_cost: 3, over_cost: 1, current: rule }
    const form = toDecisionForm(decision)
    expect(form.values.current).toBeUndefined()
    expect(baselineText(form.baseline!)).toBe('7 期前の値と同じ量')
    expect(fromDecisionForm(form, []).decision).toEqual(decision)
    // 列を書けば列, 使わないなら外す, 型を変えれば渡さない
    const typed = { ...form, values: { ...form.values, current: 'now' } }
    expect(fromDecisionForm(typed, []).decision?.current).toBe('now')
    expect(fromDecisionForm({ ...form, baseline: null }, []).decision?.current).toBeUndefined()
    const triage = { ...form, type: 'triage' as const, positive: 'a', values: {} }
    expect(fromDecisionForm(triage, ['a', 'b']).decision?.current).toBeUndefined()
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

  it('効果を確率の差で渡すときは向きを求め, 負の差も通す', () => {
    const base = { value: '1000', cost: '50', effect: '-0.2', effect_kind: 'uplift' }
    const form = { type: 'select' as const, positive: 'いる', values: base }
    const { problems } = fromDecisionForm(form, ['いる', 'いない'])
    expect(problems).toEqual([
      '「当たりをどうしたいか (減らすなら差が負, 増やすなら正が得)」を書いてください',
    ])
    const done = fromDecisionForm({ ...form, values: { ...base, effect_goal: 'reduce' } }, [
      'いる',
      'いない',
    ])
    expect(done).toEqual({
      decision: {
        type: 'select',
        positive: 'いる',
        value: 1000,
        cost: 50,
        effect: -0.2,
        effect_kind: 'uplift',
        effect_goal: 'reduce',
      },
      problems: [],
    })
    expect(visibleFields(form).some((f) => f.key === 'effect_goal')).toBe(true)
  })

  it('防げる割合では向きを渡さず, 負の効果は誤り', () => {
    const values = { value: '1000', cost: '50', effect: '-0.2', effect_goal: 'reduce' }
    const form = { type: 'select' as const, positive: 'いる', values }
    const { decision, problems } = fromDecisionForm(form, ['いる', 'いない'])
    expect(decision?.effect_goal).toBeUndefined()
    expect(problems.some((p) => p.includes('0 以上'))).toBe(true)
    expect(visibleFields(form).some((f) => f.key === 'effect_goal')).toBe(false)
  })

  it('選ばなければ意思決定しない', () => {
    expect(fromDecisionForm(toDecisionForm(null), [])).toEqual({ decision: null, problems: [] })
  })
})
