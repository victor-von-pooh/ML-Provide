import { fromForm, toForm } from './rules'

describe('結論の決め方の編集', () => {
  it('分類の決め方を行き来できる', () => {
    const rules = { min_confidence: 0.7, group_by: 'shop' }
    expect(fromForm(toForm(rules), 'classification')).toEqual({ rules, problems: [] })
  })

  it('陽性のクラスと確信の下限の組み合わせは誤り', () => {
    const form = { ...toForm({}), positive: '悪い', threshold: '0.3', minConfidence: '0.6' }
    const { problems } = fromForm(form, 'classification')
    expect(problems).toContain('陽性のクラスと確信の下限は一緒に使えません')
  })

  it('帯の最後は上限無しで, 上限は小さい順', () => {
    const form = {
      ...toForm({}),
      bands: [
        { label: '少ない', upper: '100' },
        { label: '多い', upper: '' },
      ],
    }
    expect(fromForm(form, 'regression').rules.bands).toEqual([
      { label: '少ない', upper: 100 },
      { label: '多い' },
    ])
    form.bands = [
      { label: 'a', upper: '100' },
      { label: 'b', upper: '50' },
      { label: 'c', upper: '' },
    ]
    expect(fromForm(form, 'regression').problems).toContain('帯は上限の小さい順に並べてください')
  })

  it('回帰では分類の決め方を送らない', () => {
    const form = { ...toForm({}), minConfidence: '0.5' }
    expect(fromForm(form, 'regression').rules).toEqual({})
  })
})
