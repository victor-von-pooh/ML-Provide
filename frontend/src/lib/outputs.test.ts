import { isColumnOutput, licenseText, originNote, outputDetails, placeText } from './outputs'

describe('受け持ちと出力の形', () => {
  it('版 2 までの一式と項目の無い説明は列の出力', () => {
    expect(isColumnOutput({})).toBe(true)
    expect(isColumnOutput({ column_output: true })).toBe(true)
    expect(isColumnOutput({ column_output: false })).toBe(false)
  })

  it('正解が今の処理の出力なら, 超えられないと書く', () => {
    expect(originNote({ kind: 'mask', origin: { type: 'process', name: '今の判定' } })).toBe(
      '正解は 今の判定 の出力なので, 予測はその処理との一致までしか分からず, その処理を超えられません',
    )
    expect(originNote({ kind: 'mask', origin: { type: 'process' } })).toContain('今の処理 の出力')
    expect(originNote({ kind: 'column', origin: { type: 'annotation' } })).toBeNull()
    expect(originNote({ kind: 'column' })).toBeNull()
    expect(originNote(null)).toBeNull()
  })

  it('出力の形ごとの欄を文にする', () => {
    expect(outputDetails({ form: 'mask', classes: ['背景', '物'] })).toEqual([
      '番号ごとの名前 0: 背景, 1: 物',
    ])
    expect(outputDetails({ form: 'distribution', quantiles: [0.1, 0.5, 0.9] })).toEqual([
      '水準 0.1, 0.5, 0.9',
    ])
    expect(outputDetails({ form: 'vector', shape: [16] })).toEqual(['1 件の形 (16)'])
    expect(outputDetails(undefined)).toEqual([])
  })

  it('学習済みの重みの置き場とライセンス', () => {
    const record = {
      node: 'backbone',
      hub_id: 'org/model',
      pinned_revision: 'abc123',
      license: 'cc-by-nc-4.0',
      redistributable: false,
      place: 'source',
    }
    expect(placeText(record)).toBe('出どころ (org/model @ abc123) から読む')
    expect(licenseText(record)).toEqual({ text: 'cc-by-nc-4.0 (再配布できない)', warn: true })
    expect(placeText({ node: 'a', place: 'bundle' })).toBe('一式の重みに写した')
    expect(
      licenseText({ node: 'a', place: 'bundle', redistributable: true, license: 'mit' }),
    ).toEqual({ text: 'mit (再配布できる)', warn: false })
    expect(licenseText({ node: 'a', place: 'bundle' }).warn).toBe(true)
  })
})
