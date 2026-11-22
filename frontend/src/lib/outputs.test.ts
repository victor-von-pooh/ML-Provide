import type { Description } from '../api/types'
import {
  isColumnOutput,
  licenseText,
  maskNames,
  optionsText,
  originNote,
  outputDetails,
  placeText,
  sourceLabel,
  stepsText,
} from './outputs'

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

  it('版 2 のデータセットの一式の領域は, 番号でなく領域の値ごとの名前', () => {
    const output = { form: 'mask', classes: ['background', 'circle', 'square'] }
    expect(outputDetails(output, [0, 128, 255])).toEqual([
      '値ごとの名前 0: background, 128: circle, 255: square',
    ])
    const items = {
      input: 'image',
      target: 'mask',
      prepare: 'image',
      options: {},
      values: [0, 128, 255],
      restore: null,
    }
    expect(maskNames({ output, items } as unknown as Description)).toEqual({
      0: 'background',
      128: 'circle',
      255: 'square',
    })
    // 版 3 の一式は番号のまま
    expect(maskNames({ output } as unknown as Description)).toEqual({
      0: 'background',
      1: 'circle',
      2: 'square',
    })
    expect(maskNames(undefined)).toEqual({})
  })

  it('推論するデータの欄の名前は, 版 4 の一式なら入力の項目の種類から, 他は一式の種類から', () => {
    expect(sourceLabel('image')).toContain('画像のファイル')
    expect(sourceLabel('capture')).toContain('撮影のディレクトリ')
    expect(sourceLabel('point_cloud')).toContain('点群のファイル')
    expect(sourceLabel(undefined)).toContain('同じ列のファイル')
    expect(sourceLabel(undefined, 'audio')).toContain('音声のファイル')
  })
})

describe('中身の手順', () => {
  it('手順の名前と引数をかける順に並べる', () => {
    expect(stepsText([{ op: 'resize', width: 64, height: 48 }, { op: 'grayscale' }])).toBe(
      'resize (width 64, height 48), grayscale',
    )
    expect(stepsText([])).toBe('')
  })
})

describe('optionsText', () => {
  it('名前と値を並べ, 入れ子と紛れる文字だけ引用符を付ける', () => {
    expect(
      optionsText({
        eta: 0,
        nested: { a: 1 },
        list: [1, 2],
        none: null,
        flag: true,
        sep: ',',
        empty: '',
        spaced: ' a',
        words: 'a b',
        paren: 'f(x)',
        plain: 'fast',
      }),
    ).toBe(
      'eta 0, nested {"a":1}, list [1,2], none null, flag true, sep ",", empty "", spaced " a", words "a b", paren "f(x)", plain fast',
    )
  })

  it('数, 真偽, null に見える文字に引用符を付け, 型の値と見分ける', () => {
    const typed = [
      '0',
      '-1.5',
      '1e3',
      '.5',
      '1_000',
      '0x1',
      '0o7',
      '0b1',
      '1:30',
      'inf',
      '-Infinity',
      '.NaN',
      'true',
      'False',
      'yes',
      'On',
      'n',
      'null',
      'None',
      '~',
    ]
    for (const text of typed) expect(optionsText({ v: text })).toBe(`v ${JSON.stringify(text)}`)
    expect(optionsText({ n: 0, s: '0', t: true, u: 'true', z: null, w: 'null' })).toBe(
      'n 0, s "0", t true, u "true", z null, w "null"',
    )
  })

  it('引用符や括弧で始まる文字にも引用符を付け, 普通の文字はそのまま', () => {
    expect(optionsText({ a: '"x"', b: '[1]', c: '{}', d: "'q'" })).toBe(
      'a "\\"x\\"", b "[1]", c "{}", d "\'q\'"',
    )
    for (const text of ['fast', 'linear', '1a', 'v2', 'e3', 'info', 'nope', '0xg'])
      expect(optionsText({ v: text })).toBe(`v ${text}`)
  })
})
