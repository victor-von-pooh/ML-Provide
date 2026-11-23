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
  tableColumnText,
  tableRowColumns,
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
      '2026-11-22',
      '2026-1-2',
      '2026-11-22T10:00:00',
      '2026-11-22t10:00:00.5Z',
      '2026-11-22 10:00:00 +09:00',
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
    for (const text of [
      'fast',
      'linear',
      '1a',
      'v2',
      'e3',
      'info',
      'nope',
      '0xg',
      '2026-11',
      '2026-11-22T10',
    ])
      expect(optionsText({ v: text })).toBe(`v ${text}`)
  })
})

describe('表の出力', () => {
  it('一式の説明は列の数と符号化した行の幅, 列ごとの型の文', () => {
    const columns = [
      { name: 'x', type: 'number', mean: 10, std: 2, integer: false },
      { name: 'n', type: 'number', mean: 3, std: 1.5, integer: true },
      { name: 'color', type: 'category', categories: ['blue', 'green', 'red'] },
    ]
    expect(outputDetails({ form: 'table', shape: [5], columns })).toEqual([
      '列 3, 符号化した行の幅 5',
    ])
    expect(columns.map(tableColumnText)).toEqual([
      '数 (平均 10, 標準偏差 2)',
      '整数 (平均 3, 標準偏差 1.5)',
      '種類 (3 種類: blue, green, red)',
    ])
    // 種類の多い列は先頭の 10 個と残りの数. 結果に残した列は種類の数だけ
    const many = Array.from({ length: 12 }, (_, i) => `c${i}`)
    expect(tableColumnText({ name: 'k', type: 'category', categories: many })).toBe(
      '種類 (12 種類: c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 ほか 2)',
    )
    expect(tableColumnText({ name: 'k', type: 'category', categories: 4 })).toBe('種類 (4 種類)')
    expect(tableColumnText({ name: 'n', type: 'number', integer: true })).toBe('整数')
  })

  it('結果の表は件の番号と種の後に列の符号化の順で, 判断の列を出さない', () => {
    const table = [
      { name: 'color', type: 'category', categories: 3 },
      { name: 'index', type: 'number', integer: true },
      { name: 'x', type: 'number', integer: false },
    ]
    expect(
      tableRowColumns(
        ['index', 'sample', 'seed', 'status', 'reason', 'x', 'index (入力)', 'color', 'decision'],
        table,
      ),
    ).toEqual(['index', 'sample', 'seed', 'status', 'reason', 'color', 'index (入力)', 'x'])
    // 知らない列は後ろに残す
    expect(tableRowColumns(['index', 'sample', 'x', 'extra'], table)).toEqual([
      'index',
      'sample',
      'x',
      'extra',
    ])
  })
})
