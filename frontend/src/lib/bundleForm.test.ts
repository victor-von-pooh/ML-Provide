import { describe, expect, it } from 'vitest'

import type { ProcedureEntry } from '../api/types'
import {
  buildInputs,
  buildOutputOptions,
  buildProcedure,
  EMPTY_OUTPUT,
  EMPTY_PROCEDURE,
  optionValue,
} from './bundleForm'
import { optionsText } from './outputs'

const ddim: ProcedureEntry = {
  name: 'ddim',
  builtin: true,
  description: 'DDIM',
  start: 'noise',
  time: true,
  options: ['train_steps', 'beta_start', 'beta_end', 'eta'],
}

describe('buildInputs', () => {
  it('名前のある行を出どころと番号にし, 名前の無い行は知らせる', () => {
    const { inputs, problems } = buildInputs(
      [
        { id: 1, name: 'x', source: 'state', prepend: '' },
        { id: 2, name: ' t ', source: 'time', prepend: '' },
        { id: 3, name: '', source: 'features', prepend: '' },
      ],
      false,
      true,
    )
    expect(problems).toHaveLength(1)
    expect(inputs).toEqual({ x: { source: 'state' }, t: { source: 'time' } })
  })

  it('先頭の番号を整数の並びにし, 誤りを知らせる', () => {
    expect(
      buildInputs([{ id: 4, name: 'ids', source: 'items', prepend: '2001, 5' }]).inputs,
    ).toEqual({
      ids: { source: 'items', prepend: [2001, 5] },
    })
    expect(
      buildInputs([{ id: 5, name: 'ids', source: 'items', prepend: '1.5' }]).problems,
    ).toHaveLength(1)
    expect(buildInputs([{ id: 6, name: '', source: 'items', prepend: '1' }]).problems).toHaveLength(
      1,
    )
    expect(
      buildInputs(
        [
          { id: 7, name: 'x', source: 'state', prepend: '' },
          { id: 8, name: 'x', source: 'time', prepend: '' },
        ],
        false,
        true,
      ).problems,
    ).toHaveLength(1)
    expect(buildInputs([]).inputs).toBeNull()
    // Object の性質の名前も入力の名前として扱う
    const named = buildInputs(
      ['constructor', 'toString', '__proto__'].map((name, index) => ({
        id: 100 + index,
        name,
        source: 'features',
        prepend: '',
      })),
    )
    expect(named.problems).toEqual([])
    expect(Object.keys(named.inputs ?? {})).toEqual(['constructor', 'toString', '__proto__'])
    // __proto__ も送る本体の自分の性質に残る (JSON にして確かめる)
    expect(JSON.parse(JSON.stringify(named.inputs)).__proto__).toEqual({ source: 'features' })
  })

  it('番号を足せない出どころの番号は送らず, 雑音だけの一式は手順の中身と時刻だけ', () => {
    expect(buildInputs([{ id: 90, name: 'x', source: 'features', prepend: '5' }]).inputs).toEqual({
      x: { source: 'features' },
    })
    const rows = [
      { id: 91, name: 'x', source: 'features', prepend: '' },
      { id: 92, name: 't', source: 'time', prepend: '' },
    ]
    expect(buildInputs(rows, false, true).problems).toEqual([])
    expect(buildInputs(rows, true).problems).toHaveLength(1)
    // 手順の無い一式に手順の中身と時刻, 雑音だけの一式の番号は送らない
    expect(
      buildInputs([{ id: 93, name: 'x', source: 'state', prepend: '' }]).problems,
    ).toHaveLength(1)
    expect(
      buildInputs([{ id: 94, name: 'x', source: 'state', prepend: '' }], false, true).problems,
    ).toEqual([])
    expect(
      buildInputs([{ id: 95, name: 'x', source: 'state', prepend: '7' }], true).inputs,
    ).toEqual({
      x: { source: 'state' },
    })
  })
})

describe('buildProcedure', () => {
  it('手順を選ばなければ null', () => {
    expect(buildProcedure(EMPTY_PROCEDURE, undefined)).toEqual({ procedure: null, problems: [] })
  })

  it('回数, 種, 書いた設定だけを渡す', () => {
    const { procedure, problems } = buildProcedure(
      {
        ...EMPTY_PROCEDURE,
        name: 'ddim',
        steps: '50',
        seed: '',
        options: { eta: '0.5', beta_end: '' },
      },
      ddim,
    )
    expect(problems).toEqual([])
    expect(procedure).toEqual({ name: 'ddim', steps: 50, seed: 0, options: { eta: 0.5 } })
  })

  it('回数と種の誤りを知らせる', () => {
    const form = { ...EMPTY_PROCEDURE, name: 'ddim', steps: '0', seed: '1.5' }
    expect(buildProcedure(form, ddim).problems).toHaveLength(2)
  })

  it('回数は上限まで (ちょうど上限は通る)', () => {
    const at = (steps: string) =>
      buildProcedure({ ...EMPTY_PROCEDURE, name: 'ddim', steps }, ddim, 1000).problems
    expect(at('1000')).toEqual([])
    expect(at('1001')).toEqual([`手順の回数は ${(1000).toLocaleString()} 回までです`])
  })

  it('何でも受け取る自作の手順は 名前=値 の行', () => {
    const custom = { ...ddim, name: 'lab', builtin: false, options: null }
    const { procedure, problems } = buildProcedure(
      { ...EMPTY_PROCEDURE, name: 'lab', steps: '3', extra: 'scale=2\nmode = fast\n\nbad' },
      custom,
    )
    expect(procedure?.options).toEqual({ scale: 2, mode: 'fast' })
    expect(problems).toHaveLength(1)
  })

  it('引用符で始まるのに JSON の文字として読めない設定の値は知らせる', () => {
    const { procedure, problems } = buildProcedure(
      {
        ...EMPTY_PROCEDURE,
        name: 'ddim',
        steps: '3',
        options: { eta: '"0', train_steps: '"0"', beta_end: '"a" b' },
      },
      ddim,
    )
    expect(procedure?.options).toEqual({ eta: '"0', train_steps: '0', beta_end: '"a" b' })
    expect(problems).toHaveLength(2)
    expect(problems[0]).toContain('beta_end')
    expect(problems[1]).toContain('eta')
  })
})

describe('buildOutputOptions', () => {
  it('配列の出力の形は 1 件の形', () => {
    expect(buildOutputOptions('image', { ...EMPTY_OUTPUT, shape: '1, 32, 32' }, true)).toEqual({
      options: { shape: [1, 32, 32] },
      problems: [],
    })
  })

  it('雑音から始まる手順は配列の出力の形と 1 件の形が要る', () => {
    expect(buildOutputOptions('vector', EMPTY_OUTPUT, true).problems).toHaveLength(1)
    expect(
      buildOutputOptions('class', { ...EMPTY_OUTPUT, shape: '4' }, true).problems,
    ).toHaveLength(1)
    expect(
      buildOutputOptions('vector', { ...EMPTY_OUTPUT, shape: '0' }, false).problems,
    ).toHaveLength(1)
    expect(buildOutputOptions('', EMPTY_OUTPUT, false)).toEqual({ options: null, problems: [] })
  })

  it('分布の水準と領域のクラスの名前', () => {
    expect(
      buildOutputOptions('distribution', { ...EMPTY_OUTPUT, quantiles: '0.1, 0.5, 0.9' }, false)
        .options,
    ).toEqual({
      quantiles: [0.1, 0.5, 0.9],
    })
    expect(
      buildOutputOptions('distribution', { ...EMPTY_OUTPUT, quantiles: '0.5, 0.1' }, false)
        .problems,
    ).toHaveLength(1)
    expect(
      buildOutputOptions('mask', { ...EMPTY_OUTPUT, classes: 'bg, cell' }, false).options,
    ).toEqual({
      classes: ['bg', 'cell'],
    })
  })
})

describe('optionValue', () => {
  it('数, 真偽値, 文字', () => {
    expect(optionValue('1e-4')).toBe(1e-4)
    expect(optionValue('true')).toBe(true)
    expect(optionValue(' fast ')).toBe('fast')
    expect(optionValue('0x10')).toBe('0x10')
    expect(optionValue('-.5')).toBe(-0.5)
    expect(optionValue('Infinity')).toBe('Infinity')
  })

  it('null と, JSON の引用符で囲んだ文字 (エスケープは JSON の決まり)', () => {
    expect(optionValue('null')).toBeNull()
    expect(optionValue(' "0" ')).toBe('0')
    expect(optionValue('""')).toBe('')
    expect(optionValue('" a "')).toBe(' a ')
    expect(optionValue('"say \\"hi\\" \\\\ end"')).toBe('say "hi" \\ end')
    expect(optionValue('"a')).toBe('"a')
    expect(optionValue('"a" b')).toBe('"a" b')
    expect(optionValue('"\\q"')).toBe('"\\q"')
    expect(optionValue("'0'")).toBe("'0'")
  })

  it('一式の画面に出た値を写すと元の型に戻る', () => {
    const values: Record<string, unknown> = {
      int: 0,
      float: -1.5,
      small: 1e-4,
      yes: true,
      no: false,
      none: null,
      date: '2026-11-22',
      zero: '0',
      word: 'true',
      nil: 'null',
      hex: '0x1',
      onoff: 'on',
      empty: '',
      spaced: ' a b ',
      comma: 'a,b',
      quote: 'say "hi"',
      backslash: 'a\\b',
      single: "'q'",
      plain: 'fast',
      unicode: '日本語',
      newline: 'a\nb',
    }
    for (const [key, value] of Object.entries(values)) {
      const shown = optionsText({ [key]: value }).slice(key.length + 1)
      expect(optionValue(shown)).toStrictEqual(value)
    }
  })
})
