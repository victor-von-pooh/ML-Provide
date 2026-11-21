import type { InputSource, ProcedureEntry, Procedure } from '../api/types'

/** 配列の出力の形 (雑音から始まる手順に使える. 1 件の形 shape を持てる) */
export const ARRAY_FORMS = ['mask', 'point_cloud', 'vector', 'image']

/** モデルの入力の出どころの名前 (ML-Inference の inputs の source) */
export const SOURCE_LABELS: Record<string, string> = {
  features: '表の列',
  items: '中身 (画像, 文の番号などの配列)',
  state: '手順の中身 (雑音か番号の並び)',
  time: '手順の時刻',
}

/** 一式の説明でのモデルの入力の出どころの短い名前 */
export const SHORT_SOURCE_LABELS: Record<string, string> = {
  features: '表の列',
  items: '中身',
  state: '手順の中身',
  time: '手順の時刻',
}

/** 雑音だけの一式で使える入力の出どころ (データの入力は無い) */
export const NOISE_SOURCES = ['state', 'time']

/** 入力の欄の 1 行 (行の印, 名前, 出どころ, 先頭に足す番号の文, 雑音だけの一式で自動で足したか) */
export interface InputRow {
  id: number
  auto?: boolean
  name: string
  source: string
  prepend: string
}

/** 手順の欄 (名前, 回数, 種, 設定の名前ごとの文, 何でも受け取る手順の「名前=値」の行) */
export interface ProcedureForm {
  name: string
  steps: string
  seed: string
  options: Record<string, string>
  extra: string
}

/** 出力の形ごとの欄 (分布の水準, 領域のクラスの名前, 1 件の形) */
export interface OutputForm {
  quantiles: string
  classes: string
  shape: string
}

export const EMPTY_PROCEDURE: ProcedureForm = {
  name: '',
  steps: '',
  seed: '',
  options: {},
  extra: '',
}

export const EMPTY_OUTPUT: OutputForm = { quantiles: '', classes: '', shape: '' }

let lastRowId = 0

/** 画面で扱える種の範囲を超えたときの文 (JavaScript の数で正確に送れる整数まで) */
export const SEED_RANGE_PROBLEM = `種が範囲を超えています (-${Number.MAX_SAFE_INTEGER.toLocaleString()} から ${Number.MAX_SAFE_INTEGER.toLocaleString()} の整数にしてください)`

/** 入力の欄の新しい行 (行の印は画面の中で重ならない数) */
export function newInputRow(source: string, auto = false): InputRow {
  lastRowId += 1
  return { id: lastRowId, name: '', source, prepend: '', ...(auto ? { auto } : {}) }
}

/** 先頭に番号を足せる出どころ (文の番号の並び) */
export const PREPEND_SOURCES = ['items', 'state']

/** カンマか空白で区切った並び */
function splitList(text: string): string[] {
  return text.split(/[\s,]+/).filter(Boolean)
}

/** 整数の文か (空は偽) */
function isIntegerText(text: string): boolean {
  return /^-?\d+$/.test(text.trim())
}

/**
 * 手順の設定の値の文を値にする. 数は数, true と false は真偽値, 他は文字.
 * 組み込みの手順の設定は全て数か整数
 */
export function optionValue(text: string): number | boolean | string {
  const trimmed = text.trim()
  if (trimmed === 'true') return true
  if (trimmed === 'false') return false
  // 10 進の数だけを数にする (0x10 などを数に変えない)
  const decimal = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(trimmed)
  return decimal ? Number(trimmed) : trimmed
}

/**
 * 入力の欄から inputs を作る (名前の無い行は飛ばす. 行が無ければ null).
 * noData は雑音だけの一式か (出どころは手順の中身と時刻だけ. 先頭の番号は送らない),
 * stepped は繰り返しの手順を選んだか (手順の中身と時刻は手順のある一式だけ)
 */
export function buildInputs(
  rows: InputRow[],
  noData = false,
  stepped = noData,
): {
  inputs: Record<string, InputSource> | null
  problems: string[]
} {
  const problems: string[] = []
  const inputs: Record<string, InputSource> = {}
  for (const row of rows) {
    const name = row.name.trim()
    // 番号を足せない出どころと雑音だけの一式では, 欄に残った番号を送らない
    const ids = !noData && PREPEND_SOURCES.includes(row.source) ? splitList(row.prepend) : []
    if (!name) {
      // 名前の無い行は送れない (使わない行は除く. 文は 1 つだけ)
      const problem =
        'モデルの入力の名前 (モデルの定義の inputs の名前) を書いてください. 使わない行は × で除きます'
      if (!problems.includes(problem)) problems.push(problem)
      continue
    }
    if (name in inputs) problems.push(`入力の名前 ${name} が重なっています`)
    if (noData && !NOISE_SOURCES.includes(row.source))
      problems.push(
        `雑音だけの一式の入力 ${name} の出どころは, 手順の中身か手順の時刻にしてください`,
      )
    if (!stepped && NOISE_SOURCES.includes(row.source))
      problems.push(
        `入力 ${name} の出どころの手順の中身と時刻は, 繰り返しの手順を選んだ一式だけで使えます. 手順を選ぶか, 出どころを表の列か中身にしてください`,
      )
    if (ids.some((value) => !isIntegerText(value)))
      problems.push(`入力 ${name} の足した番号は整数で書いてください`)
    inputs[name] = {
      source: row.source,
      ...(ids.length > 0 ? { prepend: ids.map(Number) } : {}),
    }
  }
  return { inputs: Object.keys(inputs).length > 0 ? inputs : null, problems }
}

/** 手順の欄から procedure を作る (手順を選ばなければ null) */
export function buildProcedure(
  form: ProcedureForm,
  entry: ProcedureEntry | undefined,
): { procedure: Procedure | null; problems: string[] } {
  const problems: string[] = []
  if (!form.name) return { procedure: null, problems }
  if (!isIntegerText(form.steps) || Number(form.steps) < 1)
    problems.push('手順の回数は 1 以上の整数で書いてください')
  if (form.seed.trim() && !isIntegerText(form.seed))
    problems.push('手順の乱数の種は整数で書いてください')
  const seed = form.seed.trim() ? Number(form.seed) : 0
  if (isIntegerText(form.seed) && !Number.isSafeInteger(seed)) problems.push(SEED_RANGE_PROBLEM)
  const options: Record<string, unknown> = {}
  // 設定の名前の分かる手順は名前ごとの欄 (空は手順の既定値)
  for (const name of entry?.options ?? []) {
    const text = form.options[name] ?? ''
    if (text.trim()) options[name] = optionValue(text)
  }
  // 何でも受け取る自作の手順は「名前=値」の行
  if (entry && entry.options === null) {
    for (const line of form.extra.split('\n')) {
      if (!line.trim()) continue
      const at = line.indexOf('=')
      const key = at > 0 ? line.slice(0, at).trim() : ''
      if (!key) {
        problems.push(`手順の設定「${line.trim()}」は 名前=値 の形で書いてください`)
        continue
      }
      options[key] = optionValue(line.slice(at + 1))
    }
  }
  const procedure: Procedure = {
    name: form.name,
    steps: Number(form.steps),
    seed,
    ...(Object.keys(options).length > 0 ? { options } : {}),
  }
  return { procedure, problems }
}

/**
 * 出力の形ごとの欄から output_options を作る (書いた欄が無ければ null).
 * needsShape は雑音から始まる手順を選んだか (1 件の形が要る)
 */
export function buildOutputOptions(
  form: string,
  fields: OutputForm,
  needsShape: boolean,
): { options: Record<string, unknown> | null; problems: string[] } {
  const problems: string[] = []
  const options: Record<string, unknown> = {}
  if (form === 'distribution' && fields.quantiles.trim()) {
    const levels = splitList(fields.quantiles).map(Number)
    const ordered = levels.every((level, index) => index === 0 || (levels[index - 1] ?? 1) < level)
    if (levels.some((level) => !(level > 0 && level < 1)) || !ordered)
      problems.push('分布の水準は 0 から 1 の間の小さい順の数で書いてください')
    options.quantiles = levels
  }
  if (form === 'mask' && fields.classes.trim()) {
    options.classes = fields.classes
      .split(',')
      .map((name) => name.trim())
      .filter(Boolean)
  }
  if (ARRAY_FORMS.includes(form) && fields.shape.trim()) {
    const sizes = splitList(fields.shape)
    if (sizes.some((size) => !isIntegerText(size) || Number(size) < 1))
      problems.push('1 件の形は正の整数の並びで書いてください (例: 1, 32, 32)')
    options.shape = sizes.map(Number)
  }
  if (needsShape && !(ARRAY_FORMS.includes(form) && fields.shape.trim()))
    problems.push(
      '雑音から始まる手順には, 配列の出力の形 (領域, 点群, ベクトル, 画像) と 1 件の形が要ります',
    )
  return { options: Object.keys(options).length > 0 ? options : null, problems }
}
