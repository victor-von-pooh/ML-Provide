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

/** 手順の回数の上限 (GET /api/info の max_procedure_steps を読む前の値) */
export const MAX_PROCEDURE_STEPS = 1000

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

/** JSON の引用符で囲んだ文を JSON の決まりで読んだ文字 (囲んでいないか読めなければ undefined) */
function quotedText(trimmed: string): string | undefined {
  if (trimmed.length < 2 || !trimmed.startsWith('"') || !trimmed.endsWith('"')) return undefined
  try {
    const value: unknown = JSON.parse(trimmed)
    return typeof value === 'string' ? value : undefined
  } catch {
    return undefined
  }
}

/** [ か { で始まる文を JSON の決まりで読んだ配列か辞書 (始まらないか読めなければ undefined) */
function nestedValue(trimmed: string): unknown[] | Record<string, unknown> | undefined {
  if (!/^[[{]/.test(trimmed)) return undefined
  try {
    const value: unknown = JSON.parse(trimmed)
    if (Array.isArray(value)) return value as unknown[]
    return value !== null && typeof value === 'object'
      ? (value as Record<string, unknown>)
      : undefined
  } catch {
    return undefined
  }
}

/** 引用符で始まるのに JSON の文字として読めない設定の値の文か ("a, "a" b, "a\q" など) */
export function brokenQuoted(text: string): boolean {
  const trimmed = text.trim()
  return trimmed.startsWith('"') && quotedText(trimmed) === undefined
}

/** [ か { で始まるのに JSON の配列か辞書として読めない設定の値の文か ([1, 2 や {a: 1} など) */
export function brokenNested(text: string): boolean {
  const trimmed = text.trim()
  return /^[[{]/.test(trimmed) && nestedValue(trimmed) === undefined
}

/**
 * 手順の設定の値の文を値にする. 数は数, true と false は真偽値, null は null,
 * JSON の引用符で囲んだ値は JSON の決まりで読んだ文字 (\" と \\ などのエスケープも),
 * [ か { で始まる値は JSON の決まりで読んだ配列か辞書 (入れ子も), 他は前後の空白を除いた文字.
 * 一式の画面に出た値 (optionsText) を写すと元の型に戻る. 組み込みの手順の設定は全て数か整数
 */
export function optionValue(
  text: string,
): number | boolean | string | null | unknown[] | Record<string, unknown> {
  const trimmed = text.trim()
  const quoted = quotedText(trimmed)
  if (quoted !== undefined) return quoted
  const nested = nestedValue(trimmed)
  if (nested !== undefined) return nested
  if (trimmed === 'true') return true
  if (trimmed === 'false') return false
  if (trimmed === 'null') return null
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
  // 名前は Map で数える (constructor, __proto__ などの名前を Object の性質と取り違えないため)
  const inputs = new Map<string, InputSource>()
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
    if (inputs.has(name)) problems.push(`入力の名前 ${name} が重なっています`)
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
    inputs.set(name, {
      source: row.source,
      ...(ids.length > 0 ? { prepend: ids.map(Number) } : {}),
    })
  }
  // fromEntries は __proto__ も自分の性質として持つ
  return { inputs: inputs.size > 0 ? Object.fromEntries(inputs) : null, problems }
}

/** 引用符で始まるのに JSON の文字として読めない設定の誤りの文 */
function quotedProblem(name: string): string {
  return `手順の設定 ${name} は, 引用符で囲んだ全体を JSON の文字として書いてください (閉じた引用符の後ろに何も付けない. 中の " と \\ は \\" と \\\\)`
}

/** [ か { で始まるのに JSON の配列か辞書として読めない設定の誤りの文 */
function nestedProblem(name: string): string {
  return `手順の設定 ${name} は, 配列か辞書なら全体を JSON の決まりで書いてください ([1, 2] や {"a": 1}. 辞書の名前と文字は " で囲む). [ か { で始まる文字は "[a" のように引用符で囲みます`
}

/** 設定の値の文の誤り (無ければ空) */
function optionProblems(name: string, text: string): string[] {
  if (brokenQuoted(text)) return [quotedProblem(name)]
  if (brokenNested(text)) return [nestedProblem(name)]
  return []
}

/** 手順の欄から procedure を作る (手順を選ばなければ null) */
export function buildProcedure(
  form: ProcedureForm,
  entry: ProcedureEntry | undefined,
  maxSteps: number = MAX_PROCEDURE_STEPS,
): { procedure: Procedure | null; problems: string[] } {
  const problems: string[] = []
  if (!form.name) return { procedure: null, problems }
  if (!isIntegerText(form.steps) || Number(form.steps) < 1)
    problems.push('手順の回数は 1 以上の整数で書いてください')
  else if (Number(form.steps) > maxSteps)
    problems.push(`手順の回数は ${maxSteps.toLocaleString()} 回までです`)
  if (form.seed.trim() && !isIntegerText(form.seed))
    problems.push('手順の乱数の種は整数で書いてください')
  const seed = form.seed.trim() ? Number(form.seed) : 0
  if (isIntegerText(form.seed) && !Number.isSafeInteger(seed)) problems.push(SEED_RANGE_PROBLEM)
  const options: Record<string, unknown> = {}
  // 設定の名前の分かる手順は名前ごとの欄 (空は手順の既定値)
  for (const name of entry?.options ?? []) {
    const text = form.options[name] ?? ''
    if (!text.trim()) continue
    problems.push(...optionProblems(name, text))
    options[name] = optionValue(text)
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
      problems.push(...optionProblems(key, line.slice(at + 1)))
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
