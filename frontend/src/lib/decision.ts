import type { Decision } from '../api/types'

/** 選んで書く欄の選択肢 (値と説明) */
export interface Choice {
  value: string
  label: string
}

/** 意思決定の欄 (名前, 説明, 省けるか, 選んで書くなら選択肢) */
export interface DecisionField {
  key: string
  label: string
  optional?: boolean
  choices?: Choice[]
}

/** 意思決定の型の名前と, 型ごとの欄 */
export const DECISION_TYPES: Record<Decision['type'], { label: string; fields: DecisionField[] }> =
  {
    quantity: {
      label: '量を決める (補充, 仕入れ, 人員)',
      fields: [
        { key: 'under_cost', label: '1 単位足りないときの損 (欠品の利益など)' },
        {
          key: 'over_cost',
          label: '1 単位余ったときの損 (捨てるなら原価, 持ち越すなら保管の費用)',
        },
        { key: 'stock', label: '今ある量 (数か列の名前)', optional: true },
        { key: 'step', label: '量の刻み (箱の単位など)', optional: true },
        { key: 'current', label: '今のやり方で足した量の列 (たどり直しで比べる)', optional: true },
      ],
    },
    select: {
      label: '対象を選ぶ (引き止め, 営業先, 点検)',
      fields: [
        { key: 'value', label: '当たり 1 件の価値 (数か列の名前)' },
        { key: 'cost', label: '行動 1 件の費用 (数か列の名前)' },
        {
          key: 'effect_kind',
          label: '効果の渡し方 (防げる割合か, 行動したとき − しないときの当たりの確率)',
          optional: true,
          choices: [
            { value: 'share', label: '防げる割合 (仮定)' },
            { value: 'uplift', label: '確率の差 (効果のモデル)' },
          ],
        },
        {
          key: 'effect',
          label: '効果 (割合は 0 から 1, 確率の差は -1 から 1. 数か列の名前)',
        },
        {
          key: 'effect_goal',
          label: '当たりをどうしたいか (減らすなら差が負, 増やすなら正が得)',
          choices: [
            { value: 'reduce', label: '減らしたい (解約など)' },
            { value: 'raise', label: '増やしたい (成約など)' },
          ],
        },
        { key: 'budget', label: '行動できる件数の上限', optional: true },
        { key: 'budget_cost', label: '費用の上限', optional: true },
        {
          key: 'current',
          label: '今のやり方で行動したかの列 (たどり直しで比べる)',
          optional: true,
        },
      ],
    },
    triage: {
      label: '振り分ける (人が確かめる, 再検査)',
      fields: [
        { key: 'miss_cost', label: '当たりを見逃す損' },
        { key: 'false_alarm_cost', label: '外れを回す損 (空振り)' },
        { key: 'capacity', label: '回せる件数の上限 (人手)', optional: true },
        { key: 'current', label: '今のやり方で回したかの列 (たどり直しで比べる)', optional: true },
      ],
    },
  }

/** 効果を確率の差 (uplift) で渡すか */
function isUplift(form: DecisionForm): boolean {
  return form.type === 'select' && form.values.effect_kind === 'uplift'
}

/** 今の書き方で使う欄 (向きは確率の差のときだけ) */
export function visibleFields(form: DecisionForm): DecisionField[] {
  if (!form.type) return []
  return DECISION_TYPES[form.type].fields.filter(
    (field) => field.key !== 'effect_goal' || isUplift(form),
  )
}

/**
 * 今のやり方の決まり (問題の設定の今のやり方. k 期前, 平均, 列).
 * 時刻の列, 系列を分ける列, 1 期の間隔も持つ (ML-Inference の決まりの形)
 */
export interface Baseline {
  type: 'lag' | 'mean' | 'column'
  periods?: number
  column?: string
  [key: string]: unknown
}

/** 画面で編集する意思決定 (欄は文字で持つ. 決まりの今のやり方は別に持つ) */
export interface DecisionForm {
  type: Decision['type'] | ''
  positive: string
  values: Record<string, string>
  baseline?: Baseline | null
}

/** 今のやり方の決まりを短い言葉にする */
export function baselineText(baseline: Baseline): string {
  if (baseline.type === 'lag') return `${baseline.periods} 期前の値と同じ量`
  if (baseline.type === 'mean') return `直前の ${baseline.periods} 期の値の平均の量`
  return `列 ${baseline.column} の値の量`
}

/** 決まりを使える型か (k 期前と平均は量だけ. 列の決まりは量だけで使う) */
function usesBaseline(form: DecisionForm): boolean {
  return form.type === 'quantity' && !!form.baseline
}

/** 意思決定を編集の形にする */
export function toDecisionForm(decision: Decision | null | undefined): DecisionForm {
  if (!decision) return { type: '', positive: '', values: {} }
  const values: Record<string, string> = {}
  let baseline: Baseline | null = null
  for (const [key, value] of Object.entries(decision)) {
    if (key === 'current' && value !== null && typeof value === 'object') {
      baseline = value as Baseline
      continue
    }
    if (key !== 'type' && key !== 'positive' && value != null) values[key] = String(value)
  }
  return { type: decision.type, positive: String(decision.positive ?? ''), values, baseline }
}

/** 今のやり方の欄が空のときに使う決まり (無ければ null) */
export function activeBaseline(form: DecisionForm): Baseline | null {
  return usesBaseline(form) && !(form.values.current ?? '').trim() ? (form.baseline ?? null) : null
}

/** 数なら数, それ以外は列の名前として渡す (全角の数字やカンマもそろえる) */
function parseValue(text: string): number | string {
  const normal = text.normalize('NFKC').trim()
  const value = Number(normal.replace(/,/g, ''))
  return normal !== '' && Number.isFinite(value) ? value : normal
}

/** 列の名前を書けない欄 (件数と刻みと上限) */
export const NUMBER_ONLY = new Set(['step', 'budget', 'budget_cost', 'capacity'])

/** 整数だけの欄 (件数) */
const INTEGER_ONLY = new Set(['budget', 'capacity'])

/** 列の名前だけを書く欄 */
const COLUMN_ONLY = new Set(['current'])

/**
 * 編集の形を API の意思決定にし, 画面で分かる誤りを返す
 *
 * 細かな確認 (列の有無, 一式の外れ方など) はライブラリが行い, 誤りの文を返す
 */
export function fromDecisionForm(
  form: DecisionForm,
  classes: string[],
): { decision: Decision | null; problems: string[] } {
  if (!form.type) return { decision: null, problems: [] }
  const problems: string[] = []
  const decision: Decision = { type: form.type }
  if (form.type !== 'quantity') {
    if (!classes.includes(form.positive)) problems.push('当たりのクラスを選んでください')
    decision.positive = form.positive
  }
  const uplift = isUplift(form)
  for (const field of visibleFields(form)) {
    const text = (form.values[field.key] ?? '').trim()
    if (!text) {
      if (!field.optional) problems.push(`「${field.label}」を書いてください`)
      continue
    }
    if (field.choices) {
      if (!field.choices.some((choice) => choice.value === text))
        problems.push(`「${field.label}」は選択肢から選んでください`)
      decision[field.key] = text
      continue
    }
    const value = COLUMN_ONLY.has(field.key) ? text : parseValue(text)
    if (NUMBER_ONLY.has(field.key) && typeof value !== 'number')
      problems.push(`「${field.label}」は数で書いてください`)
    if (typeof value === 'number' && field.key === 'effect' && uplift) {
      if (Math.abs(value) > 1)
        problems.push(`「${field.label}」は確率の差なので -1 から 1 にしてください`)
    } else if (typeof value === 'number') {
      if (value < 0) problems.push(`「${field.label}」は 0 以上にしてください`)
      if (field.key === 'effect' && value > 1)
        problems.push(`「${field.label}」は 0 から 1 にしてください`)
      if (INTEGER_ONLY.has(field.key) && !Number.isInteger(value))
        problems.push(`「${field.label}」は件数なので整数にしてください`)
    }
    decision[field.key] = value
  }
  // 今のやり方の列を書かなければ, 問題の設定の決まりで比べる
  const baseline = activeBaseline(form)
  if (baseline) decision.current = baseline
  return { decision, problems }
}
