import type { Decision } from '../api/types'

/** 意思決定の型の名前と, 型ごとの欄 (名前, 説明, 省けるか) */
export const DECISION_TYPES: Record<
  Decision['type'],
  { label: string; fields: { key: string; label: string; optional?: boolean }[] }
> = {
  quantity: {
    label: '量を決める (補充, 仕入れ, 人員)',
    fields: [
      { key: 'under_cost', label: '1 単位足りないときの損 (欠品の利益など)' },
      { key: 'over_cost', label: '1 単位余ったときの損 (捨てるなら原価, 持ち越すなら保管の費用)' },
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
      { key: 'effect', label: '行動で当たりの損を防げる割合 (0 から 1. 仮定. 列も可)' },
      { key: 'budget', label: '行動できる件数の上限', optional: true },
      { key: 'budget_cost', label: '費用の上限', optional: true },
      { key: 'current', label: '今のやり方で行動したかの列 (たどり直しで比べる)', optional: true },
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

/** 画面で編集する意思決定 (欄は文字で持つ) */
export interface DecisionForm {
  type: Decision['type'] | ''
  positive: string
  values: Record<string, string>
}

/** 意思決定を編集の形にする */
export function toDecisionForm(decision: Decision | null | undefined): DecisionForm {
  if (!decision) return { type: '', positive: '', values: {} }
  const values: Record<string, string> = {}
  for (const [key, value] of Object.entries(decision)) {
    if (key !== 'type' && key !== 'positive' && value != null) values[key] = String(value)
  }
  return { type: decision.type, positive: String(decision.positive ?? ''), values }
}

/** 数なら数, それ以外は列の名前として渡す */
function parseValue(text: string): number | string {
  const value = Number(text)
  return text.trim() !== '' && Number.isFinite(value) ? value : text.trim()
}

/** 列の名前を書けない欄 (件数と刻みと上限) */
const NUMBER_ONLY = new Set(['step', 'budget', 'budget_cost', 'capacity'])

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
  for (const field of DECISION_TYPES[form.type].fields) {
    const text = (form.values[field.key] ?? '').trim()
    if (!text) {
      if (!field.optional) problems.push(`「${field.label}」を書いてください`)
      continue
    }
    const value = COLUMN_ONLY.has(field.key) ? text : parseValue(text)
    if (NUMBER_ONLY.has(field.key) && typeof value !== 'number')
      problems.push(`「${field.label}」は数で書いてください`)
    decision[field.key] = value
  }
  return { decision, problems }
}
