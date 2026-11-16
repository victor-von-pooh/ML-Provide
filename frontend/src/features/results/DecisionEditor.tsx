import type { DecisionForm } from '../../lib/decision'
import { DECISION_TYPES } from '../../lib/decision'

const field = 'h-8 rounded-lg border border-line bg-surface px-2 text-sm'

/** 意思決定の型と損得の欄 (回帰は量, 2 クラスの分類は選ぶか振り分け) */
export function DecisionEditor({
  form,
  onChange,
  task,
  classes,
  idBase,
}: {
  form: DecisionForm
  onChange: (form: DecisionForm) => void
  task: string
  classes: string[]
  idBase: string
}) {
  const classify = task === 'classification'
  const types = (Object.keys(DECISION_TYPES) as (keyof typeof DECISION_TYPES)[]).filter((type) =>
    type === 'quantity' ? !classify : classify && classes.length === 2,
  )
  if (types.length === 0) {
    return (
      <p className="text-xs text-ink-3">
        この一式に使える意思決定の型はありません (量は回帰と予測, 選ぶと振り分けは 2 クラスの分類).
      </p>
    )
  }
  const spec = form.type ? DECISION_TYPES[form.type] : null
  return (
    <div className="flex flex-col gap-3 text-sm">
      <div className="flex flex-wrap items-center gap-2">
        <label htmlFor={`${idBase}-type`} className="text-xs font-medium text-ink-2">
          型
        </label>
        <select
          id={`${idBase}-type`}
          value={form.type}
          onChange={(event) =>
            onChange({ ...form, type: event.target.value as DecisionForm['type'] })
          }
          className={field}
        >
          <option value="">意思決定しない (予測と結論だけ)</option>
          {types.map((type) => (
            <option key={type} value={type}>
              {DECISION_TYPES[type].label}
            </option>
          ))}
        </select>
      </div>
      {spec && (
        <div className="grid gap-3 md:grid-cols-2">
          {form.type !== 'quantity' && (
            <div>
              <label
                htmlFor={`${idBase}-positive`}
                className="mb-1 block text-xs font-medium text-ink-2"
              >
                当たりのクラス (行動の対象)
              </label>
              <select
                id={`${idBase}-positive`}
                value={form.positive}
                onChange={(event) => onChange({ ...form, positive: event.target.value })}
                className={field}
              >
                <option value="">選ぶ</option>
                {classes.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </div>
          )}
          {spec.fields.map((item) => (
            <div key={item.key}>
              <label
                htmlFor={`${idBase}-${item.key}`}
                className="mb-1 block text-xs font-medium text-ink-2"
              >
                {item.label}
                {item.optional && ' (任意)'}
              </label>
              <input
                id={`${idBase}-${item.key}`}
                value={form.values[item.key] ?? ''}
                onChange={(event) =>
                  onChange({ ...form, values: { ...form.values, [item.key]: event.target.value } })
                }
                className={`${field} w-56`}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
