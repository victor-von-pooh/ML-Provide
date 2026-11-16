import { Plus, X } from 'lucide-react'

import { Button } from '../../components/ui'
import type { RulesForm } from '../../lib/rules'

const field = 'h-8 rounded-lg border border-line bg-surface px-2 text-sm'

/** 結論の決め方の編集 (分類は保留と陽性, 回帰と予測は値の帯. どちらもまとめる列) */
export function RulesEditor({
  form,
  onChange,
  task,
  classes,
  columns,
  suggestions = [],
  idBase,
}: {
  form: RulesForm
  onChange: (form: RulesForm) => void
  task: string
  classes: string[]
  columns: string[]
  suggestions?: string[]
  idBase: string
}) {
  const set = (patch: Partial<RulesForm>) => onChange({ ...form, ...patch })
  const classify = task === 'classification'
  return (
    <div className="grid gap-4 text-sm md:grid-cols-2">
      {classify && (
        <>
          <div>
            <label
              htmlFor={`${idBase}-floor`}
              className="mb-1 block text-xs font-medium text-ink-2"
            >
              確信の下限 (これより低い予測は「保留」)
            </label>
            <input
              id={`${idBase}-floor`}
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={form.minConfidence}
              placeholder="使わない"
              onChange={(event) => set({ minConfidence: event.target.value })}
              className={`${field} w-32`}
            />
          </div>
          {classes.length === 2 && (
            <fieldset>
              <legend className="mb-1 text-xs font-medium text-ink-2">
                陽性のクラスと, 陽性とする確率 (見逃しを減らすなら低く)
              </legend>
              <div className="flex gap-2">
                <label className="sr-only" htmlFor={`${idBase}-positive`}>
                  陽性のクラス
                </label>
                <select
                  id={`${idBase}-positive`}
                  value={form.positive}
                  onChange={(event) =>
                    // 陽性を選んだら確率の既定 0.5 と, 一緒に使えない下限を外す
                    set({
                      positive: event.target.value,
                      threshold: form.threshold || '0.5',
                      minConfidence: event.target.value ? '' : form.minConfidence,
                    })
                  }
                  className={field}
                >
                  <option value="">使わない (確率の最大)</option>
                  {classes.map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>
                <label className="sr-only" htmlFor={`${idBase}-threshold`}>
                  陽性とする確率
                </label>
                <input
                  id={`${idBase}-threshold`}
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  disabled={!form.positive}
                  value={form.threshold}
                  placeholder="0.5"
                  onChange={(event) => set({ threshold: event.target.value })}
                  className={`${field} w-24`}
                />
              </div>
            </fieldset>
          )}
        </>
      )}
      {!classify && (
        <fieldset className="md:col-span-2">
          <legend className="mb-1 text-xs font-medium text-ink-2">
            値の帯 (上限の小さい順. 値が上限より小さければその帯. 最後の帯は上限無し)
          </legend>
          <ul className="flex flex-col gap-1">
            {form.bands.map((band, index) => {
              const last = index === form.bands.length - 1
              return (
                <li key={index} className="flex items-center gap-2">
                  <input
                    aria-label={`帯 ${index + 1} の名前`}
                    value={band.label}
                    placeholder="名前"
                    onChange={(event) =>
                      set({
                        bands: form.bands.map((b, i) =>
                          i === index ? { ...b, label: event.target.value } : b,
                        ),
                      })
                    }
                    className={`${field} w-40`}
                  />
                  {last ? (
                    <span className="w-32 text-xs text-ink-3">上限無し</span>
                  ) : (
                    <input
                      aria-label={`帯 ${index + 1} の上限`}
                      type="number"
                      value={band.upper}
                      placeholder="上限"
                      onChange={(event) =>
                        set({
                          bands: form.bands.map((b, i) =>
                            i === index ? { ...b, upper: event.target.value } : b,
                          ),
                        })
                      }
                      className={`${field} w-32`}
                    />
                  )}
                  <Button
                    variant="ghost"
                    icon={X}
                    aria-label={`帯 ${index + 1} を消す`}
                    onClick={() => set({ bands: form.bands.filter((_, i) => i !== index) })}
                  />
                </li>
              )
            })}
          </ul>
          <Button
            variant="ghost"
            icon={Plus}
            className="mt-1"
            onClick={() =>
              set({
                // 新しい帯は最後の手前に (最後は上限無しのまま)
                bands:
                  form.bands.length === 0
                    ? [
                        { label: '少ない', upper: '' },
                        { label: '多い', upper: '' },
                      ]
                    : [
                        ...form.bands.slice(0, -1),
                        { label: '', upper: '' },
                        form.bands[form.bands.length - 1]!,
                      ],
              })
            }
          >
            帯を足す
          </Button>
        </fieldset>
      )}
      <div>
        <label htmlFor={`${idBase}-group`} className="mb-1 block text-xs font-medium text-ink-2">
          まとめる列 (値ごとに件数と内訳を出す)
        </label>
        {/* 推論の前は列が分からないので名前を書く */}
        {columns.length > 0 ? (
          <select
            id={`${idBase}-group`}
            value={form.groupBy}
            onChange={(event) => set({ groupBy: event.target.value })}
            className={field}
          >
            <option value="">まとめない</option>
            {columns.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        ) : (
          <>
            <input
              id={`${idBase}-group`}
              value={form.groupBy}
              list={`${idBase}-group-list`}
              placeholder="列の名前 (空ならまとめない)"
              onChange={(event) => set({ groupBy: event.target.value })}
              className={`${field} w-56`}
            />
            <datalist id={`${idBase}-group-list`}>
              {suggestions.map((name) => (
                <option key={name} value={name} />
              ))}
            </datalist>
          </>
        )}
      </div>
    </div>
  )
}
