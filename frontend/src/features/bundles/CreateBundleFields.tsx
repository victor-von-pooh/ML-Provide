import { Plus, X } from 'lucide-react'
import { useState, type ReactNode } from 'react'

import type { ProcedureEntry } from '../../api/types'
import {
  ARRAY_FORMS,
  newInputRow,
  NOISE_SOURCES,
  PREPEND_SOURCES,
  SOURCE_LABELS,
  TRAINED_ONLY_FORMS,
  type InputRow,
  type OutputForm,
  type ProcedureForm,
} from '../../lib/bundleForm'
import { FORM_LABELS, ROLE_LABELS } from '../../lib/outputs'

const field = 'h-8 rounded-lg border border-line bg-surface px-2 text-sm'

/** 繰り返しの手順の欄 (名前, 回数, 種, 手順ごとの設定) */
export function ProcedureFields({
  form,
  onChange,
  procedures,
  noData,
  maxSteps,
}: {
  form: ProcedureForm
  onChange: (form: ProcedureForm) => void
  procedures: ProcedureEntry[]
  noData: boolean
  maxSteps: number
}) {
  // 雑音だけの一式は雑音から始まる手順だけ
  const choices = noData ? procedures.filter((entry) => entry.start === 'noise') : procedures
  const entry = procedures.find((item) => item.name === form.name)
  return (
    <div className="grid gap-3 md:grid-cols-3">
      <div className="md:col-span-3">
        <label htmlFor="bundle-procedure" className="mb-1 block text-xs text-ink-2">
          手順 (登録簿の名前{noData ? '. 雑音から始まる手順' : ''})
        </label>
        <select
          id="bundle-procedure"
          value={form.name}
          onChange={(event) => onChange({ ...form, name: event.target.value, options: {} })}
          className={`${field} w-full`}
        >
          <option value="">{noData ? '選んでください' : '使わない (モデルを 1 回呼ぶ)'}</option>
          {choices.map((item) => (
            <option key={item.name} value={item.name}>
              {item.name} ({item.description}
              {item.builtin ? '' : '. 自作'})
            </option>
          ))}
        </select>
        {entry && (
          <p className="mt-1 text-xs text-ink-3">
            始まり: {entry.start === 'noise' ? '雑音' : '渡した番号の並び'}
            {entry.time ? '. モデルに時刻も渡す (入力の出どころに「手順の時刻」が要る)' : ''}
          </p>
        )}
      </div>
      {entry && (
        <>
          <div>
            <label htmlFor="bundle-steps" className="mb-1 block text-xs text-ink-2">
              回数 (要る. {maxSteps.toLocaleString()} 回まで. 文の生成は足す番号の数の上限)
            </label>
            <input
              id="bundle-steps"
              type="number"
              min={1}
              max={maxSteps}
              value={form.steps}
              onChange={(event) => onChange({ ...form, steps: event.target.value })}
              className={`${field} w-full`}
            />
          </div>
          <div>
            <label htmlFor="bundle-procedure-seed" className="mb-1 block text-xs text-ink-2">
              乱数の種 (空なら 0. 推論で変えられる)
            </label>
            <input
              id="bundle-procedure-seed"
              type="number"
              value={form.seed}
              onChange={(event) => onChange({ ...form, seed: event.target.value })}
              className={`${field} w-full`}
            />
          </div>
          {(entry.options ?? []).map((name) => (
            <div key={name}>
              <label
                htmlFor={`bundle-option-${name}`}
                className="mb-1 block font-mono text-xs text-ink-2"
              >
                {name}
              </label>
              <input
                id={`bundle-option-${name}`}
                value={form.options[name] ?? ''}
                placeholder={
                  entry.defaults && name in entry.defaults
                    ? `空なら既定 ${String(entry.defaults[name])}`
                    : '空なら既定値'
                }
                onChange={(event) =>
                  onChange({ ...form, options: { ...form.options, [name]: event.target.value } })
                }
                className={`${field} w-full font-mono`}
              />
            </div>
          ))}
          {entry.options === null && (
            <div className="md:col-span-3">
              <label htmlFor="bundle-option-extra" className="mb-1 block text-xs text-ink-2">
                手順の設定 (1 行に 名前=値)
              </label>
              <textarea
                id="bundle-option-extra"
                value={form.extra}
                rows={3}
                onChange={(event) => onChange({ ...form, extra: event.target.value })}
                className="w-full rounded-lg border border-line bg-surface p-2 font-mono text-sm"
              />
            </div>
          )}
        </>
      )}
    </div>
  )
}

/** 受け持ち, 出力の形, 出力の形ごとの欄 */
export function OutputFields({
  role,
  onRole,
  form,
  onForm,
  fields,
  onFields,
  noData,
}: {
  role: string
  onRole: (role: string) => void
  form: string
  onForm: (form: string) => void
  fields: OutputForm
  onFields: (fields: OutputForm) => void
  noData: boolean
}) {
  // 雑音だけの一式は配列の出力の形だけ (省くとタスクから決められない). 表は学習から作る一式だけ
  const forms = Object.entries(FORM_LABELS).filter(
    ([value]) => !TRAINED_ONLY_FORMS.includes(value) && (!noData || ARRAY_FORMS.includes(value)),
  )
  return (
    <div className="grid gap-3 md:grid-cols-2">
      <div>
        <label htmlFor="bundle-role" className="mb-1 block text-xs text-ink-2">
          受け持ち
        </label>
        <select
          id="bundle-role"
          value={role}
          onChange={(event) => onRole(event.target.value)}
          className={`${field} w-full`}
        >
          <option value="">{noData ? '作る (既定)' : '予測する (既定)'}</option>
          {Object.entries(ROLE_LABELS)
            .filter(([value]) => value !== (noData ? 'generate' : 'predict'))
            .map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
        </select>
      </div>
      <div>
        <label htmlFor="bundle-form" className="mb-1 block text-xs text-ink-2">
          {noData ? '出力の形 (要る)' : '出力の形 (正解の無いデータセットでは選ぶ)'}
        </label>
        <select
          id="bundle-form"
          value={form}
          onChange={(event) => onForm(event.target.value)}
          className={`${field} w-full`}
        >
          <option value="">{noData ? '選んでください' : 'タスクから決める'}</option>
          {forms.map(([value, label]) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>
      </div>
      {form === 'distribution' && (
        <div className="md:col-span-2">
          <label htmlFor="bundle-quantiles" className="mb-1 block text-xs text-ink-2">
            分布の水準 (分位点. 小さい順にカンマ区切り)
          </label>
          <input
            id="bundle-quantiles"
            value={fields.quantiles}
            placeholder="0.1, 0.5, 0.9"
            onChange={(event) => onFields({ ...fields, quantiles: event.target.value })}
            className={`${field} w-full font-mono`}
          />
        </div>
      )}
      {form === 'mask' && (
        <div className="md:col-span-2">
          <label htmlFor="bundle-classes" className="mb-1 block text-xs text-ink-2">
            領域のクラスの名前 (番号の順にカンマ区切り. 空なら番号)
          </label>
          <input
            id="bundle-classes"
            value={fields.classes}
            placeholder="background, cell"
            onChange={(event) => onFields({ ...fields, classes: event.target.value })}
            className={`${field} w-full`}
          />
        </div>
      )}
      {ARRAY_FORMS.includes(form) && (
        <div className="md:col-span-2">
          <label htmlFor="bundle-shape" className="mb-1 block text-xs text-ink-2">
            1 件の形 (雑音から始まる手順では要る. カンマ区切り)
          </label>
          <input
            id="bundle-shape"
            value={fields.shape}
            placeholder="1, 32, 32"
            onChange={(event) => onFields({ ...fields, shape: event.target.value })}
            className={`${field} w-full font-mono`}
          />
        </div>
      )}
    </div>
  )
}

/** モデルの入力ごとの出どころ (入力が複数のモデル, 先頭に番号を足した入力, 手順の中身と時刻) */
export function InputFields({
  rows,
  onChange,
  noData,
}: {
  rows: InputRow[]
  onChange: (rows: InputRow[]) => void
  noData: boolean
}) {
  const sources = Object.entries(SOURCE_LABELS).filter(
    ([value]) => !noData || NOISE_SOURCES.includes(value),
  )
  // 書き換えた行は自動で足した行として扱わない (外すときに消さない)
  const update = (index: number, row: InputRow) =>
    onChange(rows.map((item, at) => (at === index ? { ...row, auto: false } : item)))
  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs text-ink-3">
        モデルの定義の入力の名前ごとに, 何を渡すかを選びます. 入力が 1 つのモデルは空のままでよく,
        表なら表の列, 他は中身 (雑音だけの一式は手順の中身) を渡します.
      </p>
      {rows.map((row, index) => (
        <fieldset
          key={row.id}
          aria-label={`入力 ${index + 1}`}
          className={`grid items-end gap-2 ${noData ? 'md:grid-cols-[1fr_1.5fr_auto]' : 'md:grid-cols-[1fr_1.5fr_1fr_auto]'}`}
        >
          <div>
            <label htmlFor={`bundle-input-${index}`} className="mb-1 block text-xs text-ink-2">
              入力の名前
            </label>
            <input
              id={`bundle-input-${index}`}
              value={row.name}
              placeholder={noData ? (row.source === 'time' ? '例: t' : '例: x') : '例: ids'}
              onChange={(event) => update(index, { ...row, name: event.target.value })}
              className={`${field} w-full font-mono`}
            />
          </div>
          <div>
            <label
              htmlFor={`bundle-input-source-${index}`}
              className="mb-1 block text-xs text-ink-2"
            >
              出どころ
            </label>
            <select
              id={`bundle-input-source-${index}`}
              value={row.source}
              onChange={(event) => update(index, { ...row, source: event.target.value })}
              className={`${field} w-full`}
            >
              {sources.map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
              {/* 雑音だけに切り替える前の出どころも, 選んでいる値として見せる (問題として知らせる) */}
              {!sources.some(([value]) => value === row.source) && (
                <option value={row.source}>
                  {SOURCE_LABELS[row.source] ?? row.source} (この一式では使えない)
                </option>
              )}
            </select>
          </div>
          {/* 雑音だけの一式は番号の並びを渡さないので, 先頭の番号の欄を出さない */}
          {!noData && (
            <div>
              <label
                htmlFor={`bundle-input-prepend-${index}`}
                className="mb-1 block text-xs text-ink-2"
              >
                先頭に足した番号 ([CLS] など)
              </label>
              <input
                id={`bundle-input-prepend-${index}`}
                value={row.prepend}
                disabled={!PREPEND_SOURCES.includes(row.source)}
                placeholder="2001"
                onChange={(event) => update(index, { ...row, prepend: event.target.value })}
                className={`${field} w-full font-mono`}
              />
            </div>
          )}
          <button
            type="button"
            aria-label={`入力 ${index + 1} を除く`}
            className="mb-1 rounded p-1 text-ink-3 hover:text-ink"
            onClick={() => onChange(rows.filter((_, at) => at !== index))}
          >
            <X size={16} aria-hidden />
          </button>
        </fieldset>
      ))}
      <div>
        <button
          type="button"
          className="flex items-center gap-1 text-xs text-accent-text underline"
          onClick={() => onChange([...rows, newInputRow(noData ? 'state' : 'features')])}
        >
          <Plus size={14} aria-hidden />
          入力を足す
        </button>
      </div>
    </div>
  )
}

/**
 * 畳める欄. 開き閉じは利用者の操作を写して持ち, 自動では閉じない.
 * wantOpen が偽から真になったとき (雑音だけを選んだ) と, 誤りの数が 0 から 1 以上に
 * なったときだけ自動で開く (打っている途中に誤りが消えても, 開いた欄を閉じないため)
 */
export function AutoDetails({
  summary,
  wantOpen,
  problems,
  children,
}: {
  summary: string
  wantOpen: boolean
  problems: number
  children: ReactNode
}) {
  const [open, setOpen] = useState(wantOpen || problems > 0)
  const failing = problems > 0
  const [seen, setSeen] = useState({ wantOpen, failing })
  if (seen.wantOpen !== wantOpen || seen.failing !== failing) {
    setSeen({ wantOpen, failing })
    if ((wantOpen && !seen.wantOpen) || (failing && !seen.failing)) setOpen(true)
  }
  return (
    <details
      className="rounded-lg border border-line p-3"
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary className="cursor-pointer text-xs font-semibold text-ink-2">{summary}</summary>
      <div className="mt-3">{children}</div>
    </details>
  )
}
