import { useMutation } from '@tanstack/react-query'
import { useState } from 'react'

import { api } from '../../api/client'
import type { Bundle } from '../../api/types'
import { Dialog } from '../../components/Dialog'
import { PathField } from '../../components/PathPicker'
import { Button, ErrorBox } from '../../components/ui'
import { FORM_LABELS, ROLE_LABELS } from '../../lib/outputs'

const field = 'h-8 rounded-lg border border-line bg-surface px-2 text-sm'

/** 学習の結果 (モデルの定義, データセット, 実験) から推論の一式を作る */
export function CreateBundleDialog({
  onClose,
  onCreated,
}: {
  onClose: () => void
  onCreated: (bundle: Bundle) => void
}) {
  const [name, setName] = useState('')
  const [spec, setSpec] = useState('')
  const [dataset, setDataset] = useState('')
  const [weights, setWeights] = useState('')
  const [inputName, setInputName] = useState('')
  const [prepend, setPrepend] = useState('')
  const [output, setOutput] = useState('')
  const [center, setCenter] = useState('')
  const [scale, setScale] = useState('')
  const [trainScale, setTrainScale] = useState(false)
  const [copyWeights, setCopyWeights] = useState(true)
  // 受け持ちと出力の形 (空はタスクから. 正解の無いデータセットは出力の形が要る)
  const [role, setRole] = useState('')
  const [form, setForm] = useState('')
  const create = useMutation({
    mutationFn: () => {
      // 先頭に足す番号は, 入力の名前と番号の並びが揃ったときだけ
      const ids = prepend
        .split(/[\s,]+/)
        .filter(Boolean)
        .map((value) => Number(value))
      const inputs =
        inputName.trim() && ids.length > 0
          ? { [inputName.trim()]: { source: 'items', prepend: ids } }
          : null
      const targetScale = trainScale
        ? ('train' as const)
        : center.trim() && scale.trim()
          ? { center: Number(center), scale: Number(scale) }
          : null
      return api.createBundle({
        name: name.trim(),
        spec: spec.trim(),
        dataset: dataset.trim(),
        weights: weights.trim(),
        inputs,
        output: output.trim() || null,
        target_scale: targetScale,
        copy_weights: copyWeights,
        role: role || null,
        form: form || null,
      })
    },
    onSuccess: onCreated,
  })
  // 片方だけの欄と数でない値は, 送る前に知らせる
  const problems: string[] = []
  const ids = prepend.split(/[\s,]+/).filter(Boolean)
  if (ids.some((value) => !Number.isInteger(Number(value))))
    problems.push('足した番号は整数で書いてください')
  if (!!inputName.trim() !== ids.length > 0)
    problems.push('先頭に番号を足した入力は, 入力の名前と番号の両方を書いてください')
  if (!trainScale && !!center.trim() !== !!scale.trim())
    problems.push('目的の値の中心と幅は両方を書いてください')
  if (!trainScale && scale.trim() && !(Number(scale) > 0)) problems.push('幅は正の数にしてください')
  const ready =
    name.trim() && spec.trim() && dataset.trim() && weights.trim() && problems.length === 0
  return (
    <Dialog title="学習の結果から一式を作る" onClose={onClose}>
      <form
        className="flex flex-col gap-3"
        onSubmit={(event) => {
          event.preventDefault()
          if (ready) create.mutate()
        }}
      >
        <p className="text-xs text-ink-2">
          モデルの定義, ML-EDA (ML-Fathom の画面) で書き出したデータセット, ML-Found の実験
          (か重みのファイル) を選びます. 学習用の見本を推論できることを確かめてから,
          このアプリの置き場に作ります (元のファイルは変えません).
        </p>
        <div>
          <label htmlFor="bundle-name" className="mb-1 block text-xs font-medium text-ink-2">
            名前
          </label>
          <input
            id="bundle-name"
            value={name}
            onChange={(event) => setName(event.target.value)}
            maxLength={80}
            className={`${field} w-full`}
          />
        </div>
        <PathField
          id="bundle-spec"
          label="モデルの定義 (YAML か JSON)"
          value={spec}
          onChange={setSpec}
          pickDirs={false}
        />
        <PathField
          id="bundle-dataset"
          label="データセット (dataset.yaml を持つディレクトリ)"
          value={dataset}
          onChange={setDataset}
          pickFiles={false}
        />
        <PathField
          id="bundle-weights"
          label="実験のディレクトリか重みのファイル"
          value={weights}
          onChange={setWeights}
          hint="実験のディレクトリなら final_model.pt を使い, 状態とベストの値を記録します"
        />
        <details className="rounded-lg border border-line p-3">
          <summary className="cursor-pointer text-xs font-semibold text-ink-2">
            学習のスクリプトが前処理の外でしたこと (あれば)
          </summary>
          <div className="mt-3 grid gap-3 md:grid-cols-2">
            <div>
              <label htmlFor="bundle-input" className="mb-1 block text-xs text-ink-2">
                並びの先頭に番号を足した入力の名前
              </label>
              <input
                id="bundle-input"
                value={inputName}
                placeholder="ids"
                onChange={(event) => setInputName(event.target.value)}
                className={`${field} w-full font-mono`}
              />
            </div>
            <div>
              <label htmlFor="bundle-prepend" className="mb-1 block text-xs text-ink-2">
                足した番号 ([CLS] など. カンマ区切り)
              </label>
              <input
                id="bundle-prepend"
                value={prepend}
                placeholder="2001"
                onChange={(event) => setPrepend(event.target.value)}
                className={`${field} w-full font-mono`}
              />
            </div>
            <div>
              <label htmlFor="bundle-output" className="mb-1 block text-xs text-ink-2">
                使う出力の名前 (出力が複数のモデル)
              </label>
              <input
                id="bundle-output"
                value={output}
                placeholder="logits"
                onChange={(event) => setOutput(event.target.value)}
                className={`${field} w-full font-mono`}
              />
            </div>
            <div>
              <label htmlFor="bundle-role" className="mb-1 block text-xs text-ink-2">
                受け持ち
              </label>
              <select
                id="bundle-role"
                value={role}
                onChange={(event) => setRole(event.target.value)}
                className={`${field} w-full`}
              >
                <option value="">予測する (既定)</option>
                {Object.entries(ROLE_LABELS)
                  .filter(([value]) => value !== 'predict')
                  .map(([value, label]) => (
                    <option key={value} value={value}>
                      {label}
                    </option>
                  ))}
              </select>
            </div>
            <div>
              <label htmlFor="bundle-form" className="mb-1 block text-xs text-ink-2">
                出力の形 (正解の無いデータセットでは選ぶ)
              </label>
              <select
                id="bundle-form"
                value={form}
                onChange={(event) => setForm(event.target.value)}
                className={`${field} w-full`}
              >
                <option value="">タスクから決める</option>
                {Object.entries(FORM_LABELS).map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </div>
            <fieldset className="md:col-span-2">
              <legend className="mb-1 text-xs text-ink-2">
                目的の値を標準化して学習した場合 (回帰と予測)
              </legend>
              <label className="mb-2 flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={trainScale}
                  onChange={(event) => setTrainScale(event.target.checked)}
                />
                学習用の分割の平均と標準偏差で標準化した (中心と幅を求める)
              </label>
              <div className="flex items-center gap-2">
                <span className="text-xs text-ink-3">または (y - 中心) / 幅 の</span>
                <input
                  aria-label="中心"
                  type="number"
                  value={center}
                  disabled={trainScale}
                  placeholder="中心"
                  onChange={(event) => setCenter(event.target.value)}
                  className={`${field} w-28`}
                />
                <input
                  aria-label="幅"
                  type="number"
                  value={scale}
                  disabled={trainScale}
                  placeholder="幅"
                  onChange={(event) => setScale(event.target.value)}
                  className={`${field} w-28`}
                />
              </div>
            </fieldset>
          </div>
        </details>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={copyWeights}
            onChange={(event) => setCopyWeights(event.target.checked)}
          />
          重みを一式に写す (外すと実験の重みを指す. 実験を消すと推論できなくなる)
        </label>
        {problems.map((problem) => (
          <p key={problem} className="text-xs text-critical-text">
            {problem}
          </p>
        ))}
        {create.error && <ErrorBox error={create.error} />}
        <div className="flex justify-end">
          <Button type="submit" variant="primary" disabled={!ready || create.isPending}>
            {create.isPending ? '確かめています…' : '作る'}
          </Button>
        </div>
      </form>
    </Dialog>
  )
}
