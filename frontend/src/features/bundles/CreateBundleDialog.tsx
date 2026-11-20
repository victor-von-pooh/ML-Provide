import { useMutation, useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import { api } from '../../api/client'
import type { Bundle, DatasetDescription } from '../../api/types'
import { Dialog } from '../../components/Dialog'
import { PathField } from '../../components/PathPicker'
import { Button, ErrorBox } from '../../components/ui'
import { ENTRY_KIND_LABELS, FORM_LABELS, ORIGIN_LABELS, ROLE_LABELS } from '../../lib/outputs'
import { useDebounced } from '../../lib/useDebounced'

const field = 'h-8 rounded-lg border border-line bg-surface px-2 text-sm'

/** 空でなく, 0 以上の整数でない欄か */
function notCount(value: string, positive = false): boolean {
  if (!value.trim()) return false
  const number = Number(value)
  return !Number.isInteger(number) || number < (positive ? 1 : 0)
}

/** 版 2 のデータセットの要約 (受け持ち, 正解, 項目, 分割, 領域の値の名前) */
function ItemsSummary({ described }: { described: DatasetDescription }) {
  const target = described.target
  const origin = target?.origin?.type
  const splits = Object.entries(described.splits ?? {})
  const classes = described.classes ?? []
  return (
    <div
      role="note"
      aria-label="版 2 のデータセット"
      className="rounded-lg border border-line p-2 text-xs text-ink-2"
    >
      <p>
        版 2 のデータセット (ファイルの組): {ROLE_LABELS[described.role ?? ''] ?? described.role},
        正解 {FORM_LABELS[target?.kind ?? ''] ?? target?.kind}
        {origin && ` (出どころ: ${ORIGIN_LABELS[origin] ?? origin})`}
      </p>
      <p>
        項目:{' '}
        {(described.entries ?? [])
          .map(
            (entry) =>
              `${entry.name} (${ENTRY_KIND_LABELS[entry.kind] ?? entry.kind}, ${
                entry.role === 'input' ? '入力' : entry.role === 'target' ? '正解' : entry.role
              })`,
          )
          .join(', ')}
      </p>
      {splits.length > 0 && (
        <p>件数: {splits.map(([name, count]) => `${name} ${count ?? '–'}`).join(', ')}</p>
      )}
      {classes.length > 0 && (
        <p>
          領域の値の名前:{' '}
          {classes.map((item) => `${item.value}: ${item.name ?? '(名前なし)'}`).join(', ')}
        </p>
      )}
    </div>
  )
}

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
  // 版 2 のデータセットは, 選んだときに項目を読み, 1 件の扱いの欄を出す
  const datasetPath = useDebounced(dataset.trim())
  const described = useQuery({
    queryKey: ['dataset', datasetPath],
    queryFn: () => api.describeDataset(datasetPath),
    enabled: !!datasetPath,
    retry: false,
  })
  const itemsData = described.data?.format_version === 2 ? described.data : null
  const itemInputs = (itemsData?.entries ?? []).filter((entry) => entry.role === 'input')
  const [itemInput, setItemInput] = useState('')
  const chosenInput = itemInputs.find((entry) => entry.name === itemInput) ?? itemInputs[0]
  const needsPoints = !!chosenInput?.points
  const [points, setPoints] = useState('')
  const [targetPoints, setTargetPoints] = useState('')
  const [seed, setSeed] = useState('')
  const create = useMutation({
    mutationFn: () => {
      if (itemsData) {
        // 省いた欄 (正解の項目, 扱い方, 領域の値, 戻し方) はデータセットから
        const options: Record<string, number> = {}
        if (needsPoints) {
          options.points = Number(points)
          if (targetPoints.trim()) options.target_points = Number(targetPoints)
          if (seed.trim()) options.seed = Number(seed)
        }
        const items = {
          ...(itemInputs.length > 1 && chosenInput ? { input: chosenInput.name } : {}),
          ...(needsPoints ? { options } : {}),
        }
        return api.createBundle({
          name: name.trim(),
          spec: spec.trim(),
          dataset: dataset.trim(),
          weights: weights.trim(),
          output: output.trim() || null,
          copy_weights: copyWeights,
          items: Object.keys(items).length > 0 ? items : null,
        })
      }
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
  // 片方だけの欄と数でない値は, 送る前に知らせる (版 2 のデータセットは点の欄だけ)
  const problems: string[] = []
  const ids = prepend.split(/[\s,]+/).filter(Boolean)
  if (itemsData) {
    if (needsPoints && (!points.trim() || notCount(points, true)))
      problems.push('入力が撮影か点群なので, 選ぶ点の数 (学習と同じ数) を正の整数で書いてください')
    if (needsPoints && notCount(targetPoints, true))
      problems.push('正解から選ぶ点の数は正の整数で書いてください')
    if (needsPoints && notCount(seed))
      problems.push('点を選ぶ乱数の種は 0 以上の整数で書いてください')
  } else {
    if (ids.some((value) => !Number.isInteger(Number(value))))
      problems.push('足した番号は整数で書いてください')
    if (!!inputName.trim() !== ids.length > 0)
      problems.push('先頭に番号を足した入力は, 入力の名前と番号の両方を書いてください')
    if (!trainScale && !!center.trim() !== !!scale.trim())
      problems.push('目的の値の中心と幅は両方を書いてください')
    if (!trainScale && scale.trim() && !(Number(scale) > 0))
      problems.push('幅は正の数にしてください')
  }
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
          モデルの定義, ML-EDA (ML-Fathom の画面) で書き出したデータセット (列のデータセットか,
          画像と領域, 撮影と点群の版 2 のデータセット), ML-Found の実験 (か重みのファイル)
          を選びます. 学習用の見本を推論できることを確かめてから, このアプリの置き場に作ります
          (元のファイルは変えません).
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
        {itemsData && <ItemsSummary described={itemsData} />}
        {described.error && datasetPath === dataset.trim() && (
          <p className="text-xs text-warning-text">{described.error.message}</p>
        )}
        {itemsData && (itemInputs.length > 1 || needsPoints) && (
          <fieldset className="grid gap-3 rounded-lg border border-line p-3 md:grid-cols-2">
            <legend className="px-1 text-xs font-semibold text-ink-2">
              1 件の扱い (学習のスクリプトと同じ値)
            </legend>
            {itemInputs.length > 1 && (
              <div>
                <label htmlFor="bundle-item-input" className="mb-1 block text-xs text-ink-2">
                  モデルに渡す入力の項目
                </label>
                <select
                  id="bundle-item-input"
                  value={chosenInput?.name ?? ''}
                  onChange={(event) => setItemInput(event.target.value)}
                  className={`${field} w-full`}
                >
                  {itemInputs.map((entry) => (
                    <option key={entry.name} value={entry.name}>
                      {entry.name} ({ENTRY_KIND_LABELS[entry.kind] ?? entry.kind})
                    </option>
                  ))}
                </select>
              </div>
            )}
            {needsPoints && (
              <>
                <div>
                  <label htmlFor="bundle-points" className="mb-1 block text-xs text-ink-2">
                    入力から選ぶ点の数 (要る)
                  </label>
                  <input
                    id="bundle-points"
                    type="number"
                    min={1}
                    value={points}
                    placeholder="512"
                    onChange={(event) => setPoints(event.target.value)}
                    className={`${field} w-full`}
                  />
                </div>
                <div>
                  <label htmlFor="bundle-target-points" className="mb-1 block text-xs text-ink-2">
                    確かめで正解から選ぶ点の数 (空なら全ての点)
                  </label>
                  <input
                    id="bundle-target-points"
                    type="number"
                    min={1}
                    value={targetPoints}
                    onChange={(event) => setTargetPoints(event.target.value)}
                    className={`${field} w-full`}
                  />
                </div>
                <div>
                  <label htmlFor="bundle-seed" className="mb-1 block text-xs text-ink-2">
                    点を選ぶ乱数の種 (空なら 0)
                  </label>
                  <input
                    id="bundle-seed"
                    type="number"
                    min={0}
                    value={seed}
                    onChange={(event) => setSeed(event.target.value)}
                    className={`${field} w-full`}
                  />
                </div>
                <p className="text-xs text-ink-3 md:col-span-2">
                  ML-Quest の骨組みで学習した実験なら, config の data.input_points と
                  data.target_points, 種は data.seed + 1 (検証用の分割の種) を書きます.
                  違う数だと黙って違う予測になります
                </p>
              </>
            )}
          </fieldset>
        )}
        <PathField
          id="bundle-weights"
          label="実験のディレクトリか重みのファイル"
          value={weights}
          onChange={setWeights}
          hint="実験のディレクトリなら final_model.pt を使い, 状態とベストの値を記録します"
        />
        <details className="rounded-lg border border-line p-3">
          <summary className="cursor-pointer text-xs font-semibold text-ink-2">
            {itemsData
              ? '出力が複数のモデル (あれば)'
              : '学習のスクリプトが前処理の外でしたこと (あれば)'}
          </summary>
          <div className="mt-3 grid gap-3 md:grid-cols-2">
            {!itemsData && (
              <>
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
              </>
            )}
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
            {!itemsData && (
              <>
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
              </>
            )}
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
