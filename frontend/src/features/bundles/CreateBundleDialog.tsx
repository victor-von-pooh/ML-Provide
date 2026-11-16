import { useMutation } from '@tanstack/react-query'
import { useState } from 'react'

import { api } from '../../api/client'
import type { Bundle } from '../../api/types'
import { Dialog } from '../../components/Dialog'
import { PathField } from '../../components/PathPicker'
import { Button, ErrorBox } from '../../components/ui'

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
  const [copyWeights, setCopyWeights] = useState(true)
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
      const targetScale =
        center.trim() && scale.trim() ? { center: Number(center), scale: Number(scale) } : null
      return api.createBundle({
        name: name.trim(),
        spec: spec.trim(),
        dataset: dataset.trim(),
        weights: weights.trim(),
        inputs,
        output: output.trim() || null,
        target_scale: targetScale,
        copy_weights: copyWeights,
      })
    },
    onSuccess: onCreated,
  })
  const ready = name.trim() && spec.trim() && dataset.trim() && weights.trim()
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
          モデルの定義, ML-EDA で書き出したデータセット, ML-Found の実験 (か重みのファイル)
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
            <fieldset>
              <legend className="mb-1 text-xs text-ink-2">
                目的の値を (y - 中心) / 幅 にして学習した場合の中心と幅
              </legend>
              <div className="flex gap-2">
                <input
                  aria-label="中心"
                  type="number"
                  value={center}
                  placeholder="中心"
                  onChange={(event) => setCenter(event.target.value)}
                  className={`${field} w-28`}
                />
                <input
                  aria-label="幅"
                  type="number"
                  value={scale}
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
