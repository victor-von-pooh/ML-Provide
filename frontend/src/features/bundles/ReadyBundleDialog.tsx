import { useMutation, useQuery } from '@tanstack/react-query'
import { PackagePlus } from 'lucide-react'
import { useState } from 'react'

import { api } from '../../api/client'
import type { Bundle } from '../../api/types'
import { Dialog } from '../../components/Dialog'
import { Button, ErrorBox, Spinner } from '../../components/ui'
import { classesText, licenseAlert, preprocessText, sha256Problem } from '../../lib/ready'
import { useDebounced } from '../../lib/useDebounced'

/** 初めに入れておく実例 (torchvision の ImageNet で学習した ResNet18) */
const EXAMPLE = { library: 'torchvision', model: 'resnet18', weights: 'IMAGENET1K_V1' }

/** 欄の見た目 */
const FIELD = 'h-8 w-full rounded-lg border border-line bg-surface px-2 text-sm'

/** 欄の名前 */
function Label({ htmlFor, children }: { htmlFor: string; children: React.ReactNode }) {
  return (
    <label htmlFor={htmlFor} className="mb-1 block text-xs font-medium text-ink-2">
      {children}
    </label>
  )
}

/**
 * 学習しない既製のモデル (ライブラリの学習済みの重み) の一式を作って登録する.
 * 出どころ, モデル, 重みを選ぶと, 種類の名前, 前処理, ライセンスを下見する (重みは取りに行かない)
 */
export function ReadyBundleDialog({
  onClose,
  onCreated,
}: {
  onClose: () => void
  onCreated: (bundle: Bundle) => void
}) {
  const info = useQuery({ queryKey: ['info'], queryFn: api.info, staleTime: Infinity })
  const libraries = info.data?.ready_libraries ?? [EXAMPLE.library]
  const [library, setLibrary] = useState(EXAMPLE.library)
  const [model, setModel] = useState(EXAMPLE.model)
  const [weights, setWeights] = useState(EXAMPLE.weights)
  // 名前は書き換えるまでモデルの名前に合わせる
  const [name, setName] = useState<string | null>(null)
  const shownName = name ?? model.trim()
  const [sha256, setSha256] = useState('')
  const hashProblem = sha256Problem(sha256)

  // 打つたびに下見しないように少し待つ
  const names = useDebounced({ library, model: model.trim(), weights: weights.trim() })
  const settled =
    names.library === library && names.model === model.trim() && names.weights === weights.trim()
  const preview = useQuery({
    queryKey: ['ready-preview', names.library, names.model, names.weights],
    queryFn: () => api.readyPreview(names.library, names.model, names.weights),
    enabled: !!names.library && !!names.model && !!names.weights,
    retry: false,
  })
  const shown = settled ? preview.data : undefined
  const alert = shown ? licenseAlert(shown.license) : null

  const create = useMutation({
    mutationFn: () =>
      api.createReadyBundle({
        name: shownName.trim(),
        library,
        model: model.trim(),
        weights: weights.trim(),
        sha256: sha256.trim() || null,
      }),
    onSuccess: onCreated,
  })
  const ready = !!shown && !!shownName.trim() && !hashProblem && !create.isPending
  return (
    <Dialog title="既製のモデルから作る" onClose={onClose}>
      <p className="mb-3 text-xs text-ink-3">
        学習せずにそのまま使うモデル (ライブラリの学習済みの重み) の一式を作ります.
        種類の名前の表と前処理は重みの説明から一式に書き, 重みは写さずに名前 (版)
        とハッシュ値で固定します. 重みは初めに作るときに出どころから取りに行き, PyTorch
        の置き場に置いて次から読みます.
      </p>
      <form
        className="flex flex-col gap-3"
        onSubmit={(event) => {
          event.preventDefault()
          if (ready) create.mutate()
        }}
      >
        <div className="grid gap-3 md:grid-cols-3">
          <div>
            <Label htmlFor="ready-library">出どころ</Label>
            <select
              id="ready-library"
              value={library}
              onChange={(event) => setLibrary(event.target.value)}
              className={FIELD}
            >
              {libraries.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </div>
          <div>
            <Label htmlFor="ready-model">モデルの名前</Label>
            <input
              id="ready-model"
              value={model}
              onChange={(event) => setModel(event.target.value)}
              placeholder={EXAMPLE.model}
              className={`${FIELD} font-mono`}
            />
          </div>
          <div>
            <Label htmlFor="ready-weights">重みの名前 (版)</Label>
            <input
              id="ready-weights"
              value={weights}
              onChange={(event) => setWeights(event.target.value)}
              placeholder={EXAMPLE.weights}
              className={`${FIELD} font-mono`}
            />
          </div>
        </div>
        <p className="text-xs text-ink-3">
          今は画像の分類だけです. torchvision なら resnet18 と IMAGENET1K_V1 のように,
          モデルの名前と重みの名前 (DEFAULT はその時の版の名前に直ります) を書きます
        </p>

        <section aria-label="下見" aria-live="polite" className="rounded-lg border border-line p-3">
          <h3 className="mb-2 text-xs font-semibold text-ink-2">下見</h3>
          {(!settled || preview.isFetching) && <Spinner label="重みの説明を読んでいます" />}
          {settled && !preview.isFetching && preview.error && <ErrorBox error={preview.error} />}
          {shown && !preview.isFetching && (
            <dl className="grid grid-cols-[7rem_1fr] gap-x-2 gap-y-1 text-sm">
              <dt className="text-ink-3">種類</dt>
              <dd>
                {shown.classes.length.toLocaleString()} 種類
                <span className="block text-xs text-ink-2">{classesText(shown.classes)}</span>
              </dd>
              <dt className="text-ink-3">重みの版</dt>
              <dd className="font-mono text-xs">{shown.weights}</dd>
              <dt className="text-ink-3">前処理</dt>
              <dd className="text-xs">{preprocessText(shown.preprocess)}</dd>
              <dt className="text-ink-3">重みのファイル</dt>
              <dd className="font-mono text-xs break-all">{shown.file ?? '–'}</dd>
              <dt className="text-ink-3">ライセンス</dt>
              <dd>
                {alert ? (
                  <p role="alert" className="text-xs text-warning-text">
                    {alert}
                  </p>
                ) : (
                  <span className="text-xs">{shown.license.license ?? '–'}</span>
                )}
              </dd>
            </dl>
          )}
        </section>

        <div className="grid gap-3 md:grid-cols-2">
          <div>
            <Label htmlFor="ready-name">一式の名前</Label>
            <input
              id="ready-name"
              value={shownName}
              maxLength={80}
              onChange={(event) => setName(event.target.value)}
              className={FIELD}
            />
          </div>
          <div>
            <Label htmlFor="ready-sha256">重みのハッシュ値 (任意. sha256)</Label>
            <input
              id="ready-sha256"
              value={sha256}
              onChange={(event) => setSha256(event.target.value)}
              aria-describedby="ready-sha256-note"
              aria-invalid={!!hashProblem}
              className={`${FIELD} font-mono`}
            />
            <p id="ready-sha256-note" className="mt-1 text-xs text-ink-3">
              問題の設定や前に作った一式で決めておいた値. 書くと, 読んだ重みと違えば作りません
            </p>
            {hashProblem && (
              <p role="alert" className="mt-1 text-xs text-critical-text">
                {hashProblem}
              </p>
            )}
          </div>
        </div>
        {create.isPending && (
          <Spinner label="重みを出どころから読んでハッシュ値を求めています (初めはネットワークから取りに行きます)" />
        )}
        {create.error && <ErrorBox error={create.error} />}
        <div className="flex justify-end">
          <Button type="submit" variant="primary" icon={PackagePlus} disabled={!ready}>
            {create.isPending ? '作っています…' : '作って登録'}
          </Button>
        </div>
      </form>
    </Dialog>
  )
}
