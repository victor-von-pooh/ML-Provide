import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { FlaskConical, FolderInput, ListChecks, Package, Plus } from 'lucide-react'
import { useState } from 'react'
import { Link, useNavigate } from 'react-router'

import { api } from '../../api/client'
import type { Bundle } from '../../api/types'
import { PathPicker } from '../../components/PathPicker'
import { Button, Card, EmptyState, ErrorBox, PageHeader, Spinner } from '../../components/ui'
import { formatNumber, KIND_LABELS, TASK_LABELS } from '../../lib/format'
import { FORM_LABELS, isColumnOutput, ROLE_LABELS } from '../../lib/outputs'
import { CreateBundleDialog } from './CreateBundleDialog'
import { ExperimentsDialog } from './ExperimentsDialog'

/** 一式の行の, 種類とタスクとクラス */
function bundleSummary(bundle: Bundle): string {
  const description = bundle.description
  if (!description) return ''
  const kind = KIND_LABELS[description.kind] ?? description.kind
  // 予測 (時系列) は種類と重なるので, 種類の後は「予測」だけ
  const task =
    description.task === 'forecasting'
      ? '予測'
      : (TASK_LABELS[description.task] ?? description.task)
  // 列の出力でない一式は, 出力の形と受け持ち
  if (!isColumnOutput(description)) {
    const form = description.output?.form ?? ''
    const role = description.role ?? 'predict'
    const roleLabel = ROLE_LABELS[role] ?? role
    return `${kind}から${FORM_LABELS[form] ?? form} (${roleLabel})`
  }
  const classes = description.classes.length > 0 ? ` (${description.classes.join(', ')})` : ''
  return `${kind}の${task}${classes}`
}

/** 登録した推論の一式の一覧 */
export function BundlesPage() {
  const navigate = useNavigate()
  const client = useQueryClient()
  const bundles = useQuery({ queryKey: ['bundles'], queryFn: api.bundles })
  const [dialog, setDialog] = useState<'register' | 'experiments' | 'create' | null>(null)
  const register = useMutation({
    mutationFn: ({ path, name }: { path: string; name?: string }) => api.registerBundle(path, name),
    onSuccess: (bundle) => {
      setDialog(null)
      client.invalidateQueries({ queryKey: ['bundles'] })
      client.invalidateQueries({ queryKey: ['experiments'] })
      navigate(`/bundles/${bundle.id}`)
    },
  })
  const samples = useMutation({
    mutationFn: api.addSamples,
    onSuccess: () => client.invalidateQueries({ queryKey: ['bundles'] }),
  })
  // 全て登録済みで何も足さなかった場合の知らせ
  const nothingAdded = samples.isSuccess && samples.data.length === 0
  const opened = (bundle: Bundle) => {
    setDialog(null)
    client.invalidateQueries({ queryKey: ['bundles'] })
    navigate(`/bundles/${bundle.id}`)
  }
  const actions = (
    <>
      <Button icon={FlaskConical} disabled={samples.isPending} onClick={() => samples.mutate()}>
        {samples.isPending ? '見本を学習しています…' : '見本の一式を入れる'}
      </Button>
      <Button
        icon={ListChecks}
        onClick={() => {
          register.reset()
          setDialog('experiments')
        }}
        data-return="experiments"
      >
        実験の一覧から登録
      </Button>
      <Button icon={FolderInput} onClick={() => setDialog('register')} data-return="register">
        場所で登録
      </Button>
      <Button
        variant="primary"
        icon={Plus}
        onClick={() => setDialog('create')}
        data-return="create"
      >
        学習の結果から作る
      </Button>
    </>
  )
  return (
    <>
      <PageHeader
        title="推論の一式"
        subtitle="学習したモデル, 前処理, 重みをまとめたもの. 選んで新しいデータを推論します"
        actions={actions}
      />
      {register.isPending && <Spinner label="一式を確かめています" />}
      {nothingAdded && (
        <p role="status" className="mb-4 text-sm text-ink-2">
          見本の一式は既に入っています.
        </p>
      )}
      {samples.error && (
        <div className="mb-4">
          <ErrorBox error={samples.error} />
        </div>
      )}
      {register.error && (
        <div className="mb-4">
          <ErrorBox error={register.error} />
        </div>
      )}
      {bundles.isPending && <Spinner />}
      {bundles.error && <ErrorBox error={bundles.error} />}
      {bundles.data?.length === 0 && (
        <EmptyState icon={Package} title="一式がありません">
          学習のスクリプトが作った一式 (inference.yaml を持つディレクトリ)
          を実験の一覧か場所で登録するか, モデルの定義, データセット, 実験から作ります.
          試すだけなら「見本の一式を入れる」で, 小さな見本のデータで学習した一式 (表の分類,
          表の回帰, 時系列の先の予測, 文の分類) を入れられます.
        </EmptyState>
      )}
      {bundles.data && bundles.data.length > 0 && (
        <Card padded={false}>
          <table className="w-full text-sm">
            <caption className="sr-only">登録した推論の一式</caption>
            <thead className="text-xs text-ink-3">
              <tr className="border-b border-line">
                <th scope="col" className="px-4 py-2 text-left font-normal">
                  名前
                </th>
                <th scope="col" className="px-4 py-2 text-left font-normal">
                  データとタスク
                </th>
                <th scope="col" className="px-4 py-2 text-left font-normal">
                  学習の結果
                </th>
              </tr>
            </thead>
            <tbody>
              {bundles.data.map((bundle) => {
                const run = bundle.description?.run
                return (
                  <tr key={bundle.id} className="border-b border-line last:border-0">
                    <th scope="row" className="px-4 py-2 text-left font-medium">
                      <Link
                        to={`/bundles/${bundle.id}`}
                        className="text-accent-text hover:underline"
                      >
                        {bundle.name}
                      </Link>
                    </th>
                    <td className="px-4 py-2">
                      {bundle.ok ? (
                        bundleSummary(bundle)
                      ) : (
                        <span className="text-critical-text">読めません: {bundle.error}</span>
                      )}
                    </td>
                    <td className="px-4 py-2 text-ink-2">
                      {run?.monitor && run.best_value != null
                        ? `${run.monitor} ${formatNumber(run.best_value)} (エポック ${run.best_epoch ?? '–'})`
                        : '–'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </Card>
      )}
      {dialog === 'register' && (
        <PathPicker
          title="一式を場所で登録"
          description="inference.yaml を持つディレクトリ (箱の印) を選びます. 一式は写さずに, 場所だけを覚えます."
          pickFiles={false}
          onClose={() => setDialog(null)}
          onPick={(path) => {
            setDialog(null)
            register.mutate({ path })
          }}
        />
      )}
      {dialog === 'experiments' && (
        <ExperimentsDialog
          pending={register.isPending}
          error={register.error}
          onClose={() => setDialog(null)}
          onRegister={(path, name) => register.mutate({ path, name })}
        />
      )}
      {dialog === 'create' && (
        <CreateBundleDialog onClose={() => setDialog(null)} onCreated={opened} />
      )}
    </>
  )
}
