import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { FolderInput, Package, Plus } from 'lucide-react'
import { useState } from 'react'
import { Link, useNavigate } from 'react-router'

import { api } from '../../api/client'
import type { Bundle } from '../../api/types'
import { PathPicker } from '../../components/PathPicker'
import { Button, Card, EmptyState, ErrorBox, PageHeader, Spinner } from '../../components/ui'
import { formatNumber, KIND_LABELS, TASK_LABELS } from '../../lib/format'
import { CreateBundleDialog } from './CreateBundleDialog'

/** 一式の行の, 種類とタスクとクラス */
function bundleSummary(bundle: Bundle): string {
  const description = bundle.description
  if (!description) return ''
  const kind = KIND_LABELS[description.kind] ?? description.kind
  const task = TASK_LABELS[description.task] ?? description.task
  const classes = description.classes.length > 0 ? ` (${description.classes.join(', ')})` : ''
  return `${kind}の${task}${classes}`
}

/** 登録した推論の一式の一覧 */
export function BundlesPage() {
  const navigate = useNavigate()
  const client = useQueryClient()
  const bundles = useQuery({ queryKey: ['bundles'], queryFn: api.bundles })
  const [dialog, setDialog] = useState<'register' | 'create' | null>(null)
  const register = useMutation({
    mutationFn: (path: string) => api.registerBundle(path),
    onSuccess: (bundle) => {
      client.invalidateQueries({ queryKey: ['bundles'] })
      navigate(`/bundles/${bundle.id}`)
    },
  })
  const opened = (bundle: Bundle) => {
    setDialog(null)
    client.invalidateQueries({ queryKey: ['bundles'] })
    navigate(`/bundles/${bundle.id}`)
  }
  const actions = (
    <>
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
      {register.error && (
        <div className="mb-4">
          <ErrorBox error={register.error} />
        </div>
      )}
      {bundles.isPending && <Spinner />}
      {bundles.error && <ErrorBox error={bundles.error} />}
      {bundles.data?.length === 0 && (
        <EmptyState icon={Package} title="一式がありません">
          学習のスクリプトが作った一式 (inference.yaml を持つディレクトリ) を場所で登録するか,
          モデルの定義, データセット, 実験から作ります.
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
            register.mutate(path)
          }}
        />
      )}
      {dialog === 'create' && (
        <CreateBundleDialog onClose={() => setDialog(null)} onCreated={opened} />
      )}
    </>
  )
}
