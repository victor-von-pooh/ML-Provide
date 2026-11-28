import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { FlaskConical, FolderInput, ListChecks, Package, PackagePlus, Plus } from 'lucide-react'
import { useState } from 'react'
import { Link, useNavigate } from 'react-router'

import { useApi } from '../../api/client'
import type { Bundle } from '../../api/types'
import { PathPicker } from '../../components/PathPicker'
import { Button, Card, EmptyState, ErrorBox, PageHeader, Spinner } from '../../components/ui'
import { formatNumber, KIND_LABELS, TASK_LABELS } from '../../lib/format'
import { FORM_LABELS, isColumnOutput, ROLE_LABELS } from '../../lib/outputs'
import { classesText } from '../../lib/ready'
import { useStage } from '../../lib/stage'
import { CreateBundleDialog } from './CreateBundleDialog'
import { ExperimentsDialog } from './ExperimentsDialog'
import { ReadyBundleDialog } from './ReadyBundleDialog'

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
    // 既製の埋め込みの群分けは既製のモデルの出どころも添える
    const library = description.ready ? ` (${description.ready.library})` : ''
    return `${kind}${library}から${FORM_LABELS[form] ?? form} (${roleLabel})`
  }
  // 既製のモデルは種類が多い (ImageNet は 1000) ので数と先頭だけ
  if (description.kind === 'ready') {
    const library = description.ready ? ` (${description.ready.library})` : ''
    return `${kind}${library}の画像の${task} (${description.classes.length.toLocaleString()} 種類)`
  }
  const classes = description.classes.length > 0 ? ` (${classesText(description.classes)})` : ''
  return `${kind}の${task}${classes}`
}

/** 登録した推論の一式の一覧 */
export function BundlesPage() {
  const api = useApi()
  const navigate = useNavigate()
  const client = useQueryClient()
  const bundles = useQuery({ queryKey: ['bundles'], queryFn: api.bundles })
  const [dialog, setDialog] = useState<'register' | 'experiments' | 'create' | 'ready' | null>(null)
  // 殻で問題を選んでいれば, その問題の一式に絞る (他の一式は「全ての一式を出す」で出す)
  // 問題は, つないだ問題か出どころ (問題が記録したプロジェクトの実験の中) で合わせる
  const { project } = useStage()
  const [showAll, setShowAll] = useState(false)
  const [allFor, setAllFor] = useState(project)
  if (allFor !== project) {
    // 殻で問題を替えたら外した状態を消す (A → B → A で外したままにしない)
    setAllFor(project)
    setShowAll(false)
  }
  const filtering = project !== null && !showAll
  const shown = bundles.data?.filter(
    (bundle) =>
      !filtering ||
      bundle.problem_id === project ||
      (project !== null && (bundle.problem_ids ?? []).includes(project)),
  )
  const hidden = (bundles.data?.length ?? 0) - (shown?.length ?? 0)
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
    // 作るときに問題をつなげなかった理由は, 一式の画面で知らせる
    navigate(`/bundles/${bundle.id}`, {
      state: bundle.problem_note ? { problemNote: bundle.problem_note } : null,
    })
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
      <Button icon={PackagePlus} onClick={() => setDialog('ready')} data-return="ready">
        既製のモデルから作る
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
      {project !== null && bundles.data && bundles.data.length > 0 && (
        <label className="mb-3 flex items-center gap-2 text-xs text-ink-2">
          <input
            type="checkbox"
            checked={!filtering}
            onChange={(event) => setShowAll(event.target.checked)}
          />
          全ての一式を出す (出さなければ, 選んだ問題 {project} の一式だけ)
        </label>
      )}
      {bundles.isPending && <Spinner />}
      {bundles.error && <ErrorBox error={bundles.error} />}
      {bundles.data?.length === 0 && (
        <EmptyState icon={Package} title="一式がありません">
          学習のスクリプトが作った一式 (inference.yaml を持つディレクトリ)
          を実験の一覧か場所で登録するか, モデルの定義, データセット, 実験から作ります.
          学習しないライブラリの学習済みのモデル (torchvision の ResNet18 など)
          は「既製のモデルから作る」で作ります. 試すだけなら「見本の一式を入れる」で,
          小さな見本のデータで学習した一式 (表の分類, 表の回帰, 時系列の先の予測, 文の分類)
          を入れられます.
        </EmptyState>
      )}
      {shown && shown.length === 0 && hidden > 0 && (
        <p role="status" className="text-sm text-ink-2">
          選んだ問題の一式がありません. 「全ての一式を出す」に印を付けると他の一式が出ます.
          この問題の一式は「学習の結果から作る」か「実験の一覧から登録」で足せます.
        </p>
      )}
      {shown && shown.length > 0 && (
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
              {shown.map((bundle) => {
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
      {shown && shown.length > 0 && hidden > 0 && (
        <p className="mt-2 text-xs text-ink-3">
          選んだ問題の一式でないもの ({hidden.toLocaleString()} 件) は, 全ての一式を出すと出ます.
        </p>
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
      {dialog === 'ready' && (
        <ReadyBundleDialog onClose={() => setDialog(null)} onCreated={opened} />
      )}
      {dialog === 'create' && (
        <CreateBundleDialog onClose={() => setDialog(null)} onCreated={opened} />
      )}
    </>
  )
}
