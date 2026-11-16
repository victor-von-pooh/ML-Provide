import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Play, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router'

import { api } from '../../api/client'
import type { Bundle, Description } from '../../api/types'
import { PathField } from '../../components/PathPicker'
import { Button, Card, ErrorBox, PageHeader, Spinner } from '../../components/ui'
import { formatNumber, KIND_LABELS, TASK_LABELS } from '../../lib/format'
import { fromForm, toForm, type RulesForm } from '../../lib/rules'
import { ResultList } from '../results/ResultList'
import { RulesEditor } from '../results/RulesEditor'

/** 説明の 1 行 */
function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[9rem_1fr] gap-2 py-1">
      <dt className="text-ink-3">{label}</dt>
      <dd className="min-w-0 break-words">{children}</dd>
    </div>
  )
}

/** 一式の説明 (何を入れると何が出るか, どの学習の結果か) */
function DescriptionView({ description }: { description: Description }) {
  const run = description.run
  const inputs = Object.entries(description.model.inputs)
  return (
    <dl className="text-sm">
      <Row label="データとタスク">
        {KIND_LABELS[description.kind] ?? description.kind} /{' '}
        {TASK_LABELS[description.task] ?? description.task}
        {description.target && ` (目的の列 ${description.target})`}
      </Row>
      {description.classes.length > 0 && <Row label="クラス">{description.classes.join(', ')}</Row>}
      {description.features.length > 0 && (
        <Row label={`学習に使う列 (${description.features.length})`}>
          <span className="font-mono text-xs">{description.features.join(', ')}</span>
        </Row>
      )}
      {description.text_column && <Row label="文の列">{description.text_column}</Row>}
      <Row label="読み込み">
        <span className="font-mono text-xs">
          {description.reading.loader ?? '自動'}{' '}
          {Object.keys(description.reading.options).length > 0 &&
            JSON.stringify(description.reading.options)}
        </span>
      </Row>
      <Row label="前処理">
        <span className="font-mono text-xs">
          {[...description.steps, ...description.item_steps].join(' → ') || 'なし'}
        </span>
      </Row>
      <Row label="モデルの入力">
        <span className="font-mono text-xs">
          {inputs
            .map(
              ([name, plan]) =>
                `${name} ← ${plan.source === 'features' ? '表の列' : '中身'}${
                  plan.prepend.length > 0 ? ` (先頭に ${plan.prepend.join(', ')})` : ''
                }`,
            )
            .join(', ')}
        </span>
      </Row>
      {run && (
        <Row label="学習の結果">
          {run.status ?? '状態不明'}
          {run.monitor && run.best_value != null && (
            <>
              , {run.monitor} {formatNumber(run.best_value)} (エポック {run.best_epoch ?? '–'})
            </>
          )}
        </Row>
      )}
      {description.notes.map((note) => (
        <p key={note} className="mt-1 text-xs text-warning-text">
          {note}
        </p>
      ))}
    </dl>
  )
}

/** 推論するデータと結論の決め方を選んで推論する */
function PredictCard({ bundle, description }: { bundle: Bundle; description: Description }) {
  const navigate = useNavigate()
  const client = useQueryClient()
  const textual = description.kind === 'text' && !!description.text_column
  const [mode, setMode] = useState<'file' | 'texts'>('file')
  const [source, setSource] = useState('')
  const [texts, setTexts] = useState('')
  const [form, setForm] = useState<RulesForm>(() => toForm(description.conclusion))
  const { rules, problems } = fromForm(form, description.task)
  const predict = useMutation({
    mutationFn: () =>
      api.predict(
        bundle.id,
        mode === 'texts' ? { texts: texts.split('\n'), rules } : { source: source.trim(), rules },
      ),
    onSuccess: (result) => {
      client.invalidateQueries({ queryKey: ['results'] })
      navigate(`/results/${result.id}`)
    },
  })
  const ready = (mode === 'texts' ? texts.trim() : source.trim()) && problems.length === 0
  return (
    <Card title="推論する">
      <form
        className="flex flex-col gap-4"
        onSubmit={(event) => {
          event.preventDefault()
          if (ready) predict.mutate()
        }}
      >
        {textual && (
          <div role="radiogroup" aria-label="推論するデータ" className="flex gap-4 text-sm">
            <label className="flex items-center gap-1">
              <input type="radio" checked={mode === 'file'} onChange={() => setMode('file')} />
              ファイル
            </label>
            <label className="flex items-center gap-1">
              <input type="radio" checked={mode === 'texts'} onChange={() => setMode('texts')} />
              文を書く
            </label>
          </div>
        )}
        {mode === 'file' ? (
          <PathField
            id="predict-source"
            label="推論するデータ (学習のときと同じ列のファイルか, 画像などのディレクトリ)"
            value={source}
            onChange={setSource}
            hint="目的の列は有っても無くてもかまいません. 学習のときと同じ読み込みで読みます"
          />
        ) : (
          <div>
            <label htmlFor="predict-texts" className="mb-1 block text-xs font-medium text-ink-2">
              推論する文 (1 行に 1 つ)
            </label>
            <textarea
              id="predict-texts"
              value={texts}
              onChange={(event) => setTexts(event.target.value)}
              rows={6}
              className="w-full rounded-lg border border-line bg-surface p-2 text-sm"
            />
          </div>
        )}
        <section aria-label="結論の決め方">
          <h3 className="mb-2 text-xs font-semibold text-ink-2">
            結論の決め方 (結果の画面で変えて出し直せます)
          </h3>
          <RulesEditor
            form={form}
            onChange={setForm}
            task={description.task}
            classes={description.classes}
            columns={[]}
            idBase="predict"
          />
          {problems.map((problem) => (
            <p key={problem} className="mt-1 text-xs text-critical-text">
              {problem}
            </p>
          ))}
        </section>
        {predict.error && <ErrorBox error={predict.error} />}
        <div className="flex justify-end">
          <Button
            type="submit"
            variant="primary"
            icon={Play}
            disabled={!ready || predict.isPending}
          >
            {predict.isPending ? '推論しています…' : '推論する'}
          </Button>
        </div>
      </form>
    </Card>
  )
}

/** 一式の説明, 推論, 過去の結果 */
export function BundlePage() {
  const { id = '' } = useParams()
  const navigate = useNavigate()
  const client = useQueryClient()
  const bundle = useQuery({ queryKey: ['bundles', id], queryFn: () => api.bundle(id) })
  const results = useQuery({ queryKey: ['results', id], queryFn: () => api.results(id) })
  const remove = useMutation({
    mutationFn: () => api.deleteBundle(id),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ['bundles'] })
      navigate('/')
    },
  })
  if (bundle.isPending) return <Spinner />
  if (bundle.error) return <ErrorBox error={bundle.error} />
  const data = bundle.data
  return (
    <>
      <PageHeader
        title={data.name}
        subtitle={<span className="font-mono">{data.path}</span>}
        breadcrumb={<Link to="/">推論の一式</Link>}
        actions={
          <Button
            variant="danger"
            icon={Trash2}
            onClick={() => {
              const message = data.managed
                ? 'この一式を消しますか? (このアプリが作った一式なので, ファイルも消します. 結果は残ります)'
                : '登録を外しますか? (一式のファイルは残ります)'
              if (window.confirm(message)) remove.mutate()
            }}
          >
            {data.managed ? '消す' : '登録を外す'}
          </Button>
        }
      />
      {remove.error && <ErrorBox error={remove.error} />}
      {!data.ok || !data.description ? (
        <ErrorBox error={`一式を読めません: ${data.error ?? ''}`} />
      ) : (
        <div className="grid gap-4 xl:grid-cols-[1fr_22rem]">
          <div className="flex min-w-0 flex-col gap-4">
            <PredictCard bundle={data} description={data.description} />
            <Card title="一式の説明">
              <DescriptionView description={data.description} />
            </Card>
          </div>
          <Card title="この一式の結果">
            {results.isPending && <Spinner />}
            {results.error && <ErrorBox error={results.error} />}
            {results.data && <ResultList results={results.data} showBundle={false} />}
          </Card>
        </div>
      )}
    </>
  )
}
