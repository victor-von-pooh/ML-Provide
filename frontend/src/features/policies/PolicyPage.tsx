import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Play, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router'

import { useApi } from '../../api/client'
import type { PolicyGuards, PolicySpec } from '../../api/types'
import { Button, Card, ErrorBox, PageHeader, Spinner } from '../../components/ui'
import {
  actionLabel,
  FALLBACK_LABELS,
  parseObservation,
  spaceLabel,
  spaceSize,
  STATUS_LABELS,
} from '../../lib/policy'
import { BatchCard } from './PolicyResults'

/** 範囲の並び (null は制限なし) の見せ方 */
function bounds(values: (number | null)[] | null | undefined): string {
  if (!values) return '制限なし'
  return `[${values.map((value) => (value === null ? '–' : String(value))).join(', ')}]`
}

/** 守りの中身 */
function GuardsList({ guards }: { guards: PolicyGuards }) {
  const fallback = guards.fallback
  const rows: [string, string][] = [
    ['止める条件の下限', bounds(guards.observation_low)],
    ['止める条件の上限', bounds(guards.observation_high)],
    ['行動の下限', bounds(guards.action_low)],
    ['行動の上限', bounds(guards.action_high)],
    ['選んでよい行動', guards.allowed_actions ? guards.allowed_actions.join(', ') : '制限なし'],
    [
      '外れたとき',
      `${FALLBACK_LABELS[fallback.kind] ?? fallback.kind}${
        fallback.kind === 'action' ? ` (${JSON.stringify(fallback.action)})` : ''
      }`,
    ],
  ]
  return (
    <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 text-sm">
      {rows.map(([label, value]) => (
        <div key={label} className="contents">
          <dt className="text-ink-3">{label}</dt>
          <dd className="font-mono break-all">{value}</dd>
        </div>
      ))}
    </dl>
  )
}

/** 観測を入れて行動と守りの結果を見る */
function ActCard({ id, spec }: { id: string; spec: PolicySpec }) {
  const api = useApi()
  const size = spaceSize(spec.observation_space)
  // 初めの値は観測の大きさの 0 の並び (大きな観測は空)
  const [text, setText] = useState(size <= 64 ? Array(size).fill('0').join(', ') : '')
  const [problem, setProblem] = useState('')
  const act = useMutation({ mutationFn: (observation: unknown) => api.act(id, observation) })
  const result = act.data
  return (
    <Card title="観測から行動を決める">
      <form
        className="flex flex-col gap-2"
        onSubmit={(event) => {
          event.preventDefault()
          const parsed = parseObservation(text)
          if ('error' in parsed) {
            // 前の観測の結果を残さない
            act.reset()
            setProblem(parsed.error)
            return
          }
          setProblem('')
          act.mutate(parsed.value)
        }}
      >
        <label htmlFor="observation" className="text-sm text-ink-2">
          観測 ({spaceLabel(spec.observation_space)}. カンマか空白で区切った数か, JSON の配列)
        </label>
        <textarea
          id="observation"
          value={text}
          onChange={(event) => {
            // 書き換えた観測に前の結果を並べない
            act.reset()
            setText(event.target.value)
          }}
          rows={3}
          aria-invalid={problem ? true : undefined}
          aria-describedby={problem ? 'observation-problem' : undefined}
          className="rounded-lg border border-line bg-surface px-2 py-1 font-mono text-sm"
        />
        {problem && (
          <p id="observation-problem" role="alert" className="text-xs text-critical-text">
            {problem}
          </p>
        )}
        <div>
          <Button type="submit" variant="primary" icon={Play} disabled={act.isPending}>
            {act.isPending ? '決めています…' : '行動を決める'}
          </Button>
        </div>
      </form>
      {act.error && (
        <div className="mt-3">
          <ErrorBox error={act.error} />
        </div>
      )}
      {result && (
        <div role="status" className="mt-4 flex flex-col gap-2 text-sm">
          <p>
            <span className="text-ink-3">行動 </span>
            <span className="font-mono text-base font-semibold">{actionLabel(result.action)}</span>
          </p>
          <p>
            <span className="text-ink-3">守り </span>
            <span className={result.status === 'ok' ? '' : 'font-medium text-warning-text'}>
              {STATUS_LABELS[result.status] ?? result.status}
            </span>
          </p>
          {result.reasons.length > 0 && (
            <ul className="list-disc pl-5 text-ink-2">
              {result.reasons.map((reason, index) => (
                <li key={index}>{reason}</li>
              ))}
            </ul>
          )}
        </div>
      )}
    </Card>
  )
}

/** 方策の一式の画面 */
export function PolicyPage() {
  const api = useApi()
  const { id = '' } = useParams()
  const navigate = useNavigate()
  const client = useQueryClient()
  const policy = useQuery({ queryKey: ['policies', id], queryFn: () => api.policy(id) })
  const remove = useMutation({
    mutationFn: () => api.deletePolicy(id),
    onSuccess: () => {
      client.removeQueries({ queryKey: ['policies', id] })
      client.invalidateQueries({ queryKey: ['policies'], exact: true })
      navigate('/policies')
    },
  })
  if (policy.isPending) return <Spinner />
  if (policy.error) return <ErrorBox error={policy.error} />
  const data = policy.data
  const spec = data.spec
  return (
    <>
      <PageHeader
        title={data.name}
        subtitle={<span className="font-mono">{data.path}</span>}
        breadcrumb={<Link to="/policies">方策の一式</Link>}
        actions={
          <Button
            variant="danger"
            icon={Trash2}
            onClick={() => {
              if (window.confirm('登録を外しますか? (一式のファイルは残ります)')) remove.mutate()
            }}
          >
            登録を外す
          </Button>
        }
      />
      {remove.error && <ErrorBox error={remove.error} />}
      {!data.ok && <ErrorBox error={`読めません: ${data.error ?? ''}`} live={false} />}
      {spec && (
        <div className="flex flex-col gap-4">
          <Card title="方策">
            <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 text-sm">
              <dt className="text-ink-3">観測</dt>
              <dd>{spaceLabel(spec.observation_space)}</dd>
              <dt className="text-ink-3">行動</dt>
              <dd>{spaceLabel(spec.action_space)}</dd>
              <dt className="text-ink-3">出力の読み方</dt>
              <dd>
                {spec.head === 'categorical' ? '最もスコアの高い行動' : '出力をそのまま行動に'}
              </dd>
              {data.unused_keys && data.unused_keys.length > 0 && (
                <>
                  <dt className="text-ink-3">読まなかった重み</dt>
                  <dd className="font-mono break-all">{data.unused_keys.join(', ')}</dd>
                </>
              )}
            </dl>
          </Card>
          <Card title="守り">
            <GuardsList guards={spec.guards} />
          </Card>
          {data.ok && <ActCard id={data.id} spec={spec} />}
          {data.ok && <BatchCard id={data.id} />}
        </div>
      )}
    </>
  )
}
