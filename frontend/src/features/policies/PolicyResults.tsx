import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { Download, Files, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router'

import { useApi } from '../../api/client'
import type { PolicyAct, PolicyResult } from '../../api/types'
import { PathField } from '../../components/PathPicker'
import { Button, Card, ErrorBox, PageHeader, Spinner } from '../../components/common'
import { Crumbs, PageStatus } from '../../components/ui'
import { STATUS_LABELS } from '../../lib/policy'

// 表の 1 ページの行の数
const PAGE = 50

// 守りの結果の並び (件数を出す順. API と同じ)
const STATUSES: PolicyAct['status'][] = ['ok', 'clipped', 'fallback', 'stop', 'handover']

/** 守りの結果ごとの件数の横棒 */
export function StatusBars({ result }: { result: PolicyResult }) {
  const most = Math.max(...STATUSES.map((status) => result.counts[status] ?? 0), 1)
  return (
    <ul aria-label="守りの結果ごとの件数" className="flex flex-col gap-1.5">
      {STATUSES.map((status) => {
        const count = result.counts[status] ?? 0
        return (
          <li key={status} className="grid grid-cols-[9rem_1fr_7rem] items-center gap-2">
            <span className="text-sm">{STATUS_LABELS[status]}</span>
            <span className="h-3 rounded bg-surface-2" aria-hidden>
              <span
                className={clsx('block h-3 rounded', status === 'ok' ? 'bg-accent' : 'bg-warning')}
                style={{ width: `${(count / most) * 100}%` }}
              />
            </span>
            <span className="tabular-nums text-right text-sm text-ink-2">
              {count.toLocaleString()} 件
            </span>
          </li>
        )
      })}
    </ul>
  )
}

/** 方策の結果の一覧 (新しい順. 方策の名前を出すかを選べる) */
export function PolicyResultList({
  results,
  showPolicy,
}: {
  results: PolicyResult[]
  showPolicy: boolean
}) {
  if (results.length === 0) return <p className="text-sm text-ink-3">まだ結果がありません.</p>
  return (
    <ul className="flex flex-col divide-y divide-line">
      {results.map((result) => (
        <li key={result.id} className="py-2">
          <Link
            to={`/policy-results/${result.id}`}
            className="text-sm font-medium text-accent-text hover:underline"
          >
            {result.created_at ? new Date(result.created_at).toLocaleString() : result.id}
          </Link>
          {showPolicy && result.policy?.name && (
            <span className="ml-2 text-xs text-ink-3">{result.policy.name}</span>
          )}
          <div className="truncate text-xs text-ink-3">{result.source}</div>
          <div className="text-xs text-ink-2">
            {result.rows.toLocaleString()} 件 /{' '}
            {STATUSES.filter((status) => (result.counts[status] ?? 0) > 0)
              .map((status) => `${STATUS_LABELS[status]} ${result.counts[status]}`)
              .join(', ')}
          </div>
        </li>
      ))}
    </ul>
  )
}

/** 観測のファイル (CSV の行か .npy の並び) でまとめて行動を決め, 結果を残す */
export function BatchCard({ id }: { id: string }) {
  const api = useApi()
  const navigate = useNavigate()
  const client = useQueryClient()
  const [source, setSource] = useState('')
  const results = useQuery({
    queryKey: ['policy-results', id],
    queryFn: () => api.policyResults(id),
  })
  const batch = useMutation({
    mutationFn: (path: string) => api.policyBatch(id, path),
    onSuccess: (next) => {
      client.invalidateQueries({ queryKey: ['policy-results'] })
      navigate(`/policy-results/${next.id}`)
    },
  })
  return (
    <Card title="観測のファイルでまとめて決める">
      <form
        className="flex flex-col gap-2"
        onSubmit={(event) => {
          event.preventDefault()
          if (source.trim()) batch.mutate(source.trim())
        }}
      >
        <PathField
          id="batch-source"
          label="観測のファイル (.csv は 1 行に 1 つの観測. .npy は先頭の次元を観測の並びに)"
          value={source}
          onChange={(value) => {
            batch.reset()
            setSource(value)
          }}
          pickFiles
        />
        <p className="text-xs text-ink-3">
          1 件ずつ守りを当てた行動, 守りの結果, 理由の表と, 守りの結果ごとの件数を出し,
          結果を置き場に残します. CSV の 1 行目に数でない欄があれば列の名前として読みます.
        </p>
        <div>
          <Button
            type="submit"
            variant="primary"
            icon={Files}
            disabled={!source.trim() || batch.isPending}
          >
            {batch.isPending ? '決めています…' : 'まとめて決める'}
          </Button>
        </div>
      </form>
      {batch.error && (
        <div className="mt-3">
          <ErrorBox error={batch.error} />
        </div>
      )}
      <h3 className="mt-4 mb-1 text-sm font-medium">この方策の結果</h3>
      {results.isPending && <Spinner />}
      {results.error && <ErrorBox error={results.error} live={false} />}
      {results.data && <PolicyResultList results={results.data} showPolicy={false} />}
    </Card>
  )
}

/** 方策の結果の表 (守りの結果で絞り, ページに分ける) */
function ActionsTable({ id }: { id: string }) {
  const api = useApi()
  const [page, setPage] = useState(0)
  const [status, setStatus] = useState('')
  const rows = useQuery({
    queryKey: ['policy-result-rows', id, page, status],
    queryFn: () =>
      api.policyResultRows(id, { offset: page * PAGE, limit: PAGE, status: status || undefined }),
    placeholderData: keepPreviousData,
  })
  const total = rows.data?.total ?? 0
  const pages = Math.max(1, Math.ceil(total / PAGE))
  return (
    <Card
      title="1 件ずつの行動"
      padded={false}
      actions={
        <>
          <label className="sr-only" htmlFor="policy-rows-status">
            守りの結果で絞る
          </label>
          <select
            id="policy-rows-status"
            value={status}
            onChange={(event) => {
              setStatus(event.target.value)
              setPage(0)
            }}
            className="h-8 rounded-lg border border-line bg-surface px-2 text-sm"
          >
            <option value="">全ての守りの結果</option>
            {STATUSES.map((name) => (
              <option key={name} value={name}>
                {STATUS_LABELS[name]}
              </option>
            ))}
          </select>
        </>
      }
    >
      {rows.error && <ErrorBox error={rows.error} live={false} />}
      {rows.isPending && <Spinner />}
      {rows.data && (
        <>
          <div className="max-h-[32rem] overflow-auto">
            <table className="w-full text-xs">
              <caption className="sr-only">観測ごとの行動と守りの結果の表</caption>
              <thead className="sticky top-0 bg-surface text-ink-3">
                <tr className="border-b border-line">
                  {['#', '観測', '行動', '守り', '理由'].map((name) => (
                    <th key={name} scope="col" className="px-3 py-2 text-left font-normal">
                      {name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.data.rows.map((row) => (
                  <tr key={row.index} className="border-b border-line last:border-0">
                    <td className="tabular-nums px-3 py-1.5">{row.index}</td>
                    <td className="max-w-72 truncate px-3 py-1.5 font-mono" title={row.observation}>
                      {row.observation}
                    </td>
                    <td className="max-w-48 truncate px-3 py-1.5 font-mono" title={row.action}>
                      {row.action === 'null' ? '–' : row.action}
                    </td>
                    <td
                      className={clsx(
                        'px-3 py-1.5 whitespace-nowrap',
                        row.status !== 'ok' && 'font-medium text-warning-text',
                      )}
                    >
                      {STATUS_LABELS[row.status] ?? row.status}
                    </td>
                    <td className="px-3 py-1.5 text-ink-2">{row.reasons}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex items-center justify-between border-t border-line px-4 py-2 text-xs text-ink-3">
            <span>
              {total.toLocaleString()} 件中 {total === 0 ? 0 : page * PAGE + 1}–
              {Math.min(total, (page + 1) * PAGE)}
            </span>
            <span className="flex gap-2">
              <Button disabled={page === 0} onClick={() => setPage(page - 1)}>
                前へ
              </Button>
              <Button disabled={page + 1 >= pages} onClick={() => setPage(page + 1)}>
                次へ
              </Button>
            </span>
          </div>
        </>
      )}
    </Card>
  )
}

/** 方策の結果の画面 (守りの結果ごとの件数と 1 件ずつの表) */
export function PolicyResultPage() {
  const api = useApi()
  const { id = '' } = useParams()
  const navigate = useNavigate()
  const client = useQueryClient()
  const result = useQuery({
    queryKey: ['policy-results', 'one', id],
    queryFn: () => api.policyResult(id),
  })
  const policyId = result.data?.policy_id ?? ''
  // 登録を外しても結果は残るので, 方策が登録されているときだけ方策へ戻る
  const policy = useQuery({
    queryKey: ['policies', policyId],
    queryFn: () => api.policy(policyId),
    enabled: !!policyId,
    retry: false,
  })
  const remove = useMutation({
    mutationFn: () => api.deletePolicyResult(id),
    onSuccess: () => {
      client.removeQueries({ queryKey: ['policy-results', 'one', id] })
      client.invalidateQueries({ queryKey: ['policy-results'] })
      navigate(policy.isSuccess ? `/policies/${policyId}` : '/results')
    },
  })
  if (result.isPending || result.error) {
    return (
      <PageStatus
        id={id}
        thing="方策の結果"
        error={result.error}
        crumbs={[{ to: '/results', label: '結果' }]}
      />
    )
  }
  const data = result.data
  return (
    <>
      <PageHeader
        title={data.policy?.name ?? '方策の結果'}
        subtitle={
          <span>
            {data.created_at && new Date(data.created_at).toLocaleString()} ·{' '}
            <span className="font-mono">{data.source}</span>
          </span>
        }
        breadcrumb={
          <Crumbs
            items={[
              policy.isSuccess
                ? { to: `/policies/${policyId}`, label: data.policy?.name ?? '方策の一式' }
                : { to: '/results', label: '結果' },
            ]}
          />
        }
        actions={
          <>
            <a
              href={api.policyResultDownloadUrl(id)}
              className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-line bg-surface px-3 text-sm font-medium hover:bg-surface-2"
            >
              <Download className="size-4" aria-hidden />
              CSV
            </a>
            <Button
              variant="danger"
              icon={Trash2}
              onClick={() => window.confirm('この結果を消しますか?') && remove.mutate()}
            >
              消す
            </Button>
          </>
        }
      />
      {remove.error && <ErrorBox error={remove.error} />}
      <div className="flex flex-col gap-4">
        <Card title={`守りの結果 (${data.rows.toLocaleString()} 件)`} name="守りの結果">
          <StatusBars result={data} />
        </Card>
        <ActionsTable id={id} />
      </div>
    </>
  )
}
