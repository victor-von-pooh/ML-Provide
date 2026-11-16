import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { Download, RefreshCw, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router'

import { api } from '../../api/client'
import type { Counts, Result } from '../../api/types'
import { Button, Card, ErrorBox, PageHeader, Spinner, StatTile } from '../../components/ui'
import { formatPercent, formatValue } from '../../lib/format'
import { fromForm, toForm, type RulesForm } from '../../lib/rules'
import { RulesEditor } from './RulesEditor'

// 判断が決まらなかったことを表す名前 (ML-Inference と同じ)
const NEUTRAL = new Set(['保留', '予測なし'])

// 表の 1 ページの行の数
const PAGE = 50

// 予測と判断の列 (元の列と分けて右に寄せる)
const OWN = new Set(['index', 'status', 'reason', 'prediction', 'confidence', 'decision'])

/** 判断ごとの件数と割合の横棒 */
function DecisionBars({ counts }: { counts: Counts }) {
  if (counts.decisions.length === 0) return null
  const most = Math.max(...counts.decisions.map((item) => item.count))
  return (
    <ul aria-label="判断の内訳" className="flex flex-col gap-1.5">
      {counts.decisions.map((item) => (
        <li key={item.decision} className="grid grid-cols-[8rem_1fr_7rem] items-center gap-2">
          <span className="truncate text-sm" title={item.decision}>
            {item.decision}
          </span>
          <span className="h-3 rounded bg-surface-2" aria-hidden>
            <span
              className={clsx(
                'block h-3 rounded',
                NEUTRAL.has(item.decision) ? 'bg-ink-3/50' : 'bg-accent',
              )}
              style={{ width: `${(item.count / most) * 100}%` }}
            />
          </span>
          <span className="tabular text-right text-sm text-ink-2">
            {item.count.toLocaleString()} ({formatPercent(item.share)})
          </span>
        </li>
      ))}
    </ul>
  )
}

/** まとめる列の値ごとの表 */
function GroupTable({ result }: { result: Result }) {
  const groups = result.summary?.groups
  if (!groups || groups.length === 0) return null
  const regression = result.task !== 'classification'
  const names = [...new Set(groups.flatMap((group) => group.decisions.map((d) => d.decision)))]
  return (
    <Card title={`${result.rules?.group_by ?? ''} ごと`} padded={false}>
      <div className="max-h-96 overflow-auto">
        <table className="w-full text-sm">
          <caption className="sr-only">まとめる列の値ごとの件数と内訳</caption>
          <thead className="sticky top-0 bg-surface text-xs text-ink-3">
            <tr className="border-b border-line">
              <th scope="col" className="px-4 py-2 text-left font-normal">
                値
              </th>
              <th scope="col" className="px-4 py-2 text-right font-normal">
                予測
              </th>
              {regression && (
                <>
                  <th scope="col" className="px-4 py-2 text-right font-normal">
                    合計
                  </th>
                  <th scope="col" className="px-4 py-2 text-right font-normal">
                    平均
                  </th>
                </>
              )}
              {names.map((name) => (
                <th key={name} scope="col" className="px-4 py-2 text-right font-normal">
                  {name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {groups.map((group) => (
              <tr key={group.group} className="border-b border-line last:border-0">
                <th scope="row" className="px-4 py-1.5 text-left font-medium">
                  {group.group}
                </th>
                <td className="tabular px-4 py-1.5 text-right">
                  {group.predicted.toLocaleString()}
                </td>
                {regression && (
                  <>
                    <td className="tabular px-4 py-1.5 text-right">
                      {formatValue(group.values?.sum)}
                    </td>
                    <td className="tabular px-4 py-1.5 text-right">
                      {formatValue(group.values?.mean)}
                    </td>
                  </>
                )}
                {names.map((name) => {
                  const found = group.decisions.find((d) => d.decision === name)
                  return (
                    <td key={name} className="tabular px-4 py-1.5 text-right">
                      {found ? found.count.toLocaleString() : '–'}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

/** 予測の表 (予測しなかった行と判断で絞り, ページに分ける) */
function RowsTable({ result }: { result: Result }) {
  const [page, setPage] = useState(0)
  const [status, setStatus] = useState('')
  const [decision, setDecision] = useState('')
  const rows = useQuery({
    queryKey: ['rows', result.id, page, status, decision, result.rules],
    queryFn: () =>
      api.rows(result.id, {
        offset: page * PAGE,
        limit: PAGE,
        status: status || undefined,
        decision: decision || undefined,
      }),
    placeholderData: keepPreviousData,
  })
  const decisions = [
    ...(result.summary?.decisions.map((item) => item.decision) ?? []),
    ...((result.summary?.skipped ?? 0) > 0 ? ['予測なし'] : []),
  ]
  const columns = rows.data
    ? [
        ...rows.data.columns.filter((name) => OWN.has(name) || name.startsWith('p:')),
        ...rows.data.columns.filter((name) => !OWN.has(name) && !name.startsWith('p:')),
      ]
    : []
  const total = rows.data?.total ?? 0
  const pages = Math.max(1, Math.ceil(total / PAGE))
  return (
    <Card
      title="予測の表"
      padded={false}
      actions={
        <>
          <label className="sr-only" htmlFor="rows-status">
            状態で絞る
          </label>
          <select
            id="rows-status"
            value={status}
            onChange={(event) => {
              setStatus(event.target.value)
              setPage(0)
            }}
            className="h-8 rounded-lg border border-line bg-surface px-2 text-sm"
          >
            <option value="">全ての行</option>
            <option value="ok">予測した行</option>
            <option value="skipped">予測しなかった行</option>
          </select>
          <label className="sr-only" htmlFor="rows-decision">
            判断で絞る
          </label>
          <select
            id="rows-decision"
            value={decision}
            onChange={(event) => {
              setDecision(event.target.value)
              setPage(0)
            }}
            className="h-8 rounded-lg border border-line bg-surface px-2 text-sm"
          >
            <option value="">全ての判断</option>
            {decisions.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </>
      }
    >
      {rows.error && <ErrorBox error={rows.error} />}
      {rows.isPending && <Spinner />}
      {rows.data && (
        <>
          <div className="max-h-[32rem] overflow-auto">
            <table className="w-full text-xs">
              <caption className="sr-only">予測と判断の表</caption>
              <thead className="sticky top-0 bg-surface text-ink-3">
                <tr className="border-b border-line">
                  {columns.map((name) => (
                    <th
                      key={name}
                      scope="col"
                      className="px-3 py-2 text-left font-normal whitespace-nowrap"
                    >
                      {name === 'index' ? '#' : name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.data.rows.map((row) => (
                  <tr
                    key={String(row.index)}
                    className={clsx(
                      'border-b border-line last:border-0',
                      row.status === 'skipped' && 'text-ink-3',
                    )}
                  >
                    {columns.map((name) => (
                      <td
                        key={name}
                        className="max-w-72 truncate px-3 py-1.5"
                        title={String(row[name] ?? '')}
                      >
                        {formatValue(row[name], name === 'confidence' || name.startsWith('p:'))}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex items-center justify-between border-t border-line px-4 py-2 text-xs text-ink-3">
            <span>
              {total.toLocaleString()} 行中 {total === 0 ? 0 : page * PAGE + 1}–
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

/** 決め方を変えて結論を出し直す */
function ConcludeCard({ result }: { result: Result }) {
  const client = useQueryClient()
  const task = result.task ?? ''
  const [form, setForm] = useState<RulesForm>(() => toForm(result.rules))
  const { rules, problems } = fromForm(form, task, result.classes ?? [])
  const conclude = useMutation({
    mutationFn: () => api.conclude(result.id, rules),
    onSuccess: (next) => {
      // この結果は返った値に置き換え, 表と一覧 (この結果を除く) を読み直す
      client.setQueryData(['results', 'one', result.id], next)
      client.invalidateQueries({ queryKey: ['rows', result.id] })
      client.invalidateQueries({
        queryKey: ['results'],
        predicate: (query) => query.queryKey[1] !== 'one',
      })
    },
  })
  const columns = (result.columns ?? []).filter((name) => !OWN.has(name) && !name.startsWith('p:'))
  return (
    <Card
      title="結論の決め方"
      actions={
        <Button
          variant="primary"
          icon={RefreshCw}
          disabled={problems.length > 0 || conclude.isPending}
          onClick={() => conclude.mutate()}
        >
          {conclude.isPending ? '出し直しています…' : '出し直す'}
        </Button>
      }
    >
      <RulesEditor
        form={form}
        onChange={setForm}
        task={task}
        classes={result.classes ?? []}
        columns={columns}
        idBase="conclude"
      />
      {problems.map((problem) => (
        <p key={problem} className="mt-1 text-xs text-critical-text">
          {problem}
        </p>
      ))}
      {conclude.error && (
        <div className="mt-2">
          <ErrorBox error={conclude.error} />
        </div>
      )}
    </Card>
  )
}

/** 推論の結果 (結論の文, 内訳, まとめ, 予測の表) */
export function ResultPage() {
  const { id = '' } = useParams()
  const navigate = useNavigate()
  const client = useQueryClient()
  const result = useQuery({ queryKey: ['results', 'one', id], queryFn: () => api.result(id) })
  // 一式を消しても結果は残るので, 一式があるときだけ一式へ戻る
  const bundles = useQuery({ queryKey: ['bundles'], queryFn: api.bundles })
  const bundleAlive = !!bundles.data?.some((bundle) => bundle.id === result.data?.bundle_id)
  const remove = useMutation({
    mutationFn: () => api.deleteResult(id),
    onSuccess: () => {
      // 消した結果は取り直さずに外し, 一覧だけを読み直す
      client.removeQueries({ queryKey: ['results', 'one', id] })
      client.invalidateQueries({
        queryKey: ['results'],
        predicate: (query) => query.queryKey[1] !== 'one',
      })
      navigate(bundleAlive ? `/bundles/${result.data?.bundle_id}` : '/results')
    },
  })
  if (result.isPending) return <Spinner />
  if (result.error) return <ErrorBox error={result.error} />
  const data = result.data
  const summary = data.summary
  const values = summary?.values
  // まとめる列ごとの文 (ML-Inference は先頭の 10 個まで) の前までが全体の文
  const overall = data.sentences.length - Math.min(summary?.groups?.length ?? 0, 10)
  return (
    <>
      <PageHeader
        title={data.bundle?.name ?? '結果'}
        subtitle={
          <span>
            {data.created_at && new Date(data.created_at).toLocaleString()} ·{' '}
            <span className="font-mono">{data.source}</span>
          </span>
        }
        breadcrumb={
          bundleAlive ? (
            <Link to={`/bundles/${data.bundle_id}`}>{data.bundle?.name ?? '一式'}</Link>
          ) : (
            <Link to="/results">結果</Link>
          )
        }
        actions={
          <>
            <a
              href={api.downloadUrl(id)}
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
        <Card title="結論">
          <ul className="mb-4 flex flex-col gap-1 text-base">
            {data.sentences.slice(0, overall).map((sentence) => (
              <li key={sentence}>{sentence}</li>
            ))}
          </ul>
          {/* まとめる列ごとの文 (下の表と同じ中身) は畳む */}
          {overall < data.sentences.length && (
            <details className="mb-4 text-sm text-ink-2">
              <summary className="cursor-pointer">{data.rules?.group_by} ごとの文</summary>
              <ul className="mt-1 flex flex-col gap-0.5">
                {data.sentences.slice(overall).map((sentence) => (
                  <li key={sentence}>{sentence}</li>
                ))}
              </ul>
            </details>
          )}
          {summary && <DecisionBars counts={summary} />}
        </Card>
        {summary && (
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <StatTile label="行" value={summary.rows.toLocaleString()} />
            <StatTile label="予測した" value={summary.predicted.toLocaleString()} />
            <StatTile label="予測しなかった" value={summary.skipped.toLocaleString()} />
            {values ? (
              <StatTile
                label="予測の平均"
                value={formatValue(values.mean)}
                hint={`合計 ${formatValue(values.sum)}`}
              />
            ) : (
              <StatTile label="判断の種類" value={summary.decisions.length.toLocaleString()} />
            )}
          </div>
        )}
        {/* 出し直した後は, 使った決め方で編集を作り直す */}
        <ConcludeCard key={`conclude:${JSON.stringify(data.rules ?? {})}`} result={data} />
        <GroupTable result={data} />
        {/* 出し直した後は, 絞り込みとページを初めに戻す */}
        <RowsTable key={`rows:${JSON.stringify(data.rules ?? {})}`} result={data} />
      </div>
    </>
  )
}
