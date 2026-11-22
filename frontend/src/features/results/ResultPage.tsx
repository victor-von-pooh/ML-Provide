import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { Download, RefreshCw, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router'

import { api } from '../../api/client'
import type { Counts, Result } from '../../api/types'
import { Button, Card, ErrorBox, PageHeader, Spinner, StatTile } from '../../components/ui'
import { formatPercent, formatValue } from '../../lib/format'
import { FORM_LABELS, maskNames } from '../../lib/outputs'
import { READY_UNSUPPORTED } from '../../lib/ready'
import { fromForm, toForm, type RulesForm } from '../../lib/rules'
import { OutputNotices } from '../bundles/OutputView'
import { ArtifactCard } from './ArtifactCard'
import { DecisionCard } from './DecisionCard'
import { ExplainCard } from './ExplainCard'
import { RemeasureCard } from './RemeasureCard'
import { RulesEditor } from './RulesEditor'

// 判断が決まらなかったことを表す名前 (ML-Inference と同じ)
const NEUTRAL = new Set(['保留', '予測なし'])

// 表の 1 ページの行の数
const PAGE = 50

// 種類ごとの確率の列 (p:<種類>) を表に出す数の上限 (多ければ CSV だけ. ImageNet は 1000)
const MAX_PROBABILITY_COLUMNS = 20

// 予測と判断の列 (元の列と分けて右に寄せる)
const OWN = new Set([
  'index',
  'status',
  'reason',
  'step',
  'prediction',
  'confidence',
  'decision',
  'action',
  'level',
  'expected_cost',
  'point_expected_cost',
  'gain_if_acted',
  'expected_gain',
])

/** 判断ごとの件数と割合の横棒 */
function DecisionBars({ counts }: { counts: Counts }) {
  // 件数だけの結論 (列の出力でない一式) は判断の内訳を持たない
  if (!counts.decisions || counts.decisions.length === 0) return null
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
  const names = [
    ...new Set(groups.flatMap((group) => (group.decisions ?? []).map((d) => d.decision))),
  ]
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
                  const found = group.decisions?.find((d) => d.decision === name)
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
function RowsTable({ result, onView }: { result: Result; onView?: (row: number) => void }) {
  const column = result.column_output !== false
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
    ...(result.summary?.decisions?.map((item) => item.decision) ?? []),
    ...((result.summary?.skipped ?? 0) > 0 ? ['予測なし'] : []),
  ]
  const probabilities = rows.data?.columns.filter((name) => name.startsWith('p:')) ?? []
  const manyProbabilities = probabilities.length > MAX_PROBABILITY_COLUMNS
  const columns = rows.data
    ? [
        ...rows.data.columns.filter(
          (name) => OWN.has(name) || (name.startsWith('p:') && !manyProbabilities),
        ),
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
            <option value="ok">{column ? '予測した行' : '推論した行'}</option>
            <option value="skipped">{column ? '予測しなかった行' : '推論しなかった行'}</option>
          </select>
          {column && (
            <>
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
          )}
        </>
      }
    >
      {rows.error && <ErrorBox error={rows.error} />}
      {rows.isPending && <Spinner />}
      {rows.data && (
        <>
          {manyProbabilities && (
            <p className="border-b border-line px-4 py-2 text-xs text-ink-3">
              種類が {probabilities.length.toLocaleString()} あるので, 種類ごとの確率の列 (p:種類)
              は表に出さず, CSV に入れます
            </p>
          )}
          <div className="max-h-[32rem] overflow-auto">
            <table className="w-full text-xs">
              <caption className="sr-only">予測と判断の表</caption>
              <thead className="sticky top-0 bg-surface text-ink-3">
                <tr className="border-b border-line">
                  {onView && (
                    <th scope="col" className="px-3 py-2 text-left font-normal">
                      <span className="sr-only">出力を見る</span>
                    </th>
                  )}
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
                    {onView && (
                      <td className="px-3 py-1">
                        {row.artifact ? (
                          <button
                            type="button"
                            className="text-accent-text underline"
                            aria-label={`${String(row.index)} 行目の出力を見る`}
                            onClick={() => {
                              onView(Number(row.index))
                              // 表の上にある出力のカードへ移る
                              document.getElementById('artifact-row')?.focus()
                            }}
                          >
                            見る
                          </button>
                        ) : null}
                      </td>
                    )}
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
  // 配列の出力で見る行 (選ぶまでは推論した最初の行. 別の結果に移ったら戻す)
  const [artifactRow, setArtifactRow] = useState<number | null>(null)
  const [artifactOf, setArtifactOf] = useState(id)
  if (artifactOf !== id) {
    setArtifactOf(id)
    setArtifactRow(null)
  }
  const navigate = useNavigate()
  const client = useQueryClient()
  const result = useQuery({ queryKey: ['results', 'one', id], queryFn: () => api.result(id) })
  // 一式を消しても結果は残るので, 一式があるときだけ一式へ戻る
  const bundleId = result.data?.bundle_id ?? ''
  const bundle = useQuery({
    queryKey: ['bundles', bundleId],
    queryFn: () => api.bundle(bundleId),
    enabled: !!bundleId,
    retry: false,
  })
  const bundleAlive = bundle.isSuccess
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
  // (API が数えた数. 無い前の API では文の数から)
  const overall =
    data.overall_count ?? data.sentences.length - Math.min(summary?.groups?.length ?? 0, 10)
  // 列の出力でない一式の結果は件数だけ (決め方, 意思決定, 列の効きは使えない)
  const column = data.column_output !== false
  // 既製のモデルの一式の結果は, 意思決定, たどり直し, 外れ方の確かめを使えない
  const readyModel = data.kind === 'ready'
  const form = data.artifacts?.form ?? data.output_form ?? ''
  return (
    <>
      <PageHeader
        title={data.bundle?.name ?? '結果'}
        subtitle={
          <span>
            {data.forecast && '何段も先の予測 · '}
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
        <OutputNotices target={data.target_info} warnings={data.pretrained_warnings} />
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
        {/* 決め直した後は, 使った決め方で編集を作り直す */}
        {readyModel && <p className="text-xs text-ink-2">{READY_UNSUPPORTED}</p>}
        {column && !readyModel && (
          <DecisionCard
            key={`decision:${JSON.stringify(data.decision?.decision ?? null)}`}
            result={data}
            bundleAlive={bundleAlive}
          />
        )}
        {column && !readyModel && (
          <RemeasureCard
            key={`remeasure:${data.id}`}
            result={data}
            bundleAlive={bundleAlive}
            forecastable={!!bundle.data?.description?.forecastable}
          />
        )}
        {summary && (
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <StatTile label="行" value={summary.rows.toLocaleString()} />
            <StatTile
              label={column ? '予測した' : '推論した'}
              value={summary.predicted.toLocaleString()}
            />
            <StatTile
              label={column ? '予測しなかった' : '推論しなかった'}
              value={summary.skipped.toLocaleString()}
            />
            {!column ? (
              <StatTile label="出力の形" value={FORM_LABELS[form] ?? form} />
            ) : values ? (
              <StatTile
                label="予測の平均"
                value={formatValue(values.mean)}
                hint={`合計 ${formatValue(values.sum)}`}
              />
            ) : (
              <StatTile
                label="判断の種類"
                value={(summary.decisions?.length ?? 0).toLocaleString()}
              />
            )}
          </div>
        )}
        {/* 出し直した後は, 使った決め方で編集を作り直す */}
        {column && (
          <ConcludeCard key={`conclude:${JSON.stringify(data.rules ?? {})}`} result={data} />
        )}
        {column && <GroupTable result={data} />}
        {column && <ExplainCard result={data} bundleAlive={bundleAlive} />}
        {data.artifacts && (
          <ArtifactCard
            result={data}
            row={artifactRow}
            onRow={setArtifactRow}
            names={maskNames(bundle.data?.description)}
            note={
              bundle.data?.description?.items?.restore === 'shift'
                ? '点は元の座標 (正規化の平行移動を戻した, 撮影の世界の座標) です'
                : null
            }
          />
        )}
        {/* 出し直した後は, 絞り込みとページを初めに戻す */}
        <RowsTable
          key={`rows:${JSON.stringify(data.rules ?? {})}:${JSON.stringify(data.decision?.decision ?? null)}`}
          result={data}
          onView={data.artifacts ? setArtifactRow : undefined}
        />
      </div>
    </>
  )
}
