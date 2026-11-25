import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

import { api } from '../../api/client'
import type { AnomalyEvaluation, AnomalySummary, Judgment, Result } from '../../api/types'
import { Button, Card, ErrorBox, Spinner } from '../../components/ui'
import {
  ANOMALY_UNSUPPORTED,
  EVALUATION_SPLITS,
  JUDGMENT_LABELS,
  loadReviewer,
  reviewedText,
  reviewerProblem,
  saveReviewer,
  triageSummary,
} from '../../lib/anomaly'
import { formatNumber, formatPercent, formatValue } from '../../lib/format'

/** 表の見出しの 1 つ */
function Heading({ children, right = false }: { children: React.ReactNode; right?: boolean }) {
  return (
    <th
      scope="col"
      className={`px-3 py-2 font-normal whitespace-nowrap ${right ? 'text-right' : 'text-left'}`}
    >
      {children}
    </th>
  )
}

/** 判断の印 (異常は注意の色) */
function JudgmentText({ judgment }: { judgment: string | null | undefined }) {
  if (!judgment) return <span className="text-ink-3">未確認</span>
  return (
    <span className={judgment === 'anomaly' ? 'text-critical-text' : 'text-good-text'}>
      {JUDGMENT_LABELS[judgment] ?? judgment}
    </span>
  )
}

/** 実験の上位 K 件の適合率と PR-AUC の表 (分割ごと). 無ければ異常の正解の無い実験と書く */
export function AnomalyEvaluationsTable({
  evaluations,
}: {
  evaluations: Record<string, AnomalyEvaluation>
}) {
  const entries = Object.entries(evaluations ?? {})
  if (entries.length === 0)
    return (
      <p className="text-sm text-ink-2">
        異常の正解の無い実験なので, 度合いの分布と上位の行を人が見て確かめます
      </p>
    )
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <caption className="sr-only">実験の評価 (分割ごとの上位 K 件の適合率と PR-AUC)</caption>
        <thead className="text-xs text-ink-3">
          <tr className="border-b border-line">
            <Heading>分割</Heading>
            <Heading right>行</Heading>
            <Heading right>異常の行</Heading>
            <Heading right>K</Heading>
            <Heading right>上位 K 件の適合率</Heading>
            <Heading right>PR-AUC</Heading>
          </tr>
        </thead>
        <tbody>
          {entries.map(([split, item]) => (
            <tr key={split} className="border-b border-line">
              <th scope="row" className="px-3 py-1.5 text-left font-medium">
                {EVALUATION_SPLITS[split] ?? split}
              </th>
              <td className="tabular px-3 py-1.5 text-right">{formatNumber(item.rows)}</td>
              <td className="tabular px-3 py-1.5 text-right">{formatNumber(item.anomalies)}</td>
              <td className="tabular px-3 py-1.5 text-right">{formatNumber(item.k)}</td>
              <td className="tabular px-3 py-1.5 text-right">
                {item.precision_at_k == null ? '–' : item.precision_at_k.toFixed(3)}
              </td>
              <td className="tabular px-3 py-1.5 text-right">
                {item.average_precision == null ? '–' : item.average_precision.toFixed(3)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-ink-3">
        上位 K 件の適合率は, 度合いの上位 K 件のうち本当に異常だった行の割合です. PR-AUC
        は度合いの順で見た異常の見つけやすさ (1 に近いほど異常の行が上位に集まる) です
      </p>
    </div>
  )
}

/** 度合いの分布 (新しい行と学習用の正常の行) */
function DistributionCard({ summary }: { summary: AnomalySummary }) {
  const found = summary.distribution
  const normal = summary.normal
  const points = [
    ['中央', 'p50'],
    ['90 % 点', 'p90'],
    ['99 % 点', 'p99'],
    ['最大', 'max'],
  ] as const
  return (
    <Card title="度合いの分布">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <caption className="sr-only">
            度合いの分布 (新しい行と学習用の正常の行の中央, 90 % 点, 99 % 点, 最大)
          </caption>
          <thead className="text-xs text-ink-3">
            <tr className="border-b border-line">
              <Heading>行</Heading>
              {points.map(([label]) => (
                <Heading key={label} right>
                  {label}
                </Heading>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-line">
              <th scope="row" className="px-3 py-1.5 text-left font-medium">
                新しい行 ({formatNumber(summary.predicted)} 行)
              </th>
              {points.map(([label, key]) => (
                <td key={label} className="tabular px-3 py-1.5 text-right">
                  {found ? formatValue(found[key]) : '–'}
                </td>
              ))}
            </tr>
            <tr className="border-b border-line">
              <th scope="row" className="px-3 py-1.5 text-left font-medium">
                学習用の正常の行 ({formatNumber(normal.rows)} 行)
              </th>
              {points.map(([label, key]) => (
                <td key={label} className="tabular px-3 py-1.5 text-right">
                  {formatValue(normal[key])}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-sm">
        学習用の正常の行の 99 % 点 ({formatValue(normal.p99)}) を超えた行:{' '}
        <span className="tabular font-semibold">{formatNumber(summary.beyond_normal)} 行</span>
        {summary.predicted > 0 && ` (${formatPercent(summary.beyond_normal / summary.predicted)})`}
      </p>
      <p className="mt-1 text-xs text-ink-3">
        度合いは大きいほど異常で, 確率ではありません.
        学習用の正常の行の分布から外れた度合いは異常の目安で, 異常かは人の確認で決めます
      </p>
    </Card>
  )
}

/** 度合いの上位の行 (一式の K 件) */
function TopRowsCard({ summary }: { summary: AnomalySummary }) {
  return (
    <Card title={`度合いの上位の行 (${summary.top.length} 件)`} padded={false}>
      {summary.top.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <caption className="sr-only">
              度合いの上位の行 (順位, 行の位置, 行の id, 度合い, 最も外れた列)
            </caption>
            <thead className="text-xs text-ink-3">
              <tr className="border-b border-line">
                <Heading right>順位</Heading>
                <Heading right>#</Heading>
                <Heading>{summary.id ?? '行の id'}</Heading>
                <Heading right>度合い</Heading>
                <Heading>最も外れた列 (度合いのうちの割合)</Heading>
              </tr>
            </thead>
            <tbody>
              {summary.top.map((row, index) => (
                <tr key={row.row} className="border-b border-line">
                  <td className="tabular px-3 py-1.5 text-right">{index + 1}</td>
                  <td className="tabular px-3 py-1.5 text-right">{row.row}</td>
                  <td className="px-3 py-1.5 font-mono text-xs">{formatNumber(row.id)}</td>
                  <td className="tabular px-3 py-1.5 text-right">{formatValue(row.score)}</td>
                  <td className="px-3 py-1.5">
                    <span className="font-mono text-xs">{row.column ?? '–'}</span> (
                    {formatPercent(row.column_share)})
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="p-4 text-sm text-ink-2">度合いを出した行がありません</p>
      )}
    </Card>
  )
}

/**
 * 人の確認. 振り分けの文, 確かめた人の名前の欄, 人の確認に回した行の待ち行列と,
 * 1 件ずつ「異常」か「正常」を付けるボタン
 */
function ReviewCard({ result }: { result: Result }) {
  const client = useQueryClient()
  const triage = triageSummary(result)
  const [reviewer, setReviewer] = useState(loadReviewer)
  const problem = reviewerProblem(reviewer)
  const queue = useQuery({ queryKey: ['queue', result.id], queryFn: () => api.queue(result.id) })
  const review = useMutation({
    mutationFn: ({ row, judgment }: { row: string | number; judgment: Judgment }) =>
      api.addReview(result.id, row, judgment, reviewer.trim()),
    onSuccess: () => {
      saveReviewer(reviewer.trim())
      client.invalidateQueries({ queryKey: ['queue', result.id] })
      client.invalidateQueries({ queryKey: ['reviews', result.id] })
    },
  })
  const rows = queue.data?.rows ?? []
  const done = rows.filter((row) => row.judgment).length
  return (
    <Card title="人の確認">
      <div className="flex flex-col gap-3">
        {result.decision ? (
          <div>
            <ul className="flex flex-col gap-1 text-sm">
              {result.decision.sentences.map((sentence) => (
                <li key={sentence}>{sentence}</li>
              ))}
            </ul>
            <ul className="mt-1 flex flex-col gap-0.5 text-xs text-ink-3">
              {result.decision.notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          </div>
        ) : (
          <p className="text-sm text-ink-2">
            振り分けずに推論した結果なので, 人の確認に回した行はありません
          </p>
        )}
        <p className="text-xs text-ink-2">{ANOMALY_UNSUPPORTED}</p>
        {triage && (
          <>
            <div className="max-w-xs">
              <label htmlFor="reviewer" className="mb-1 block text-xs font-medium text-ink-2">
                確かめた人の名前 (記録に残します)
              </label>
              <input
                id="reviewer"
                value={reviewer}
                onChange={(event) => setReviewer(event.target.value)}
                autoComplete="name"
                aria-describedby={problem ? 'reviewer-problem' : undefined}
                className="h-8 w-full rounded-lg border border-line bg-surface px-2 text-sm"
              />
              {problem && (
                <p id="reviewer-problem" className="mt-1 text-xs text-ink-3">
                  {problem}
                </p>
              )}
            </div>
            {queue.error && <ErrorBox error={queue.error} live={false} />}
            {review.error && <ErrorBox error={review.error} />}
            {queue.isPending ? (
              <Spinner label="人の確認に回した行を読んでいます" />
            ) : rows.length > 0 ? (
              <>
                <p role="status" className="text-xs text-ink-3">
                  {rows.length} 件のうち {done} 件を確かめました. 同じ行を何度確かめてもよく,
                  最後の記録が今の判断です. 時刻は押したときの時刻です
                </p>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <caption className="sr-only">
                      人の確認に回した行 (度合いの高い順. 順位, 行の id, 度合い, 最も外れた列,
                      今の判断)
                    </caption>
                    <thead className="text-xs text-ink-3">
                      <tr className="border-b border-line">
                        <Heading right>順位</Heading>
                        <Heading right>#</Heading>
                        <Heading>{queue.data?.id ?? '行の id'}</Heading>
                        <Heading right>度合い</Heading>
                        <Heading>最も外れた列</Heading>
                        <Heading>今の判断</Heading>
                        <th scope="col" className="px-3 py-2 font-normal">
                          <span className="sr-only">判断を付ける</span>
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((row) => {
                        const name = `${queue.data?.id ?? '行'} ${String(row.id ?? row.row)}`
                        return (
                          <tr key={row.row} className="border-b border-line">
                            <td className="tabular px-3 py-1.5 text-right">
                              {formatNumber(row.rank)}
                            </td>
                            <td className="tabular px-3 py-1.5 text-right">{row.row}</td>
                            <th
                              scope="row"
                              className="px-3 py-1.5 text-left font-mono text-xs font-normal"
                            >
                              {formatNumber(row.id ?? row.row)}
                            </th>
                            <td className="tabular px-3 py-1.5 text-right">
                              {formatValue(row.score)}
                            </td>
                            <td className="px-3 py-1.5">
                              <span className="font-mono text-xs">{row.column ?? '–'}</span> (
                              {formatPercent(row.column_share)})
                            </td>
                            <td className="px-3 py-1.5 whitespace-nowrap">
                              <JudgmentText judgment={row.judgment} />
                              {row.reviewer && (
                                <span className="block text-xs text-ink-3">
                                  {row.reviewer} · {reviewedText(row.reviewed_at)}
                                </span>
                              )}
                            </td>
                            <td className="px-3 py-1.5">
                              <span className="flex justify-end gap-2">
                                {(['anomaly', 'normal'] as const).map((judgment) => (
                                  <Button
                                    key={judgment}
                                    aria-label={`${name} を${JUDGMENT_LABELS[judgment]}とする`}
                                    disabled={!!problem || review.isPending}
                                    onClick={() => review.mutate({ row: row.key, judgment })}
                                  >
                                    {JUDGMENT_LABELS[judgment]}
                                  </Button>
                                ))}
                              </span>
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </>
            ) : (
              !queue.error && <p className="text-sm text-ink-2">人の確認に回した行はありません</p>
            )}
          </>
        )}
      </div>
    </Card>
  )
}

/** 確認の記録 (書いた順) */
function ReviewRecordsCard({ result }: { result: Result }) {
  const reviews = useQuery({
    queryKey: ['reviews', result.id],
    queryFn: () => api.reviews(result.id),
  })
  const records = reviews.data?.records ?? []
  const column = (result.summary as AnomalySummary | null)?.id ?? '行の id'
  return (
    <Card title={`確認の記録 (${records.length} 件)`} padded={false}>
      {reviews.error && <ErrorBox error={reviews.error} live={false} />}
      {reviews.isPending ? (
        <Spinner label="確認の記録を読んでいます" />
      ) : records.length > 0 ? (
        <div className="max-h-[24rem] overflow-auto">
          <table className="w-full text-sm">
            <caption className="sr-only">
              確認の記録 (書いた順. 行の id, 判断, 確かめた人, 時刻)
            </caption>
            <thead className="sticky top-0 bg-surface text-xs text-ink-3">
              <tr className="border-b border-line">
                <Heading right>順</Heading>
                <Heading>{column}</Heading>
                <Heading>判断</Heading>
                <Heading>確かめた人</Heading>
                <Heading>時刻</Heading>
              </tr>
            </thead>
            <tbody>
              {records.map((record, index) => (
                <tr key={`${index}:${record.id}`} className="border-b border-line">
                  <td className="tabular px-3 py-1.5 text-right">{index + 1}</td>
                  <td className="px-3 py-1.5 font-mono text-xs">{record.id}</td>
                  <td className="px-3 py-1.5">
                    <JudgmentText judgment={record.judgment} />
                  </td>
                  <td className="px-3 py-1.5">{record.reviewer}</td>
                  <td className="px-3 py-1.5 whitespace-nowrap">
                    {reviewedText(record.reviewed_at)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        !reviews.error && <p className="p-4 text-sm text-ink-2">まだ記録はありません</p>
      )}
      <p className="border-t border-line px-4 py-2 text-xs text-ink-3">
        記録は結果の隣の reviews.csv に 1 件ずつ足します (結果を消すと記録も消えます)
      </p>
    </Card>
  )
}

/** 異常の度合いの結果の, 度合いの分布, 実験の評価, 上位の行, 人の確認と記録 */
export function AnomalyCards({ result, summary }: { result: Result; summary: AnomalySummary }) {
  return (
    <>
      <DistributionCard summary={summary} />
      <Card title="実験の評価">
        <AnomalyEvaluationsTable evaluations={summary.evaluations} />
      </Card>
      <TopRowsCard summary={summary} />
      <ReviewCard result={result} />
      <ReviewRecordsCard result={result} />
    </>
  )
}
