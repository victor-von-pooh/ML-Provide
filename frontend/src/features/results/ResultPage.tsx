import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { Download, RefreshCw, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router'

import { api } from '../../api/client'
import type { Counts, Result } from '../../api/types'
import { Button, Card, ErrorBox, PageHeader, Spinner, StatTile } from '../../components/ui'
import { formatPercent, formatValue } from '../../lib/format'
import {
  FORM_LABELS,
  maskNames,
  TABLE_HEADINGS,
  tableColumnText,
  tableRowColumns,
} from '../../lib/outputs'
import {
  CLUSTER_HEADINGS,
  CLUSTER_STATUS_LABELS,
  CLUSTER_UNSUPPORTED,
  clusterName,
  clusterRepresentatives,
  clusterRowColumns,
  clusterRows,
  DEFAULT_REPRESENTATIVES,
  REPRESENTATIVE_COUNTS,
} from '../../lib/clusters'
import {
  metricText,
  RANK_HEADINGS,
  RANK_STATUS_LABELS,
  RANK_UNSUPPORTED,
  rankColumnsText,
  rankRowColumns,
} from '../../lib/ranking'
import { READY_UNSUPPORTED } from '../../lib/ready'
import { anomalySummary, triageSummary } from '../../lib/anomaly'
import { treatSummary, upliftSummary } from '../../lib/uplift'
import { groupColumns } from '../../lib/remeasure'
import { rowReasonable } from '../../lib/rowReason'
import { fromForm, toForm, type RulesForm } from '../../lib/rules'
import { OutputNotices } from '../bundles/OutputView'
import { AnomalyCards } from './AnomalyCards'
import { ArtifactCard } from './ArtifactCard'
import { UpliftCards, UpliftTerms } from './UpliftCards'
import {
  ClusterRepresentativesTable,
  ClusterStabilityView,
  ClusterSummaryTable,
} from './ClusterCards'
import { DecisionCard } from './DecisionCard'
import { ExplainCard } from './ExplainCard'
import { RemeasureCard } from './RemeasureCard'
import { RowReasonCard } from './RowReasonCard'
import { RulesEditor } from './RulesEditor'

// 判断が決まらなかったことを表す名前 (ML-Inference と同じ)
const NEUTRAL = new Set(['保留', '予測なし'])

// 表の 1 ページの行の数
const PAGE = 50

// 種類ごとの確率の列 (p:<種類>) を表に出す数の上限 (多ければ CSV だけ. ImageNet は 1000)
const MAX_PROBABILITY_COLUMNS = 20

// 予測と判断の列 (元の列と分けて右に寄せる. 外れ方を分ける列にもしない. API の OWN_COLUMNS と同じ)
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
function RowsTable({
  result,
  onView,
  onReason,
  cluster = '',
  onCluster,
  focus = null,
  onFocus,
}: {
  result: Result
  onView?: (row: number) => void
  onReason?: (row: number) => void
  /** 群分けの結果で絞る群の番号 (空は全ての群. 群の要約の「行を見る」と同じ値) */
  cluster?: string
  onCluster?: (group: string) => void
  /** 1 行だけを見る行の位置 (群の代表の行の「行を見る」. null は絞らない) */
  focus?: number | null
  onFocus?: (row: number | null) => void
}) {
  const column = result.column_output !== false
  const [page, setPage] = useState(0)
  const [status, setStatus] = useState('')
  const [decision, setDecision] = useState('')
  // 群分けの結果は行ごとの群と距離 (見出しと状態は日本語. 群で絞れる)
  const grouped = result.output_form === 'cluster'
  const clusterCount = result.summary?.k ?? result.grouping?.k ?? 0
  // 群の要約から絞る群を変えたら, ページを初めに戻す
  // (代表の行の「行を見る」で 1 行に絞るときも同じ)
  const [seenCluster, setSeenCluster] = useState(cluster)
  const [seenFocus, setSeenFocus] = useState(focus)
  if (seenCluster !== cluster || seenFocus !== focus) {
    setSeenCluster(cluster)
    setSeenFocus(focus)
    setPage(0)
  }
  const rows = useQuery({
    queryKey: ['rows', result.id, page, status, decision, cluster, focus, result.rules],
    queryFn: () =>
      api.rows(result.id, {
        offset: page * PAGE,
        limit: PAGE,
        status: status || undefined,
        decision: decision || undefined,
        group: grouped && cluster !== '' ? Number(cluster) : undefined,
        row: focus ?? undefined,
      }),
    placeholderData: keepPreviousData,
  })
  const decisions = [
    ...(result.summary?.decisions?.map((item) => item.decision) ?? []),
    ...((result.summary?.skipped ?? 0) > 0 ? ['予測なし'] : []),
  ]
  const probabilities = rows.data?.columns.filter((name) => name.startsWith('p:')) ?? []
  const manyProbabilities = probabilities.length > MAX_PROBABILITY_COLUMNS
  // 表の出力の結果は作った表 (件の番号と種の後に列の符号化の順の列)
  const table = result.output_form === 'table' && !!result.table_columns
  // 並べるの結果は利用者と順位ごとの品物と点数 (見出しと状態は日本語)
  const ranking = result.output_form === 'ranking'
  const headings = ranking
    ? RANK_HEADINGS
    : grouped
      ? CLUSTER_HEADINGS
      : table
        ? TABLE_HEADINGS
        : null
  const statusLabels = ranking ? RANK_STATUS_LABELS : grouped ? CLUSTER_STATUS_LABELS : null
  const columns = !rows.data
    ? []
    : ranking
      ? rankRowColumns(rows.data.columns)
      : grouped
        ? clusterRowColumns(rows.data.columns)
        : table
          ? tableRowColumns(rows.data.columns, result.table_columns ?? [])
          : [
              ...rows.data.columns.filter(
                (name) =>
                  (OWN.has(name) && (column || name !== 'decision')) ||
                  (name.startsWith('p:') && !manyProbabilities),
              ),
              ...rows.data.columns.filter((name) => !OWN.has(name) && !name.startsWith('p:')),
            ]
  const total = rows.data?.total ?? 0
  const pages = Math.max(1, Math.ceil(total / PAGE))
  return (
    <Card
      title={
        ranking ? '並べた品物' : grouped ? '群に割り当てた行' : table ? '作った表' : '予測の表'
      }
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
              onFocus?.(null)
            }}
            className="h-8 rounded-lg border border-line bg-surface px-2 text-sm"
          >
            <option value="">全ての行</option>
            <option value="ok">
              {ranking
                ? '並べた利用者の行'
                : grouped
                  ? '割り当てた行'
                  : column
                    ? '予測した行'
                    : '推論した行'}
            </option>
            <option value="skipped">
              {ranking
                ? '並べられなかった利用者'
                : grouped
                  ? '割り当てられなかった行'
                  : column
                    ? '予測しなかった行'
                    : '推論しなかった行'}
            </option>
          </select>
          {grouped && onCluster && (
            <>
              <label className="sr-only" htmlFor="rows-cluster">
                群で絞る
              </label>
              <select
                id="rows-cluster"
                value={cluster}
                onChange={(event) => onCluster(event.target.value)}
                className="h-8 rounded-lg border border-line bg-surface px-2 text-sm"
              >
                <option value="">全ての群</option>
                {Array.from({ length: clusterCount }, (_, group) => (
                  <option key={group} value={String(group)}>
                    {clusterName(group)}
                  </option>
                ))}
              </select>
            </>
          )}
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
                  onFocus?.(null)
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
          {table && (
            <p className="border-b border-line px-4 py-2 text-xs text-ink-3">
              1 件が 1 行で, 列は学習のときの列の並び (
              {(result.table_columns ?? [])
                .map((entry) => `${entry.name}: ${tableColumnText(entry)}`)
                .join(', ')}
              ). 推論しなかった行の値は空です. 表の全ての行は CSV で持ち出せます
            </p>
          )}
          {ranking && (
            <p className="border-b border-line px-4 py-2 text-xs text-ink-3">
              利用者ごとに, 点数の高い順の品物が 1 行ずつ並びます (同じ点数なら番号の表で先の品物).
              並べられなかった利用者は 1 行で, 理由を出します. 表の全ての行は CSV で持ち出せます
            </p>
          )}
          {grouped && (
            <p className="border-b border-line px-4 py-2 text-xs text-ink-3">
              1 行ごとに, 最も近い群の中心の番号 (群) と, その中心までの埋め込みの空間の距離
              (小さいほど群の中心に近い) を出します. 割り当てられなかった行は理由を出します.
              表の全ての行は CSV で持ち出せます
            </p>
          )}
          {focus !== null && (
            <p
              role="status"
              className="flex flex-wrap items-center gap-2 border-b border-line bg-surface-2 px-4 py-2 text-xs text-ink-2"
            >
              群の代表の行 #{focus} だけを見ています
              {onFocus && (
                <button
                  type="button"
                  className="text-accent-text underline"
                  onClick={() => onFocus(null)}
                >
                  1 行の絞りを外す
                </button>
              )}
            </p>
          )}
          {manyProbabilities && (
            <p className="border-b border-line px-4 py-2 text-xs text-ink-3">
              種類が {probabilities.length.toLocaleString()} あるので, 種類ごとの確率の列 (p:種類)
              は表に出さず, CSV に入れます
            </p>
          )}
          <div className="max-h-[32rem] overflow-auto">
            <table className="w-full text-xs">
              <caption className="sr-only">
                {ranking
                  ? '並べた品物 (利用者ごとの上位の品物と点数)'
                  : grouped
                    ? '群に割り当てた行 (行ごとの群と距離)'
                    : table
                      ? '作った表 (件ごとの列の値)'
                      : '予測と判断の表'}
              </caption>
              <thead className="sticky top-0 bg-surface text-ink-3">
                <tr className="border-b border-line">
                  {onView && (
                    <th scope="col" className="px-3 py-2 text-left font-normal">
                      <span className="sr-only">出力を見る</span>
                    </th>
                  )}
                  {onReason && (
                    <th scope="col" className="px-3 py-2 text-left font-normal">
                      <span className="sr-only">予測の理由</span>
                    </th>
                  )}
                  {columns.map((name) => (
                    <th
                      key={name}
                      scope="col"
                      className="px-3 py-2 text-left font-normal whitespace-nowrap"
                    >
                      {headings ? (headings[name] ?? name) : name === 'index' ? '#' : name}
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
                    {onReason && (
                      <td className="px-3 py-1">
                        {row.status === 'ok' ? (
                          <button
                            type="button"
                            className="text-accent-text underline"
                            aria-label={`${String(row.index)} 行目の予測の理由`}
                            onClick={() => {
                              onReason(Number(row.index))
                              // 表の上にある理由のカードへ移る
                              document.getElementById('reason-row')?.focus()
                            }}
                          >
                            理由
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
                        {statusLabels && name === 'status'
                          ? (statusLabels[String(row[name])] ?? String(row[name] ?? ''))
                          : grouped && name === 'group' && typeof row[name] === 'number'
                            ? clusterName(row[name])
                            : formatValue(
                                row[name],
                                name === 'confidence' ||
                                  name.startsWith('p:') ||
                                  name === 'distance',
                              )}
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

/** 並べるの結果の件数, 使った表と, 正解の反応の表から数えた Recall@K と NDCG */
function RankingCard({ result }: { result: Result }) {
  const ranking = result.ranking
  if (!ranking) return null
  const metrics = ranking.metrics
  return (
    <Card title="並べ方と評価">
      <dl className="text-sm">
        <div className="grid grid-cols-[11rem_1fr] gap-2 py-1">
          <dt className="text-ink-3">並べた件数</dt>
          <dd>利用者ごとに上位 {ranking.k} 件</dd>
        </div>
        <div className="grid grid-cols-[11rem_1fr] gap-2 py-1">
          <dt className="text-ink-3">前に反応した品物</dt>
          <dd className="min-w-0 break-words">
            {ranking.seen ? (
              <>
                除いた (<span className="font-mono text-xs">{ranking.seen}</span>)
              </>
            ) : (
              '除いていない'
            )}
          </dd>
        </div>
        <div className="grid grid-cols-[11rem_1fr] gap-2 py-1">
          <dt className="text-ink-3">正解の反応の表</dt>
          <dd className="min-w-0 break-words">
            {ranking.truth ? (
              <span className="font-mono text-xs">{ranking.truth}</span>
            ) : (
              'なし (推論の画面で正解の反応の表を選ぶと, Recall@K と NDCG を数えます)'
            )}
          </dd>
        </div>
        <div className="grid grid-cols-[11rem_1fr] gap-2 py-1">
          <dt className="text-ink-3">反応の表の列</dt>
          <dd className="font-mono text-xs">{rankColumnsText(ranking.columns)}</dd>
        </div>
      </dl>
      {metrics && (
        <>
          <div className="mt-3 grid grid-cols-2 gap-3 md:grid-cols-3">
            <StatTile
              label={`Recall@${metrics.k} (上位 ${metrics.k} 件に入った正解の割合)`}
              value={metricText(metrics.recall_at_k)}
            />
            <StatTile
              label={`NDCG (上位 ${metrics.k} 件の正解の位置で見た並びの良さ)`}
              value={metricText(metrics.ndcg)}
            />
            <StatTile label="数えた利用者" value={metrics.users.toLocaleString()} />
          </div>
          <p className="mt-2 text-xs text-ink-3">
            正解のある利用者ごとに, 上位 {metrics.k} 件に入った正解の割合 (Recall@K) と,
            正解の位置で重みを付けた並びの良さ (NDCG. 正解を上から並べたときを 1) を数え,
            利用者で平均した値です
            {ranking.seen ? '. 前に反応した品物は正解から除いています' : ''}
            {metrics.users === 0 ? '. 正解のある利用者がいないので 0 です' : ''}
          </p>
        </>
      )}
    </Card>
  )
}

/**
 * 群分けの結果の群の代表の行. 件数は既定が結論の 3 件で, 他の件数 (と結論に代表の行の
 * 無い結果) は API で求める
 */
function RepresentativesCard({
  result,
  onPick,
}: {
  result: Result
  onPick: (row: number, group: number) => void
}) {
  const [count, setCount] = useState(DEFAULT_REPRESENTATIVES)
  const kept = clusterRepresentatives(result)
  const useKept = count === DEFAULT_REPRESENTATIVES && kept !== undefined
  const asked = useQuery({
    queryKey: ['representatives', result.id, count],
    queryFn: () => api.representatives(result.id, count),
    enabled: !useKept,
  })
  // 結論の 3 件は件数が 3 のときだけ使い, 他の件数は求め終わるまで読み込み中を出す
  // (選んだ件数と違う件数の表を出さないため)
  const groups = useKept ? kept : asked.data?.groups
  const columns = result.grouping?.columns ?? []
  return (
    <Card
      title="群の代表の行"
      padded={false}
      actions={
        <>
          <label className="sr-only" htmlFor="representatives-count">
            代表の行の件数
          </label>
          <select
            id="representatives-count"
            value={count}
            onChange={(event) => setCount(Number(event.target.value))}
            className="h-8 rounded-lg border border-line bg-surface px-2 text-sm"
          >
            {REPRESENTATIVE_COUNTS.map((value) => (
              <option key={value} value={value}>
                群ごとに {value} 件
              </option>
            ))}
          </select>
        </>
      }
    >
      <p className="border-b border-line px-4 py-2 text-xs text-ink-3">
        群ごとに, 群の中心に近い順の割り当てた行と, 群を作る列の値です
        (距離は群の中心までの埋め込みの空間の距離. 小さいほど群らしい行).
        群の平均や最も多い値と並べて, 群を具体的な行で解釈できます. 「行を見る」で,
        割り当てた行の表をその行だけにします
      </p>
      {!useKept && asked.error && <ErrorBox error={asked.error} live={false} />}
      {groups ? (
        groups.length > 0 ? (
          <ClusterRepresentativesTable
            groups={groups}
            columns={columns}
            summary={clusterRows(result)}
            onPick={onPick}
            caption="群の代表の行 (群ごとに中心に近い順の行, 距離, 群を作る列の値)"
          />
        ) : (
          <p className="p-4 text-sm text-ink-2">群に割り当てた行が無いので, 代表の行はありません</p>
        )
      ) : (
        !asked.error && <Spinner label="代表の行を求めています" />
      )}
    </Card>
  )
}

/** 群分けの結果の群の要約 (新しいデータの群ごとの件数と特徴) と学習の群の安定性 */
function ClusterCards({
  result,
  onPick,
  onPickRow,
}: {
  result: Result
  onPick: (group: number) => void
  onPickRow: (row: number, group: number) => void
}) {
  const rows = clusterRows(result)
  const stability = result.summary?.stability
  const columns = result.grouping?.columns ?? []
  return (
    <>
      <Card title="群の要約" padded={false}>
        {rows.length > 0 ? (
          <>
            <p className="border-b border-line px-4 py-2 text-xs text-ink-3">
              割り当てた行の群ごとの件数と割合, 群を作る列ごとの特徴 (数の列は平均,
              種類の列は最も多い値と群の中の割合) です. 学習のときの割合と比べると,
              新しいデータで増えた群と減った群が分かります. 安定性は学習のときに作り直した群との
              Jaccard 係数の平均 (1 に近いほど作り直しても同じ行が集まる群) です. 行の無い群も 0
              件で並べます
            </p>
            <ClusterSummaryTable
              rows={rows}
              columns={columns}
              learned={result.grouping?.learned}
              onPick={onPick}
              caption="群の要約 (群ごとの件数, 割合, 安定性, 列の特徴と全体)"
            />
          </>
        ) : (
          <p className="p-4 text-sm text-ink-2">群に割り当てた行が無いので, 群の要約はありません</p>
        )}
      </Card>
      <RepresentativesCard result={result} onPick={onPickRow} />
      {stability && (
        <Card title="群の安定性 (学習のとき)">
          <ClusterStabilityView stability={stability} />
        </Card>
      )}
    </>
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
  // 理由を求める行 (押すまでは無し)
  const [reasonRow, setReasonRow] = useState<number | null>(null)
  // 群分けの結果の表で絞る群 (群の要約の「行を見る」と表の欄で選ぶ. 空は全ての群)
  const [cluster, setCluster] = useState('')
  // 1 行だけを見る行の位置 (群の代表の行の「行を見る」. null は絞らない)
  const [focusRow, setFocusRow] = useState<number | null>(null)
  const [artifactOf, setArtifactOf] = useState(id)
  if (artifactOf !== id) {
    setArtifactOf(id)
    setArtifactRow(null)
    setReasonRow(null)
    setCluster('')
    setFocusRow(null)
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
  // 並べるの結果は利用者ごとの上位の品物 (数えるのは利用者の人数)
  const ranking = data.output_form === 'ranking'
  // 群分けの結果は行ごとの群 (結論は群の要約と学習の群の安定性)
  const grouped = data.output_form === 'cluster'
  // 異常の度合いの結果は行ごとの度合い (結論は度合いの分布と上位の行, 意思決定は人の確認への振り分け)
  const anomaly = anomalySummary(data)
  const triage = triageSummary(data)
  // 効果の結果は行ごとの効果 (結論は効果の分布と上位の行, 意思決定は効果の上位から配る判断)
  const uplift = upliftSummary(data)
  const treat = treatSummary(data)
  const form = data.artifacts?.form ?? data.output_form ?? ''
  // 外れ方を種類ごとに分けられる列 (回帰と予測の一式の, 予測と判断と目的の列を除いた列)
  const target = bundle.data?.description?.target
  const groups = groupColumns(data.task, data.columns ?? [], target, OWN)
  // 1 行の予測の理由 (表と時系列の, 書いた文でないファイルを 1 つ先まで予測した結果)
  const reasonable = rowReasonable(data, bundleAlive)
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
          {uplift && (
            <UpliftTerms outcome={uplift.outcome} k={uplift.top_k?.k ?? data.uplift?.k ?? 0.3} />
          )}
          {summary && <DecisionBars counts={summary} />}
        </Card>
        {/* 決め直した後は, 使った決め方で編集を作り直す */}
        {readyModel && <p className="text-xs text-ink-2">{READY_UNSUPPORTED}</p>}
        {ranking && <p className="text-xs text-ink-2">{RANK_UNSUPPORTED}</p>}
        {ranking && <RankingCard result={data} />}
        {grouped && <p className="text-xs text-ink-2">{CLUSTER_UNSUPPORTED}</p>}
        {grouped && (
          <ClusterCards
            result={data}
            onPick={(group) => {
              setCluster(String(group))
              setFocusRow(null)
              document.getElementById('result-rows')?.scrollIntoView({ block: 'start' })
            }}
            onPickRow={(row, group) => {
              setCluster(String(group))
              setFocusRow(row)
              document.getElementById('result-rows')?.scrollIntoView({ block: 'start' })
            }}
          />
        )}
        {anomaly && <AnomalyCards result={data} summary={anomaly} />}
        {uplift && <UpliftCards result={data} summary={uplift} />}
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
            groupColumns={groups}
          />
        )}
        {summary && (
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <StatTile label={ranking ? '利用者' : '行'} value={summary.rows.toLocaleString()} />
            <StatTile
              label={
                ranking
                  ? '並べた利用者'
                  : grouped
                    ? '割り当てた'
                    : anomaly
                      ? '度合いを出した'
                      : uplift
                        ? '効果を見積もった'
                        : column
                          ? '予測した'
                          : '推論した'
              }
              value={summary.predicted.toLocaleString()}
            />
            <StatTile
              label={
                ranking
                  ? '並べられなかった利用者'
                  : grouped
                    ? '割り当てられなかった'
                    : anomaly
                      ? '度合いを出せなかった'
                      : uplift
                        ? '効果を出せなかった'
                        : column
                          ? '予測しなかった'
                          : '推論しなかった'
              }
              value={summary.skipped.toLocaleString()}
            />
            {ranking ? (
              <StatTile label="並べた件数" value={`${summary.k ?? data.ranking?.k ?? '–'} 件`} />
            ) : grouped ? (
              <StatTile label="群の数" value={`${summary.k ?? data.grouping?.k ?? '–'} 個`} />
            ) : anomaly ? (
              <StatTile
                label="人の確認に回す"
                value={`${triage?.acted ?? 0} 件`}
                hint={triage ? `上限 ${triage.capacity} 件` : '振り分けなし'}
              />
            ) : uplift ? (
              <StatTile
                label="配る"
                value={`${treat?.acted ?? 0} 件`}
                hint={treat ? `上限 ${treat.budget} 件` : '配らない'}
              />
            ) : !column ? (
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
        {reasonable && <RowReasonCard result={data} row={reasonRow} onRow={setReasonRow} />}
        {/* 出し直した後は, 絞り込みとページを初めに戻す */}
        <div id="result-rows" className="scroll-mt-4">
          <RowsTable
            key={`rows:${JSON.stringify(data.rules ?? {})}:${JSON.stringify(data.decision?.decision ?? null)}`}
            result={data}
            onView={data.artifacts ? setArtifactRow : undefined}
            onReason={reasonable ? setReasonRow : undefined}
            cluster={cluster}
            onCluster={
              grouped
                ? (group) => {
                    setCluster(group)
                    setFocusRow(null)
                  }
                : undefined
            }
            focus={grouped ? focusRow : null}
            onFocus={grouped ? setFocusRow : undefined}
          />
        </div>
      </div>
    </>
  )
}
