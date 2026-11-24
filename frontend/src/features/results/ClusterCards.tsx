import type { ClusterRow, ClusterStability } from '../../api/types'
import {
  ALL_ROWS,
  clusterName,
  diffText,
  featureHeading,
  scoreText,
  stabilityLines,
  stabilityMethodText,
} from '../../lib/clusters'
import { formatPercent, formatValue } from '../../lib/format'

/**
 * 群の要約の表. 群ごとの件数, 割合 (横棒), 学習のときの割合 (渡した場合), 安定性,
 * 群を作る列ごとの特徴と, 最後に全体の行. onPick を渡すと, 群の行に「行を見る」を出す
 */
export function ClusterSummaryTable({
  rows,
  columns,
  learned,
  onPick,
  caption,
}: {
  rows: ClusterRow[]
  columns: string[]
  learned?: ClusterRow[]
  onPick?: (group: number) => void
  caption: string
}) {
  // 学習のときの群ごとの割合 (群の番号で引く)
  const learnedShare = new Map(
    (learned ?? [])
      .filter((row) => typeof row.group === 'number')
      .map((row) => [row.group, row.share]),
  )
  const headings = columns.map((name) => featureHeading(name, [...rows, ...(learned ?? [])]))
  return (
    <div className="max-h-[32rem] overflow-auto">
      <table className="w-full text-sm">
        <caption className="sr-only">{caption}</caption>
        <thead className="sticky top-0 bg-surface text-xs text-ink-3">
          <tr className="border-b border-line">
            <th scope="col" className="px-4 py-2 text-left font-normal">
              群
            </th>
            <th scope="col" className="px-3 py-2 text-right font-normal">
              件数
            </th>
            <th scope="col" className="px-3 py-2 text-left font-normal">
              割合
            </th>
            {learned && (
              <th scope="col" className="px-3 py-2 text-right font-normal whitespace-nowrap">
                学習のときの割合
              </th>
            )}
            <th scope="col" className="px-3 py-2 text-right font-normal whitespace-nowrap">
              安定性
            </th>
            {headings.map((heading) => (
              <th
                key={heading}
                scope="col"
                className="px-3 py-2 text-left font-normal whitespace-nowrap"
              >
                {heading}
              </th>
            ))}
            {onPick && (
              <th scope="col" className="px-3 py-2 font-normal">
                <span className="sr-only">行を見る</span>
              </th>
            )}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const total = row.group === ALL_ROWS
            const group = row.group
            return (
              <tr
                key={String(group)}
                className={total ? 'border-t-2 border-line bg-surface-2' : 'border-b border-line'}
              >
                <th scope="row" className="px-4 py-1.5 text-left font-medium whitespace-nowrap">
                  {clusterName(group)}
                </th>
                <td className="tabular px-3 py-1.5 text-right">{row.rows.toLocaleString()}</td>
                <td className="px-3 py-1.5">
                  <span className="flex items-center gap-2">
                    {!total && (
                      <span className="h-2.5 w-24 rounded bg-surface-2" aria-hidden>
                        <span
                          className="block h-2.5 rounded bg-accent"
                          style={{ width: `${Math.min(100, row.share * 100)}%` }}
                        />
                      </span>
                    )}
                    <span className="tabular">{formatPercent(row.share)}</span>
                  </span>
                </td>
                {learned && (
                  <td className="tabular px-3 py-1.5 text-right">
                    {total ? '–' : formatPercent(learnedShare.get(group))}
                  </td>
                )}
                <td className="tabular px-3 py-1.5 text-right">{scoreText(row.stability)}</td>
                {columns.map((name) => (
                  <td
                    key={name}
                    className="max-w-64 truncate px-3 py-1.5"
                    title={String(row[name] ?? '')}
                  >
                    {row[name] === null || row[name] === undefined ? '–' : formatValue(row[name])}
                  </td>
                ))}
                {onPick && (
                  <td className="px-3 py-1.5 text-right">
                    {typeof group === 'number' && row.rows > 0 && (
                      <button
                        type="button"
                        className="text-xs whitespace-nowrap text-accent-text underline"
                        aria-label={`${clusterName(group)} の行を見る`}
                        onClick={() => onPick(group)}
                      >
                        行を見る
                      </button>
                    )}
                  </td>
                )}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

/**
 * 学習の群の安定性. 調整ランド指数, 最も低い調整ランド指数, シルエット係数の,
 * モデルの値, 比べる相手の値, 差 (モデル − 比べる相手) と, 作り直し方と比べる相手
 */
export function ClusterStabilityView({ stability }: { stability: ClusterStability }) {
  const baseline = stability.baseline || '比べる相手'
  return (
    <div className="flex flex-col gap-2">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <caption className="sr-only">学習の群の安定性 (モデル, 比べる相手, 差)</caption>
          <thead className="text-xs text-ink-3">
            <tr className="border-b border-line">
              <th scope="col" className="py-2 pr-3 text-left font-normal">
                指標
              </th>
              <th scope="col" className="px-3 py-2 text-right font-normal">
                モデル
              </th>
              <th scope="col" className="px-3 py-2 text-right font-normal">
                比べる相手
              </th>
              <th scope="col" className="px-3 py-2 text-right font-normal whitespace-nowrap">
                差 (モデル − 比べる相手)
              </th>
            </tr>
          </thead>
          <tbody>
            {stabilityLines(stability).map((line) => (
              <tr key={line.key} className="border-b border-line last:border-0">
                <th scope="row" className="py-1.5 pr-3 text-left font-normal">
                  <span className="font-medium">{line.label}</span>
                  <span className="block text-xs text-ink-3">{line.note}</span>
                </th>
                <td className="tabular px-3 py-1.5 text-right font-semibold">
                  {scoreText(line.model)}
                </td>
                <td className="tabular px-3 py-1.5 text-right">{scoreText(line.baseline)}</td>
                <td className="tabular px-3 py-1.5 text-right">{diffText(line.diff)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-ink-3">
        学習のときに {stabilityMethodText(stability)}, 群の数 {stability.k} で数えた値です
        {stability.epoch != null ? ` (エポック ${stability.epoch})` : ''}. 比べる相手は「
        {baseline}」です. 新しいデータでは群を作り直さないので, 群の安定性は学習のときの値です
      </p>
    </div>
  )
}
