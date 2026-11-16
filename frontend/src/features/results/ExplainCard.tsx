import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Sigma } from 'lucide-react'

import { api } from '../../api/client'
import type { Result } from '../../api/types'
import { Button, Card, ErrorBox } from '../../components/ui'
import { formatValue } from '../../lib/format'

/** 列ごとの効き (どの列が予測を動かしているか) */
export function ExplainCard({ result }: { result: Result }) {
  const client = useQueryClient()
  const explain = useMutation({
    mutationFn: () => api.explain(result.id),
    onSuccess: (next) => client.setQueryData(['results', 'one', result.id], next),
  })
  // 表と時系列の, ファイルを推論した結果だけ
  if (result.kind && !['table', 'timeseries'].includes(result.kind)) return null
  // 何段も先の予測の結果には出さない (並べ替えて推論し直すのは 1 つ先の予測のため)
  if (result.forecast) return null
  const explanation = result.explanation
  const most = Math.max(...(explanation?.columns.map((item) => item.effect ?? 0) ?? [0]), 1e-12)
  const unit = explanation?.measure === 'probability' ? '確率' : '値'
  return (
    <Card
      title="列の効き (どの列が予測を動かしているか)"
      actions={
        <Button icon={Sigma} disabled={explain.isPending} onClick={() => explain.mutate()}>
          {explain.isPending ? '求めています…' : explanation ? '求め直す' : '求める'}
        </Button>
      }
    >
      {explain.error && <ErrorBox error={explain.error} />}
      {!explanation && !explain.error && (
        <p className="text-sm text-ink-3">
          列を 1 つずつ行の間で並べ替えて推論し直し, 予測の{' '}
          {result.task === 'classification' ? '確率' : '値'}
          がどれだけ動くかを求めます (正解は要りません. 列の数 × 3 回の推論をします).
        </p>
      )}
      {explanation && (
        <>
          <ul aria-label="列の効き" className="flex flex-col gap-1.5">
            {explanation.columns.map((item) => (
              <li key={item.column} className="grid grid-cols-[12rem_1fr_9rem] items-center gap-2">
                <span className="truncate text-sm" title={item.error ?? item.column}>
                  {item.column}
                  {item.kind === 'series' && (
                    <span className="ml-1 text-xs text-ink-3">(系列の名前)</span>
                  )}
                </span>
                <span className="h-3 rounded bg-surface-2" aria-hidden>
                  <span
                    className="block h-3 rounded bg-accent"
                    style={{ width: `${((item.effect ?? 0) / most) * 100}%` }}
                  />
                </span>
                <span className="tabular text-right text-sm text-ink-2">
                  {item.kind === 'time'
                    ? '求めない'
                    : item.effect == null
                      ? '予測できなくなる'
                      : formatValue(item.effect, explanation.measure === 'probability')}
                  {item.lost > 0 && (
                    <span className="block text-xs text-warning-text">
                      {Math.round(item.lost).toLocaleString()} 行が予測できなくなる
                    </span>
                  )}
                </span>
              </li>
            ))}
          </ul>
          <p className="mt-2 text-xs text-ink-3">
            並べ替えたときの予測の{unit}の変わり方の平均 ({explanation.rows.toLocaleString()} 行, 3
            回の平均). この一式がこのデータでその列にどれだけ頼っているかで,
            原因の強さではありません. 前処理で使わない列は 0 になります.
            系列を分ける列は系列の名前だけを入れ替えて求めます.
          </p>
        </>
      )}
    </Card>
  )
}
