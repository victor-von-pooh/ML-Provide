import { useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { Sigma } from 'lucide-react'
import { useState } from 'react'

import { api } from '../../api/client'
import type { Result } from '../../api/types'
import { Button, Card, ErrorBox, Spinner } from '../../components/ui'
import { formatValue } from '../../lib/format'

/** 選んだ 1 行の予測の理由 (列ごとにその行の値が予測をどちらへどれだけ動かしたか) */
export function RowReasonCard({
  result,
  row,
  onRow,
}: {
  result: Result
  row: number | null
  onRow: (row: number) => void
}) {
  // 求めるのは選んだ行だけ (推論の回数は列の数 × 回数なので, 押すまで求めない)
  // 表の「理由」で選んだ行は欄にも写す
  const [draft, setDraft] = useState(row === null ? '' : String(row))
  const [copied, setCopied] = useState(row)
  if (copied !== row) {
    setCopied(row)
    setDraft(row === null ? '' : String(row))
  }
  const reason = useQuery({
    queryKey: ['row-reason', result.id, row],
    queryFn: () => api.rowReason(result.id, row ?? 0),
    enabled: row !== null,
    retry: false,
    staleTime: Infinity,
  })
  const total = result.summary?.rows ?? 0
  const data = reason.data
  const most = Math.max(...(data?.effects.map((item) => Math.abs(item.effect ?? 0)) ?? [0]), 1e-12)
  const probability = data?.measure === 'probability'
  const unit = probability ? '確率' : '値'
  const typed = Number(draft)
  const valid = draft !== '' && Number.isInteger(typed) && typed >= 0 && typed < total
  return (
    <Card title="1 行の予測の理由">
      <form
        className="mb-3 flex items-center gap-2 text-sm"
        onSubmit={(event) => {
          event.preventDefault()
          if (valid) onRow(typed)
        }}
      >
        <label htmlFor="reason-row">行の番号</label>
        <input
          id="reason-row"
          type="number"
          min={0}
          max={Math.max(0, total - 1)}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          className="h-8 w-24 rounded-lg border border-line bg-surface px-2"
        />
        <Button type="submit" icon={Sigma} disabled={!valid || reason.isFetching}>
          {reason.isFetching ? '求めています…' : '理由を求める'}
        </Button>
        <span className="text-xs text-ink-3">予測の表の「理由」でも選べます</span>
      </form>
      {row === null && (
        <p className="text-sm text-ink-3">
          選んだ行の列を 1 つずつ他の行の値に替えて推論し直し, 予測の
          {result.task === 'classification' ? '確率' : '値'}
          がどちらへどれだけ動くかを求めます. 全ての行ではなく, 選んだ行と比べる行だけを推論します.
        </p>
      )}
      {reason.isFetching && <Spinner />}
      {reason.error && <ErrorBox error={reason.error} live={false} />}
      {data && !reason.isFetching && (
        <section aria-label={`${data.row} 行目の予測の理由`} id="row-reason" tabIndex={-1}>
          <p className="text-sm">
            <span className="font-medium">{data.row} 行目</span>
            {data.prediction !== null && (
              <>
                {' '}
                の予測 <span className="tabular">{formatValue(data.prediction)}</span>
              </>
            )}
            : {data.reason ?? '効いた列はありません'}
          </p>
          <ul aria-label="列ごとの効き" className="mt-2 flex flex-col gap-1.5">
            {data.effects.map((item) => {
              const effect = item.effect ?? 0
              const width = `${(Math.abs(effect) / most) * 50}%`
              return (
                <li
                  key={item.column}
                  className="grid grid-cols-[12rem_1fr_7rem] items-center gap-2"
                >
                  <span
                    className="truncate text-sm"
                    title={data.errors[item.column] ?? item.column}
                  >
                    {item.column}
                  </span>
                  {/* 中央から右が予測を上げた, 左が下げた */}
                  <span className="relative h-3 rounded bg-surface-2" aria-hidden>
                    <span
                      className={clsx(
                        'absolute top-0 h-3 rounded',
                        effect >= 0 ? 'left-1/2 bg-accent' : 'right-1/2 bg-serious',
                      )}
                      style={{ width }}
                    />
                  </span>
                  <span className="tabular text-right text-sm text-ink-2">
                    {item.effect === null
                      ? '求められない'
                      : `${effect > 0 ? '+' : ''}${formatValue(effect, probability)}`}
                  </span>
                </li>
              )
            })}
          </ul>
          {Object.entries(data.errors).map(([name, text]) => (
            <p key={name} className="mt-1 text-xs text-ink-3">
              {name}: {text}
            </p>
          ))}
          <p className="mt-2 text-xs text-ink-3">
            その列の値を比べる行の値に替えたときの予測の{unit}の変わり方 (この行と比べる{' '}
            {(data.rows - 1).toLocaleString()} 行, 列ごとに {data.repeats} 回の平均).
            正ならこの行の値が予測を上げた. 列ごとに 1 つずつ替えて求めるので,
            足しても予測にはならず, 原因の強さでもありません.
          </p>
        </section>
      )}
    </Card>
  )
}
