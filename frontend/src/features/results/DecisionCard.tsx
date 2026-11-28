import { useMutation, useQueryClient } from '@tanstack/react-query'
import { History, RefreshCw } from 'lucide-react'
import { useState } from 'react'

import { useApi } from '../../api/client'
import type { BacktestGuard, Result } from '../../api/types'
import { Button, Card, ErrorBox } from '../../components/common'
import { FieldProblems } from '../../components/ui'
import { fromDecisionForm, toDecisionForm, type DecisionForm } from '../../lib/decision'
import { formatValue } from '../../lib/format'
import { guardAmount, guardLine, guardVerdict } from '../../lib/guard'
import { DecisionEditor } from './DecisionEditor'

/** 意思決定 (何をどれだけするか, 前提, たどり直し) と決め直し */
export function DecisionCard({ result, bundleAlive }: { result: Result; bundleAlive: boolean }) {
  const api = useApi()
  const client = useQueryClient()
  const decided = result.decision
  const [form, setForm] = useState<DecisionForm>(() => toDecisionForm(decided?.decision))
  const { decision, problems } = fromDecisionForm(form, result.classes ?? [])
  // 結果を返った値に置き換え, 表を読み直す
  const refresh = (next: Result) => {
    client.setQueryData(['results', 'one', result.id], next)
    client.invalidateQueries({ queryKey: ['rows', result.id] })
  }
  const decide = useMutation({
    mutationFn: () => api.decide(result.id, decision),
    onSuccess: refresh,
  })
  const replay = useMutation({ mutationFn: () => api.backtest(result.id), onSuccess: refresh })
  const canReplay = !!decided && !!result.has_target && !result.forecast && bundleAlive
  return (
    <Card
      title="意思決定 (だから何をするか)"
      actions={
        <>
          {canReplay && (
            <Button
              icon={History}
              disabled={replay.isPending || decide.isPending}
              onClick={() => replay.mutate()}
            >
              {replay.isPending ? 'たどり直しています…' : '過去のデータでたどり直す'}
            </Button>
          )}
          <Button
            variant="primary"
            icon={RefreshCw}
            disabled={problems.length > 0 || decide.isPending || replay.isPending || !bundleAlive}
            onClick={() => decide.mutate()}
          >
            {decide.isPending ? '決めています…' : '決め直す'}
          </Button>
        </>
      }
    >
      {decided ? (
        <>
          <ul className="mb-2 flex flex-col gap-1 text-base">
            {decided.sentences.map((sentence) => (
              <li key={sentence}>{sentence}</li>
            ))}
          </ul>
          <details className="mb-4 text-sm" open>
            <summary className="cursor-pointer text-xs font-semibold text-ink-2">前提</summary>
            <ul className="mt-1 flex list-disc flex-col gap-0.5 pl-5 text-ink-2">
              {decided.notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          </details>
        </>
      ) : (
        <p className="mb-4 text-sm text-ink-3">
          予測から「だから何をするか」を決めます. 損得と制約 (問題の値) を書くと, 行ごとの行動と,
          その前提での見込みの得と損を出します.
        </p>
      )}
      {result.backtest && (
        <section aria-label="たどり直し" className="mb-4 rounded-lg border border-line p-3">
          <h3 className="mb-2 text-xs font-semibold text-ink-2">
            過去のデータでたどり直した結果 ({result.backtest.rows.toLocaleString()} 行)
          </h3>
          <table className="mb-2 text-sm">
            <caption className="sr-only">
              方針ごとの得と損 (選ぶ型の得は効果の仮定のもとでの値)
            </caption>
            <tbody>
              {Object.entries(result.backtest.policies).map(([name, value]) => (
                <tr key={name}>
                  <th scope="row" className="pr-4 text-left font-normal">
                    {name}
                  </th>
                  <td className="tabular-nums text-right">
                    {value < 0 ? `損 ${formatValue(-value)}` : `得 ${formatValue(value)}`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {result.backtest.guard && <GuardBox guard={result.backtest.guard} />}
          <ul className="flex flex-col gap-0.5 text-xs text-ink-2">
            {result.backtest.sentences.slice(1).map((sentence) => (
              <li key={sentence}>{sentence}</li>
            ))}
          </ul>
        </section>
      )}
      {replay.error && <ErrorBox error={replay.error} />}
      <DecisionEditor
        form={form}
        onChange={setForm}
        task={result.task ?? ''}
        classes={result.classes ?? []}
        idBase="decide"
      />
      <FieldProblems idBase="decide" problems={problems} />
      {decide.error && (
        <div className="mt-2">
          <ErrorBox error={decide.error} />
        </div>
      )}
    </Card>
  )
}

/** たどり直しで数えた, 問題の設定の守る指標 (この方針の値, 守る線, 守れたか) */
function GuardBox({ guard }: { guard: BacktestGuard }) {
  const tone =
    guard.kept === true ? 'text-good-text' : guard.kept === false ? 'text-critical-text' : ''
  return (
    <div aria-label="守る指標" role="group" className="mb-2 text-sm">
      <p className="mb-1 text-xs font-semibold text-ink-2">
        守る指標「{guard.name}」{guard.description && ` (${guard.description})`}
      </p>
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-0.5">
        <dt>この方針での値</dt>
        <dd className="tabular-nums">{guardAmount(guard, guard.value)}</dd>
        <dt>守る線</dt>
        <dd className="tabular-nums">{guardLine(guard)}</dd>
        <dt>守れたか</dt>
        <dd className={tone}>{guardVerdict(guard)}</dd>
      </dl>
      {guard.note && <p className="mt-1 text-xs text-ink-2">{guard.note}</p>}
    </div>
  )
}
