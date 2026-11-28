import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Ruler, Save } from 'lucide-react'
import { useState } from 'react'

import { useApi } from '../../api/client'
import type { Remeasure, Result } from '../../api/types'
import { Button, Card, ErrorBox } from '../../components/ui'

/** 書き戻す前の確かめ (登録した場所のファイルを書き換えるため) */
const SAVE_CONFIRM =
  '一式の外れ方の記録を, この実績で測り直したものに書き換えます (登録した場所の inference.yaml を書き換えます. 学習の実験の中の一式は実験を書き換えず, 推論の段階の置き場に写してから書き戻します. 一式の画面で作ったときの記録に戻せます). 書き戻しますか?'

/** 測り直しの入力 (何段先までと時刻の列, 分ける列. 測らなければ空) */
interface Inputs {
  horizon?: number
  time?: string
  groups?: string
}

/** 外れ方の確かめ (この実績で, 一式の外れ方の記録が当たっているか) と書き戻し */
export function RemeasureCard({
  result,
  bundleAlive,
  forecastable,
  groupColumns,
}: {
  result: Result
  bundleAlive: boolean
  forecastable: boolean
  /** 外れ方を種類ごとに分けられる列 (回帰と予測の一式だけ. 分けられなければ空) */
  groupColumns: string[]
}) {
  const api = useApi()
  const client = useQueryClient()
  // 最後に終わった確かめか書き戻しの結果と, そのときの段の数と時刻の列
  // (書き戻しは確かめたときの入力で行う)
  const [shown, setShown] = useState<{ view: Remeasure; inputs: Inputs } | null>(null)
  // 何段も先の外れ方 (何段も先まで予測できる一式だけ. 空なら測らない)
  const columns = result.columns ?? []
  const [horizon, setHorizon] = useState('')
  const [time, setTime] = useState(() => columns.find((c) => /date|日付|time/i.test(c)) ?? '')
  // 外れ方を種類ごとに分ける列 (空なら全ての行で 1 つ)
  const [groups, setGroups] = useState('')
  const steps = horizon.normalize('NFKC').trim()
  const stepCount = steps === '' ? undefined : Number(steps)
  const stepProblem =
    stepCount !== undefined && (!Number.isInteger(stepCount) || stepCount < 2)
      ? '「何段先まで」は 2 以上の整数にしてください'
      : stepCount !== undefined && !time
        ? '何段も先の外れ方には, 時刻の列を選んでください'
        : null
  const current: Inputs = {
    horizon: stepCount,
    time: stepCount !== undefined ? time : undefined,
    groups: groups || undefined,
  }
  const check = useMutation({
    mutationFn: (inputs: Inputs) =>
      api.remeasure(result.id, false, inputs.horizon, inputs.time, inputs.groups),
    onSuccess: (view, inputs) => setShown({ view, inputs }),
  })
  const save = useMutation({
    mutationFn: (inputs: Inputs) =>
      api.remeasure(result.id, true, inputs.horizon, inputs.time, inputs.groups),
    onSuccess: (view, inputs) => {
      setShown({ view, inputs })
      // 一式の説明 (外れ方の記録) を読み直す
      client.invalidateQueries({ queryKey: ['bundles'] })
    },
  })
  // 入力を変えたら, 前の結果と誤りを消す
  const changed = () => {
    setShown(null)
    check.reset()
    save.reset()
  }
  // 目的の列のある入力のファイルの結果だけ (何段も先の予測は除く)
  if (!result.has_target || result.forecast || !bundleAlive) return null
  const busy = check.isPending || save.isPending
  const error = check.error ?? save.error
  const view = shown?.view
  return (
    <Card
      title="外れ方の確かめ (この実績で記録が当たっているか)"
      actions={
        <>
          <Button
            icon={Ruler}
            disabled={busy || !!stepProblem}
            onClick={() => check.mutate(current)}
          >
            {check.isPending ? '確かめています…' : '確かめる'}
          </Button>
          {shown && !shown.view.written && (
            <Button
              variant="primary"
              icon={Save}
              disabled={busy}
              onClick={() => window.confirm(SAVE_CONFIRM) && save.mutate(shown.inputs)}
            >
              {save.isPending ? '書き戻しています…' : '一式に書き戻す'}
            </Button>
          )}
        </>
      }
    >
      {forecastable && (
        <div className="mb-3 flex flex-wrap items-end gap-3 text-sm">
          <div>
            <label htmlFor={`steps-${result.id}`} className="mb-1 block text-xs text-ink-2">
              何段先まで (任意. 何段も先の予測で量を決めるなら, 段ごとの外れ方も測る)
            </label>
            <input
              id={`steps-${result.id}`}
              inputMode="numeric"
              value={horizon}
              disabled={busy}
              aria-invalid={!!stepProblem}
              aria-describedby={stepProblem ? `steps-problem-${result.id}` : undefined}
              onChange={(event) => {
                setHorizon(event.target.value)
                changed()
              }}
              className="h-8 w-24 rounded-lg border border-line bg-surface px-2"
            />
          </div>
          <div>
            <label htmlFor={`time-${result.id}`} className="mb-1 block text-xs text-ink-2">
              予測を始める時刻の列
            </label>
            <select
              id={`time-${result.id}`}
              value={time}
              disabled={busy}
              onChange={(event) => {
                setTime(event.target.value)
                changed()
              }}
              className="h-8 rounded-lg border border-line bg-surface px-2"
            >
              <option value="">選ぶ</option>
              {columns.map((column) => (
                <option key={column} value={column}>
                  {column}
                </option>
              ))}
            </select>
          </div>
        </div>
      )}
      {groupColumns.length > 0 && (
        <div className="mb-3 text-sm">
          <label htmlFor={`groups-${result.id}`} className="mb-1 block text-xs text-ink-2">
            外れ方を分ける列 (任意. 店や品のように行の種類で外れ方が違うなら, 種類ごとに測る. 20
            行より少ない種類は全ての行の外れ方)
          </label>
          <select
            id={`groups-${result.id}`}
            value={groups}
            disabled={busy}
            onChange={(event) => {
              setGroups(event.target.value)
              changed()
            }}
            className="h-8 rounded-lg border border-line bg-surface px-2"
          >
            <option value="">分けない</option>
            {groupColumns.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
        </div>
      )}
      {stepProblem && (
        <p id={`steps-problem-${result.id}`} className="mb-2 text-xs text-critical-text">
          {stepProblem}
        </p>
      )}
      {error && <ErrorBox error={error} />}
      {!shown && !error && (
        <p className="text-sm text-ink-3">
          意思決定は, 一式を作るときに測った予測の外れ方 (確率の当たり具合) を前提にします.
          この入力の実際の値で, その記録がまだ当たっているかを確かめ,
          外れていれば測り直して一式に書き戻せます.
        </p>
      )}
      {view && (
        <ul aria-label="外れ方の確かめの結果" className="flex flex-col gap-1 text-sm">
          {view.sentences.map((sentence) => (
            <li key={sentence}>{sentence}</li>
          ))}
        </ul>
      )}
      {/* 読み上げのため, 知らせの場所は常に置き, 中身だけを替える */}
      <p role="status" aria-label="外れ方の確かめの状態" className="mt-2 text-xs text-ink-3">
        {view &&
          (view.written
            ? `一式に書き戻しました${view.copied_to ? ` (学習の実験は変えず, 推論の段階の置き場に写した一式 ${view.copied_to} に書き戻し, この一式の登録を写した方に移しました)` : ''}. これからの推論と決め直しで使います (この結果の意思決定は決め直すと変わります. 一式の画面で作ったときの記録に戻せます).`
            : '確かめただけで, 一式は変えていません. 測った実績で方針をたどり直すと良く見えすぎるので, 書き戻した後は別の期間の実績で確かめてください.')}
      </p>
    </Card>
  )
}
