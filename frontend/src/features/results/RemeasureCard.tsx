import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Ruler, Save } from 'lucide-react'
import { useState } from 'react'

import { api } from '../../api/client'
import type { Remeasure, Result } from '../../api/types'
import { Button, Card, ErrorBox } from '../../components/ui'

/** 書き戻す前の確認 (登録した場所のファイルを書き換えるため) */
const SAVE_CONFIRM =
  '一式の外れ方の記録を, この実績で測り直したものに書き換えます (登録した場所の inference.yaml を書き換えます. 一式の画面で作ったときの記録に戻せます). 書き戻しますか?'

/** 外れ方の確かめ (この実績で, 一式の外れ方の記録が当たっているか) と書き戻し */
export function RemeasureCard({ result, bundleAlive }: { result: Result; bundleAlive: boolean }) {
  const client = useQueryClient()
  // 最後に終わった確かめか書き戻しの結果
  const [shown, setShown] = useState<Remeasure | null>(null)
  const check = useMutation({
    mutationFn: () => api.remeasure(result.id, false),
    onSuccess: setShown,
  })
  const save = useMutation({
    mutationFn: () => api.remeasure(result.id, true),
    onSuccess: (next) => {
      setShown(next)
      // 一式の説明 (外れ方の記録) を読み直す
      client.invalidateQueries({ queryKey: ['bundles'] })
    },
  })
  // 目的の列のある入力のファイルの結果だけ (何段も先の予測は除く)
  if (!result.has_target || result.forecast || !bundleAlive) return null
  const busy = check.isPending || save.isPending
  const error = check.error ?? save.error
  return (
    <Card
      title="外れ方の確かめ (この実績で記録が当たっているか)"
      actions={
        <>
          <Button icon={Ruler} disabled={busy} onClick={() => check.mutate()}>
            {check.isPending ? '確かめています…' : '確かめる'}
          </Button>
          {shown && !shown.written && (
            <Button
              variant="primary"
              icon={Save}
              disabled={busy}
              onClick={() => window.confirm(SAVE_CONFIRM) && save.mutate()}
            >
              {save.isPending ? '書き戻しています…' : '一式に書き戻す'}
            </Button>
          )}
        </>
      }
    >
      {error && <ErrorBox error={error} />}
      {!shown && !error && (
        <p className="text-sm text-ink-3">
          意思決定は, 一式を作るときに測った予測の外れ方 (確率の当たり具合) を前提にします.
          この入力の実際の値で, その記録がまだ当たっているかを確かめ,
          外れていれば測り直して一式に書き戻せます.
        </p>
      )}
      {shown && (
        <ul aria-label="外れ方の確かめの結果" className="flex flex-col gap-1 text-sm">
          {shown.sentences.map((sentence) => (
            <li key={sentence}>{sentence}</li>
          ))}
        </ul>
      )}
      {/* 読み上げのため, 知らせの場所は常に置き, 中身だけを替える */}
      <p role="status" aria-label="外れ方の確かめの状態" className="mt-2 text-xs text-ink-3">
        {shown &&
          (shown.written
            ? '一式に書き戻しました. これからの推論と決め直しで使います (この結果の意思決定は決め直すと変わります. 一式の画面で作ったときの記録に戻せます).'
            : '確かめただけで, 一式は変えていません. 測った実績で方針をたどり直すと良く見えすぎるので, 書き戻した後は別の期間の実績で確かめてください.')}
      </p>
    </Card>
  )
}
