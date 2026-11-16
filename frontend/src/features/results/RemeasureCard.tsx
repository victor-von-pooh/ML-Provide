import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Ruler, Save } from 'lucide-react'

import { api } from '../../api/client'
import type { Result } from '../../api/types'
import { Button, Card, ErrorBox } from '../../components/ui'

/** 外れ方の確かめ (この実績で, 一式の外れ方の記録が当たっているか) と書き戻し */
export function RemeasureCard({ result, bundleAlive }: { result: Result; bundleAlive: boolean }) {
  const client = useQueryClient()
  const check = useMutation({ mutationFn: () => api.remeasure(result.id, false) })
  const save = useMutation({
    mutationFn: () => api.remeasure(result.id, true),
    // 一式の説明 (外れ方の記録) を読み直す
    onSuccess: () => client.invalidateQueries({ queryKey: ['bundles'] }),
  })
  // 目的の列のある入力のファイルの結果だけ (何段も先の予測は除く)
  if (!result.has_target || result.forecast || !bundleAlive) return null
  const shown = save.data ?? check.data
  const busy = check.isPending || save.isPending
  return (
    <Card
      title="外れ方の確かめ (この実績で記録が当たっているか)"
      actions={
        <>
          <Button icon={Ruler} disabled={busy} onClick={() => check.mutate()}>
            {check.isPending ? '確かめています…' : '確かめる'}
          </Button>
          {check.data && !save.data && (
            <Button variant="primary" icon={Save} disabled={busy} onClick={() => save.mutate()}>
              {save.isPending ? '書き戻しています…' : '一式に書き戻す'}
            </Button>
          )}
        </>
      }
    >
      {(check.error || save.error) && <ErrorBox error={check.error ?? save.error} />}
      {!shown && !check.error && (
        <p className="text-sm text-ink-3">
          意思決定は, 一式を作るときに測った予測の外れ方 (確率の当たり具合) を前提にします.
          この入力の実際の値で, その記録がまだ当たっているかを確かめ,
          外れていれば測り直して一式に書き戻せます.
        </p>
      )}
      {shown && (
        <>
          <ul className="flex flex-col gap-1 text-sm">
            {shown.sentences.map((sentence) => (
              <li key={sentence}>{sentence}</li>
            ))}
          </ul>
          <p role="status" className="mt-2 text-xs text-ink-3">
            {shown.written
              ? '一式に書き戻しました. これからの推論と決め直しで使います (この結果の意思決定は決め直すと変わります).'
              : '確かめただけで, 一式は変えていません. 測った実績で方針をたどり直すと良く見えすぎるので, 書き戻した後は別の期間の実績で確かめてください.'}
          </p>
        </>
      )}
    </Card>
  )
}
