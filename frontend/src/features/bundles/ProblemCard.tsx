import { useMutation, useQueryClient } from '@tanstack/react-query'
import { ClipboardList, Link2, Unlink } from 'lucide-react'
import { useState } from 'react'

import { api } from '../../api/client'
import type { ProblemState } from '../../api/types'
import { PathField } from '../../components/PathPicker'
import { Button, Card, ErrorBox } from '../../components/ui'

/**
 * 一式につないだ問題の設定 (損得, 根拠, 一式との食い違い) を出し, つなぐか外す.
 * つないだ一式は, 推論の意思決定と結論の決め方の既定が問題の設定の値になる
 */
export function ProblemCard({ bundleId, state }: { bundleId: string; state: ProblemState }) {
  const client = useQueryClient()
  const [path, setPath] = useState('')
  const link = useMutation({
    mutationFn: (target: string | null) => api.linkProblem(bundleId, target),
    onSuccess: (next) => {
      setPath('')
      client.setQueryData(['bundle-problem', bundleId], next)
      client.invalidateQueries({ queryKey: ['bundles'] })
    },
  })
  const linked = state.linked
  const hint = state.hint
  return (
    <Card
      title={
        <span className="flex items-center gap-1.5">
          <ClipboardList className="size-4 text-accent" aria-hidden />
          問題の設定{linked ? `: ${linked.name}` : ''}
        </span>
      }
    >
      {linked ? (
        <div className="space-y-2 text-sm">
          {linked.handled ? (
            <p className="text-xs text-ink-3">
              推論の意思決定と結論の決め方は, この問題の設定の今の値が既定です. 問題の設定を
              ML-Pathfinder で直すと, 次にこの画面を開いたときから使います.
            </p>
          ) : (
            <p className="text-xs text-ink-2">受け持ち: {linked.role}</p>
          )}
          <p className={linked.handled ? undefined : 'text-warning-text'}>{linked.text}</p>
          {linked.basis ? (
            <p className="text-xs text-ink-2">損得の根拠: {linked.basis}</p>
          ) : (
            linked.decision && (
              <p className="text-xs text-warning-text">
                損得の根拠が書かれていません. 根拠の無い値では,
                意思決定はもっともらしいだけになります
              </p>
            )
          )}
          {linked.issues.length > 0 && (
            <ul aria-label="一式との食い違い" className="list-disc pl-5 text-xs text-warning-text">
              {linked.issues.map((text) => (
                <li key={text}>{text}</li>
              ))}
            </ul>
          )}
          <p className="truncate font-mono text-xs text-ink-3" title={linked.path}>
            {linked.path}
          </p>
        </div>
      ) : state.path ? (
        <div className="space-y-2">
          <ErrorBox
            live={false}
            error={`${state.error ?? '問題の設定を読めません'}. 推論の既定は一式の既定値に戻しています`}
          />
          <p className="truncate font-mono text-xs text-ink-3" title={state.path}>
            {state.path}
          </p>
        </div>
      ) : (
        <form
          className="space-y-3"
          onSubmit={(event) => {
            event.preventDefault()
            if (path.trim()) link.mutate(path.trim())
          }}
        >
          <p className="text-xs text-ink-3">
            問題の設定 (ML-Pathfinder の problem.yaml) をつなぐと,
            損得と結論の決め方を問題の設定の今の値で既定にし,
            損得の根拠と一式との食い違いを出します.
          </p>
          {hint && (
            <div className="rounded-lg border border-line p-3 text-xs">
              <p className="text-ink-2">一式の上のディレクトリに問題の設定がありました.</p>
              {state.hint_note && <p className="mt-1 text-ink-2">{state.hint_note}</p>}
              <p id="problem-hint" className="mt-1 truncate font-mono text-ink-3" title={hint}>
                {hint}
              </p>
              <Button
                className="mt-2"
                icon={Link2}
                aria-describedby="problem-hint"
                disabled={link.isPending}
                onClick={() => link.mutate(hint)}
              >
                これをつなぐ
              </Button>
            </div>
          )}
          <PathField
            id="problem-path"
            label="問題の設定 (problem.yaml か, それを持つディレクトリ)"
            value={path}
            onChange={setPath}
            pickDirs
            pickFiles
          />
          <div className="flex justify-end">
            <Button type="submit" icon={Link2} disabled={!path.trim() || link.isPending}>
              {link.isPending ? 'つないでいます…' : 'つなぐ'}
            </Button>
          </div>
        </form>
      )}
      {state.path && (
        <div className="mt-3 flex justify-end">
          <Button icon={Unlink} disabled={link.isPending} onClick={() => link.mutate(null)}>
            つなぎを外す
          </Button>
        </div>
      )}
      {link.error && (
        <div className="mt-3">
          <ErrorBox error={link.error} />
        </div>
      )}
      <p role="status" className="mt-3 text-xs text-good-text empty:hidden">
        {link.isSuccess && !link.isPending
          ? `${link.variables === null ? 'つなぎを外しました' : 'つなぎました'}. 推論の意思決定と結論の決め方の既定を読み直しました`
          : ''}
      </p>
    </Card>
  )
}
