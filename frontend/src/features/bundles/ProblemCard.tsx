import { useMutation, useQueryClient } from '@tanstack/react-query'
import { ClipboardList, Link2, Unlink } from 'lucide-react'
import { useState } from 'react'

import { useApi } from '../../api/client'
import type { ProblemState } from '../../api/types'
import { PathField } from '../../components/PathPicker'
import { Button, Card, ErrorBox } from '../../components/common'
import { useStage } from '../../lib/stage'

/**
 * 一式につないだ問題 (損得, 根拠, 一式との食い違い) を出し, つなぐか外す.
 * つないだ一式は, 推論の意思決定と結論の決め方の既定が問題の値になる
 * (既製のモデルの一式は結論の決め方だけ. 意思決定を使えないため)
 */
export function ProblemCard({
  bundleId,
  state,
  readyModel = false,
}: {
  bundleId: string
  state: ProblemState
  readyModel?: boolean
}) {
  const api = useApi()
  const client = useQueryClient()
  const { project, onProjectChange } = useStage()
  const [path, setPath] = useState('')
  const link = useMutation({
    // 外した後の候補にも殻の選びを使う
    mutationFn: (target: string | null) => api.linkProblem(bundleId, target, project),
    onSuccess: (next) => {
      setPath('')
      // 他の殻の選びで読んだ写しは古いので, 次に使うときに読み直す
      client.invalidateQueries({ queryKey: ['bundle-problem', bundleId], refetchType: 'none' })
      client.setQueryData(['bundle-problem', bundleId, project], next)
      // 殻の選びが替わった後の値も同じ (つないだ一式は殻の選びで変わらない)
      if (next.problem_id !== null)
        client.setQueryData(['bundle-problem', bundleId, next.problem_id], next)
      client.invalidateQueries({ queryKey: ['bundles'] })
      // 置き場の別の問題をつないだら, 殻の選びをそれにする (同じ識別子と,
      // 置き場の外の問題では知らせない)
      if (next.problem_id !== null && next.problem_id !== project) {
        onProjectChange(next.problem_id)
      }
    },
  })
  const linked = state.linked
  const hint = state.hint
  return (
    <Card
      title={
        <span className="flex items-center gap-1.5">
          <ClipboardList className="size-4 text-accent" aria-hidden />
          問題{linked ? `: ${linked.name}` : ''}
        </span>
      }
      name="問題"
    >
      {linked ? (
        <div className="space-y-2 text-sm">
          {linked.handled ? (
            <p className="text-xs text-ink-3">
              推論の{readyModel ? '' : '意思決定と'}結論の決め方は, この問題の今の値が既定です.
              問題を問題の設定の段階で直すと, 次にこの画面を開いたときから使います.
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
            error={`${state.error ?? '問題を読めません'}. 推論の既定は一式の既定値に戻しています`}
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
            {readyModel
              ? '問題 (作り方が既製のモデルをそのままの問題) をつなぐと, 結論の決め方を問題の今の値で既定にし, 決めておいた出どころ, モデル, 重み, 重みのハッシュ値と一式との食い違いを出します (意思決定は使えません).'
              : '問題 (問題の設定の段階の problem.yaml) をつなぐと, 損得と結論の決め方を問題の今の値で既定にし, 損得の根拠と一式との食い違いを出します.'}
          </p>
          {hint && (
            <div className="rounded-lg border border-line p-3 text-xs">
              <p className="text-ink-2">
                {state.hint_project
                  ? '一式の上のディレクトリに問題が無いので, 選んだ問題を候補にしました.'
                  : '一式の上のディレクトリに問題がありました.'}
              </p>
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
            label="問題 (problem.yaml か, それを持つディレクトリ)"
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
