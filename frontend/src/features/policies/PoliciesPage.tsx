import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Bot, FolderInput, ListChecks } from 'lucide-react'
import { useState } from 'react'
import { Link, useNavigate } from 'react-router'

import { useApi } from '../../api/client'
import { PathPicker } from '../../components/PathPicker'
import { ExperimentsDialog } from '../bundles/ExperimentsDialog'
import { Button, Card, EmptyState, ErrorBox, PageHeader, Spinner } from '../../components/ui'
import { FALLBACK_LABELS, spaceLabel } from '../../lib/policy'

/** 登録した強化学習の方策の一式の一覧 */
export function PoliciesPage() {
  const api = useApi()
  const navigate = useNavigate()
  const client = useQueryClient()
  const policies = useQuery({ queryKey: ['policies'], queryFn: api.policies })
  const [dialog, setDialog] = useState<'experiments' | 'path' | null>(null)
  const register = useMutation({
    mutationFn: ({ path, name }: { path: string; name?: string }) => api.registerPolicy(path, name),
    onSuccess: (policy) => {
      setDialog(null)
      client.invalidateQueries({ queryKey: ['policies'] })
      client.invalidateQueries({ queryKey: ['policy-experiments'] })
      navigate(`/policies/${policy.id}`)
    },
  })
  return (
    <>
      <PageHeader
        title="方策の一式"
        subtitle="強化学習で学習した方策. 観測を入れて, 守りを当てた行動を見ます"
        actions={
          <>
            <Button
              variant="primary"
              icon={ListChecks}
              onClick={() => {
                register.reset()
                setDialog('experiments')
              }}
              data-return="policy-experiments"
            >
              実験の一覧から登録
            </Button>
            <Button
              icon={FolderInput}
              onClick={() => {
                register.reset()
                setDialog('path')
              }}
              data-return="register-policy"
            >
              場所で登録
            </Button>
          </>
        }
      />
      {register.isPending && <Spinner label="方策の一式を確かめています" />}
      {register.error && (
        <div className="mb-4">
          <ErrorBox error={register.error} />
        </div>
      )}
      {policies.isPending && <Spinner />}
      {policies.error && <ErrorBox error={policies.error} />}
      {policies.data?.length === 0 && (
        <EmptyState icon={Bot} title="方策の一式がありません">
          学習の基盤が書き出した方策の一式 (policy.json と policy.pt を持つ policy のディレクトリ)
          を, 実験の一覧か場所で登録します. 方策は観測から行動と守りの結果を返すだけで,
          環境や機械には触れません.
        </EmptyState>
      )}
      {policies.data && policies.data.length > 0 && (
        <Card padded={false}>
          <table className="w-full text-sm">
            <caption className="sr-only">登録した方策の一式</caption>
            <thead className="text-xs text-ink-3">
              <tr className="border-b border-line">
                <th scope="col" className="px-4 py-2 text-left font-normal">
                  名前
                </th>
                <th scope="col" className="px-4 py-2 text-left font-normal">
                  観測と行動
                </th>
                <th scope="col" className="px-4 py-2 text-left font-normal">
                  外れたとき
                </th>
              </tr>
            </thead>
            <tbody>
              {policies.data.map((policy) => (
                <tr key={policy.id} className="border-b border-line last:border-0">
                  <th scope="row" className="px-4 py-2 text-left font-medium">
                    <Link
                      to={`/policies/${policy.id}`}
                      className="text-accent-text hover:underline"
                    >
                      {policy.name}
                    </Link>
                  </th>
                  <td className="px-4 py-2">
                    {policy.spec ? (
                      `${spaceLabel(policy.spec.observation_space)} から ${spaceLabel(policy.spec.action_space)}`
                    ) : (
                      <span className="text-critical-text">読めません: {policy.error}</span>
                    )}
                  </td>
                  <td className="px-4 py-2 text-ink-2">
                    {policy.spec
                      ? (FALLBACK_LABELS[policy.spec.guards.fallback.kind] ??
                        policy.spec.guards.fallback.kind)
                      : '–'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
      {dialog === 'experiments' && (
        <ExperimentsDialog
          kind="policy"
          pending={register.isPending}
          error={register.error}
          onClose={() => setDialog(null)}
          onRegister={(path, name) => register.mutate({ path, name })}
        />
      )}
      {dialog === 'path' && (
        <PathPicker
          title="方策の一式を場所で登録"
          description="policy.json と policy.pt を持つディレクトリを選びます. 一式は写さずに, 場所だけを覚えます."
          pickFiles={false}
          onClose={() => setDialog(null)}
          onPick={(path) => {
            setDialog(null)
            register.mutate({ path })
          }}
        />
      )}
    </>
  )
}
