import { useQuery } from '@tanstack/react-query'
import { Columns3 } from 'lucide-react'
import { useState } from 'react'
import { useNavigate } from 'react-router'

import { api } from '../../api/client'
import { Button, Card, ErrorBox, PageHeader, Spinner } from '../../components/ui'
import { PolicyResultList } from '../policies/PolicyResults'
import { ResultList } from './ResultList'

// 一度に比べられる結果の数の上限 (API の MAX_COMPARE と同じ)
const MAX_COMPARE = 6

/** 全ての一式の結果と, 方策の結果 (新しい順). 一式の結果は選んで比べる */
export function ResultsPage() {
  const navigate = useNavigate()
  const results = useQuery({ queryKey: ['results', 'all'], queryFn: () => api.results() })
  const policies = useQuery({
    queryKey: ['policy-results', 'all'],
    queryFn: () => api.policyResults(),
  })
  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set())
  // 消えた結果は選びから外す
  const chosen = (results.data ?? []).map((r) => r.id).filter((id) => selected.has(id))
  const toggle = (id: string) =>
    setSelected((current) => {
      const next = new Set(current)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  const ready = chosen.length >= 2 && chosen.length <= MAX_COMPARE
  return (
    <>
      <PageHeader title="結果" subtitle="推論した結果. 一式を消しても結果は残ります" />
      <div className="flex flex-col gap-4">
        <Card
          title="推論の一式の結果"
          actions={
            <Button
              icon={Columns3}
              disabled={!ready}
              onClick={() => navigate(`/results/compare?ids=${chosen.join(',')}`)}
            >
              選んだ {chosen.length} つを比べる
            </Button>
          }
        >
          <p className="mb-2 text-xs text-ink-3">
            同じ一式か同じ入力のファイルの結果を 2 つから {MAX_COMPARE} つ選ぶと, 結論, 判断の内訳,
            意思決定を並べて比べます.
          </p>
          {results.isPending && <Spinner />}
          {results.error && <ErrorBox error={results.error} />}
          {results.data && (
            <ResultList results={results.data} showBundle selected={selected} onToggle={toggle} />
          )}
        </Card>
        <Card title="方策の結果">
          {policies.isPending && <Spinner />}
          {policies.error && <ErrorBox error={policies.error} />}
          {policies.data && <PolicyResultList results={policies.data} showPolicy />}
        </Card>
      </div>
    </>
  )
}
