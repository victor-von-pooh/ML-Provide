import type { UseQueryResult } from '@tanstack/react-query'
import { Columns3 } from 'lucide-react'
import { useState } from 'react'
import { useNavigate } from 'react-router'

import type { Result } from '../../api/types'
import { Button, Card, ErrorBox, Spinner } from '../../components/ui'
import { ResultList } from './ResultList'

/** 一度に比べられる結果の数の上限 (API の MAX_COMPARE と同じ) */
export const MAX_COMPARE = 6

/** 結果の一覧の枠. 結果を 2 つから上限まで選んで比べの画面に移る */
export function CompareResultsCard({
  title,
  results,
  showBundle,
  hint,
}: {
  title: string
  results: UseQueryResult<Result[]>
  showBundle: boolean
  /** 何を選べば比べられるかの案内 */
  hint: string
}) {
  const navigate = useNavigate()
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
    <Card
      title={title}
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
      <p className="mb-2 text-xs text-ink-3">{hint}</p>
      {results.isPending && <Spinner />}
      {results.error && <ErrorBox error={results.error} />}
      {results.data && (
        <ResultList
          results={results.data}
          showBundle={showBundle}
          selected={selected}
          onToggle={toggle}
          full={chosen.length >= MAX_COMPARE}
        />
      )}
    </Card>
  )
}
