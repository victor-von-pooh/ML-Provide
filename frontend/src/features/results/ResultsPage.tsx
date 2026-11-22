import { useQuery } from '@tanstack/react-query'

import { api } from '../../api/client'
import { Card, ErrorBox, PageHeader, Spinner } from '../../components/ui'
import { PolicyResultList } from '../policies/PolicyResults'
import { ResultList } from './ResultList'

/** 全ての一式の結果と, 方策の結果 (新しい順) */
export function ResultsPage() {
  const results = useQuery({ queryKey: ['results', 'all'], queryFn: () => api.results() })
  const policies = useQuery({
    queryKey: ['policy-results', 'all'],
    queryFn: () => api.policyResults(),
  })
  return (
    <>
      <PageHeader title="結果" subtitle="推論した結果. 一式を消しても結果は残ります" />
      <div className="flex flex-col gap-4">
        <Card title="推論の一式の結果">
          {results.isPending && <Spinner />}
          {results.error && <ErrorBox error={results.error} />}
          {results.data && <ResultList results={results.data} showBundle />}
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
