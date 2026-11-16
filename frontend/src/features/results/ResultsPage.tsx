import { useQuery } from '@tanstack/react-query'

import { api } from '../../api/client'
import { Card, ErrorBox, PageHeader, Spinner } from '../../components/ui'
import { ResultList } from './ResultList'

/** 全ての一式の結果 (新しい順) */
export function ResultsPage() {
  const results = useQuery({ queryKey: ['results', 'all'], queryFn: () => api.results() })
  return (
    <>
      <PageHeader title="結果" subtitle="推論した結果. 一式を消しても結果は残ります" />
      <Card>
        {results.isPending && <Spinner />}
        {results.error && <ErrorBox error={results.error} />}
        {results.data && <ResultList results={results.data} showBundle />}
      </Card>
    </>
  )
}
