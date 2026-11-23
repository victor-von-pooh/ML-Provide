import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query'
import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router'

import type { Result } from '../../api/types'
import { CompareResultsCard } from './CompareResultsCard'

// 同じ一式の 3 つの結果 (時刻の無い結果は id を出す)
const results: Result[] = ['r1', 'r2', 'r3'].map((id) => ({
  id,
  bundle: null,
  bundle_id: 'b1',
  source: 'new.csv',
  created_at: null,
  task: 'classification',
  summary: null,
  sentences: [],
}))

/** 一式の画面と同じく, 一式の結果を読んで枠に渡す */
function Bundle() {
  const query = useQuery({ queryKey: ['results', 'b1'], queryFn: () => Promise.resolve(results) })
  return <CompareResultsCard title="この一式の結果" results={query} showBundle={false} hint="" />
}

/** 移った先の場所 */
function Where() {
  const location = useLocation()
  return <p>{`${location.pathname}${location.search}`}</p>
}

describe('結果の一覧の枠で比べる結果を選ぶ', () => {
  it('2 つ選ぶと比べられ, 選んだ順ではなく一覧の順で比べの画面に移る', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter initialEntries={['/bundles/b1']}>
          <Routes>
            <Route path="/bundles/:id" element={<Bundle />} />
            <Route path="/results/compare" element={<Where />} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    )
    const last = await screen.findByLabelText('r3 の結果を比べる')
    const button = screen.getByRole('button', { name: /つを比べる/ })
    expect(button).toBeDisabled()
    fireEvent.click(last)
    expect(button).toBeDisabled()
    fireEvent.click(screen.getByLabelText('r1 の結果を比べる'))
    expect(button).toHaveTextContent('選んだ 2 つを比べる')
    expect(button).toBeEnabled()
    fireEvent.click(button)
    expect(await screen.findByText('/results/compare?ids=r1,r3')).toBeInTheDocument()
  })
})
