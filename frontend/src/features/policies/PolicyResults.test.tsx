import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router'

import type { PolicyResult, PolicyResultRows } from '../../api/types'
import { PolicyResultPage } from './PolicyResults'

// CartPole の方策で 3 件の観測を決めた結果 (1 件は元の制御に戻した)
const result: PolicyResult = {
  id: 'r1',
  policy_id: 'p1',
  policy: { name: 'cartpole', path: '/outputs/cartpole/policy' },
  source: '/data/observations.csv',
  created_at: '2026-01-01T00:00:00',
  rows: 3,
  counts: { ok: 2, clipped: 0, fallback: 0, stop: 0, handover: 1 },
}

const all: PolicyResultRows['rows'] = [
  { index: 0, observation: '[0.0, 0.0, 0.0, 0.0]', action: '1', status: 'ok', reasons: '' },
  {
    index: 1,
    observation: '[2.5, 0.0, 0.0, 0.0]',
    action: 'null',
    status: 'handover',
    reasons: '観測の要素 0 が範囲の外 (2.5)',
  },
  { index: 2, observation: '[0.1, 0.0, 0.0, 0.0]', action: '0', status: 'ok', reasons: '' },
]

const rows = vi.fn((_id: string, params: { status?: string }) =>
  Promise.resolve({
    total: all.filter((row) => !params.status || row.status === params.status).length,
    offset: 0,
    rows: all.filter((row) => !params.status || row.status === params.status),
  }),
)

vi.mock('../../api/client', async () =>
  (await import('../../test/mockClient')).mockClient({
    api: {
      policyResult: () => Promise.resolve(result),
      policy: () => Promise.reject(new Error('無い')),
      policyResultRows: (id: string, params: { status?: string }) => rows(id, params),
      policyResultDownloadUrl: (id: string) => `/api/policy-results/${id}/download`,
    },
  }),
)

describe('方策の結果の画面', () => {
  it('守りの結果ごとの件数と 1 件ずつの表を出し, 守りの結果で絞る', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter initialEntries={['/policy-results/r1']}>
          <Routes>
            <Route path="/policy-results/:id" element={<PolicyResultPage />} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    )
    const bars = await screen.findByRole('list', { name: '守りの結果ごとの件数' })
    expect(
      within(bars)
        .getAllByRole('listitem')
        .map((item) => item.textContent),
    ).toEqual([
      'そのまま2 件',
      '範囲に収めた0 件',
      '決めた行動にした0 件',
      '止めた0 件',
      '元の制御に戻した1 件',
    ])
    expect(screen.getByRole('link', { name: 'CSV' })).toHaveAttribute(
      'href',
      '/api/policy-results/r1/download',
    )
    const table = await screen.findByRole('table', { name: '観測ごとの行動と守りの結果の表' })
    expect(within(table).getAllByRole('row')).toHaveLength(4)
    expect(within(table).getByText('観測の要素 0 が範囲の外 (2.5)')).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText('守りの結果で絞る'), { target: { value: 'handover' } })
    expect(await screen.findByText('1 件中 1–1')).toBeInTheDocument()
    expect(rows).toHaveBeenLastCalledWith('r1', { offset: 0, limit: 50, status: 'handover' })
    // 登録の無い方策の結果は結果の一覧へ戻る
    expect(screen.getByRole('link', { name: '結果' })).toHaveAttribute('href', '/results')
  })
})
