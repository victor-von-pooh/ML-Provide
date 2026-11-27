import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router'

import type { Comparison, Result } from '../../api/types'
import { ComparePage } from './ComparePage'

// 同じ一式と入力で, 確信の下限だけが違う 2 つ
const first: Result = {
  id: 'r1',
  bundle: { name: '評判', path: '/bundles/reviews' },
  bundle_id: 'b1',
  source: 'new.csv',
  created_at: null,
  task: 'classification',
  rules: { min_confidence: 0.55 },
  summary: { rows: 40, predicted: 40, skipped: 0, decisions: [] },
  sentences: [],
}
const second: Result = { ...first, id: 'r2', rules: {} }

let view: Comparison
const compare = vi.fn((_ids: string[]) => Promise.resolve(view))

vi.mock('../../api/client', async () =>
  (await import('../../test/mockClient')).mockClient({
    api: { compare: (ids: string[]) => compare(ids) },
  }),
)

/** 比べの画面を ids 付きで描く */
function show() {
  render(
    <QueryClientProvider client={new QueryClient()}>
      <MemoryRouter initialEntries={['/results/compare?ids=r1,r2']}>
        <Routes>
          <Route path="/results/compare" element={<ComparePage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('結果どうしの比べの画面', () => {
  beforeEach(() => compare.mockClear())

  it('選んだ結果を並べ, 違う行に印を付け, 違う行だけに絞れる', async () => {
    view = { results: [first, second], comparable: true, basis: ['bundle', 'source'], reasons: [] }
    show()
    const row = (await screen.findByText('保留にする確信の下限')).closest('tr')!
    expect(within(row).getByText('違う')).toBeInTheDocument()
    expect(within(row).getByText('0.55')).toBeInTheDocument()
    expect(compare).toHaveBeenCalledWith(['r1', 'r2'])
    expect(screen.getByText(/同じ一式, 同じ入力のファイルの 2 つの結果/)).toBeInTheDocument()
    expect(screen.getByText('一式')).toBeInTheDocument()
    fireEvent.click(screen.getByLabelText('違う行だけを出す'))
    expect(screen.queryByText('一式')).not.toBeInTheDocument()
    expect(screen.getByText('保留にする確信の下限')).toBeInTheDocument()
  })

  it('群分けの結果どうしは群の大きさと列の特徴を並べる', async () => {
    const groups = [
      { group: 0, rows: 2, share: 1, stability: 0.9, spend: 5 },
      { group: '全体', rows: 2, share: 1, stability: null, spend: 5 },
    ] as unknown as NonNullable<Result['summary']>['groups']
    const cluster: Result = {
      ...first,
      task: null,
      rules: {},
      output_form: 'cluster',
      summary: { rows: 2, predicted: 2, skipped: 0, k: 1, groups },
      grouping: {
        k: 1,
        columns: ['spend'],
        learned: [
          { group: 0, rows: 4, share: 1, stability: 0.9, spend: 4 },
          { group: '全体', rows: 4, share: 1, stability: null, spend: 4 },
        ],
      },
    }
    view = {
      results: [cluster, { ...cluster, id: 'r2' }],
      comparable: true,
      basis: ['bundle'],
      reasons: [],
      grouping: true,
    }
    show()
    expect(
      await screen.findByText('群の大きさ (件数, 割合, 学習のときの割合との差)'),
    ).toBeInTheDocument()
    expect(screen.getByText('群 0 (学習のとき 100.0%)')).toBeInTheDocument()
    expect(screen.getByText('全体の列の特徴')).toBeInTheDocument()
    expect(screen.queryByText('判断の内訳')).not.toBeInTheDocument()
  })

  it('比べられない組は理由と結果へのリンクだけを出す', async () => {
    view = {
      results: [first, { ...second, bundle_id: 'b2', source: 'other.csv' }],
      comparable: false,
      basis: [],
      reasons: ['一式も入力も違う結果は比べられません'],
    }
    show()
    expect(await screen.findByText('一式も入力も違う結果は比べられません')).toBeInTheDocument()
    expect(screen.getAllByRole('link', { name: /結果 \d/ })).toHaveLength(2)
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
  })
})
