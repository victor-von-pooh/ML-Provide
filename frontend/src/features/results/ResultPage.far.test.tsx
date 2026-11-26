import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router'

import type { Result, Rows } from '../../api/types'
import { ResultPage } from './ResultPage'

// 遠さの上限を持つ群分けの一式 (版 16) で 3 行を 2 つの群に割り当て, 群 1 の 1 行が遠い結果
const groups = [
  { group: 0, rows: 2, share: 2 / 3, stability: 0.9, spend: 1010 },
  { group: 1, rows: 1, share: 1 / 3, stability: 0.92, spend: 90000 },
  { group: '全体', rows: 3, share: 1, stability: null, spend: 30673 },
]
const result = {
  id: 'r1',
  bundle: { name: 'customers', path: '/bundles/customers' },
  bundle_id: 'g1',
  source: '/data/customers.csv',
  created_at: '2026-01-01T00:00:00',
  task: '',
  classes: [],
  rules: {},
  summary: {
    rows: 3,
    predicted: 3,
    skipped: 0,
    k: 2,
    groups,
    representatives: [
      { group: 0, rows: [{ row: 0, distance: 0.01, values: { spend: 1000 } }] },
      { group: 1, rows: [{ row: 2, distance: 3.5, values: { spend: 90000 } }] },
    ],
    far: {
      quantile: 0.99,
      rows: 1,
      groups: [
        { group: 0, rows: 0, threshold: 0.5604 },
        { group: 1, rows: 1, threshold: 0.6332 },
      ],
    },
  },
  sentences: [
    '3 行のうち 3 行を 2 個の群に割り当てた',
    '群 0 2 行 (66.7 %), 群 1 1 行 (33.3 %)',
    '1 行は群の中心から学習のときより遠い (群ごとの学習の行の距離の 99 % 点を超える. 群 1 1 行)',
  ],
  overall_count: 3,
  columns: ['customer', 'spend', 'status', 'reason', 'group', 'distance', 'far', 'decision'],
  kind: 'group',
  role: 'group',
  output_form: 'cluster',
  column_output: false,
  artifacts: null,
  grouping: {
    k: 2,
    columns: ['spend'],
    learned: groups,
    far: { quantile: 0.99, rows: 60, thresholds: [0.5604, 0.6332] },
  },
} as unknown as Result

const rows: Rows = {
  total: 3,
  offset: 0,
  columns: ['index', 'customer', 'spend', 'status', 'reason', 'group', 'distance', 'far'],
  rows: [
    {
      index: 0,
      customer: 'a',
      spend: 1000,
      status: 'ok',
      reason: '',
      group: 0,
      distance: 0.01,
      far: false,
    },
    {
      index: 1,
      customer: 'b',
      spend: 1020,
      status: 'ok',
      reason: '',
      group: 0,
      distance: 0.02,
      far: false,
    },
    {
      index: 2,
      customer: 'z',
      spend: 90000,
      status: 'ok',
      reason: '',
      group: 1,
      distance: 3.5,
      far: true,
    },
  ],
}

const rowsCall = vi.fn((_id: string, _params: Record<string, unknown>) => Promise.resolve(rows))

vi.mock('../../api/client', () => ({
  api: {
    result: () => Promise.resolve(result),
    bundle: () => Promise.reject(new Error('無い')),
    rows: (id: string, params: Record<string, unknown>) => rowsCall(id, params),
    representatives: () => new Promise(() => {}),
    downloadUrl: (id: string) => `/api/results/${id}/download`,
  },
}))

// 「行を見る」で表へ移る (jsdom は scrollIntoView を持たない)
Element.prototype.scrollIntoView = vi.fn()

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/results/r1']}>
        <Routes>
          <Route path="/results/:id" element={<ResultPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('群の中心から遠い行のある群分けの結果の画面', () => {
  it('群ごとの遠い行の件数と上限, 行の印を出し, 遠さで行を絞る', async () => {
    renderPage()
    expect(await screen.findByText(/^1 行は群の中心から学習のときより遠い/)).toBeInTheDocument()
    expect(screen.getByText('中心から遠い行 1 行')).toBeInTheDocument()

    // 群ごとの遠い行の件数と距離の上限 (最後に全体) と, 上限の決め方
    const table = screen.getByRole('table', {
      name: '群の中心から遠い行 (群ごとの遠い行の件数と距離の上限, 全体)',
    })
    expect(
      within(table)
        .getAllByRole('columnheader')
        .map((cell) => cell.textContent),
    ).toEqual(['群', '遠い行', '距離の上限', '遠い行を見る'])
    const lines = within(table).getAllByRole('row').slice(1)
    expect(lines).toHaveLength(3)
    expect(lines[0]).toHaveTextContent('群 0')
    expect(lines[0]).toHaveTextContent('0.560')
    expect(lines[1]).toHaveTextContent('0.633')
    expect(lines[2]).toHaveTextContent('全体')
    expect(lines[2]).toHaveTextContent('1')
    expect(screen.getByText(/学習の行 60 行の群の中心までの距離の 99 % 点を上限にし/)).toBeVisible()
    // 遠い行の無い群には行を見るを出さない
    expect(screen.queryByRole('button', { name: '群 0 の遠い行を見る' })).not.toBeInTheDocument()

    // 割り当てた行の表は距離の次に中心から遠いかの列
    const rowsTable = await screen.findByRole('table', {
      name: '群に割り当てた行 (行ごとの群と距離)',
    })
    expect(
      within(rowsTable)
        .getAllByRole('columnheader')
        .map((cell) => cell.textContent),
    ).toEqual(['#', '群', '距離', '中心から遠い', '状態', '理由', 'customer', 'spend'])
    const body = within(rowsTable).getAllByRole('row').slice(1)
    expect(body[0]).not.toHaveTextContent('遠い')
    expect(body[2]).toHaveTextContent('遠い')

    // 遠さで絞る欄
    fireEvent.change(screen.getByLabelText('遠さで絞る'), { target: { value: 'false' } })
    await waitFor(() =>
      expect(rowsCall).toHaveBeenLastCalledWith('r1', expect.objectContaining({ far: false })),
    )
    fireEvent.change(screen.getByLabelText('遠さで絞る'), { target: { value: '' } })
    await waitFor(() =>
      expect(rowsCall).toHaveBeenLastCalledWith('r1', expect.objectContaining({ far: undefined })),
    )

    // 群の行の「行を見る」で, その群の遠い行だけにする (代表の行の 1 行の絞りは外す)
    fireEvent.click(screen.getByRole('button', { name: '行を見る (群 1 の代表の行 #2)' }))
    await waitFor(() =>
      expect(rowsCall).toHaveBeenLastCalledWith('r1', expect.objectContaining({ row: 2 })),
    )
    fireEvent.click(screen.getByRole('button', { name: '群 1 の遠い行を見る' }))
    await waitFor(() =>
      expect(rowsCall).toHaveBeenLastCalledWith(
        'r1',
        expect.objectContaining({ group: 1, far: true, row: undefined, offset: 0 }),
      ),
    )
    expect(screen.getByLabelText('群で絞る')).toHaveValue('1')
    expect(screen.getByLabelText('遠さで絞る')).toHaveValue('true')
  })
})
