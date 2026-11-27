import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router'

import type { Result, Rows } from '../../api/types'
import { ResultPage } from './ResultPage'

// 利用者 3 人を 2 件ずつ並べた結果 (1 人は番号の表に無い. 正解の反応の表あり)
const result: Result = {
  id: 'r1',
  bundle: { name: 'shop', path: '/bundles/shop' },
  bundle_id: 's1',
  source: '利用者 3 人',
  created_at: '2026-01-01T00:00:00',
  task: '',
  classes: [],
  rules: {},
  summary: { rows: 3, predicted: 2, skipped: 1, k: 2, excluded: true },
  sentences: [
    '利用者 3 人のうち 2 人に, 上位 2 件の品物を並べた (1 人は並べられず)',
    '前に反応した品物は除いて並べた',
  ],
  columns: ['user', 'status', 'reason', 'rank', 'item', 'score', 'decision'],
  kind: 'rank',
  role: 'rank',
  output_form: 'ranking',
  column_output: false,
  artifacts: null,
  ranking: {
    k: 2,
    excluded: true,
    columns: { user: 'member', item: 'item_id', reaction: 'qty' },
    seen: '/data/seen.csv',
    truth: '/data/later.csv',
    metrics: { users: 2, k: 2, recall_at_k: 0.75, ndcg: 0.6131 },
  },
}

const unknown = '番号の表に無い利用者です (学習用の反応に無い利用者は並べられません)'

const rows: Rows = {
  total: 5,
  offset: 0,
  columns: ['index', 'user', 'status', 'reason', 'rank', 'item', 'score', 'decision'],
  rows: [
    {
      index: 0,
      user: 'u0',
      status: 'ok',
      reason: '',
      rank: 1,
      item: 'i3',
      score: 2.5,
      decision: '',
    },
    {
      index: 1,
      user: 'u0',
      status: 'ok',
      reason: '',
      rank: 2,
      item: 'i1',
      score: 1.5,
      decision: '',
    },
    {
      index: 2,
      user: '007',
      status: 'ok',
      reason: '',
      rank: 1,
      item: 'i0',
      score: 3,
      decision: '',
    },
    {
      index: 3,
      user: '007',
      status: 'ok',
      reason: '',
      rank: 2,
      item: 'i5',
      score: 1,
      decision: '',
    },
    {
      index: 4,
      user: 'nobody',
      status: 'skipped',
      reason: unknown,
      rank: null,
      item: null,
      score: null,
      decision: '予測なし',
    },
  ],
}

vi.mock('../../api/client', async () =>
  (await import('../../test/mockClient')).mockClient({
    api: {
      result: () => Promise.resolve(result),
      bundle: () => Promise.reject(new Error('無い')),
      rows: () => Promise.resolve(rows),
      downloadUrl: (id: string) => `/api/results/${id}/download`,
    },
  }),
)

describe('並べるの結果の画面', () => {
  it('利用者ごとの上位の品物と点数, 並べられなかった理由, Recall@K と NDCG を出す', async () => {
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
    expect(
      await screen.findByText(
        '利用者 3 人のうち 2 人に, 上位 2 件の品物を並べた (1 人は並べられず)',
      ),
    ).toBeInTheDocument()
    expect(screen.getByText('並べた利用者')).toBeInTheDocument()
    expect(screen.getAllByText('並べられなかった利用者')).toHaveLength(2)
    expect(screen.getByText('2 件')).toBeInTheDocument()

    // 評価と使った表
    expect(screen.getByText('Recall@2 (上位 2 件に入った正解の割合)')).toBeInTheDocument()
    expect(screen.getByText('NDCG (上位 2 件の正解の位置で見た並びの良さ)')).toBeInTheDocument()
    expect(screen.getByText('0.750')).toBeInTheDocument()
    expect(screen.getByText('0.613')).toBeInTheDocument()
    expect(screen.getByText('/data/seen.csv')).toBeInTheDocument()
    expect(screen.getByText('/data/later.csv')).toBeInTheDocument()

    // 表の見出しと状態は日本語で, 判断の列は出さない
    const table = await screen.findByRole('table', {
      name: '並べた品物 (利用者ごとの上位の品物と点数)',
    })
    const headings = within(table)
      .getAllByRole('columnheader')
      .map((cell) => cell.textContent)
    expect(headings).toEqual(['#', '利用者', '順位', '品物', '点数', '状態', '理由'])
    const body = within(table).getAllByRole('row').slice(1)
    expect(body).toHaveLength(5)
    expect(body[2]).toHaveTextContent('007')
    expect(body[0]).toHaveTextContent('並べた')
    expect(body[4]).toHaveTextContent('並べられず')
    expect(body[4]).toHaveTextContent(unknown)
    expect(screen.queryByText('ok')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('判断で絞る')).not.toBeInTheDocument()
    // 結論の決め方と意思決定の欄は出さない
    expect(screen.queryByText('結論の決め方')).not.toBeInTheDocument()
    expect(screen.getByText(/利用者ごとの品物の一覧を出すので/)).toBeInTheDocument()
  })
})
