import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router'

import type { Result, Rows } from '../../api/types'
import { ResultPage } from './ResultPage'

// 3 行を 3 つの群に割り当てた結果 (群 2 には行が無く, 1 行は割り当てられず)
const learned = [
  {
    group: 0,
    rows: 20,
    share: 0.5,
    stability: 0.9,
    spend: 1000.5,
    region: 'north (100 %)',
  },
  { group: 1, rows: 10, share: 0.25, stability: 0.92, spend: 2000, region: 'south (100 %)' },
  { group: 2, rows: 10, share: 0.25, stability: 0.94, spend: 3000, region: 'east (100 %)' },
  { group: '全体', rows: 40, share: 1, stability: null, spend: 1750, region: 'north (50 %)' },
]
const groups = [
  { group: 0, rows: 2, share: 2 / 3, stability: 0.9, spend: 1010.25, region: 'north (100 %)' },
  { group: 1, rows: 1, share: 1 / 3, stability: 0.92, spend: 1990, region: 'south (100 %)' },
  { group: 2, rows: 0, share: 0, stability: 0.94, spend: null, region: null },
  { group: '全体', rows: 3, share: 1, stability: null, spend: 1336.83, region: 'north (67 %)' },
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
    rows: 4,
    predicted: 3,
    skipped: 1,
    k: 3,
    groups,
    representatives: [
      {
        group: 0,
        rows: [
          { row: 0, distance: 0.0123, values: { spend: 1000, region: 'north' } },
          { row: 1, distance: 0.02, values: { spend: 1020, region: 'north' } },
        ],
      },
      { group: 1, rows: [{ row: 2, distance: 0.03, values: { spend: 1990, region: 'south' } }] },
      { group: 2, rows: [] },
    ],
    stability: {
      k: 3,
      method: 'seed',
      repeats: 5,
      ari: 0.97,
      ari_baseline: 0.99,
      ari_diff: -0.02,
      ari_min: 0.95,
      ari_min_baseline: 0.98,
      ari_min_diff: -0.03,
      silhouette: 0.62,
      silhouette_baseline: 0.55,
      silhouette_diff: 0.07,
      baseline: '列の特徴をそのまま k-means で分けた群',
      epoch: 12,
    },
  },
  sentences: [
    '4 行のうち 3 行を 3 個の群に割り当てた (1 行は割り当てられず)',
    '群 0 2 行 (66.7 %), 群 1 1 行 (33.3 %), 群 2 0 行 (0.0 %)',
  ],
  overall_count: 2,
  columns: ['customer', 'spend', 'region', 'status', 'reason', 'group', 'distance', 'decision'],
  kind: 'group',
  role: 'group',
  output_form: 'cluster',
  column_output: false,
  artifacts: null,
  grouping: { k: 3, columns: ['spend', 'region'], learned },
} as unknown as Result

const reason = "列 spend の値 '?' を数にできません"

const rows: Rows = {
  total: 4,
  offset: 0,
  columns: [
    'index',
    'customer',
    'spend',
    'region',
    'status',
    'reason',
    'group',
    'distance',
    'decision',
  ],
  rows: [
    {
      index: 0,
      customer: 'a',
      spend: 1000,
      region: 'north',
      status: 'ok',
      reason: '',
      group: 0,
      distance: 0.0123,
      decision: '',
    },
    {
      index: 1,
      customer: 'b',
      spend: 1020,
      region: 'north',
      status: 'ok',
      reason: '',
      group: 0,
      distance: 0.02,
      decision: '',
    },
    {
      index: 2,
      customer: 'c',
      spend: 1990,
      region: 'south',
      status: 'ok',
      reason: '',
      group: 1,
      distance: 0.03,
      decision: '',
    },
    {
      index: 3,
      customer: 'x',
      spend: '?',
      region: 'north',
      status: 'skipped',
      reason,
      group: null,
      distance: null,
      decision: '予測なし',
    },
  ],
}

const rowsCall = vi.fn((_id: string, _params: Record<string, unknown>) => Promise.resolve(rows))

// 件数を変えて求めた代表の行 (群 0 は 1 件だけ)
const representativesCall = vi.fn((_id: string, count: number) =>
  Promise.resolve({
    count,
    groups: [
      { group: 0, rows: [{ row: 0, distance: 0.0123, values: { spend: 1000, region: 'north' } }] },
      { group: 1, rows: [{ row: 2, distance: 0.03, values: { spend: 1990, region: 'south' } }] },
      { group: 2, rows: [] },
    ],
  }),
)

vi.mock('../../api/client', () => ({
  api: {
    result: () => Promise.resolve(result),
    bundle: () => Promise.reject(new Error('無い')),
    rows: (id: string, params: Record<string, unknown>) => rowsCall(id, params),
    representatives: (id: string, count: number) => representativesCall(id, count),
    downloadUrl: (id: string) => `/api/results/${id}/download`,
  },
}))

// 「行を見る」で表へ移る (jsdom は scrollIntoView を持たない)
const scrolled = vi.fn()
Element.prototype.scrollIntoView = scrolled

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

describe('群分けの結果の画面', () => {
  it('群の代表の行を群の要約と並べ, 件数を変え, 行を見るで 1 行に絞る', async () => {
    renderPage()
    const table = await screen.findByRole('table', {
      name: '群の代表の行 (群ごとに中心に近い順の行, 距離, 群の要約の列の値)',
    })
    expect(
      within(table)
        .getAllByRole('columnheader')
        .map((cell) => cell.textContent),
    ).toEqual(['群', '順位', '#', '距離', 'spend', 'region', '行を見る'])
    const lines = within(table).getAllByRole('row').slice(1)
    // 群 0 の 2 行, 群 1 の 1 行, 行の無い群 2 の 1 行
    expect(lines).toHaveLength(4)
    expect(lines[0]).toHaveTextContent('群 0')
    expect(lines[0]).toHaveTextContent('0.012')
    expect(lines[0]).toHaveTextContent('north')
    expect(lines[1]).toHaveTextContent('1,020')
    expect(lines[3]).toHaveTextContent('群 2')
    expect(lines[3]).toHaveTextContent('割り当てた行はありません')
    // 既定の 3 件は結論のまま (API で求めない)
    expect(screen.getByLabelText('代表の行の件数')).toHaveValue('3')
    expect(representativesCall).not.toHaveBeenCalled()

    // 件数を変えると API で求める
    fireEvent.change(screen.getByLabelText('代表の行の件数'), { target: { value: '1' } })
    // 求め終わるまでは結論の 3 件を出さず, 読み込み中を出す
    expect(screen.getByText('代表の行を求めています')).toBeInTheDocument()
    expect(
      screen.queryByRole('table', {
        name: '群の代表の行 (群ごとに中心に近い順の行, 距離, 群の要約の列の値)',
      }),
    ).not.toBeInTheDocument()
    await waitFor(() => expect(representativesCall).toHaveBeenLastCalledWith('r1', 1))
    await waitFor(() =>
      expect(
        within(
          screen.getByRole('table', {
            name: '群の代表の行 (群ごとに中心に近い順の行, 距離, 群の要約の列の値)',
          }),
        )
          .getAllByRole('row')
          .slice(1),
      ).toHaveLength(3),
    )

    // 行を見るで, 割り当てた行の表をその行だけにし, 群で絞る欄もその群にする
    fireEvent.click(screen.getByRole('button', { name: '行を見る (群 1 の代表の行 #2)' }))
    await waitFor(() =>
      expect(rowsCall).toHaveBeenLastCalledWith(
        'r1',
        expect.objectContaining({ group: 1, row: 2, offset: 0 }),
      ),
    )
    expect(screen.getByLabelText('群で絞る')).toHaveValue('1')
    expect(screen.getByText(/群の代表の行 #2 だけを見ています/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '1 行の絞りを外す' }))
    await waitFor(() =>
      expect(rowsCall).toHaveBeenLastCalledWith(
        'r1',
        expect.objectContaining({ group: 1, row: undefined }),
      ),
    )
    expect(screen.queryByText(/だけを見ています/)).not.toBeInTheDocument()

    // 群で絞る欄を変えても 1 行の絞りは外れる
    fireEvent.click(screen.getByRole('button', { name: '行を見る (群 0 の代表の行 #0)' }))
    await waitFor(() =>
      expect(rowsCall).toHaveBeenLastCalledWith('r1', expect.objectContaining({ row: 0 })),
    )
    fireEvent.change(screen.getByLabelText('群で絞る'), { target: { value: '' } })
    await waitFor(() =>
      expect(rowsCall).toHaveBeenLastCalledWith(
        'r1',
        expect.objectContaining({ group: undefined, row: undefined }),
      ),
    )
  })

  it('群の要約, 学習の群の安定性, 行ごとの群を出し, 群で行を絞る', async () => {
    renderPage()
    expect(
      await screen.findByText('4 行のうち 3 行を 3 個の群に割り当てた (1 行は割り当てられず)'),
    ).toBeInTheDocument()
    expect(screen.getByText('割り当てた')).toBeInTheDocument()
    expect(screen.getByText('割り当てられなかった')).toBeInTheDocument()
    expect(screen.getByText('3 個')).toBeInTheDocument()
    expect(screen.getByText(/行ごとに最も近い群を出すので/)).toBeInTheDocument()

    // 群の要約 (群ごとの件数, 割合, 学習のときの割合, 安定性, 列の特徴と全体)
    const summary = screen.getByRole('table', {
      name: '群の要約 (群ごとの件数, 割合, 安定性, 列の特徴と全体)',
    })
    const headings = within(summary)
      .getAllByRole('columnheader')
      .map((cell) => cell.textContent)
    expect(headings).toEqual([
      '群',
      '件数',
      '割合',
      '学習のときの割合',
      '安定性',
      'spend (平均)',
      'region (最も多い値)',
      '行を見る',
    ])
    const lines = within(summary).getAllByRole('row').slice(1)
    expect(lines).toHaveLength(4)
    expect(lines[0]).toHaveTextContent('群 0')
    expect(lines[0]).toHaveTextContent('66.7%')
    expect(lines[0]).toHaveTextContent('50.0%')
    expect(lines[0]).toHaveTextContent('0.900')
    expect(lines[0]).toHaveTextContent('north (100 %)')
    // 行の無い群も 0 件で並び, 行を見るは出さない
    expect(lines[2]).toHaveTextContent('群 2')
    expect(screen.queryByRole('button', { name: '群 2 の行を見る' })).not.toBeInTheDocument()
    expect(lines[3]).toHaveTextContent('全体')

    // 群の安定性 (モデル, 比べる相手, 差)
    const stability = screen.getByRole('table', {
      name: '学習の群の安定性 (モデル, 比べる相手, 差)',
    })
    const ari = within(stability).getAllByRole('row')[1]
    expect(ari).toHaveTextContent('調整ランド指数 (平均)')
    expect(ari).toHaveTextContent('0.970')
    expect(ari).toHaveTextContent('0.990')
    expect(ari).toHaveTextContent('-0.020')
    expect(within(stability).getAllByRole('row')[3]).toHaveTextContent('+0.070')
    expect(screen.getByText(/種を変えて作り直す, 5 回/)).toHaveTextContent(
      '列の特徴をそのまま k-means で分けた群',
    )

    // 行ごとの群 (見出しと状態は日本語で, 判断の列は出さない)
    const table = await screen.findByRole('table', { name: '群に割り当てた行 (行ごとの群と距離)' })
    expect(
      within(table)
        .getAllByRole('columnheader')
        .map((cell) => cell.textContent),
    ).toEqual(['#', '群', '距離', '状態', '理由', 'customer', 'spend', 'region'])
    const body = within(table).getAllByRole('row').slice(1)
    expect(body[0]).toHaveTextContent('群 0')
    expect(body[0]).toHaveTextContent('割り当てた')
    expect(body[3]).toHaveTextContent('割り当てられず')
    expect(body[3]).toHaveTextContent(reason)
    expect(screen.queryByLabelText('判断で絞る')).not.toBeInTheDocument()
    expect(screen.queryByText('結論の決め方')).not.toBeInTheDocument()
    // 遠さの上限を持たない一式 (版 9) の結果には遠い行の欄と絞る欄が無い
    expect(screen.queryByText('群の中心から遠い行')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('遠さで絞る')).not.toBeInTheDocument()

    // 群の要約の「行を見る」で, 表をその群に絞る
    fireEvent.click(screen.getByRole('button', { name: '群 1 の行を見る' }))
    await waitFor(() =>
      expect(rowsCall).toHaveBeenLastCalledWith('r1', expect.objectContaining({ group: 1 })),
    )
    expect(screen.getByLabelText('群で絞る')).toHaveValue('1')
    expect(scrolled).toHaveBeenCalled()
    fireEvent.change(screen.getByLabelText('群で絞る'), { target: { value: '' } })
    await waitFor(() =>
      expect(rowsCall).toHaveBeenLastCalledWith(
        'r1',
        expect.objectContaining({ group: undefined }),
      ),
    )
  })
})
