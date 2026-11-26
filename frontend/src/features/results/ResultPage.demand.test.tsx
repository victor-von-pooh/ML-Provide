import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router'

import type { OrderRows, Result, Rows } from '../../api/types'
import { ResultPage } from './ResultPage'

// 3 商品のうち 2 商品の需要を予測し, 1 商品を上限で切った結果
const evaluation = {
  rows: 8,
  products: 4,
  mae: 4.5,
  pinball_loss: 1.25,
  order_cost: 40,
  baseline_cost: 50,
  order_cost_saving: 10,
  order_cost_ratio: 0.8,
  capped: 0,
}
const result = {
  id: 'r1',
  bundle: { name: 'demand', path: '/bundles/demand' },
  bundle_id: 'd1',
  source: '/data/daily.csv',
  created_at: '2026-01-01T00:00:00',
  task: '',
  classes: [],
  rules: {},
  summary: {
    rows: 3,
    predicted: 2,
    skipped: 1,
    id: 'record_id',
    product: 'product_id',
    date: 'date',
    stock: 'stock',
    horizon: 7,
    levels: [0.1, 0.5, 0.9],
    dates: { min: '2026-03-11', max: '2026-03-11' },
    prediction: { mean: 20, total: 40 },
    previous: { mean: 18, counted: 2 },
    train: { rows: 20, products: 4, demand_mean: 27.45 },
    evaluations: {
      valid: evaluation,
      test: { ...evaluation, baseline_cost: 0, order_cost_ratio: null },
    },
  },
  sentences: ['3 商品のうち 2 商品の先の 7 日の需要の分位点を予測した'],
  overall_count: 1,
  columns: ['record_id', 'status', 'reason', 'order'],
  kind: 'demand',
  role: 'predict',
  output_form: 'distribution',
  column_output: false,
  artifacts: null,
  demand: {
    horizon: 7,
    levels: [0.1, 0.5, 0.9],
    quantiles: [12, 20, 31],
    low: 0.1,
    high: 0.9,
    width: { counted: 2, mean: 19, min: 15, p10: 15.4, p50: 19, p90: 22.6, max: 23 },
  },
  decision: {
    decision: { type: 'order', shortage_cost: 9, holding_cost: 1, max_order: 25 },
    summary: {
      rows: 2,
      undecided: 0,
      ordered: 2,
      total: 41,
      capped: 1,
      level: 0.9,
      used_level: 0.9,
      outside: false,
      expected_cost: 12.5,
      compared: 2,
      compared_cost: 12.5,
      baseline_cost: 30,
      saving: 17.5,
      ratio: 0.4167,
      shortage_cost: 9,
      holding_cost: 1,
      max_order: 25,
    },
    sentences: [
      '書いた損 (品切れ 9, 売れ残り 1. 商品 1 つあたり) と, 先の 7 日の需要の分位点の予測が正しいとすれば, 見込みの損が最も小さいのは水準 τ = 0.9 の分位点から在庫を引いた数',
      '2 商品のうち 2 商品で発注する (発注の数の和 41. 水準 0.9, 上限 25). 見込みの損の和は 12.5',
    ],
    notes: [
      '書いた損 (品切れ 9, 売れ残り 1. 商品 1 つあたり) と, 先の 7 日の需要の分位点の予測が正しいとすれば, 見込みの損が最も小さいのは水準 τ = 0.9 の分位点から在庫を引いた数',
    ],
  },
} as unknown as Result

const rows: Rows = { total: 0, offset: 0, columns: ['index'], rows: [] }
const orders = vi.fn((_id: string, offset: number, _limit: number): Promise<OrderRows> =>
  Promise.resolve({
    id: 'record_id',
    product: 'product_id',
    levels: [0.1, 0.5, 0.9],
    shortage_cost: 9,
    holding_cost: 1,
    max_order: 25,
    total: 2,
    offset,
    rows: [
      {
        row: 1,
        id: 'r1_69',
        product_id: 'p01',
        quantiles: [16, 25, 34],
        order_date: '2026-03-11',
        on_hand: 2,
        previous_demand: 21,
        prediction: 25,
        quantile: 34,
        order: 25,
        capped: true,
        expected_cost: 8,
        baseline_order: 19,
        baseline_cost: 20,
      },
      {
        row: 0,
        id: 10234,
        product_id: 'p00',
        quantiles: [8, 15, 23],
        order_date: '2026-03-11',
        on_hand: 7,
        previous_demand: null,
        prediction: 15,
        quantile: 23,
        order: 16,
        capped: false,
        expected_cost: 4.5,
        baseline_order: null,
        baseline_cost: null,
      },
    ],
  }),
)

vi.mock('../../api/client', () => ({
  api: {
    result: () => Promise.resolve(result),
    bundle: () => Promise.reject(new Error('無い')),
    rows: () => Promise.resolve(rows),
    orders: (id: string, offset: number, limit: number) => orders(id, offset, limit),
    downloadUrl: (id: string) => `/api/results/${id}/download`,
  },
}))

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

describe('需要の結果の画面', () => {
  beforeEach(() => orders.mockClear())

  it('言葉の説明, 見込みの損の比べ, 需要の分位点, 実験の評価, 発注の数を出し, 列の出力だけの欄を出さない', async () => {
    renderPage()
    // 結論に初めて出る言葉の説明
    const glossary = await screen.findByLabelText('言葉の説明')
    for (const term of [
      '需要',
      '分位点',
      '品切れの損',
      '売れ残りの損',
      '水準 τ',
      '発注の数の上限',
      '見込みの損',
    ])
      expect(within(glossary).getByText(term)).toBeInTheDocument()
    expect(glossary).toHaveTextContent('先の 7 日')
    expect(glossary).toHaveTextContent('在庫の数の列 stock')
    expect(glossary).toHaveTextContent('前の 7 日に売れた数の合計')
    expect(glossary).toHaveTextContent('実際の損は発注の後の売れた数で確かめて')

    // 見込みの損の比べと上限に当たった商品
    expect(screen.getByText('水準 τ', { selector: 'dt.text-ink-2' }).nextSibling).toHaveTextContent(
      '0.9',
    )
    expect(screen.getByText('見込みの損の和').nextSibling).toHaveTextContent('12.5')
    expect(screen.getByText('前の期間と同じ数の発注の見込みの損').nextSibling).toHaveTextContent(
      '30 (この決め方なら 12.5. 2 商品. 差 17.5, 比 0.4167 倍)',
    )
    expect(screen.getByText('上限に当たった商品').nextSibling).toHaveTextContent('1 商品')
    expect(screen.getByText(/見込みの損の和は 12.5/)).toBeInTheDocument()
    expect(screen.queryByText(/学んだ水準の範囲の外なので/)).not.toBeInTheDocument()

    // 需要の分位点の平均と幅
    const means = screen.getByRole('table', { name: /^需要の分位点の平均/ })
    expect(within(means).getAllByRole('row')[1]).toHaveTextContent('分位点の平均 (2 商品)')
    expect(within(means).getAllByRole('row')[1]).toHaveTextContent('31')
    const width = screen.getByRole('table', { name: /^需要の分位点の幅/ })
    expect(within(width).getAllByRole('row')[1]).toHaveTextContent('水準 0.1 から 0.9 まで')
    expect(screen.getByText(/学習用の分割の先の期間の需要の平均/)).toHaveTextContent('27.45')
    expect(screen.getByText(/前の 7 日の売れた数の合計の平均/)).toHaveTextContent('18')

    const evaluations = screen.getByRole('table', { name: /^実験の評価/ })
    const [, valid, test] = within(evaluations).getAllByRole('row')
    expect(valid).toHaveTextContent('4.5')
    expect(valid).toHaveTextContent('0.8 倍')
    expect(test).toHaveTextContent('無し (前の期間と同じ数の発注の損が 0)')

    // 発注の数の一覧 (上限に当たった商品に印)
    const list = await screen.findByRole('table', { name: /^発注の数/ })
    const [, first, second] = within(list).getAllByRole('row')
    expect(first).toHaveTextContent('r1_69')
    expect(first).toHaveTextContent('p01')
    expect(first).toHaveTextContent('16, 25, 34')
    expect(first).toHaveTextContent('上限に当たった')
    expect(second).toHaveTextContent('10234')
    expect(second).not.toHaveTextContent('10,234')
    expect(second).not.toHaveTextContent('上限に当たった')
    expect(orders).toHaveBeenCalledWith('r1', 0, 100)
    expect(screen.getByText('需要を予測した')).toBeInTheDocument()
    expect(screen.getByText('発注の数の和 41, 上限に当たった 1 商品')).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: '結論の決め方を変える' })).not.toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: /意思決定/ })).not.toBeInTheDocument()
  })
})
