import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router'

import type { PriceRows, Result, Rows } from '../../api/types'
import { ResultPage } from './ResultPage'

// 3 行のうち 2 行の売れた数を予測し, 1 行の値段を上げた結果
const evaluation = {
  rows: 10,
  products: 4,
  mae: 27.7,
  rmse: 31.5,
  elasticity: -0.487,
  elasticity_counted: 4,
  profit: 224.9,
  current_profit: 184.8,
  profit_gain: 40.1,
  profit_ratio: 1.217,
  changed: 4,
  max_change: 0.1,
}
const result = {
  id: 'r1',
  bundle: { name: 'prices', path: '/bundles/prices' },
  bundle_id: 'p1',
  source: '/data/products.csv',
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
    cost: 'unit_cost',
    units: { mean: 10, total: 20 },
    elasticity: { mean: -1.5, counted: 2 },
    train: { products: 6, changed: 6, elasticity: -0.32 },
    evaluations: { valid: evaluation, test: { ...evaluation, profit_ratio: null } },
    measured: { rows: 2, excluded: 0, mae: 1.5, rmse: 2 },
  },
  sentences: ['3 行のうち 2 行の売れた数を今の値段 (行の値段の列) で予測した (1 行は出せず)'],
  overall_count: 1,
  columns: ['record_id', 'status', 'reason', 'current_price', 'price', 'profit'],
  kind: 'price',
  role: 'effect',
  output_form: 'effect',
  column_output: false,
  artifacts: null,
  price: {
    step: 0.01,
    elasticity: {
      counted: 2,
      mean: -1.5,
      min: -2,
      p10: -1.9,
      p50: -1.5,
      p90: -1.1,
      max: -1,
      negative: 1,
    },
  },
  decision: {
    decision: { type: 'price', candidates: [-0.1, 0, 0.1, 0.2], max_change: 0.1 },
    summary: {
      rows: 2,
      undecided: 0,
      changed: 1,
      raised: 1,
      lowered: 0,
      total: 130,
      current_total: 120,
      gain: 10,
      ratio: 1.0833,
      candidates: [-0.1, 0, 0.1, 0.2],
      max_change: 0.1,
      limited: 1,
    },
    sentences: [
      '売れた数は値段を入力に入れたモデルの予測で, 試しに値段を変えた記録 (無作為に近い) を前提にしている',
      '2 行のうち 1 行の値段を変える (上げる 1 行, 下げる 0 行. 候補 [-0.1, 0, 0.1, 0.2], 変える幅の上限 0.1). 見込みの利益の和は 130 (今の値段のままなら 120. 差 10, 比 1.083)',
    ],
    notes: [
      '売れた数は値段を入力に入れたモデルの予測で, 試しに値段を変えた記録 (無作為に近い) を前提にしている',
    ],
  },
} as unknown as Result

const rows: Rows = { total: 0, offset: 0, columns: ['index'], rows: [] }
const prices = vi.fn((_id: string, offset: number, _limit: number): Promise<PriceRows> =>
  Promise.resolve({
    id: 'record_id',
    product: 'product_id',
    candidates: [-0.1, 0, 0.1, 0.2],
    max_change: 0.1,
    total: 2,
    offset,
    rows: [
      {
        row: 0,
        id: 'r0007',
        product_id: 'p00',
        current_price: 100,
        cost: 60,
        change: 0.1,
        price: 110,
        units: 2,
        price_units: 2,
        current_profit: 80,
        profit: 100,
        elasticity: -1,
      },
      {
        row: 1,
        id: 10234,
        product_id: 'p01',
        current_price: 80,
        cost: 60,
        change: 0,
        price: 80,
        units: 2,
        price_units: 2,
        current_profit: 40,
        profit: 40,
        elasticity: -2,
      },
    ],
  }),
)

vi.mock('../../api/client', () => ({
  api: {
    result: () => Promise.resolve(result),
    bundle: () => Promise.reject(new Error('無い')),
    rows: () => Promise.resolve(rows),
    prices: (id: string, offset: number, limit: number) => prices(id, offset, limit),
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

describe('値段の結果の画面', () => {
  beforeEach(() => prices.mockClear())

  it('言葉の説明, 利益の比べ, 弾力性の分布, 実験の評価, 選んだ値段を出し, 列の出力だけの欄を出さない', async () => {
    renderPage()
    // 結論に初めて出る言葉の説明
    const glossary = await screen.findByLabelText('言葉の説明')
    for (const term of ['値段', '売れた数', '原価', '利益 (見込み)', '弾力性', '変える割合'])
      expect(within(glossary).getByText(term)).toBeInTheDocument()
    expect(glossary).toHaveTextContent('原価の列 unit_cost')
    expect(glossary).toHaveTextContent('値段を 1 % 上げたとき')
    expect(glossary).toHaveTextContent('変える幅の上限は, 選んでよい割合の絶対値の上限')
    expect(glossary).toHaveTextContent('試しに値段を変えた記録を前提')

    // 利益の比べと上限を超える候補の方が良い行
    expect(screen.getByText('選んだ値段の見込みの利益の和').nextSibling).toHaveTextContent('130')
    expect(screen.getByText('上限を超える候補の方が利益の大きい行').nextSibling).toHaveTextContent(
      '1 行',
    )
    expect(screen.getByText('-10 %, 0 %, +10 %, +20 %')).toBeInTheDocument()
    expect(
      screen.getByText('変える幅の上限', { selector: 'dt.text-ink-2' }).nextSibling,
    ).toHaveTextContent('±10 %')
    expect(screen.getByText(/見込みの利益の和は 130/)).toBeInTheDocument()

    const elasticity = screen.getByRole('table', { name: /^弾力性の分布/ })
    expect(within(elasticity).getAllByRole('row')[1]).toHaveTextContent('予測した行 (2 行)')
    expect(within(elasticity).getAllByRole('row')[1]).toHaveTextContent('100.0 %')
    expect(screen.getByText(/学習用の分割の弾力性の平均/)).toHaveTextContent('-0.32 %')

    const evaluations = screen.getByRole('table', { name: /^実験の評価/ })
    const [, valid, test] = within(evaluations).getAllByRole('row')
    expect(valid).toHaveTextContent('27.7')
    expect(valid).toHaveTextContent('1.217 倍')
    expect(test).toHaveTextContent('無し (今の値段の利益の和が 0 以下)')
    expect(screen.getByText('渡した表の売れた数で数え直した誤差')).toBeInTheDocument()

    const list = await screen.findByRole('table', { name: /^選んだ値段/ })
    const [, first, second] = within(list).getAllByRole('row')
    expect(first).toHaveTextContent('r0007')
    expect(first).toHaveTextContent('p00')
    expect(first).toHaveTextContent('+10 %')
    expect(second).toHaveTextContent('10234')
    expect(second).not.toHaveTextContent('10,234')
    expect(prices).toHaveBeenCalledWith('r1', 0, 100)
    expect(screen.getByText('売れた数を予測した')).toBeInTheDocument()
    expect(screen.getByText('上げる 1 行, 下げる 0 行')).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: '結論の決め方を変える' })).not.toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: /意思決定/ })).not.toBeInTheDocument()
  })
})
