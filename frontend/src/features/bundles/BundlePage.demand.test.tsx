import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router'

import type { Bundle } from '../../api/types'
import { BundlePage } from './BundlePage'

// 需要の一式 (describe_bundle の値)
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
const demand = {
  id: 'd1',
  name: 'demand',
  path: '/bundles/demand',
  managed: false,
  added_at: '2026-01-01T00:00:00',
  ok: true,
  error: null,
  uses_problem: false,
  description: {
    path: '/bundles/demand',
    name: 'demand',
    created_at: null,
    kind: 'demand',
    task: '',
    target: null,
    classes: [],
    features: ['category', 'units_lag1', 'units_lag7', 'units_mean7', 'date_weekday'],
    text_column: null,
    reading: { loader: null, options: {} },
    steps: ['lag', 'datetime_parts', 'standardize'],
    item_steps: [],
    model: {
      name: null,
      inputs: {},
      output: null,
      kind: 'value',
    },
    run: null,
    conclusion: {},
    decision: { type: 'order', shortage_cost: 3, holding_cost: 1, max_order: 60 },
    check: null,
    notes: [],
    format_version: 14,
    role: 'predict',
    output: { form: 'distribution', quantiles: [0.1, 0.25, 0.5, 0.75, 0.9] },
    column_output: false,
    target_info: null,
    items: null,
    ready: null,
    rank: null,
    group: null,
    anomaly: null,
    uplift: null,
    price: null,
    demand: {
      features: ['category', 'units_lag1', 'units_lag7', 'units_mean7', 'date_weekday'],
      product: 'product_id',
      date: 'date',
      units: 'units',
      stock: 'stock',
      id: 'record_id',
      horizon: 7,
      interval: 7,
      levels: [0.1, 0.25, 0.5, 0.75, 0.9],
      shortage_cost: 3,
      holding_cost: 1,
      max_order: 60,
      rows: 36,
      train: { rows: 20, products: 4, demand_mean: 27.45 },
      evaluations: { valid: evaluation },
      check: { rows: 36, agreement: 1, max_diff: 0, pinball_loss: 12.7 },
    },
  },
} as unknown as Bundle

const bundleProblem = vi.fn()
const predict = vi.fn()

vi.mock('../../api/client', async () =>
  (await import('../../test/mockClient')).mockClient({
    api: {
      bundle: () => Promise.resolve(demand),
      results: () => Promise.resolve([]),
      bundleProblem: () => bundleProblem(),
      info: () => new Promise(() => {}),
      predict: (id: string, body: unknown) => predict(id, body),
    },
  }),
)

/** 一覧から開いた一式の画面を描く */
function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  client.setQueryData(['bundles'], [demand])
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/bundles/d1']}>
        <Routes>
          <Route path="/bundles/:id" element={<BundlePage />} />
          <Route path="/results/:id" element={<p>結果の画面</p>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('需要の一式の画面', () => {
  beforeEach(() => {
    bundleProblem.mockReset()
    predict.mockReset()
  })

  it('列, 先の期間と学んだ水準, 損と上限, 学習用の分割, 実験の評価を言葉の説明と出し, 問題の設定を読まない', async () => {
    show()
    expect(await screen.findByText(/需要の決め方と前処理のレシピ/)).toBeInTheDocument()
    expect(
      screen.getByText('category, units_lag1, units_lag7, units_mean7, date_weekday'),
    ).toBeInTheDocument()
    // 推論の欄が言葉の初めて出る所なので, 欄の上に言葉の説明と前提. 説明の行にも添える
    const glossary = screen.getByLabelText('言葉の説明')
    for (const term of ['需要', '分位点', '品切れの損', '売れ残りの損', '水準 τ', '発注の数の上限'])
      expect(within(glossary).getByText(term)).toBeInTheDocument()
    expect(glossary).toHaveTextContent('在庫の数の列 stock')
    expect(screen.getAllByText(/分位点は, 需要がその数以下になる見込み/)).toHaveLength(2)
    expect(screen.getAllByText(/1 度に発注してよい数の限度/)).toHaveLength(2)
    expect(screen.getByText('0.1, 0.25, 0.5, 0.75, 0.9')).toBeInTheDocument()
    expect(screen.getByText(/3 と 1 \(水準 τ 0.75\)/)).toBeInTheDocument()
    expect(screen.getByText(/100.0 % が実験の発注の数と合う/)).toBeInTheDocument()
    expect(screen.queryByText('モデルの入力')).not.toBeInTheDocument()
    expect(screen.queryByText('読み込み')).not.toBeInTheDocument()
    expect(screen.queryByText(/学習に使う列/)).not.toBeInTheDocument()
    const evaluations = screen.getByRole('table', { name: /^実験の評価/ })
    expect(within(evaluations).getAllByRole('row')[1]).toHaveTextContent('0.8 倍')

    // 推論の欄は記録の表のファイルと, 損と上限 (既定は一式の値). 結論の決め方と意思決定は出さない
    expect(screen.getByLabelText(/^発注の数を決める日ごとの記録の表/)).toBeVisible()
    expect(screen.getByLabelText(/^品切れの損/)).toHaveValue(3)
    expect(screen.getByLabelText(/^売れ残りの損/)).toHaveValue(1)
    expect(screen.getByLabelText(/^発注の数の上限/)).toHaveValue(60)
    expect(screen.getByText(/^水準 τ = 0.75/)).toBeInTheDocument()
    expect(screen.queryByRole('region', { name: '意思決定' })).not.toBeInTheDocument()
    expect(screen.queryByRole('region', { name: '結論の決め方' })).not.toBeInTheDocument()
    expect(screen.queryByText(/結論は推論した件数だけです/)).not.toBeInTheDocument()
    expect(bundleProblem).not.toHaveBeenCalled()
  })

  it('損と上限を確かめ, 記録の表と発注の判断を渡して推論する', async () => {
    predict.mockResolvedValue({ id: 'r1' })
    show()
    fireEvent.change(await screen.findByLabelText(/^発注の数を決める日ごとの記録の表/), {
      target: { value: '/data/daily.csv' },
    })
    const shortage = screen.getByLabelText(/^品切れの損/)
    const holding = screen.getByLabelText(/^売れ残りの損/)
    const limit = screen.getByLabelText(/^発注の数の上限/)
    fireEvent.change(holding, { target: { value: '0' } })
    expect(document.getElementById('predict-problem-0')).toHaveTextContent(
      '売れ残りの損は 0 より大きい数',
    )
    expect(screen.getByRole('button', { name: '推論する' })).toBeDisabled()
    fireEvent.change(holding, { target: { value: '1' } })
    fireEvent.change(limit, { target: { value: '2.5' } })
    expect(document.getElementById('predict-problem-0')).toHaveTextContent('0 以上の整数')
    expect(screen.getByRole('button', { name: '推論する' })).toBeDisabled()
    // 学んだ水準の外の τ (19 / 20 = 0.95) は端の水準を使うと知らせる
    fireEvent.change(shortage, { target: { value: '19' } })
    fireEvent.change(limit, { target: { value: '0' } })
    expect(screen.getByText(/^水準 τ = 0.95/)).toBeInTheDocument()
    expect(screen.getByText(/端の水準 0.9 の分位点を使います/)).toBeInTheDocument()
    expect(screen.getByText(/どの商品も発注しません/)).toBeInTheDocument()
    fireEvent.change(shortage, { target: { value: '9' } })
    fireEvent.change(limit, { target: { value: '25' } })
    expect(screen.queryByText(/端の水準/)).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '推論する' }))
    await waitFor(() =>
      expect(predict).toHaveBeenCalledWith('d1', {
        source: '/data/daily.csv',
        decision: { type: 'order', shortage_cost: 9, holding_cost: 1, max_order: 25 },
      }),
    )
    expect(await screen.findByText('結果の画面')).toBeInTheDocument()
  })
})
