import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router'

import type { Bundle } from '../../api/types'
import { BundlePage } from './BundlePage'

// 値段の一式 (describe_bundle の値)
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
const prices = {
  id: 'p1',
  name: 'prices',
  path: '/bundles/prices',
  managed: false,
  added_at: '2026-01-01T00:00:00',
  ok: true,
  error: null,
  description: {
    path: '/bundles/prices',
    name: 'prices',
    created_at: null,
    kind: 'price',
    task: '',
    target: null,
    classes: [],
    features: ['category', 'weekday', 'price'],
    text_column: null,
    reading: { loader: null, options: {} },
    steps: ['drop_columns', 'standardize'],
    item_steps: [],
    model: {
      name: null,
      inputs: { x: { source: 'features', prepend: [] } },
      output: null,
      kind: 'value',
    },
    run: null,
    conclusion: {},
    decision: { type: 'price', candidates: [-0.1, 0, 0.1, 0.2], max_change: 0.1 },
    check: null,
    notes: [],
    format_version: 13,
    role: 'effect',
    output: { form: 'effect' },
    column_output: false,
    target_info: null,
    items: null,
    ready: null,
    rank: null,
    group: null,
    anomaly: null,
    uplift: null,
    price: {
      features: ['category', 'weekday', 'price'],
      price: 'price',
      units: 'units',
      cost: 'unit_cost',
      product: 'product_id',
      id: 'record_id',
      candidates: [-0.1, 0, 0.1, 0.2],
      max_change: 0.1,
      elasticity_step: 0.01,
      current: 'median',
      rows: 14,
      train: { products: 6, changed: 6, elasticity: -0.32 },
      evaluations: { valid: evaluation },
      check: { rows: 6, agreement: 1, max_diff: 0 },
    },
  },
} as unknown as Bundle

const bundleProblem = vi.fn()
const predict = vi.fn()

vi.mock('../../api/client', () => ({
  api: {
    bundle: () => Promise.resolve(prices),
    results: () => Promise.resolve([]),
    bundleProblem: () => bundleProblem(),
    info: () => new Promise(() => {}),
    predict: (id: string, body: unknown) => predict(id, body),
  },
}))

/** 一覧から開いた一式の画面を描く */
function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  client.setQueryData(['bundles'], [prices])
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/bundles/p1']}>
        <Routes>
          <Route path="/bundles/:id" element={<BundlePage />} />
          <Route path="/results/:id" element={<p>結果の画面</p>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('値段の一式の画面', () => {
  beforeEach(() => {
    bundleProblem.mockReset()
    predict.mockReset()
  })

  it('列, 値段の候補と上限, 学習用の分割, 実験の評価を言葉の説明と出し, 問題の設定を読まない', async () => {
    show()
    expect(await screen.findByText(/値段の決め方と前処理のレシピ/)).toBeInTheDocument()
    expect(screen.getByText('category, weekday, price')).toBeInTheDocument()
    // 推論の欄が言葉の初めて出る所なので, 欄の上に言葉の説明と前提. 説明の行にも添える
    const glossary = screen.getByLabelText('言葉の説明')
    for (const term of ['値段', '売れた数', '原価', '利益 (見込み)', '弾力性', '変える割合'])
      expect(within(glossary).getByText(term)).toBeInTheDocument()
    expect(glossary).toHaveTextContent('原価の列 unit_cost')
    expect(glossary).toHaveTextContent('試しに値段を変えた記録を前提')
    expect(screen.getAllByText(/値段は商品 1 つあたりの売り値です/)).toHaveLength(2)
    expect(screen.getAllByText(/売れた数は, その値段で売れる商品の数/)).toHaveLength(2)
    expect(screen.getAllByText(/変える割合は今の値段に対する値段の変え方/)).toHaveLength(2)
    expect(screen.getAllByText(/選んでよい値上げと値下げの大きさの限度/)).toHaveLength(2)
    expect(screen.getAllByText(/値段を 1 % 上げたときに予測した売れた数/)).toHaveLength(2)
    expect(screen.getByText('-10 %, 0 %, +10 %, +20 %')).toBeInTheDocument()
    expect(screen.getByText('商品ごとの記録の値段の中央値')).toBeInTheDocument()
    expect(screen.getByText(/100.0 % が実験の選んだ割合と合う/)).toBeInTheDocument()
    expect(screen.queryByText('モデルの入力')).not.toBeInTheDocument()
    expect(screen.queryByText('読み込み')).not.toBeInTheDocument()
    expect(screen.queryByText(/学習に使う列/)).not.toBeInTheDocument()
    const evaluations = screen.getByRole('table', { name: /^実験の評価/ })
    expect(within(evaluations).getAllByRole('row')[1]).toHaveTextContent('1.217 倍')

    // 推論の欄は表のファイルと, 割合の小数の候補と上限 (既定は一式の値). 結論の決め方と意思決定は出さない
    expect(screen.getByLabelText(/^値段を選ぶ商品の表/)).toBeVisible()
    expect(screen.getByLabelText(/^値段の候補/)).toHaveValue('-0.1, 0, 0.1, 0.2')
    expect(screen.getByLabelText(/^変える幅の上限/)).toHaveValue(0.1)
    expect(screen.getByText(/^% では -10 %, 0 %, \+10 %, \+20 %/)).toBeInTheDocument()
    expect(screen.getByText(/^% では ±10 %/)).toBeInTheDocument()
    expect(screen.getByText(/上限を超える候補 \+20 % は選びません/)).toBeInTheDocument()
    expect(screen.queryByRole('region', { name: '意思決定' })).not.toBeInTheDocument()
    expect(screen.queryByRole('region', { name: '結論の決め方' })).not.toBeInTheDocument()
    expect(screen.queryByText(/結論は推論した件数だけです/)).not.toBeInTheDocument()
    expect(bundleProblem).not.toHaveBeenCalled()
  })

  it('候補と上限を確かめ, 行の表と値段の判断を渡して推論する', async () => {
    predict.mockResolvedValue({ id: 'r1' })
    show()
    fireEvent.change(await screen.findByLabelText(/^値段を選ぶ商品の表/), {
      target: { value: '/data/products.csv' },
    })
    const candidates = screen.getByLabelText(/^値段の候補/)
    const limit = screen.getByLabelText(/^変える幅の上限/)
    fireEvent.change(candidates, { target: { value: '0.1, 0.2' } })
    expect(screen.getByRole('alert')).toHaveTextContent('今の値段 (0) を含めて')
    expect(screen.getByText('一式の既定は -10 %, 0 %, +10 %, +20 %')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '推論する' })).toBeDisabled()
    fireEvent.change(candidates, { target: { value: '-0.05, 0, 0.05, 0.2' } })
    fireEvent.change(limit, { target: { value: '' } })
    expect(screen.getByRole('alert')).toHaveTextContent('省けません')
    expect(screen.getByText('一式の既定は ±10 % (省けません)')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '推論する' })).toBeDisabled()
    fireEvent.change(limit, { target: { value: '0.2' } })
    expect(screen.queryByText(/上限を超える候補 .* は選びません/)).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '推論する' }))
    await waitFor(() =>
      expect(predict).toHaveBeenCalledWith('p1', {
        source: '/data/products.csv',
        decision: { type: 'price', candidates: [-0.05, 0, 0.05, 0.2], max_change: 0.2 },
      }),
    )
    expect(await screen.findByText('結果の画面')).toBeInTheDocument()
  })
})
