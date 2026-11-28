import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router'

import type { Bundle } from '../../api/types'
import { BundlePage } from './BundlePage'

// クーポンの効果の一式 (describe_bundle の値)
const coupons = {
  id: 'u1',
  name: 'coupons',
  path: '/bundles/coupons',
  managed: false,
  added_at: '2026-01-01T00:00:00',
  ok: true,
  error: null,
  uses_problem: false,
  description: {
    path: '/bundles/coupons',
    name: 'coupons',
    created_at: null,
    kind: 'uplift',
    task: '',
    target: null,
    classes: [],
    features: ['age', 'visits'],
    text_column: null,
    reading: { loader: null, options: {} },
    steps: ['standardize'],
    item_steps: [],
    model: {
      name: null,
      inputs: { x: { source: 'features', prepend: [] } },
      output: null,
      kind: 'value',
    },
    run: null,
    conclusion: {},
    decision: { type: 'treat', budget: 8 },
    check: null,
    notes: [],
    format_version: 12,
    role: 'effect',
    output: { form: 'effect' },
    column_output: false,
    target_info: null,
    items: null,
    ready: null,
    rank: null,
    group: null,
    anomaly: null,
    uplift: {
      learner: 't_learner',
      outcome: 'binary',
      features: ['age', 'visits'],
      id: 'customer_id',
      treatment: 'coupon',
      budget: 8,
      k: 0.3,
      rows: 400,
      train: {
        rows: 200,
        mean: 0.36,
        min: -0.3,
        p10: -0.02,
        p50: 0.32,
        p90: 0.83,
        max: 0.97,
        positive: 0.89,
      },
      evaluations: {
        valid: {
          rows: 100,
          treated: 48,
          control: 52,
          k: 0.3,
          qini: 0.139,
          uplift_at_k: 0.402,
          effect: 0.327,
          estimated: 0.377,
        },
        test: {
          rows: 100,
          treated: 48,
          control: 52,
          k: 0.3,
          qini: 0.245,
          uplift_at_k: null,
          effect: 0.327,
          estimated: 0.3,
        },
      },
      curves: {
        valid: {
          rows: [0, 50, 100],
          fraction: [0, 0.5, 1],
          qini: [0, 20, 32],
          random: [0, 16, 32],
        },
      },
      check: { rows: 400, agreement: 1, max_diff: 0 },
    },
  },
} as unknown as Bundle

const bundleProblem = vi.fn()
const predict = vi.fn()

vi.mock('../../api/client', async () =>
  (await import('../../test/mockClient')).mockClient({
    api: {
      bundle: () => Promise.resolve(coupons),
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
  client.setQueryData(['bundles'], [coupons])
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/bundles/u1']}>
        <Routes>
          <Route path="/bundles/:id" element={<BundlePage />} />
          <Route path="/results/:id" element={<p>結果の画面</p>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('効果の一式の画面', () => {
  beforeEach(() => {
    bundleProblem.mockReset()
    predict.mockReset()
  })

  it('効果の決め方, 列, 配る数の上限, 学習用の行の効果, 実験の評価と曲線を出し, 問題の設定を読まない', async () => {
    show()
    expect(await screen.findByText(/効果の決め方と前処理のレシピ/)).toBeInTheDocument()
    expect(screen.getByText(/2 つの予測の差 \(T-learner\)/)).toBeInTheDocument()
    expect(
      screen.getAllByText(/処置したとき \(クーポンを配るなど\) の結果の予測/).length,
    ).toBeGreaterThan(0)
    expect(screen.getAllByText(/処置の群は学習の記録で処置した行/).length).toBeGreaterThan(0)
    expect(screen.getByText('age, visits')).toBeInTheDocument()
    expect(screen.getByText('coupon')).toBeInTheDocument()
    expect(screen.getByText(/8 件 \(実験の評価は上位 30 %\)/)).toBeInTheDocument()
    expect(screen.getByText(/効果が正の行 89.0 %/)).toBeInTheDocument()
    expect(screen.getByText(/100.0 % が実験の効果と合う/)).toBeInTheDocument()
    expect(screen.queryByText('モデルの入力')).not.toBeInTheDocument()
    expect(screen.queryByText('読み込み')).not.toBeInTheDocument()
    expect(screen.queryByText(/学習に使う列/)).not.toBeInTheDocument()
    const evaluations = screen.getByRole('table', {
      name: '実験の評価 (分割ごとの Qini の係数と上位 30 % の効果の差)',
    })
    const [, valid, test] = within(evaluations).getAllByRole('row')
    expect(valid).toHaveTextContent('検証用')
    expect(valid).toHaveTextContent('0.139')
    expect(valid).toHaveTextContent('0.402')
    expect(test).toHaveTextContent('無し (上位に片方の群が無い)')
    expect(screen.getByRole('img', { name: /^検証用の Qini の曲線/ })).toBeInTheDocument()
    expect(screen.getByText('でたらめに配る線')).toBeInTheDocument()

    // 推論の欄は表のファイルと配る数の上限だけで, 結論の決め方と意思決定は出さない
    expect(screen.getByLabelText(/^効果を見積もる行の表/)).toBeVisible()
    expect(screen.getByLabelText(/^配る数の上限/)).toHaveValue(8)
    expect(screen.queryByRole('region', { name: '意思決定' })).not.toBeInTheDocument()
    expect(screen.queryByRole('region', { name: '結論の決め方' })).not.toBeInTheDocument()
    expect(screen.queryByText(/結論は推論した件数だけです/)).not.toBeInTheDocument()
    expect(bundleProblem).not.toHaveBeenCalled()
  })

  it('配る数の上限を確かめ, 行の表と配る判断を渡して推論する', async () => {
    predict.mockResolvedValue({ id: 'r1' })
    show()
    fireEvent.change(await screen.findByLabelText(/^効果を見積もる行の表/), {
      target: { value: '/data/customers.csv' },
    })
    const budget = screen.getByLabelText(/^配る数の上限/)
    fireEvent.change(budget, { target: { value: '0' } })
    expect(budget).toHaveAccessibleDescription(expect.stringContaining('1 から 100,000 の整数'))
    expect(budget).toHaveAttribute('aria-invalid', 'true')
    fireEvent.change(budget, { target: { value: '100001' } })
    expect(budget).toHaveAccessibleDescription(expect.stringContaining('1 から 100,000 の整数'))
    expect(budget).toHaveAttribute('aria-invalid', 'true')
    expect(screen.getByRole('button', { name: '推論する' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '推論する' })).toBeDisabled()
    fireEvent.change(budget, { target: { value: '3' } })
    fireEvent.click(screen.getByRole('button', { name: '推論する' }))
    await waitFor(() =>
      expect(predict).toHaveBeenCalledWith('u1', {
        source: '/data/customers.csv',
        decision: { type: 'treat', budget: 3 },
      }),
    )
    expect(await screen.findByText('結果の画面')).toBeInTheDocument()
  })
})
