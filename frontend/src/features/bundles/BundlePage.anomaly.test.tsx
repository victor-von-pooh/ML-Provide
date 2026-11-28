import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router'

import type { Bundle } from '../../api/types'
import { BundlePage } from './BundlePage'

// 距離に基づく異常の度合いの一式 (describe_bundle の値)
const sensors = {
  id: 'a1',
  name: 'sensors',
  path: '/bundles/sensors',
  managed: false,
  added_at: '2026-01-01T00:00:00',
  ok: true,
  error: null,
  uses_problem: false,
  description: {
    path: '/bundles/sensors',
    name: 'sensors',
    created_at: null,
    kind: 'anomaly',
    task: '',
    target: null,
    classes: [],
    features: ['temp', 'vibration', 'current'],
    text_column: null,
    reading: { loader: null, options: {} },
    steps: ['standardize'],
    item_steps: [],
    model: { name: null, inputs: {}, output: null, kind: 'value' },
    run: null,
    conclusion: {},
    decision: { type: 'triage', capacity: 3 },
    check: null,
    notes: [],
    format_version: 11,
    role: 'predict',
    output: { form: 'value' },
    column_output: false,
    target_info: null,
    items: null,
    ready: null,
    rank: null,
    group: null,
    anomaly: {
      score: 'distance',
      features: ['temp', 'vibration', 'current'],
      id: 'reading_id',
      capacity: 3,
      k: 3,
      rows: 64,
      normal: { rows: 36, mean: 0.9, p50: 0.82, p90: 1.8, p95: 2.1, p99: 3.39, max: 3.8 },
      evaluations: {
        valid: { rows: 13, anomalies: 1, k: 3, precision_at_k: 0.333, average_precision: 0.9 },
      },
      check: { rows: 64, agreement: 1, max_diff: 0 },
    },
  },
} as unknown as Bundle

const bundleProblem = vi.fn()
const predict = vi.fn()

vi.mock('../../api/client', async () =>
  (await import('../../test/mockClient')).mockClient({
    api: {
      bundle: () => Promise.resolve(sensors),
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
  client.setQueryData(['bundles'], [sensors])
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/bundles/a1']}>
        <Routes>
          <Route path="/bundles/:id" element={<BundlePage />} />
          <Route path="/results/:id" element={<p>結果の画面</p>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('異常の度合いの一式の画面', () => {
  beforeEach(() => {
    bundleProblem.mockReset()
    predict.mockReset()
  })

  it('度合いの決め方, 列, 件数の上限, 正常の行の分布, 実験の評価を出し, 問題の設定を読まない', async () => {
    show()
    expect(await screen.findByText(/度合いの決め方と前処理のレシピ/)).toBeInTheDocument()
    expect(screen.getByText('学習用の正常の行の中心からの距離')).toBeInTheDocument()
    expect(screen.getByText('temp, vibration, current')).toBeInTheDocument()
    expect(screen.getByText('reading_id')).toBeInTheDocument()
    expect(screen.getByText('standardize')).toBeInTheDocument()
    expect(screen.getByText(/1 回に 3 件/)).toBeInTheDocument()
    expect(screen.getByText(/99 % 点 3.39/)).toBeInTheDocument()
    expect(screen.getByText(/100.0 % が実験の度合いと合う/)).toBeInTheDocument()
    expect(screen.queryByText('モデルの入力')).not.toBeInTheDocument()
    expect(screen.queryByText('読み込み')).not.toBeInTheDocument()
    expect(screen.queryByText(/学習に使う列/)).not.toBeInTheDocument()
    const evaluations = screen.getByRole('table', {
      name: '実験の評価 (分割ごとの上位 K 件の適合率と PR-AUC)',
    })
    expect(within(evaluations).getAllByRole('row')[1]).toHaveTextContent('検証用')
    expect(within(evaluations).getAllByRole('row')[1]).toHaveTextContent('0.333')
    expect(within(evaluations).getAllByRole('row')[1]).toHaveTextContent('0.900')

    // 推論の欄は表のファイルと件数の上限だけで, 結論の決め方と意思決定は出さない
    expect(screen.getByLabelText(/^度合いを出す行の表/)).toBeVisible()
    expect(screen.getByLabelText(/^人の確かめに回す件数の上限/)).toHaveValue(3)
    expect(screen.queryByRole('region', { name: '意思決定' })).not.toBeInTheDocument()
    expect(screen.queryByRole('region', { name: '結論の決め方' })).not.toBeInTheDocument()
    expect(screen.queryByText(/結論は推論した件数だけです/)).not.toBeInTheDocument()
    expect(bundleProblem).not.toHaveBeenCalled()
  })

  it('件数の上限を確かめ, 行の表と振り分けを渡して推論する', async () => {
    predict.mockResolvedValue({ id: 'r1' })
    show()
    fireEvent.change(await screen.findByLabelText(/^度合いを出す行の表/), {
      target: { value: '/data/readings.csv' },
    })
    const capacity = screen.getByLabelText(/^人の確かめに回す件数の上限/)
    fireEvent.change(capacity, { target: { value: '0' } })
    expect(screen.getByRole('alert')).toHaveTextContent('1 から 1000 の整数')
    expect(screen.getByRole('button', { name: '推論する' })).toBeDisabled()
    fireEvent.change(capacity, { target: { value: '5' } })
    fireEvent.click(screen.getByRole('button', { name: '推論する' }))
    await waitFor(() =>
      expect(predict).toHaveBeenCalledWith('a1', {
        source: '/data/readings.csv',
        decision: { type: 'triage', capacity: 5 },
      }),
    )
    expect(await screen.findByText('結果の画面')).toBeInTheDocument()
  })
})
