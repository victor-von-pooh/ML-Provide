import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router'

import type { Result, Rows, TreatedRows } from '../../api/types'
import { ResultPage } from './ResultPage'

// 4 行のうち 3 行の効果を見積もり, 上位 2 行に配った結果
const train = {
  rows: 200,
  mean: 0.36,
  min: -0.3,
  p10: -0.02,
  p50: 0.32,
  p90: 0.83,
  max: 0.97,
  positive: 0.89,
}
const result = {
  id: 'r1',
  bundle: { name: 'coupons', path: '/bundles/coupons' },
  bundle_id: 'u1',
  source: '/data/customers.csv',
  created_at: '2026-01-01T00:00:00',
  task: '',
  classes: [],
  rules: {},
  summary: {
    rows: 4,
    predicted: 3,
    skipped: 1,
    outcome: 'binary',
    id: 'customer_id',
    distribution: {
      mean: 0.4,
      min: -0.1,
      p10: 0.0,
      p50: 0.5,
      p90: 0.7,
      max: 0.75,
      positive: 0.667,
    },
    train,
    top_k: { k: 0.3, rows: 1, mean: 0.75, lowest: 0.75 },
    top: [
      { row: 2, id: 'c12', effect: 0.75, treated: 0.9, control: 0.15 },
      { row: 0, id: 'c10', effect: 0.5, treated: 0.7, control: 0.2 },
      { row: 3, id: 'c13', effect: -0.1, treated: 0.1, control: 0.2 },
    ],
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
    },
    measured: {
      rows: 3,
      treated: 2,
      control: 1,
      excluded: 0,
      k: 0.3,
      qini: 0.5,
      uplift_at_k: null,
      effect: 0.2,
    },
  },
  sentences: [
    '4 行のうち 3 行の効果 (処置したときの結果 − しないときの結果. 確率の差) を見積もった (1 行は出せず)',
  ],
  overall_count: 1,
  columns: [
    'customer_id',
    'status',
    'reason',
    'effect',
    'treated',
    'control',
    'rank',
    'decision',
    'action',
  ],
  kind: 'uplift',
  role: 'effect',
  output_form: 'effect',
  column_output: false,
  artifacts: null,
  uplift: {
    outcome: 'binary',
    k: 0.3,
    curves: {
      valid: { rows: [0, 50, 100], fraction: [0, 0.5, 1], qini: [0, 20, 32], random: [0, 16, 32] },
    },
  },
  decision: {
    decision: { type: 'treat', budget: 2 },
    summary: {
      rows: 3,
      acted: 2,
      budget: 2,
      lowest: 0.5,
      gain: 1.25,
      left_positive: 0,
      nonpositive: 0,
    },
    sentences: [
      '効果は処置したときとしないときの 2 つの予測の差の見積もりで, 処置を無作為に配った記録を前提にしている',
      '効果の高い順に 2 件に配る (配る数の上限 2 件). 配る行の効果は 0.5 以上で, 効果の和は 1.25 (見積もりで増える結果)',
    ],
    notes: [
      '効果は処置したときとしないときの 2 つの予測の差の見積もりで, 処置を無作為に配った記録を前提にしている',
    ],
  },
} as unknown as Result

const rows: Rows = { total: 0, offset: 0, columns: ['index'], rows: [] }
const treated = vi.fn((_id: string, offset: number, _limit: number): Promise<TreatedRows> =>
  Promise.resolve({
    id: 'customer_id',
    budget: 2,
    total: 2,
    offset,
    rows: [
      { row: 2, id: 'c12', rank: 1, effect: 0.75, treated: 0.9, control: 0.15 },
      { row: 0, id: 'c10', rank: 2, effect: 0.5, treated: 0.7, control: 0.2 },
    ],
  }),
)

vi.mock('../../api/client', () => ({
  api: {
    result: () => Promise.resolve(result),
    bundle: () => Promise.reject(new Error('無い')),
    rows: () => Promise.resolve(rows),
    treated: (id: string, offset: number, limit: number) => treated(id, offset, limit),
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

describe('効果の結果の画面', () => {
  beforeEach(() => treated.mockClear())

  it('言葉の説明, 効果の分布, 実験の評価と曲線, 上位の行, 配る行を出し, 列の出力だけの欄を出さない', async () => {
    renderPage()
    // 結論に初めて出る言葉の説明
    const glossary = await screen.findByLabelText('言葉の説明')
    expect(glossary).toHaveTextContent(/効果は, 処置したとき .* \(確率の差\)/)
    expect(glossary).toHaveTextContent('処置の群は学習の記録で処置した行')
    expect(glossary).toHaveTextContent('Qini の曲線は')
    expect(glossary).toHaveTextContent('上位 30 % の効果の差')
    expect(glossary).toHaveTextContent('無作為に配った記録を前提にした見積もり')

    const distribution = screen.getByRole('table', { name: /^効果の分布/ })
    const [, found, learned] = within(distribution).getAllByRole('row')
    expect(found).toHaveTextContent('新しい行 (3 行)')
    expect(found).toHaveTextContent('66.7 %')
    expect(learned).toHaveTextContent('学習用の行 (200 行)')
    expect(screen.getByText(/^効果の高い上位 30 % \(1 行\):/)).toBeInTheDocument()

    const evaluations = screen.getByRole('table', { name: /^実験の評価/ })
    expect(within(evaluations).getAllByRole('row')[1]).toHaveTextContent('0.139')
    expect(screen.getByRole('img', { name: /^検証用の Qini の曲線/ })).toBeInTheDocument()
    expect(screen.getByText('渡した表の処置と結果で数え直した評価')).toBeInTheDocument()

    const top = screen.getByRole('table', { name: /^効果の上位の行/ })
    expect(within(top).getAllByRole('row')).toHaveLength(4)
    expect(within(top).getAllByRole('row')[1]).toHaveTextContent('c12')

    const list = await screen.findByRole('table', { name: /^配る行/ })
    expect(within(list).getAllByRole('row')).toHaveLength(3)
    expect(within(list).getAllByRole('row')[1]).toHaveTextContent('c12')
    expect(treated).toHaveBeenCalledWith('r1', 0, 100)
    expect(screen.getByText(/効果の高い順に 2 件に配る/)).toBeInTheDocument()
    expect(screen.getByText('効果を見積もった')).toBeInTheDocument()
    expect(screen.getByText('上限 2 件')).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: '結論の決め方を変える' })).not.toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: /意思決定/ })).not.toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: '人の確認' })).not.toBeInTheDocument()
  })

  it('曲線の図は矢印のキーで点を選ぶと値を出し, 表でも読める', async () => {
    renderPage()
    const chart = await screen.findByRole('img', { name: /^検証用の Qini の曲線/ })
    fireEvent.keyDown(chart, { key: 'ArrowRight' })
    fireEvent.keyDown(chart, { key: 'ArrowRight' })
    await waitFor(() => expect(screen.getByText(/上位 50 % \(50 行\)/)).toBeInTheDocument())
    const values = screen.getByRole('table', { name: '検証用の Qini の曲線の値', hidden: true })
    expect(within(values).getAllByRole('row', { hidden: true })).toHaveLength(4)
  })
})
