import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router'

import type { Result, ReviewQueue, ReviewQueueRow, ReviewRecord, Rows } from '../../api/types'
import { ResultPage } from './ResultPage'

// 4 行のうち 3 行の度合いを出し, 上位 2 行を人の確認に回した結果
const normal = { rows: 36, mean: 0.9, p50: 0.82, p90: 1.8, p95: 2.1, p99: 3.39, max: 3.8 }
const result = {
  id: 'r1',
  bundle: { name: 'sensors', path: '/bundles/sensors' },
  bundle_id: 'a1',
  source: '/data/readings.csv',
  created_at: '2026-01-01T00:00:00',
  task: '',
  classes: [],
  rules: {},
  summary: {
    rows: 4,
    predicted: 3,
    skipped: 1,
    score: 'distance',
    id: 'reading_id',
    distribution: { p50: 0.69, p90: 725, p99: 938, max: 943 },
    normal,
    beyond_normal: 2,
    top: [
      { row: 3, id: 'r013', score: 942.5, column: 'vibration', column_share: 0.9 },
      { row: 2, id: 'r012', score: 904.9, column: 'vibration', column_share: 0.8 },
    ],
    evaluations: {},
  },
  sentences: [
    '4 行のうち 3 行の異常の度合い (学習用の正常の行の中心からの距離) を出した (1 行は出せず)',
  ],
  overall_count: 1,
  columns: ['reading_id', 'temp', 'status', 'reason', 'score', 'rank', 'column', 'column_share'],
  kind: 'anomaly',
  role: 'predict',
  output_form: 'value',
  column_output: false,
  artifacts: null,
  decision: {
    decision: { type: 'triage', capacity: 2 },
    summary: { rows: 3, acted: 2, capacity: 2, lowest: 904.9, normal_p99: 3.39, left_beyond: 0 },
    sentences: [
      '度合いの高い順に 2 件を人の確認に回す (人が 1 回に確かめる件数の上限 2 件). 回す行の度合いは 905 以上',
    ],
    notes: ['度合いは確率ではないので, 回す件数は損得のしきい値でなく人手の上限で決めた'],
  },
} as unknown as Result

const rows: Rows = { total: 0, offset: 0, columns: ['index'], rows: [] }

let records: ReviewRecord[] = []
// 待ち行列を差し替える値 (null は記録から作る待ち行列)
let queueOverride: ReviewQueue | null = null
const queueRows = (): ReviewQueue => ({
  id: 'reading_id',
  capacity: 2,
  rows: [
    {
      row: 3,
      key: 'r013',
      id: 'r013',
      score: 942.5,
      rank: 1,
      column: 'vibration',
      column_share: 0.9,
    },
    {
      row: 2,
      key: 'r012',
      id: 'r012',
      score: 904.9,
      rank: 2,
      column: 'vibration',
      column_share: 0.8,
    },
  ].map((row) => {
    const last = records.filter((record) => record.id === row.key).at(-1)
    return {
      ...row,
      judgment: (last?.judgment as 'anomaly' | 'normal' | undefined) ?? null,
      reviewer: last?.reviewer ?? null,
      reviewed_at: last?.reviewed_at ?? null,
    }
  }),
})
const addReview = vi.fn((_id: string, row: string | number, judgment: string, reviewer: string) => {
  const record = {
    id: String(row),
    judgment,
    reviewer,
    reviewed_at: '2026-01-02T09:00:00+09:00',
  }
  records = [...records, record]
  return Promise.resolve(record)
})

vi.mock('../../api/client', () => ({
  api: {
    result: () => Promise.resolve(result),
    bundle: () => Promise.reject(new Error('無い')),
    rows: () => Promise.resolve(rows),
    queue: () => Promise.resolve(queueOverride ?? queueRows()),
    reviews: () => Promise.resolve({ records }),
    addReview: (id: string, row: string | number, judgment: string, reviewer: string) =>
      addReview(id, row, judgment, reviewer),
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

describe('異常の度合いの結果の画面', () => {
  beforeEach(() => {
    records = []
    queueOverride = null
    addReview.mockClear()
    window.localStorage.clear()
  })

  it('度合いの分布, 実験の評価, 上位の行, 振り分けを出し, 列の出力だけの欄を出さない', async () => {
    renderPage()
    const distribution = await screen.findByRole('table', {
      name: '度合いの分布 (新しい行と学習用の正常の行の中央, 90 % 点, 99 % 点, 最大)',
    })
    const [, found, learned] = within(distribution).getAllByRole('row')
    expect(found).toHaveTextContent('新しい行 (3 行)')
    expect(found).toHaveTextContent('943')
    expect(learned).toHaveTextContent('3.39')
    expect(screen.getByText('2 行')).toBeInTheDocument()
    expect(screen.getByText(/異常の正解の無い実験なので/)).toBeInTheDocument()
    const top = screen.getByRole('table', {
      name: '度合いの上位の行 (順位, 行の位置, 行の id, 度合い, 最も外れた列)',
    })
    expect(within(top).getAllByRole('row')[1]).toHaveTextContent('r013')
    expect(within(top).getAllByRole('row')[1]).toHaveTextContent('90.0%')
    expect(screen.getByText(/度合いの高い順に 2 件を人の確認に回す/)).toBeInTheDocument()
    expect(screen.getByText('人の確認に回す')).toBeInTheDocument()
    expect(screen.getByText('度合いを出した')).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: '結論の決め方を変える' })).not.toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: /意思決定/ })).not.toBeInTheDocument()
  })

  it('名前を書いてから 1 件ずつ判断を付け, 待ち行列と記録を読み直す', async () => {
    renderPage()
    const queue = await screen.findByRole('table', { name: /^人の確認に回した行/ })
    expect(within(queue).getAllByRole('row')).toHaveLength(3)
    expect(within(queue).getAllByText('未確認')).toHaveLength(2)
    expect(await screen.findByText('まだ記録はありません')).toBeInTheDocument()

    // 名前が空の間は押せない
    const first = screen.getByRole('button', { name: 'reading_id r013 を異常とする' })
    expect(first).toBeDisabled()
    fireEvent.change(screen.getByLabelText(/^確かめた人の名前/), { target: { value: ' 佐藤 ' } })
    expect(first).toBeEnabled()
    fireEvent.click(first)
    await waitFor(() => expect(addReview).toHaveBeenCalledWith('r1', 'r013', 'anomaly', '佐藤'))

    // 今の判断と記録 (名前はブラウザに覚える)
    await waitFor(() =>
      expect(
        within(screen.getByRole('table', { name: /^人の確認に回した行/ })).getAllByRole('row')[1],
      ).toHaveTextContent('異常佐藤'),
    )
    const log = await screen.findByRole('table', { name: /^確認の記録/ })
    expect(within(log).getAllByRole('row')).toHaveLength(2)
    expect(within(log).getAllByRole('row')[1]).toHaveTextContent('佐藤')
    expect(window.localStorage.getItem('ml-provide:reviewer')).toBe('佐藤')

    // 同じ行を正常に付け直すと, 最後の記録が今の判断
    fireEvent.click(screen.getByRole('button', { name: 'reading_id r013 を正常とする' }))
    await waitFor(() => expect(addReview).toHaveBeenCalledTimes(2))
    await waitFor(() =>
      expect(
        within(screen.getByRole('table', { name: /^確認の記録/ })).getAllByRole('row'),
      ).toHaveLength(3),
    )
    expect(screen.getByText(/2 件のうち 1 件を確かめました/)).toBeInTheDocument()
  })

  it('行の id の重なる結果は, 名前の欄と待ち行列を出さずに理由を出す', async () => {
    queueOverride = {
      id: 'reading_id',
      capacity: 2,
      problem: '行の id の列 reading_id に同じ値の行があります (r013)',
      rows: [],
    }
    renderPage()
    expect(await screen.findByRole('note', { name: '人の確認を使えない理由' })).toHaveTextContent(
      'r013',
    )
    expect(screen.queryByLabelText(/^確かめた人の名前/)).not.toBeInTheDocument()
    expect(screen.queryByRole('table', { name: /^人の確認に回した行/ })).not.toBeInTheDocument()
  })

  it('行の id の欠けた行は判断を付けられない', async () => {
    const base = queueRows()
    const [top, next] = base.rows as [ReviewQueueRow, ReviewQueueRow]
    queueOverride = { ...base, rows: [{ ...top, key: null, id: null }, next] }
    renderPage()
    fireEvent.change(await screen.findByLabelText(/^確かめた人の名前/), {
      target: { value: '佐藤' },
    })
    const queue = await screen.findByRole('table', { name: /^人の確認に回した行/ })
    const lines = within(queue).getAllByRole('row')
    const blank = lines[1]!
    const other = lines[2]!
    expect(within(blank).getByText(/行の id が欠けているので記録できません/)).toBeInTheDocument()
    for (const button of within(blank).getAllByRole('button')) expect(button).toBeDisabled()
    for (const button of within(other).getAllByRole('button')) expect(button).toBeEnabled()
  })
})
