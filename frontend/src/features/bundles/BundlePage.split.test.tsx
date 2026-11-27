import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router'

import type { Bundle, Description, ProblemState, Result, TestSplit } from '../../api/types'
import { BundlePage } from './BundlePage'

// 実験から登録した表の分類の一式 (describe_bundle の値)
const description = {
  path: '/project/outputs/churn/2026-11-28/10-00-00/inference',
  name: '解約',
  created_at: null,
  kind: 'table',
  task: 'classification',
  target: 'label',
  classes: ['続く', '解約'],
  features: [],
  text_column: null,
  reading: { loader: 'csv', options: {} },
  steps: [],
  item_steps: [],
  model: { name: 'm', inputs: {}, output: null, kind: 'logits' },
  run: null,
  conclusion: {},
  check: null,
  notes: [],
  format_version: 3,
  role: 'predict',
  output: { form: 'class' },
  column_output: true,
  target_info: { kind: 'column' },
  items: null,
} as unknown as Description

const bundle: Bundle = {
  id: 'b1',
  name: '解約',
  path: description.path,
  managed: false,
  added_at: '2026-01-01T00:00:00',
  ok: true,
  error: null,
  description,
  uses_problem: false,
}

const problem: ProblemState = {
  path: null,
  linked: null,
  error: null,
  hint: null,
  hint_note: null,
  problem_id: null,
  hint_project: false,
}

const found: TestSplit = {
  split: 'test',
  dataset: '/project/data/dataset',
  rows: 30,
  experiment: '/project/outputs/churn/2026-11-28/10-00-00',
  reason: null,
}

// 返す分割 (テストごとに差し替える) と, 推論に送った中身
let shownSplit: TestSplit = found
const predict = vi.fn((_id: string, _body: Record<string, unknown>) =>
  Promise.resolve({ id: 'r1' } as Result),
)

vi.mock('../../api/client', async () =>
  (await import('../../test/mockClient')).mockClient({
    api: {
      bundle: () => Promise.resolve(bundle),
      results: () => Promise.resolve([]),
      bundleProblem: () => Promise.resolve(problem),
      info: () => new Promise(() => {}),
      testSplit: () => Promise.resolve(shownSplit),
      predict: (id: string, body: Record<string, unknown>) => predict(id, body),
    },
  }),
)

/** 一式の画面を描く関数 */
function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/bundles/b1']}>
        <Routes>
          <Route path="/bundles/:id" element={<BundlePage />} />
          <Route path="/results/:id" element={<p>結果の画面</p>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('元のデータセットのテスト用の分割', () => {
  beforeEach(() => {
    shownSplit = found
    predict.mockClear()
  })

  it('選ぶと場所の欄の代わりに分割を出し, 名前だけを送って推論する', async () => {
    show()
    const choice = await screen.findByRole('radio', { name: '元のデータセットのテスト用の分割' })
    fireEvent.click(choice)
    const section = screen.getByRole('region', { name: 'テスト用の分割' })
    expect(section).toHaveTextContent('(30 行)')
    expect(section).toHaveTextContent('/project/data/dataset')
    expect(screen.queryByLabelText(/推論するデータ \(/)).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: '推論する' }))
    expect(await screen.findByText('結果の画面')).toBeInTheDocument()
    expect(predict).toHaveBeenCalledTimes(1)
    const body = predict.mock.calls[0]?.[1] ?? {}
    expect(body.split).toBe('test')
    expect(body).not.toHaveProperty('source')
    expect(body).not.toHaveProperty('dataset')
  })

  it('見つからない実験の一式は, 選び方を出さずに理由を添える', async () => {
    shownSplit = {
      ...found,
      split: null,
      dataset: null,
      rows: null,
      reason: '学習のデータセットにテスト用の分割がありません',
    }
    show()
    expect(
      await screen.findByText(/テスト用の分割は選べません: 学習のデータセットにテスト用の分割/),
    ).toBeInTheDocument()
    expect(screen.queryByRole('radio', { name: '元のデータセットのテスト用の分割' })).toBeNull()
    expect(screen.getByLabelText(/推論するデータ \(/)).toBeInTheDocument()
  })
})
