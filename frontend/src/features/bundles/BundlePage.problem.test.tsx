import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router'

import type { Bundle, Description, ProblemState } from '../../api/types'
import { BundlePage } from './BundlePage'

/** データの種類だけを変えた一式 (表の分類か雑音だけ. 問題の設定の欄を出すかは API の値) */
function makeBundle(id: string, kind: string, usesProblem: boolean): Bundle {
  const description = {
    path: `/bundles/${id}`,
    name: id,
    created_at: null,
    kind,
    task: 'classification',
    target: 'label',
    classes: ['a', 'b'],
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
  return {
    id,
    name: id,
    path: `/bundles/${id}`,
    managed: true,
    added_at: '2026-01-01T00:00:00',
    ok: true,
    error: null,
    description,
    uses_problem: usesProblem,
  }
}

const problem: ProblemState = { path: null, linked: null, error: null, hint: null, hint_note: null }

// 一式の読み込みはテストで終える時を決める
const pending: Array<() => void> = []
const bundles: Record<string, Bundle> = {
  t1: makeBundle('t1', 'table', true),
  n1: makeBundle('n1', 'noise', false),
}
const bundleProblem = vi.fn((_id: string) => Promise.resolve(problem))

vi.mock('../../api/client', () => ({
  api: {
    bundle: (id: string) =>
      new Promise<Bundle>((resolve) => {
        pending.push(() => resolve(bundles[id] as Bundle))
      }),
    results: () => Promise.resolve([]),
    bundleProblem: (id: string) => bundleProblem(id),
    info: () => new Promise(() => {}),
  },
}))

/** 一式の画面を描く (一覧の読んだ値があれば先に置く) */
function renderPage(id: string, listed?: Bundle[]) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  if (listed) client.setQueryData(['bundles'], listed)
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[`/bundles/${id}`]}>
        <Routes>
          <Route path="/bundles/:id" element={<BundlePage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('一式の画面の問題の設定の読み込み', () => {
  beforeEach(() => {
    pending.length = 0
    bundleProblem.mockClear()
  })

  it('データのある一式は, 一式を読み終える前に問題の設定も読む', async () => {
    renderPage('t1')
    await waitFor(() => expect(bundleProblem).toHaveBeenCalledWith('t1'))
    expect(pending).toHaveLength(1)
    pending[0]?.()
    expect(await screen.findByRole('heading', { name: 't1' })).toBeInTheDocument()
  })

  it('一覧で問題の設定の欄を出さないと分かる一式は, 問題の設定を読まない', async () => {
    renderPage('n1', [bundles.n1 as Bundle])
    await waitFor(() => expect(pending).toHaveLength(1))
    pending[0]?.()
    expect(await screen.findByRole('heading', { name: 'n1' })).toBeInTheDocument()
    expect(bundleProblem).not.toHaveBeenCalled()
    expect(screen.queryByText('問題の設定を読んでいます…')).toBeNull()
  })
})
