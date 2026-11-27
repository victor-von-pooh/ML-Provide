import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router'

import type { Bundle } from '../../api/types'
import { BundlePage } from './BundlePage'

// 利用者 4 人と品物 6 件の並べるの一式 (describe_bundle の値)
const shop: Bundle = {
  id: 's1',
  name: 'shop',
  path: '/bundles/shop',
  managed: false,
  added_at: '2026-01-01T00:00:00',
  ok: true,
  error: null,
  uses_problem: false,
  description: {
    path: '/bundles/shop',
    name: 'shop',
    created_at: null,
    kind: 'rank',
    task: '',
    target: null,
    classes: [],
    features: [],
    text_column: null,
    reading: { loader: null, options: {} },
    steps: [],
    item_steps: [],
    model: {
      name: 'two_tower_retrieval',
      inputs: {
        user: { source: 'items', prepend: [] },
        item: { source: 'items', prepend: [] },
      },
      output: null,
      kind: 'value',
    },
    run: { path: '/outputs/shop/run', status: 'completed' },
    conclusion: {},
    decision: null,
    check: null,
    notes: [],
    format_version: 8,
    role: 'rank',
    output: { form: 'ranking' },
    column_output: false,
    target_info: null,
    items: null,
    ready: null,
    rank: {
      users: 4,
      items: 6,
      width: 16,
      scale: 20,
      k: 3,
      columns: { user: 'member', item: 'item_id', reaction: 'qty' },
    },
  },
}

const bundleProblem = vi.fn()
const predict = vi.fn()

vi.mock('../../api/client', async () =>
  (await import('../../test/mockClient')).mockClient({
    api: {
      bundle: () => Promise.resolve(shop),
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
  client.setQueryData(['bundles'], [shop])
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/bundles/s1']}>
        <Routes>
          <Route path="/bundles/:id" element={<BundlePage />} />
          <Route path="/results/:id" element={<p>結果の画面</p>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('並べるの一式の画面', () => {
  beforeEach(() => {
    bundleProblem.mockReset()
    predict.mockReset()
  })

  it('番号の表と件数の既定と反応の表の列を出し, 結論の決め方と意思決定の欄を出さない', async () => {
    show()
    expect(await screen.findByText(/利用者 4 人と品物 6 件/)).toBeInTheDocument()
    expect(screen.getByText('3 件 (問題の設定の件数)')).toBeInTheDocument()
    expect(screen.getByText('利用者 member, 品物 item_id, 反応の強さ qty')).toBeInTheDocument()
    expect(screen.getByText(/qty が 0 より大きい行を反応とします/)).toBeInTheDocument()
    expect(screen.queryByText('モデルの入力')).not.toBeInTheDocument()
    expect(screen.queryByText('読み込み')).not.toBeInTheDocument()
    // 件数の既定は一式の rank.k
    expect(screen.getByLabelText(/^並べる件数 K/)).toHaveValue(3)
    expect(screen.getByText(/数や種類を予測する一式で使うため, ここでは使えません/)).toBeVisible()
    expect(screen.getByText(/問題の設定の欄はありません/)).toBeVisible()
    expect(screen.queryByRole('region', { name: '意思決定' })).not.toBeInTheDocument()
    expect(screen.queryByRole('region', { name: '結論の決め方' })).not.toBeInTheDocument()
    // 問題の設定は読まない
    expect(bundleProblem).not.toHaveBeenCalled()
  })

  it('書いた利用者, 件数, 前に反応した品物の表, 正解の反応の表を渡して並べる', async () => {
    predict.mockResolvedValue({ id: 'r1' })
    show()
    const submit = await screen.findByRole('button', { name: '並べる' })
    expect(submit).toBeDisabled()
    fireEvent.change(screen.getByRole('textbox', { name: /^並べる利用者/ }), {
      target: { value: 'u0\n\n 007 \n' },
    })
    expect(submit).toBeEnabled()

    // 件数の誤りと, 除くのに表が無い誤り
    fireEvent.change(screen.getByLabelText(/^並べる件数 K/), { target: { value: '0' } })
    expect(screen.getByRole('alert')).toHaveTextContent('並べる件数は 1 から 1,000 の整数')
    fireEvent.change(screen.getByLabelText(/^並べる件数 K/), { target: { value: '2' } })
    fireEvent.click(screen.getByLabelText('前に反応した品物を除く'))
    expect(screen.getByRole('alert')).toHaveTextContent('前に反応した品物の表を選んでください')
    expect(submit).toBeDisabled()
    fireEvent.change(screen.getByLabelText(/^前に反応した品物の表/), {
      target: { value: '/data/seen.csv' },
    })
    fireEvent.change(screen.getByLabelText(/^正解の反応の表/), {
      target: { value: '/data/later.csv' },
    })
    fireEvent.click(submit)
    await waitFor(() =>
      expect(predict).toHaveBeenCalledWith('s1', {
        users: ['u0', '007'],
        k: 2,
        seen: '/data/seen.csv',
        truth: '/data/later.csv',
      }),
    )
    expect(await screen.findByText('結果の画面')).toBeInTheDocument()
  })

  it('利用者の表のファイルで並べ, 除かなければ前の反応の表を送らない', async () => {
    predict.mockResolvedValue({ id: 'r2' })
    show()
    fireEvent.click(await screen.findByLabelText('利用者の表のファイル'))
    fireEvent.change(screen.getByLabelText(/^利用者の表 \(利用者の列 member/), {
      target: { value: '/data/members.csv' },
    })
    fireEvent.click(screen.getByRole('button', { name: '並べる' }))
    await waitFor(() =>
      expect(predict).toHaveBeenCalledWith('s1', { source: '/data/members.csv', k: 3 }),
    )
  })
})
