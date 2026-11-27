import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router'

import type { Policy } from '../../api/types'
import { PolicyPage } from './PolicyPage'

// CartPole の方策の一式 (観測 4 つ, 行動 2 通り, 外れたら元の制御に戻す)
const cartpole: Policy = {
  id: 'p1',
  name: 'cartpole',
  path: '/outputs/cartpole/policy',
  added_at: '2026-01-01T00:00:00',
  ok: true,
  error: null,
  unused_keys: [],
  spec: {
    format_version: 1,
    kind: 'policy',
    head: 'categorical',
    observation_space: {
      type: 'box',
      shape: [4],
      low: [-4.8, null, -0.4, null],
      high: [4.8, null, 0.4, null],
    },
    action_space: { type: 'discrete', n: 2, start: 0 },
    guards: {
      observation_low: [-2.4, null, -0.21, null],
      observation_high: [2.4, null, 0.21, null],
      allowed_actions: null,
      fallback: { kind: 'handover', action: null },
    },
  },
}

const act = vi.fn((_id: string, observation: unknown) =>
  Promise.resolve(
    ((observation as number[])[0] ?? 0) > 2.4
      ? { action: null, status: 'handover', reasons: ['観測の要素 0 が範囲の外 (2.5)'] }
      : { action: 1, status: 'ok', reasons: [] },
  ),
)

const policyBatch = vi.fn((_id: string, _source: string) =>
  Promise.resolve({
    id: 'r1',
    policy_id: 'p1',
    policy: { name: 'cartpole', path: '/outputs/cartpole/policy' },
    source: '/data/observations.csv',
    created_at: null,
    rows: 3,
    counts: { ok: 2, clipped: 0, fallback: 0, stop: 0, handover: 1 },
  }),
)

vi.mock('../../api/client', async () =>
  (await import('../../test/mockClient')).mockClient({
    api: {
      policy: () => Promise.resolve(cartpole),
      act: (id: string, observation: unknown) => act(id, observation),
      policyResults: () => Promise.resolve([]),
      policyBatch: (id: string, source: string) => policyBatch(id, source),
    },
  }),
)

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/policies/p1']}>
        <Routes>
          <Route path="/policies/:id" element={<PolicyPage />} />
          <Route path="/policy-results/:id" element={<p>方策の結果の画面</p>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('方策の一式の画面', () => {
  it('空間と守りを出し, 観測から行動と守りの結果を出す', async () => {
    renderPage()
    expect(await screen.findByText('番号 0 から 1 (2 通り)')).toBeInTheDocument()
    expect(screen.getByText('元の制御に戻す')).toBeInTheDocument()
    const field = screen.getByLabelText(/^観測 \(/)
    expect(field).toHaveValue('0, 0, 0, 0')

    fireEvent.click(screen.getByRole('button', { name: '行動を決める' }))
    expect(await screen.findByText('そのまま')).toBeInTheDocument()
    expect(act).toHaveBeenLastCalledWith('p1', [0, 0, 0, 0])

    fireEvent.change(field, { target: { value: '2.5, 0, 0, 0' } })
    fireEvent.click(screen.getByRole('button', { name: '行動を決める' }))
    expect(await screen.findByText('元の制御に戻した')).toBeInTheDocument()
    expect(screen.getByText('行動なし')).toBeInTheDocument()
    expect(screen.getByText('観測の要素 0 が範囲の外 (2.5)')).toBeInTheDocument()
  })

  it('観測を書き換えると前の結果を消し, 数でない観測は送らずに誤りを出す', async () => {
    renderPage()
    const field = await screen.findByLabelText(/^観測 \(/)
    fireEvent.click(screen.getByRole('button', { name: '行動を決める' }))
    expect(await screen.findByText('そのまま')).toBeInTheDocument()
    act.mockClear()
    fireEvent.change(field, { target: { value: '1, x' } })
    expect(screen.queryByText('そのまま')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '行動を決める' }))
    expect(await screen.findByText('x は数として読めません')).toBeInTheDocument()
    expect(act).not.toHaveBeenCalled()
  })

  it('観測のファイルでまとめて決めると, 残した結果の画面に移る', async () => {
    renderPage()
    const button = await screen.findByRole('button', { name: 'まとめて決める' })
    expect(button).toBeDisabled()
    expect(await screen.findByText('まだ結果がありません.')).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText(/^観測のファイル/), {
      target: { value: ' /data/observations.csv ' },
    })
    fireEvent.click(button)
    expect(await screen.findByText('方策の結果の画面')).toBeInTheDocument()
    expect(policyBatch).toHaveBeenCalledWith('p1', '/data/observations.csv')
  })
})
