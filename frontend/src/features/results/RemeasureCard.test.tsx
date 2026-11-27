import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'

import type { Remeasure, Result } from '../../api/types'
import { RemeasureCard } from './RemeasureCard'

// 来店数の見本の過去のデータ (目的の列あり) の結果
const result: Result = {
  id: 'r1',
  bundle: { name: '来店数', path: '/bundles/shop' },
  bundle_id: 'b1',
  source: 'past.csv',
  created_at: null,
  task: 'regression',
  summary: null,
  sentences: [],
  columns: ['shop', 'a', 'visitors', 'prediction'],
  has_target: true,
}

// 店ごとに分けて測った確かめの結果
const view: Remeasure = {
  rows: 120,
  before: { coverage: { '80': 0.8, '90': 0.9 } },
  sentences: ['shop ごとの外れ方を 3 種類で測った'],
  written: false,
  copied_to: null,
}

const remeasure = vi.fn(
  (_id: string, _write: boolean, _horizon?: number, _time?: string, _groups?: string) =>
    Promise.resolve(view),
)

vi.mock('../../api/client', async () =>
  (await import('../../test/mockClient')).mockClient({
    api: {
      remeasure: (id: string, write: boolean, horizon?: number, time?: string, groups?: string) =>
        remeasure(id, write, horizon, time, groups),
    },
  }),
)

/** 確かめの欄を描く関数 */
function show(groupColumns: string[]) {
  const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } })
  render(
    <QueryClientProvider client={client}>
      <RemeasureCard result={result} bundleAlive forecastable={false} groupColumns={groupColumns} />
    </QueryClientProvider>,
  )
}

describe('外れ方の確かめの分ける列', () => {
  beforeEach(() => remeasure.mockClear())

  it('選んだ列を測り直しに渡し, 種類ごとの文を出す', async () => {
    show(['shop', 'a'])
    const select = screen.getByLabelText(/外れ方を分ける列/)
    expect(screen.getByRole('option', { name: '分けない' })).toBeInTheDocument()
    fireEvent.change(select, { target: { value: 'shop' } })
    fireEvent.click(screen.getByRole('button', { name: '確かめる' }))
    await waitFor(() =>
      expect(remeasure).toHaveBeenCalledWith('r1', false, undefined, undefined, 'shop'),
    )
    expect(await screen.findByText('shop ごとの外れ方を 3 種類で測った')).toBeInTheDocument()

    // 書き戻しも確かめたときの列で
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    fireEvent.click(screen.getByRole('button', { name: '一式に書き戻す' }))
    await waitFor(() =>
      expect(remeasure).toHaveBeenLastCalledWith('r1', true, undefined, undefined, 'shop'),
    )
  })

  it('分けない場合は列を渡さない', async () => {
    show(['shop'])
    fireEvent.click(screen.getByRole('button', { name: '確かめる' }))
    await waitFor(() =>
      expect(remeasure).toHaveBeenCalledWith('r1', false, undefined, undefined, undefined),
    )
  })

  it('分けられる列が無ければ (分類の一式) 欄を出さない', () => {
    show([])
    expect(screen.queryByLabelText(/外れ方を分ける列/)).not.toBeInTheDocument()
  })
})
