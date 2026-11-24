import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, within } from '@testing-library/react'

import type { BacktestGuard, Result } from '../../api/types'
import { DecisionCard } from './DecisionCard'

vi.mock('../../api/client', () => ({ api: {} }))

// 日々の売上の見本の過去のデータで, 量を決めてたどり直した結果
function resultWith(guard: BacktestGuard | null): Result {
  return {
    id: 'r1',
    bundle: { name: '日々の売上', path: '/bundles/daily' },
    bundle_id: 'b1',
    source: 'past.csv',
    created_at: null,
    task: 'forecasting',
    summary: null,
    sentences: [],
    columns: ['date', 'shop', 'sales', 'prediction'],
    has_target: true,
    decision: {
      decision: { type: 'quantity', under_cost: 200, over_cost: 150 },
      summary: { rows: 30, acting: 30 },
      sentences: ['量を決めた'],
      notes: [],
    },
    backtest: {
      rows: 30,
      policies: { この方針: -1200, 予測のまま: -1500 },
      guard,
      sentences: [
        '過去の 30 行でたどり直した',
        '守る指標「欠品の率」(足りなかった行の割合) は, この方針で 12.3 % (守る線は 10 % 以下. 守れなかった)',
      ],
    },
  }
}

/** 意思決定の欄を描く関数 */
function show(guard: BacktestGuard | null) {
  const client = new QueryClient()
  render(
    <QueryClientProvider client={client}>
      <DecisionCard result={resultWith(guard)} bundleAlive />
    </QueryClientProvider>,
  )
}

describe('たどり直しの守る指標', () => {
  it('この方針の値, 守る線, 守れたかを % で並べ, 量の名前は出さない', () => {
    show({
      name: '欠品の率',
      unit: null,
      counted: 'shortage_rate',
      description: '足りなかった行の割合',
      value: 0.1234,
      line: 0.1,
      better: 'lower',
      kept: false,
      note: null,
    })
    const box = screen.getByRole('group', { name: '守る指標' })
    expect(
      within(box).getByText(/守る指標「欠品の率」 \(足りなかった行の割合\)/),
    ).toBeInTheDocument()
    expect(within(box).getByText('12.3 %')).toBeInTheDocument()
    expect(within(box).getByText('10 % 以下')).toBeInTheDocument()
    expect(within(box).getByText('守れなかった')).toBeInTheDocument()
    expect(box.textContent).not.toContain('shortage_rate')
    expect(screen.getByText(/この方針で 12.3 %/)).toBeInTheDocument()
  })

  it('数える量を指していなければ, 課題の場で数えると添える', () => {
    show({
      name: '顧客の満足',
      unit: null,
      counted: null,
      description: null,
      value: null,
      line: null,
      better: null,
      kept: null,
      note: '過去のデータでは数えられない. 課題の場で数える',
    })
    const box = screen.getByRole('group', { name: '守る指標' })
    expect(within(box).getByText('数えていない')).toBeInTheDocument()
    expect(within(box).getByText(/課題の場で数える/)).toBeInTheDocument()
  })

  it('問題の設定をつないでいなければ守る指標の欄は無い', () => {
    show(null)
    expect(screen.queryByRole('group', { name: '守る指標' })).toBeNull()
  })
})
