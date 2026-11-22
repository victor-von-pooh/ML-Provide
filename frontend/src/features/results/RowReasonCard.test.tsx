import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen } from '@testing-library/react'
import { useState } from 'react'

import type { Result, RowReason } from '../../api/types'
import { rowReasonable } from '../../lib/rowReason'
import { RowReasonCard } from './RowReasonCard'

// 来店数の見本の新しいデータの結果
const result: Result = {
  id: 'r1',
  bundle: { name: '来店数', path: '/bundles/shop' },
  bundle_id: 'b1',
  source: 'new.csv',
  created_at: null,
  task: 'regression',
  kind: 'table',
  summary: { rows: 50, predicted: 50, skipped: 0 },
  sentences: [],
  columns: ['shop', 'temperature', 'prediction'],
}

// 3 行目の理由 (temperature は下げ, 時刻の列は求めない)
const view: RowReason = {
  row: 3,
  measure: 'value',
  status: 'ok',
  prediction: 120.5,
  reason: 'shop (+30.2), temperature (-4.1)',
  effects: [
    { column: 'shop', effect: 30.2 },
    { column: 'temperature', effect: -4.1 },
  ],
  errors: { date: '時刻の列は求めません' },
  rows: 201,
  repeats: 10,
}

const rowReason = vi.fn((_id: string, _row: number) => Promise.resolve(view))

vi.mock('../../api/client', () => ({
  api: { rowReason: (id: string, row: number) => rowReason(id, row) },
}))

/** 行を親が持つ形で欄を描く */
function Harness() {
  const [row, setRow] = useState<number | null>(null)
  return <RowReasonCard result={result} row={row} onRow={setRow} />
}

describe('1 行の予測の理由', () => {
  beforeEach(() => rowReason.mockClear())

  it('押すまで求めず, 選んだ行だけを求めて効きと理由を出す', async () => {
    const client = new QueryClient()
    render(
      <QueryClientProvider client={client}>
        <Harness />
      </QueryClientProvider>,
    )
    expect(rowReason).not.toHaveBeenCalled()
    const button = screen.getByRole('button', { name: '求める' })
    expect(button).toBeDisabled()
    fireEvent.change(screen.getByLabelText('行 (#)'), { target: { value: '3' } })
    fireEvent.click(button)
    expect(await screen.findByText(/shop \(\+30\.2\), temperature \(-4\.1\)/)).toBeInTheDocument()
    expect(rowReason).toHaveBeenCalledTimes(1)
    expect(rowReason).toHaveBeenCalledWith('r1', 3)
    const items = screen.getAllByRole('listitem')
    expect(items.map((item) => item.textContent)).toEqual(['shop+30.2', 'temperature-4.1'])
    expect(screen.getByText(/date: 時刻の列は求めません/)).toBeInTheDocument()
    expect(screen.getByText(/この行と比べる 200 行, 列ごとに 10 回の平均/)).toBeInTheDocument()
  })

  it('行の数より大きい行は求めない', () => {
    const client = new QueryClient()
    render(
      <QueryClientProvider client={client}>
        <Harness />
      </QueryClientProvider>,
    )
    fireEvent.change(screen.getByLabelText('行 (#)'), { target: { value: '50' } })
    expect(screen.getByRole('button', { name: '求める' })).toBeDisabled()
  })

  it('表と時系列の 1 つ先の予測の結果だけで出す', () => {
    expect(rowReasonable(result, true)).toBe(true)
    expect(rowReasonable(result, false)).toBe(false)
    expect(rowReasonable({ ...result, forecast: true }, true)).toBe(false)
    expect(rowReasonable({ ...result, kind: 'text' }, true)).toBe(false)
    expect(rowReasonable({ ...result, column_output: false }, true)).toBe(false)
  })
})
