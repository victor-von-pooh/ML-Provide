import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'

import type { DatasetDescription } from '../../api/types'
import { CreateBundleDialog } from './CreateBundleDialog'

// 説明の読み込みを止めておき, 読み終える時をテストで決める
const pending: Array<(value: DatasetDescription) => void> = []
const describeDataset = vi.fn(
  (_path: string) =>
    new Promise<DatasetDescription>((resolve) => {
      pending.push(resolve)
    }),
)

vi.mock('../../api/client', () => ({
  api: {
    info: () => Promise.resolve({ procedures: [], max_procedure_steps: 1000 }),
    describeDataset: (path: string) => describeDataset(path),
    createBundle: vi.fn(),
  },
}))

/** 対話を描く */
function renderDialog() {
  // jsdom は dialog の showModal を持たないので, 開いた印だけ付ける
  HTMLDialogElement.prototype.showModal = function (this: HTMLDialogElement) {
    this.setAttribute('open', '')
  }
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={client}>
      <CreateBundleDialog onClose={vi.fn()} onCreated={vi.fn()} />
    </QueryClientProvider>,
  )
}

/** 列のデータセット (版 1) の説明 */
const columns = { format_version: 1 } as DatasetDescription

describe('学習の結果から作る対話', () => {
  beforeEach(() => {
    pending.length = 0
    describeDataset.mockClear()
  })

  it('データセットの説明を読む間と, 欄が読んだ場所と違う間は「作る」を押せない', async () => {
    renderDialog()
    fireEvent.change(screen.getByLabelText('名前'), { target: { value: 'm' } })
    fireEvent.change(screen.getByLabelText('モデルの定義 (YAML か JSON)'), {
      target: { value: '/s.yaml' },
    })
    fireEvent.change(screen.getByLabelText('実験のディレクトリか重みのファイル'), {
      target: { value: '/run' },
    })
    const dataset = screen.getByLabelText('データセット (dataset.yaml を持つディレクトリ)')
    const submit = screen.getByRole('button', { name: '作る' })

    // 打った直後 (読む前) と読んでいる間は押せない
    fireEvent.change(dataset, { target: { value: '/a' } })
    expect(submit).toBeDisabled()
    expect(screen.getByRole('status')).toHaveTextContent('データセットの説明を読み込んでいます')
    await waitFor(() => expect(describeDataset).toHaveBeenCalledWith('/a'))
    expect(submit).toBeDisabled()

    // 読み終えると押せる
    pending.shift()?.(columns)
    await waitFor(() => expect(submit).toBeEnabled())
    expect(screen.queryByRole('status')).toBeNull()

    // 別の場所に変えると, その説明を読み終えるまで押せない
    fireEvent.change(dataset, { target: { value: '/b' } })
    expect(submit).toBeDisabled()
    await waitFor(() => expect(describeDataset).toHaveBeenCalledWith('/b'))
    expect(submit).toBeDisabled()
    pending.shift()?.(columns)
    await waitFor(() => expect(submit).toBeEnabled())
  })
})
