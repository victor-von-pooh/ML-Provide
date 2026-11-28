import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'

import type { BundleCreateBody, DatasetDescription } from '../../api/types'
import { StageContext } from '../../lib/stage'
import { CreateBundleDialog } from './CreateBundleDialog'

// 雑音から始まる組み込みの手順 (登録簿の値)
const ddim = {
  name: 'ddim',
  builtin: true,
  description: 'DDIM',
  start: 'noise',
  time: true,
  options: ['train_steps', 'eta'],
  defaults: { train_steps: 1000, eta: 0 },
}

// 説明の読み込みを止めておき, 読み終える時をテストで決める
const pending: Array<(value: DatasetDescription) => void> = []
const describeDataset = vi.fn(
  (_path: string) =>
    new Promise<DatasetDescription>((resolve) => {
      pending.push(resolve)
    }),
)

// 作る要求 (送った中身を確かめる. 作り終えない)
const createBundle = vi.fn((_body: BundleCreateBody) => new Promise<never>(() => {}))

// 殻で選んだ問題をつなげるか (bad は推論に渡せない問題)
const projectProblem = vi.fn((project: string) =>
  project === 'slow'
    ? new Promise<never>(() => {})
    : Promise.resolve(
        project === 'bad'
          ? { usable: false, reason: '問題 bad は推論に渡せません' }
          : { usable: true, reason: null },
      ),
)

vi.mock('../../api/client', async () =>
  (await import('../../test/mockClient')).mockClient({
    api: {
      info: () => Promise.resolve({ procedures: [ddim], max_procedure_steps: 1000 }),
      describeDataset: (path: string) => describeDataset(path),
      createBundle: (body: BundleCreateBody) => createBundle(body),
      projectProblem: (project: string) => projectProblem(project),
    },
  }),
)

/** 対話を描く (project は殻で選んだ問題) */
function renderDialog(project: string | null = null) {
  // jsdom は dialog の showModal を持たないので, 開いた印だけ付ける
  HTMLDialogElement.prototype.showModal = function (this: HTMLDialogElement) {
    this.setAttribute('open', '')
  }
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={client}>
      <StageContext.Provider
        value={{ project, onProjectChange: vi.fn(), embedded: project !== null }}
      >
        <CreateBundleDialog onClose={vi.fn()} onCreated={vi.fn()} />
      </StageContext.Provider>
    </QueryClientProvider>,
  )
}

/** 列のデータセット (版 1) の説明 */
const columns = { format_version: 1 } as DatasetDescription

describe('学習の結果から作る対話', () => {
  beforeEach(() => {
    pending.length = 0
    describeDataset.mockClear()
    createBundle.mockClear()
    window.localStorage.clear()
  })

  /** 列のデータセットの欄を埋めて「作る」を押せるまで待つ */
  async function fillColumns() {
    fireEvent.change(screen.getByLabelText('名前'), { target: { value: 'm' } })
    fireEvent.change(screen.getByLabelText('モデルの定義 (YAML か JSON)'), {
      target: { value: '/s.yaml' },
    })
    fireEvent.change(screen.getByLabelText('実験のディレクトリか重みのファイル'), {
      target: { value: '/run' },
    })
    fireEvent.change(screen.getByLabelText('データセット (dataset.yaml を持つディレクトリ)'), {
      target: { value: '/a' },
    })
    await waitFor(() => expect(describeDataset).toHaveBeenCalledWith('/a'))
    pending.shift()?.(columns)
    const submit = screen.getByRole('button', { name: '作る' })
    await waitFor(() => expect(submit).toBeEnabled())
    return submit
  }

  it('殻で選んだ問題を, 作る一式につなぐ既定にし, 外せる', async () => {
    renderDialog('aaaa')
    const link = await screen.findByRole('checkbox', { name: /選んだ問題 aaaa をつなぐ/ })
    expect(link).toBeChecked()
    const submit = await fillColumns()
    fireEvent.click(submit)
    await waitFor(() => expect(createBundle).toHaveBeenCalledTimes(1))
    expect(createBundle.mock.calls[0]?.[0]).toMatchObject({ name: 'm', problem: 'aaaa' })
  })

  it('問題のつなぎを外すと, 問題を渡さずに作る', async () => {
    renderDialog('aaaa')
    fireEvent.click(await screen.findByRole('checkbox', { name: /選んだ問題 aaaa をつなぐ/ }))
    fireEvent.click(await fillColumns())
    await waitFor(() => expect(createBundle).toHaveBeenCalledTimes(1))
    expect(createBundle.mock.calls[0]?.[0].problem).toBeNull()
  })

  it('つなげるかを読み終えるまでは, つなぐ欄も理由も出さず, 問題を渡さない', async () => {
    renderDialog('slow')
    await waitFor(() => expect(projectProblem).toHaveBeenCalledWith('slow'))
    expect(screen.queryByRole('checkbox', { name: /をつなぐ/ })).toBeNull()
    expect(screen.queryByText(/つながずに作ります/)).toBeNull()
    fireEvent.click(await fillColumns())
    await waitFor(() => expect(createBundle).toHaveBeenCalledTimes(1))
    expect(createBundle.mock.calls[0]?.[0].problem).toBeNull()
  })

  it('推論に渡せない問題は, つなぐ欄を出さずに理由を添え, 問題を渡さない', async () => {
    renderDialog('bad')
    expect(
      await screen.findByText(/選んだ問題 bad はつながずに作ります: 問題 bad は推論に渡せません/),
    ).toBeInTheDocument()
    expect(screen.queryByRole('checkbox', { name: /をつなぐ/ })).toBeNull()
    fireEvent.click(await fillColumns())
    await waitFor(() => expect(createBundle).toHaveBeenCalledTimes(1))
    expect(createBundle.mock.calls[0]?.[0].problem).toBeNull()
  })

  it('殻で問題を選んでいなければ, つなぐ欄を出さず問題を渡さない', async () => {
    renderDialog()
    expect(screen.queryByRole('checkbox', { name: /をつなぐ/ })).toBeNull()
    fireEvent.click(await fillColumns())
    await waitFor(() => expect(createBundle).toHaveBeenCalledTimes(1))
    expect(createBundle.mock.calls[0]?.[0].problem).toBeNull()
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

  it('雑音だけの一式の誤りに, 繰り返しの手順, 出力の形, 入力の出どころの欄の名前が付く', async () => {
    renderDialog()
    fireEvent.click(screen.getByLabelText('雑音だけから作る (データセットを使わない)'))
    const procedure = screen.getByLabelText(/^手順 \(登録簿の名前/)
    await waitFor(() => expect(procedure).toContainHTML('ddim'))
    fireEvent.change(procedure, { target: { value: 'ddim' } })
    fireEvent.change(screen.getByLabelText(/^回数/), { target: { value: '1001' } })
    fireEvent.change(screen.getByLabelText('出力の形 (要る)'), { target: { value: 'vector' } })
    fireEvent.change(screen.getByLabelText(/^1 件の形/), { target: { value: '0' } })
    expect(
      screen.getByText('「繰り返しの手順」: 手順の回数は 1,000 回までです'),
    ).toBeInTheDocument()
    expect(screen.getByText(/^「出力の形」: 1 件の形は正の整数の並び/)).toBeInTheDocument()
    expect(
      screen.getByText(/^「モデルの入力ごとの出どころ」: モデルの入力の名前/),
    ).toBeInTheDocument()
  })

  it('出力の形に表は出さない (学習から作る一式だけ)', async () => {
    renderDialog()
    const form = screen.getByLabelText(/^出力の形 \(正解の無い/)
    expect(form).toContainHTML('vector')
    expect(form).not.toContainHTML('value="table"')
    fireEvent.click(screen.getByLabelText('雑音だけから作る (データセットを使わない)'))
    const noData = screen.getByLabelText('出力の形 (要る)')
    expect(noData).toContainHTML('vector')
    expect(noData).not.toContainHTML('value="table"')
  })

  it('列のデータセットの誤りには「受け持ちと出力の形」の欄の名前が付く', async () => {
    renderDialog()
    fireEvent.change(screen.getByLabelText('データセット (dataset.yaml を持つディレクトリ)'), {
      target: { value: '/a' },
    })
    await waitFor(() => expect(describeDataset).toHaveBeenCalledWith('/a'))
    pending.shift()?.(columns)
    fireEvent.change(screen.getByLabelText(/^出力の形 \(正解の無い/), {
      target: { value: 'distribution' },
    })
    fireEvent.change(screen.getByLabelText(/^分布の水準/), { target: { value: '0.9, 0.1' } })
    expect(
      await screen.findByText(
        '「受け持ちと出力の形」: 分布の水準は 0 から 1 の間の小さい順の数で書いてください',
      ),
    ).toBeInTheDocument()
  })
})
