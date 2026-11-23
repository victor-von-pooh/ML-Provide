import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'

import type { ReadyPreview } from '../../api/types'
import { ReadyBundleDialog } from './ReadyBundleDialog'

// torchvision の ResNet18 の下見 (種類は 1000. ライセンスは機械で読めないので警告)
const preview: ReadyPreview = {
  library: 'torchvision',
  model: 'resnet18',
  weights: 'IMAGENET1K_V1',
  classes: Array.from({ length: 1000 }, (_, i) => (i === 0 ? 'tench' : `c${i}`)),
  input: 'image',
  preprocess: {
    resize: 256,
    crop: 224,
    interpolation: 'bilinear',
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
  },
  url: 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
  file: 'resnet18-f37072fd.pth',
  license: {
    license: null,
    status: 'unknown',
    redistributable: false,
    gated: false,
    alert: true,
    message: 'ライセンス: 不明',
    source: '読めなかった',
    url: null,
  },
}

const readyPreview = vi.fn((_library: string, _model: string, _weights: string) =>
  Promise.resolve(preview),
)
const createReadyBundle = vi.fn((_body: unknown) => Promise.resolve({ id: 'r1', name: 'resnet18' }))

vi.mock('../../api/client', () => ({
  api: {
    info: () => Promise.resolve({ ready_libraries: ['torchvision'] }),
    readyPreview: (library: string, model: string, weights: string) =>
      readyPreview(library, model, weights),
    createReadyBundle: (body: unknown) => createReadyBundle(body),
  },
}))

/** 対話を描く */
function renderDialog(onCreated = vi.fn()) {
  // jsdom は dialog の showModal を持たないので, 開いた印だけ付ける
  HTMLDialogElement.prototype.showModal = function (this: HTMLDialogElement) {
    this.setAttribute('open', '')
  }
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={client}>
      <ReadyBundleDialog onClose={vi.fn()} onCreated={onCreated} />
    </QueryClientProvider>,
  )
  return onCreated
}

describe('既製のモデルから作る対話', () => {
  beforeEach(() => {
    readyPreview.mockClear()
    createReadyBundle.mockClear()
  })

  it('ResNet18 を下見して種類の数, 前処理, ライセンスの警告を出し, 作って登録する', async () => {
    const onCreated = renderDialog()
    expect(await screen.findByText('1,000 種類')).toBeInTheDocument()
    expect(readyPreview).toHaveBeenCalledWith('torchvision', 'resnet18', 'IMAGENET1K_V1')
    expect(screen.getByText(/tench, c1/)).toBeInTheDocument()
    expect(screen.getByText(/ほか 990 種類/)).toBeInTheDocument()
    expect(screen.getByText(/224 × 224/)).toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent('ライセンス: 不明 (情報: 読めなかった)')
    expect(screen.getByLabelText('一式の名前')).toHaveValue('resnet18')

    fireEvent.click(screen.getByRole('button', { name: '作って登録' }))
    await waitFor(() => expect(onCreated).toHaveBeenCalled())
    expect(createReadyBundle).toHaveBeenCalledWith({
      name: 'resnet18',
      library: 'torchvision',
      model: 'resnet18',
      weights: 'IMAGENET1K_V1',
      sha256: null,
    })
  })

  it('形の違うハッシュ値では作れない', async () => {
    renderDialog()
    await screen.findByText('1,000 種類')
    fireEvent.change(screen.getByLabelText(/重みのハッシュ値/), { target: { value: 'abc' } })
    expect(screen.getByText(/16 進の 64 文字/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '作って登録' })).toBeDisabled()
  })
})
