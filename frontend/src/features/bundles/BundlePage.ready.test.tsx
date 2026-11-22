import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router'

import type { Bundle } from '../../api/types'
import { BundlePage } from './BundlePage'

// torchvision の ResNet18 の既製のモデルの一式 (describe_bundle の値. 重みは置き場に無い)
const resnet: Bundle = {
  id: 'r1',
  name: 'resnet18',
  path: '/bundles/resnet18',
  managed: true,
  added_at: '2026-01-01T00:00:00',
  ok: true,
  error: null,
  description: {
    path: '/bundles/resnet18',
    name: 'resnet18',
    created_at: null,
    kind: 'ready',
    task: 'classification',
    target: null,
    classes: Array.from({ length: 1000 }, (_, i) => (i === 0 ? 'tench' : `c${i}`)),
    features: [],
    text_column: null,
    reading: { loader: null, options: {} },
    steps: [],
    item_steps: [],
    model: { name: 'resnet18', inputs: {}, output: null, kind: 'logits' },
    run: null,
    conclusion: {},
    decision: null,
    check: null,
    notes: ['重みは一式に写さず, 使うときに出どころから読みます'],
    format_version: 6,
    role: 'predict',
    output: { form: 'class' },
    column_output: true,
    target_info: null,
    items: null,
    pretrained: [{ node: 'model', place: 'source', sha256: 'ab'.repeat(32), status: 'unknown' }],
    ready: {
      library: 'torchvision',
      model: 'resnet18',
      weights: 'IMAGENET1K_V1',
      input: 'image',
      preprocess: {
        resize: 256,
        crop: 224,
        interpolation: 'bilinear',
        mean: [0.485, 0.456, 0.406],
        std: [0.229, 0.224, 0.225],
      },
      url: null,
      file: 'resnet18-f37072fd.pth',
      license: {
        license: null,
        status: 'unknown',
        redistributable: false,
        gated: false,
        alert: true,
        message: 'ライセンス: 不明',
        source: 'torchvision',
        url: null,
      },
      weights_file: '/hub/checkpoints/resnet18-f37072fd.pth',
      weights_cached: false,
    },
  },
}

const bundleProblem = vi.fn()

vi.mock('../../api/client', () => ({
  api: {
    bundle: () => Promise.resolve(resnet),
    results: () => Promise.resolve([]),
    bundleProblem: () => bundleProblem(),
    info: () => new Promise(() => {}),
  },
}))

describe('既製のモデルの一式の画面', () => {
  it('ライセンスの警告, 重みの置き場, 画像の欄を出し, 意思決定と問題の設定の欄を出さない', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    // 一覧から開いた一式 (種類が分かるので問題の設定を要求しない)
    client.setQueryData(['bundles'], [resnet])
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter initialEntries={['/bundles/r1']}>
          <Routes>
            <Route path="/bundles/:id" element={<BundlePage />} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    )
    expect(await screen.findByRole('note', { name: 'ライセンスの警告' })).toHaveTextContent(
      'ライセンス: 不明 (情報: torchvision)',
    )
    expect(screen.getByText(/置き場にありません/)).toBeInTheDocument()
    expect(screen.getByText(/ほか 990 種類/)).toBeInTheDocument()
    expect(screen.getByLabelText(/推論する画像/)).toBeInTheDocument()
    expect(screen.getByText(/意思決定, 過去のデータでのたどり直し/)).toBeInTheDocument()
    expect(screen.queryByRole('region', { name: '意思決定' })).not.toBeInTheDocument()
    expect(screen.getByRole('region', { name: '結論の決め方' })).toBeInTheDocument()
    expect(screen.queryByText('読み込み')).not.toBeInTheDocument()
    expect(bundleProblem).not.toHaveBeenCalled()
  })
})
