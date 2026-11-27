import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router'

import type { Bundle, ProblemState } from '../../api/types'
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

vi.mock('../../api/client', async () =>
  (await import('../../test/mockClient')).mockClient({
    api: {
      bundle: () => Promise.resolve(resnet),
      results: () => Promise.resolve([]),
      bundleProblem: () => bundleProblem(),
      info: () => new Promise(() => {}),
    },
  }),
)

// 作り方が既製のモデルをそのままの問題の設定をつないだ状態 (重みのハッシュ値が一式と違う)
const linkedState: ProblemState = {
  path: '/project/problem.yaml',
  problem_id: null,
  hint_project: false,
  linked: {
    name: '写真の仕分け',
    path: '/project/problem.yaml',
    version: 'v1',
    created_at: '2026-01-01T00:00:00',
    role: '予測する',
    handled: true,
    decision: null,
    conclusion: { min_confidence: 0.8 },
    horizon: null,
    basis: null,
    text: '既製のモデルの一式は意思決定を使えないので, 問題の設定の意思決定は使いません. 結論の決め方は問題の設定の値を既定にします',
    issues: [
      '既製のモデル: 問題の設定の重みのハッシュ値 8ec1484cdbb8… と, 一式が読んだ重みのハッシュ値 abababababab… が違います',
    ],
  },
  error: null,
  hint: null,
  hint_note: null,
}

/** 一覧から開いた一式の画面を描く */
function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
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
}

describe('既製のモデルの一式の画面', () => {
  beforeEach(() => bundleProblem.mockReset())

  it('ライセンスの警告, 重みの置き場, 画像の欄を出し, 意思決定の欄を出さない', async () => {
    bundleProblem.mockResolvedValue({
      path: null,
      linked: null,
      error: null,
      hint: null,
      hint_note: null,
    })
    show()
    expect(await screen.findByRole('note', { name: 'ライセンスの警告' })).toHaveTextContent(
      'ライセンス: 不明 (情報: torchvision)',
    )
    expect(screen.getByText(/置き場にありません/)).toBeInTheDocument()
    expect(screen.getByText(/ほか 990 種類/)).toBeInTheDocument()
    expect(await screen.findByLabelText(/推論する画像/)).toBeInTheDocument()
    expect(screen.getByText(/意思決定, 過去のデータでのたどり直し/)).toBeInTheDocument()
    expect(screen.queryByRole('region', { name: '意思決定' })).not.toBeInTheDocument()
    expect(screen.getByRole('region', { name: '結論の決め方' })).toBeInTheDocument()
    expect(screen.queryByText('読み込み')).not.toBeInTheDocument()
    // 問題の設定の欄は, 既製のモデルの問題の設定をつなげると書く
    expect(screen.getByText(/作り方が既製のモデルをそのままの問題の設定/)).toBeInTheDocument()
  })

  it('つないだ問題の設定の結論の決め方が既定になり, 重みの食い違いを出す', async () => {
    bundleProblem.mockResolvedValue(linkedState)
    show()
    expect(await screen.findByLabelText(/確信の下限/)).toHaveValue(0.8)
    expect(screen.getByText(/問題の設定の意思決定は使いません/)).toBeInTheDocument()
    expect(screen.getByText(/推論の結論の決め方は/)).toBeInTheDocument()
    expect(screen.getByRole('list', { name: '一式との食い違い' })).toHaveTextContent(
      '重みのハッシュ値 8ec1484cdbb8',
    )
    expect(screen.queryByRole('region', { name: '意思決定' })).not.toBeInTheDocument()
  })
})
