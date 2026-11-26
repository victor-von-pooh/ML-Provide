import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router'

import type { Bundle } from '../../api/types'
import { BundlePage } from './BundlePage'

// 3 つの群の群分けの一式 (describe_bundle の値)
const customers = {
  id: 'g1',
  name: 'customers',
  path: '/bundles/customers',
  managed: false,
  added_at: '2026-01-01T00:00:00',
  ok: true,
  error: null,
  description: {
    path: '/bundles/customers',
    name: 'customers',
    created_at: null,
    kind: 'group',
    task: '',
    target: null,
    classes: [],
    features: [],
    text_column: null,
    reading: { loader: null, options: {} },
    steps: [],
    item_steps: [],
    model: {
      name: 'contrastive_encoder',
      inputs: { x: { source: 'items', prepend: [] } },
      output: null,
      kind: 'value',
    },
    run: { path: '/outputs/customers/run', status: 'completed' },
    conclusion: {},
    decision: null,
    check: null,
    notes: [],
    format_version: 9,
    role: 'group',
    output: { form: 'cluster' },
    column_output: false,
    target_info: null,
    items: null,
    ready: null,
    rank: null,
    group: {
      k: 2,
      columns: ['spend', 'region'],
      features: 3,
      width: 16,
      id: 'customer',
      rows: 40,
      stability: {
        k: 2,
        method: 'sample',
        repeats: 3,
        ari: 0.8,
        ari_baseline: 0.7,
        ari_diff: 0.1,
        ari_min: 0.6,
        ari_min_baseline: 0.5,
        ari_min_diff: 0.1,
        silhouette: 0.4,
        silhouette_baseline: 0.3,
        silhouette_diff: 0.1,
        baseline: '列の特徴をそのまま k-means で分けた群',
        epoch: 4,
      },
      summary: [
        { group: 0, rows: 30, share: 0.75, stability: 0.8, spend: 100, region: 'north (90 %)' },
        { group: 1, rows: 10, share: 0.25, stability: 0.7, spend: 900, region: 'south (80 %)' },
        { group: '全体', rows: 40, share: 1, stability: null, spend: 300, region: 'north (70 %)' },
      ],
      check: { rows: 40, agreement: 1 },
    },
  },
} as unknown as Bundle

// 既製の埋め込みの群分けの一式 (版 10. 列を符号化せず, 既製のモデルの出どころを持つ)
const photos = {
  ...customers,
  id: 'g2',
  name: 'photos',
  description: {
    ...customers.description,
    name: 'photos',
    format_version: 10,
    run: null,
    model: {
      name: 'resnet18',
      inputs: { image: { source: 'items', prepend: [] } },
      output: null,
      kind: 'value',
    },
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
      license: null,
      weights_file: '/hub/checkpoints/resnet18-f37072fd.pth',
      weights_cached: true,
    },
    group: {
      ...customers.description!.group!,
      columns: ['width', 'height'],
      features: null,
      width: 512,
      id: 'path',
    },
  },
} as unknown as Bundle

// 学習の表を渡して作った群分けの一式 (版 16. 群ごとの遠い行の上限を持ち, 群 1 は行が無かった)
const farred = {
  ...customers,
  id: 'g3',
  description: {
    ...customers.description,
    format_version: 16,
    group: {
      ...customers.description!.group!,
      far: { quantile: 0.99, rows: 40, thresholds: [0.4123, null] },
    },
  },
} as unknown as Bundle

let current = customers
const bundleProblem = vi.fn()
const predict = vi.fn()

vi.mock('../../api/client', () => ({
  api: {
    bundle: () => Promise.resolve(current),
    results: () => Promise.resolve([]),
    bundleProblem: () => bundleProblem(),
    info: () => new Promise(() => {}),
    predict: (id: string, body: unknown) => predict(id, body),
  },
}))

/** 一覧から開いた一式の画面を描く */
function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  client.setQueryData(['bundles'], [current])
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[`/bundles/${current.id}`]}>
        <Routes>
          <Route path="/bundles/:id" element={<BundlePage />} />
          <Route path="/results/:id" element={<p>結果の画面</p>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('群分けの一式の画面', () => {
  beforeEach(() => {
    current = customers
    bundleProblem.mockReset()
    predict.mockReset()
  })

  it('群を作る列, 群の数, 学習の群の安定性と群の要約を出し, 問題の設定を読まない', async () => {
    show()
    expect(await screen.findByText(/列の符号化と群の中心/)).toBeInTheDocument()
    expect(screen.getByText('spend, region')).toBeInTheDocument()
    expect(screen.getByText(/2 個 \(学習で 40 行を分けた/)).toBeInTheDocument()
    expect(screen.getByText(/100.0 % が実験と同じ群/)).toBeInTheDocument()
    expect(screen.queryByText('モデルの入力')).not.toBeInTheDocument()
    expect(screen.queryByText('読み込み')).not.toBeInTheDocument()

    // 学習のときの群の安定性と群の要約
    const stability = screen.getByRole('table', {
      name: '学習の群の安定性 (モデル, 比べる相手, 差)',
    })
    expect(within(stability).getAllByRole('row')[1]).toHaveTextContent('0.800')
    expect(screen.getByText(/標本を変えて作り直す, 3 回/)).toBeInTheDocument()
    const summary = screen.getByRole('table', {
      name: '学習のときの群の要約 (群ごとの件数, 割合, 安定性, 列の特徴と全体)',
    })
    expect(within(summary).getAllByRole('row')).toHaveLength(4)
    expect(within(summary).queryByText('学習のときの割合')).not.toBeInTheDocument()

    // 推論の欄は表のファイルだけで, 結論の決め方と意思決定は出さない
    expect(screen.getByLabelText(/^群に割り当てる行の表 \(群を作る列 spend, region/)).toBeVisible()
    expect(screen.getByText(/行ごとに最も近い群を出すので/)).toBeVisible()
    expect(screen.queryByText(/結論は推論した件数だけです/)).not.toBeInTheDocument()
    expect(screen.queryByRole('region', { name: '意思決定' })).not.toBeInTheDocument()
    expect(screen.queryByRole('region', { name: '結論の決め方' })).not.toBeInTheDocument()
    // 遠さの上限を持たない一式 (版 9) には遠い行の上限の行が無い
    expect(screen.queryByText('遠い行の上限')).not.toBeInTheDocument()
    expect(bundleProblem).not.toHaveBeenCalled()
  })

  it('遠い行の上限を持つ一式 (版 16) は, 学習の行の数, 分位点, 群ごとの上限を出す', async () => {
    current = farred
    show()
    expect(await screen.findByText('遠い行の上限')).toBeInTheDocument()
    expect(
      screen.getByText(/学習の 40 行の群の中心までの距離の 99 % 点 \(群 0 0\.412, 群 1 –\)/),
    ).toBeInTheDocument()
    expect(screen.getByText(/上限より遠い行に「遠い」の印を付けます/)).toBeInTheDocument()
  })

  it('行の表のファイルを渡して群に割り当てる', async () => {
    predict.mockResolvedValue({ id: 'r1' })
    show()
    fireEvent.change(await screen.findByLabelText(/^群に割り当てる行の表/), {
      target: { value: '/data/customers.csv' },
    })
    fireEvent.click(screen.getByRole('button', { name: '推論する' }))
    await waitFor(() =>
      expect(predict).toHaveBeenCalledWith('g1', { source: '/data/customers.csv' }),
    )
    expect(await screen.findByText('結果の画面')).toBeInTheDocument()
  })

  it('既製の埋め込みの群分けの一式 (版 10) も, 出どころと学習のときの群を出して落ちない', async () => {
    current = photos
    show()
    expect(await screen.findByText(/画像 \(既製のモデル/)).toBeInTheDocument()
    expect(screen.getByText('resnet18')).toBeInTheDocument()
    // 群の行は版 10 の文 (列は群の要約の列で, 符号化しない)
    expect(screen.getByText('群の要約の列 (2)')).toBeInTheDocument()
    expect(screen.getByText(/この列は符号化しません/)).toBeInTheDocument()
    expect(screen.queryByText(/推論する表はこの列を持ちます/)).not.toBeInTheDocument()
    expect(screen.queryByText(/列の符号化と群の中心/)).not.toBeInTheDocument()
    expect(screen.getByText(/埋め込みの幅 512/)).toBeInTheDocument()
    expect(screen.queryByText(/符号化した行の幅/)).not.toBeInTheDocument()
    expect(
      screen.getByRole('table', { name: '学習の群の安定性 (モデル, 比べる相手, 差)' }),
    ).toBeInTheDocument()
    // 推論の欄は写真のファイルかディレクトリで, 列の符号化の文は出さない
    expect(screen.queryByLabelText(/^群に割り当てる行の表/)).not.toBeInTheDocument()
    expect(screen.getByText(/既製のモデルの埋め込みを最も近い群の中心に割り当てます/)).toBeVisible()
    expect(screen.queryByText(/同じ式で列を符号化し/)).not.toBeInTheDocument()
    expect(bundleProblem).not.toHaveBeenCalled()
  })

  it('既製の埋め込みの群分けの一式 (版 10) は写真のディレクトリを渡して群に分ける', async () => {
    current = photos
    predict.mockResolvedValue({ id: 'r2' })
    show()
    fireEvent.change(await screen.findByLabelText(/^群に分ける写真/), {
      target: { value: '/data/photos' },
    })
    fireEvent.click(screen.getByRole('button', { name: '推論する' }))
    await waitFor(() => expect(predict).toHaveBeenCalledWith('g2', { source: '/data/photos' }))
  })
})
