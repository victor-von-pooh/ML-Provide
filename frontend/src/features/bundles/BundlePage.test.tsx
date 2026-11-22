import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router'

import type { Bundle, ProblemState } from '../../api/types'
import { BundlePage } from './BundlePage'

// ML-EDA の版 1 の音声のデータセットから作った一式の説明 (describe_bundle の値)
const audio: Bundle = {
  id: 'a1',
  name: 'sounds',
  path: '/bundles/sounds',
  managed: true,
  added_at: '2026-01-01T00:00:00',
  ok: true,
  error: null,
  description: {
    path: '/bundles/sounds',
    name: 'sounds',
    created_at: null,
    kind: 'audio',
    task: 'classification',
    target: 'label',
    classes: ['high', 'low'],
    features: [],
    text_column: null,
    reading: { loader: 'audio_folder', options: {} },
    steps: [],
    item_steps: ['resample', 'mix_channels', 'fix_length'],
    model: { name: 'flat', inputs: {}, output: null, kind: 'logits' },
    run: null,
    conclusion: {},
    check: null,
    notes: [],
    format_version: 3,
    role: 'predict',
    output: { form: 'class' },
    column_output: true,
    target_info: { kind: 'column' },
    items: null,
  },
}

// 画面に返す一式 (テストごとに差し替える)
let shown: Bundle = audio

const problem: ProblemState = { path: null, linked: null, error: null, hint: null, hint_note: null }

vi.mock('../../api/client', () => ({
  api: {
    bundle: () => Promise.resolve(shown),
    results: () => Promise.resolve([]),
    bundleProblem: () => Promise.resolve(problem),
    info: () => new Promise(() => {}),
  },
}))

/** 一式の画面を描く関数 */
function show() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/bundles/a1']}>
        <Routes>
          <Route path="/bundles/:id" element={<BundlePage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('音声の一式の画面', () => {
  beforeEach(() => {
    shown = audio
  })

  it('種類を「音声」と出し, 推論するデータの案内を音声の文にする', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter initialEntries={['/bundles/a1']}>
          <Routes>
            <Route path="/bundles/:id" element={<BundlePage />} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    )
    expect(await screen.findByText(/音声 \//)).toBeInTheDocument()
    expect(screen.queryByText(/audio \//)).not.toBeInTheDocument()
    expect(
      await screen.findByLabelText(/音声のファイルか, 音声を並べたディレクトリ/),
    ).toBeInTheDocument()
    expect(screen.queryByText(/画像などのディレクトリ/)).not.toBeInTheDocument()
  })
})

describe('外れ方の記録', () => {
  it('種類ごとに測った列と種類の数を出す', async () => {
    shown = {
      ...audio,
      description: {
        ...audio.description!,
        task: 'regression',
        target: 'visitors',
        classes: [],
        uncertainty: {
          split: 'recent',
          rows: 120,
          kind: 'absolute',
          measured_at: null,
          coverage: null,
          ece: null,
          groups: { column: 'shop', count: 3 },
        },
      },
    }
    show()
    expect(await screen.findByText(/shop ごとに 3 種類で測った/)).toBeInTheDocument()
  })
})
