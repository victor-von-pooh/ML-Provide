import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { Bundle, Description, ExperimentList, ProblemState } from './api/types'
import { ThemeProvider } from './lib/theme'
import { useTheme } from './lib/themeContext'
import { StageApp, type StageAppProps } from './StageApp'

/** 問題の設定の置き場 */
const ROOT = '/home/user/.ml-pathfinder/problems'

/** 問題の設定の欄を出す表の分類の一式 */
const BUNDLE: Bundle = {
  id: 'b1',
  name: '解約の予測',
  path: '/bundles/b1',
  managed: true,
  added_at: '2026-01-01T00:00:00',
  ok: true,
  error: null,
  description: {
    path: '/bundles/b1',
    name: 'b1',
    created_at: null,
    kind: 'table',
    task: 'classification',
    target: 'label',
    classes: ['a', 'b'],
    features: [],
    text_column: null,
    reading: { loader: 'csv', options: {} },
    steps: [],
    item_steps: [],
    model: { name: 'm', inputs: {}, output: null, kind: 'logits' },
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
  } as unknown as Description,
  uses_problem: true,
}

/** つないでいない一式の状態 (殻で選んだ問題の設定を候補にした形) */
function unlinked(project: string | null): ProblemState {
  return {
    path: null,
    problem_id: null,
    linked: null,
    error: null,
    hint: project ? `${ROOT}/${project}/problem.yaml` : null,
    hint_note: null,
    hint_project: project !== null,
  }
}

/** 2 つの問題の設定のプロジェクトの実験 */
const EXPERIMENTS: ExperimentList = {
  problems: ROOT,
  projects: 2,
  truncated: false,
  experiments: [
    {
      path: '/project-a/outputs/run/1/inference',
      project: '/project-a',
      problem: '解約の予測',
      problem_ids: ['aaaa'],
      experiment: 'run/1',
      modified: '2026-01-02T00:00:00',
      bundle_id: null,
    },
    {
      path: '/project-b/outputs/run/2/inference',
      project: '/project-b',
      problem: '総菜の作る量',
      problem_ids: ['bbbb'],
      experiment: 'run/2',
      modified: '2026-01-01T00:00:00',
      bundle_id: null,
    },
  ],
}

/**
 * API の頭 `/api/infer` の下だけに答える fetch (他の場所を呼べば 404)
 *
 * @param linkedId - つないだ後に返す問題の設定の識別子 (置き場の外なら null)
 */
function stubApi(linkedId: string | null = null) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)
    const method = init?.method ?? 'GET'
    if (!url.startsWith('/api/infer/')) return new Response(null, { status: 404 })
    const [path, query = ''] = url.slice('/api/infer'.length).split('?')
    const params = new URLSearchParams(query)
    if (path === '/info') return new Promise<Response>(() => {})
    if (path === '/bundles' && method === 'GET') return Response.json([BUNDLE])
    if (path === '/bundles/b1') return Response.json(BUNDLE)
    if (path === '/results') return Response.json([])
    if (path === '/experiments') {
      const project = params.get('project')
      return Response.json({
        ...EXPERIMENTS,
        experiments: EXPERIMENTS.experiments.filter(
          (row) => project === null || row.problem_ids.includes(project),
        ),
      })
    }
    if (path === '/bundles/b1/problem' && method === 'GET')
      return Response.json(unlinked(params.get('project')))
    if (path === '/bundles/b1/problem' && method === 'PUT') {
      const { path: linked } = JSON.parse(String(init?.body)) as { path: string }
      return Response.json({ ...unlinked(null), path: linked, problem_id: linkedId, hint: null })
    }
    return new Response(null, { status: 404 })
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

/** 殻の下の形 (経路の頭 `/infer`, API の頭 `/api/infer`) で描く */
function renderStage(props: Partial<StageAppProps> = {}) {
  const onProjectChange = vi.fn()
  const onProjectsChange = vi.fn()
  const element = (next: Partial<StageAppProps>) => (
    <StageApp
      basename="/infer"
      apiBase="/api/infer"
      project={null}
      onProjectChange={onProjectChange}
      onProjectsChange={onProjectsChange}
      embedded
      {...props}
      {...next}
    />
  )
  const view = render(element({}))
  return {
    ...view,
    onProjectChange,
    onProjectsChange,
    rerenderWith: (next: Partial<StageAppProps>) => view.rerender(element(next)),
  }
}

beforeEach(() => {
  // jsdom は対話の枠を開けないので, 開いた印だけを付ける
  HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
    this.open = true
  }
  window.history.replaceState(null, '', '/infer')
  document.documentElement.classList.remove('dark')
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('StageApp', () => {
  it('API を全て apiBase の下に呼び, 経路を basename の下に置く', async () => {
    const fetchMock = stubApi()
    renderStage()
    expect(await screen.findByRole('link', { name: '解約の予測' })).toHaveAttribute(
      'href',
      '/infer/bundles/b1',
    )
    expect(screen.getByRole('link', { name: '結果' })).toHaveAttribute('href', '/infer/results')
    const urls = fetchMock.mock.calls.map(([input]) => String(input))
    expect(urls.length).toBeGreaterThan(0)
    for (const url of urls) expect(url.startsWith('/api/infer/')).toBe(true)
  })

  it('embedded ではアプリの名前と明暗の切り替えを出さず, 左の欄の移動は出す', async () => {
    stubApi()
    renderStage()
    await screen.findByRole('link', { name: '解約の予測' })
    expect(screen.queryByText('ML-Provide')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /表示にする/ })).not.toBeInTheDocument()
    expect(screen.getByRole('link', { name: '推論の一式' })).toBeInTheDocument()
  })

  it('単独ではアプリの名前と明暗の切り替えを出し, 切り替えで html のクラスを書く', async () => {
    stubApi()
    renderStage({ embedded: false })
    expect(screen.getByText('ML-Provide')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '暗い表示にする' }))
    expect(document.documentElement.classList.contains('dark')).toBe(true)
    expect(window.localStorage.getItem('theme')).toBe('dark')
  })

  it('読むだけの明暗は html のクラスを書き換えず, 殻の切り替えを読み直す', async () => {
    function Shown() {
      return <p>{useTheme().theme}</p>
    }
    render(
      <ThemeProvider readOnly>
        <Shown />
      </ThemeProvider>,
    )
    expect(screen.getByText('light')).toBeInTheDocument()
    expect(document.documentElement.classList.contains('dark')).toBe(false)
    document.documentElement.classList.add('dark')
    expect(await screen.findByText('dark')).toBeInTheDocument()
    document.documentElement.classList.remove('dark')
    expect(await screen.findByText('light')).toBeInTheDocument()
  })

  it('部品の範囲に data-stage-root を付け, 対話の枠もその中に出す', async () => {
    stubApi()
    const { container } = renderStage()
    await screen.findByRole('link', { name: '解約の予測' })
    const root = container.querySelector('[data-stage-root]')
    expect(root).not.toBeNull()
    expect(root).toContainElement(screen.getByRole('navigation', { name: '移動' }))
    fireEvent.click(screen.getByRole('button', { name: '実験の一覧から登録' }))
    expect(root).toContainElement(await screen.findByRole('dialog'))
  })

  it('project を渡すと, 実験の一覧をその問題の設定の実験に絞り, 外せる', async () => {
    const fetchMock = stubApi()
    renderStage({ project: 'aaaa' })
    fireEvent.click(await screen.findByRole('button', { name: '実験の一覧から登録' }))
    const dialog = await screen.findByRole('dialog')
    expect(await within(dialog).findByText('run/1')).toBeInTheDocument()
    expect(within(dialog).queryByText('run/2')).not.toBeInTheDocument()
    // 絞りは上限で打ち切る前に効くように API に渡す
    const asked = fetchMock.mock.calls.map(([input]) => String(input))
    expect(asked).toContain('/api/infer/experiments?project=aaaa')
    fireEvent.click(within(dialog).getByRole('checkbox', { name: /全ての問題の設定の実験を出す/ }))
    expect(await within(dialog).findByText('run/2')).toBeInTheDocument()
  })

  it('project が null なら実験の一覧を絞らず, 絞りの欄も出さない', async () => {
    stubApi()
    renderStage()
    fireEvent.click(await screen.findByRole('button', { name: '実験の一覧から登録' }))
    const dialog = await screen.findByRole('dialog')
    expect(await within(dialog).findByText('run/1')).toBeInTheDocument()
    expect(within(dialog).getByText('run/2')).toBeInTheDocument()
    expect(within(dialog).queryByRole('checkbox')).not.toBeInTheDocument()
  })

  it('project を問題の設定の候補に渡し, 置き場の別の問題の設定をつなぐと知らせる', async () => {
    window.history.replaceState(null, '', '/infer/bundles/b1')
    const fetchMock = stubApi('bbbb')
    const { onProjectChange } = renderStage({ project: 'aaaa' })
    expect(await screen.findByText(/選んだプロジェクトの問題の設定を候補にしました/)).toBeVisible()
    const asked = fetchMock.mock.calls.map(([input]) => String(input))
    expect(asked).toContain('/api/infer/bundles/b1/problem?project=aaaa')
    fireEvent.click(screen.getByRole('button', { name: 'これをつなぐ' }))
    await waitFor(() => expect(onProjectChange).toHaveBeenCalledWith('bbbb'))
    // つなぐ (外す) 要求にも渡し, 外した後の候補に使う
    const linked = fetchMock.mock.calls.find(([, init]) => init?.method === 'PUT')
    expect(String(linked?.[0])).toBe('/api/infer/bundles/b1/problem?project=aaaa')
    expect(onProjectChange).toHaveBeenCalledTimes(1)
  })

  it('同じ識別子と, 置き場の外の問題の設定をつないだときは知らせない', async () => {
    for (const linkedId of ['aaaa', null]) {
      window.history.replaceState(null, '', '/infer/bundles/b1')
      stubApi(linkedId)
      const { onProjectChange, unmount } = renderStage({ project: 'aaaa' })
      fireEvent.click(await screen.findByRole('button', { name: 'これをつなぐ' }))
      expect(await screen.findByText(/つなぎました/)).toBeInTheDocument()
      expect(onProjectChange).not.toHaveBeenCalled()
      unmount()
    }
  })

  it('一式を作る欄の書きかけは, 外されるときと project が替わるときに待たずに下書きに書く', async () => {
    stubApi()
    const view = renderStage({ project: 'aaaa' })
    fireEvent.click(await screen.findByRole('button', { name: '学習の結果から作る' }))
    fireEvent.change(await screen.findByLabelText('名前'), { target: { value: '書きかけ' } })
    expect(window.localStorage.getItem('ml-provide:drafts')).toBeNull()
    // 殻でプロジェクトを替えると, 待たずに書く
    view.rerenderWith({ project: 'bbbb' })
    expect(window.localStorage.getItem('ml-provide:drafts')).toContain('書きかけ')
    // 外されるときも待たずに書く
    fireEvent.change(screen.getByLabelText('名前'), { target: { value: '書きかけ 2' } })
    view.unmount()
    expect(window.localStorage.getItem('ml-provide:drafts')).toContain('書きかけ 2')

    // 次に開くと戻し, 捨てると空の欄にして下書きも消す
    renderStage()
    fireEvent.click(await screen.findByRole('button', { name: '学習の結果から作る' }))
    expect(await screen.findByLabelText('名前')).toHaveValue('書きかけ 2')
    // 戻しただけでは書き直さない (下書きの時刻を保つ)
    const stored = window.localStorage.getItem('ml-provide:drafts')
    await act(async () => {
      await new Promise((resolve) => window.setTimeout(resolve, 1100))
    })
    expect(window.localStorage.getItem('ml-provide:drafts')).toBe(stored)
    fireEvent.click(screen.getByRole('button', { name: '書きかけを捨てる' }))
    expect(screen.getByLabelText('名前')).toHaveValue('')
    await act(async () => {})
    expect(window.localStorage.getItem('ml-provide:drafts') ?? '{}').not.toContain('書きかけ')
  })
})
