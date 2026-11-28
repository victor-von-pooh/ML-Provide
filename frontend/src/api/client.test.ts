import { afterEach, describe, expect, it, vi } from 'vitest'

import { request } from './client'

/** fetch を 1 回だけ, 決まった応答で差し替える */
function respond(status: number, statusText: string, body: string) {
  vi.stubGlobal(
    'fetch',
    vi.fn(() => Promise.resolve(new Response(body, { status, statusText }))),
  )
}

describe('API の誤りの文', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('見つからない物の文はそのまま出す', async () => {
    respond(404, 'Not Found', JSON.stringify({ detail: '一式 abc が見つかりません' }))
    await expect(request('/api/bundles/abc')).rejects.toThrow('一式 abc が見つかりません')
  })

  it('既定の 404 は API の経路の誤りと出す', async () => {
    respond(404, 'Not Found', JSON.stringify({ detail: 'Not Found' }))
    await expect(request('/api/nowhere')).rejects.toThrow('API の経路が見つかりません (404)')
  })

  it('入力の検証の誤りは欄ごとに 1 行にする', async () => {
    const detail = [
      { loc: ['body', 'name'], msg: 'Field required' },
      { loc: ['query', 'limit'], msg: 'Input should be a valid integer' },
    ]
    respond(422, 'Unprocessable Entity', JSON.stringify({ detail }))
    await expect(request('/api/bundles')).rejects.toThrow(
      'name: Field required\nlimit: Input should be a valid integer',
    )
  })

  it('本文が JSON でなければ状態の番号と文', async () => {
    respond(500, 'Internal Server Error', 'oops')
    await expect(request('/api/info')).rejects.toThrow('500 Internal Server Error')
  })
})
