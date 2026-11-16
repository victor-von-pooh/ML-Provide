import type { Bundle, BundleCreateBody, Listing, Result, Rows, Rules } from './types'

/** API がエラーを返したときの例外 */
export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

/** エラーのレスポンスから表示する文を取り出す */
async function readError(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json()
    if (body && typeof body === 'object' && 'detail' in body) {
      const detail = (body as { detail: unknown }).detail
      if (typeof detail === 'string') return detail
      return JSON.stringify(detail)
    }
  } catch {
    // JSON でない場合は状態の文を使う
  }
  return `${response.status} ${response.statusText}`
}

/** JSON の API を呼ぶ */
export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!response.ok) {
    throw new ApiError(response.status, await readError(response))
  }
  // 中身の無い応答 (消した場合の 204) は undefined
  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  body: JSON.stringify(body),
})

export const api = {
  browse: (path: string) => request<Listing>(`/api/browse?path=${encodeURIComponent(path)}`),
  bundles: () => request<Bundle[]>('/api/bundles'),
  bundle: (id: string) => request<Bundle>(`/api/bundles/${id}`),
  registerBundle: (path: string, name?: string) =>
    request<Bundle>('/api/bundles', json('POST', { path, name })),
  createBundle: (body: BundleCreateBody) =>
    request<Bundle>('/api/bundles/create', json('POST', body)),
  addSamples: () => request<Bundle[]>('/api/samples', json('POST', {})),
  deleteBundle: (id: string) => request<void>(`/api/bundles/${id}`, { method: 'DELETE' }),
  predict: (id: string, body: { source?: string; texts?: string[]; rules?: Rules }) =>
    request<Result>(`/api/bundles/${id}/predict`, json('POST', body)),
  results: (bundle?: string) =>
    request<Result[]>(`/api/results${bundle ? `?bundle=${encodeURIComponent(bundle)}` : ''}`),
  result: (id: string) => request<Result>(`/api/results/${id}`),
  rows: (
    id: string,
    params: { offset: number; limit: number; status?: string; decision?: string },
  ) => {
    const search = new URLSearchParams()
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== '') search.set(key, String(value))
    }
    return request<Rows>(`/api/results/${id}/rows?${search.toString()}`)
  },
  conclude: (id: string, rules: Rules) =>
    request<Result>(`/api/results/${id}/conclude`, json('POST', { rules })),
  explain: (id: string) => request<Result>(`/api/results/${id}/explain`, json('POST', {})),
  downloadUrl: (id: string) => `/api/results/${id}/download`,
  deleteResult: (id: string) => request<void>(`/api/results/${id}`, { method: 'DELETE' }),
}
