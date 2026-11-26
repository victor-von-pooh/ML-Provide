import type {
  AppInfo,
  ArtifactPreview,
  Bundle,
  BundleCreateBody,
  Comparison,
  DatasetDescription,
  Decision,
  ExperimentList,
  Judgment,
  Listing,
  OrderRows,
  Policy,
  PolicyAct,
  PolicyExperimentRow,
  PolicyResult,
  PolicyResultRows,
  PriceRows,
  ProblemState,
  ReadyCreateBody,
  ReadyPreview,
  Remeasure,
  Representatives,
  Result,
  ReviewQueue,
  ReviewRecord,
  RowReason,
  Rows,
  Rules,
  TreatedRows,
} from './types'

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
  info: () => request<AppInfo>('/api/info'),
  browse: (path: string) => request<Listing>(`/api/browse?path=${encodeURIComponent(path)}`),
  bundles: () => request<Bundle[]>('/api/bundles'),
  bundle: (id: string) => request<Bundle>(`/api/bundles/${id}`),
  registerBundle: (path: string, name?: string) =>
    request<Bundle>('/api/bundles', json('POST', { path, name })),
  createBundle: (body: BundleCreateBody) =>
    request<Bundle>('/api/bundles/create', json('POST', body)),
  readyPreview: (library: string, model: string, weights: string) =>
    request<ReadyPreview>(
      `/api/ready/preview?${new URLSearchParams({ library, model, weights }).toString()}`,
    ),
  createReadyBundle: (body: ReadyCreateBody) =>
    request<Bundle>('/api/bundles/ready', json('POST', body)),
  experiments: () => request<ExperimentList>('/api/experiments'),
  policyExperiments: () =>
    request<ExperimentList<PolicyExperimentRow>>('/api/experiments/policies'),
  describeDataset: (path: string) =>
    request<DatasetDescription>(`/api/datasets/describe?path=${encodeURIComponent(path)}`),
  addSamples: () => request<Bundle[]>('/api/samples', json('POST', {})),
  restoreUncertainty: (id: string) =>
    request<Bundle>(`/api/bundles/${id}/restore-uncertainty`, json('POST', {})),
  deleteBundle: (id: string) => request<void>(`/api/bundles/${id}`, { method: 'DELETE' }),
  bundleProblem: (id: string) => request<ProblemState>(`/api/bundles/${id}/problem`),
  linkProblem: (id: string, path: string | null) =>
    request<ProblemState>(`/api/bundles/${id}/problem`, json('PUT', { path })),
  predict: (
    id: string,
    body: {
      source?: string
      texts?: string[]
      /** 雑音だけの一式の件数と種 */
      count?: number
      seed?: number
      rules?: Rules
      forecast?: boolean
      /** 効果の一式は効果の上位から配る判断, 値段の一式は値段の判断, 需要の一式は発注の判断 */
      decision?:
        | Decision
        | Record<string, never>
        | { type: 'treat'; budget: number }
        | { type: 'price'; candidates: number[]; max_change: number }
        | { type: 'order'; shortage_cost: number; holding_cost: number; max_order: number }
      /** 並べるの一式の利用者 (書いた値), 件数, 前に反応した品物の表と正解の反応の表 */
      users?: string[]
      k?: number
      seen?: string
      truth?: string
    },
  ) => request<Result>(`/api/bundles/${id}/predict`, json('POST', body)),
  results: (bundle?: string) =>
    request<Result[]>(`/api/results${bundle ? `?bundle=${encodeURIComponent(bundle)}` : ''}`),
  result: (id: string) => request<Result>(`/api/results/${id}`),
  compare: (ids: string[]) =>
    request<Comparison>(`/api/results/compare?ids=${ids.map(encodeURIComponent).join(',')}`),
  rows: (
    id: string,
    params: {
      offset: number
      limit: number
      status?: string
      decision?: string
      group?: number
      row?: number
    },
  ) => {
    const search = new URLSearchParams()
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== '') search.set(key, String(value))
    }
    return request<Rows>(`/api/results/${id}/rows?${search.toString()}`)
  },
  representatives: (id: string, count: number) =>
    request<Representatives>(`/api/results/${id}/representatives?count=${count}`),
  treated: (id: string, offset: number, limit: number) =>
    request<TreatedRows>(`/api/results/${id}/treated?offset=${offset}&limit=${limit}`),
  prices: (id: string, offset: number, limit: number) =>
    request<PriceRows>(`/api/results/${id}/prices?offset=${offset}&limit=${limit}`),
  orders: (id: string, offset: number, limit: number) =>
    request<OrderRows>(`/api/results/${id}/orders?offset=${offset}&limit=${limit}`),
  queue: (id: string) => request<ReviewQueue>(`/api/results/${id}/queue`),
  reviews: (id: string) => request<{ records: ReviewRecord[] }>(`/api/results/${id}/reviews`),
  addReview: (id: string, row: string | number, judgment: Judgment, reviewer: string) =>
    request<ReviewRecord>(`/api/results/${id}/reviews`, json('POST', { row, judgment, reviewer })),
  conclude: (id: string, rules: Rules) =>
    request<Result>(`/api/results/${id}/conclude`, json('POST', { rules })),
  decide: (id: string, decision: Decision | null) =>
    request<Result>(`/api/results/${id}/decide`, json('POST', { decision })),
  backtest: (id: string) => request<Result>(`/api/results/${id}/backtest`, json('POST', {})),
  remeasure: (id: string, write: boolean, horizon?: number, time?: string, groups?: string) =>
    request<Remeasure>(
      `/api/results/${id}/remeasure`,
      json('POST', { write, horizon, time, groups }),
    ),
  explain: (id: string) => request<Result>(`/api/results/${id}/explain`, json('POST', {})),
  rowReason: (id: string, row: number) =>
    request<RowReason>(`/api/results/${id}/rows/${row}/reason`, json('POST', {})),
  downloadUrl: (id: string) => `/api/results/${id}/download`,
  artifact: (id: string, row: number) =>
    request<ArtifactPreview>(`/api/results/${id}/artifacts/${row}`),
  artifactUrl: (id: string, row: number) => `/api/results/${id}/artifacts/${row}/file`,
  deleteResult: (id: string) => request<void>(`/api/results/${id}`, { method: 'DELETE' }),
  policies: () => request<Policy[]>('/api/policies'),
  policy: (id: string) => request<Policy>(`/api/policies/${id}`),
  registerPolicy: (path: string, name?: string) =>
    request<Policy>('/api/policies', json('POST', { path, name })),
  deletePolicy: (id: string) => request<void>(`/api/policies/${id}`, { method: 'DELETE' }),
  act: (id: string, observation: unknown) =>
    request<PolicyAct>(`/api/policies/${id}/act`, json('POST', { observation })),
  policyBatch: (id: string, source: string) =>
    request<PolicyResult>(`/api/policies/${id}/batch`, json('POST', { source })),
  policyResults: (policy?: string) =>
    request<PolicyResult[]>(
      `/api/policy-results${policy ? `?policy=${encodeURIComponent(policy)}` : ''}`,
    ),
  policyResult: (id: string) => request<PolicyResult>(`/api/policy-results/${id}`),
  policyResultRows: (id: string, params: { offset: number; limit: number; status?: string }) => {
    const query = new URLSearchParams({
      offset: String(params.offset),
      limit: String(params.limit),
    })
    if (params.status) query.set('status', params.status)
    return request<PolicyResultRows>(`/api/policy-results/${id}/rows?${query}`)
  },
  policyResultDownloadUrl: (id: string) => `/api/policy-results/${id}/download`,
  deletePolicyResult: (id: string) =>
    request<void>(`/api/policy-results/${id}`, { method: 'DELETE' }),
}
