import { createContext, useContext } from 'react'

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

/** 殻で選んだ問題の設定の識別子の検索の文 (選んでいなければ空) */
const projectQuery = (project: string | null): string =>
  project ? `?project=${encodeURIComponent(project)}` : ''

const json = (method: string, body: unknown): RequestInit => ({
  method,
  body: JSON.stringify(body),
})

/**
 * API の頭 (単独では `/api`, 殻の下では `/api/infer`) の下に送る呼び出しの一式を作る
 *
 * @param base - API の頭 (末尾の `/` は無し)
 */
export function createApi(base: string) {
  return {
    info: () => request<AppInfo>(`${base}/info`),
    browse: (path: string) => request<Listing>(`${base}/browse?path=${encodeURIComponent(path)}`),
    bundles: () => request<Bundle[]>(`${base}/bundles`),
    bundle: (id: string) => request<Bundle>(`${base}/bundles/${id}`),
    registerBundle: (path: string, name?: string) =>
      request<Bundle>(`${base}/bundles`, json('POST', { path, name })),
    createBundle: (body: BundleCreateBody) =>
      request<Bundle>(`${base}/bundles/create`, json('POST', body)),
    readyPreview: (library: string, model: string, weights: string) =>
      request<ReadyPreview>(
        `${base}/ready/preview?${new URLSearchParams({ library, model, weights }).toString()}`,
      ),
    createReadyBundle: (body: ReadyCreateBody) =>
      request<Bundle>(`${base}/bundles/ready`, json('POST', body)),
    /** 実験の一覧 (project は殻で選んだ問題の設定の識別子. 渡せばそのプロジェクトの実験だけ) */
    experiments: (project: string | null = null) =>
      request<ExperimentList>(`${base}/experiments${projectQuery(project)}`),
    policyExperiments: (project: string | null = null) =>
      request<ExperimentList<PolicyExperimentRow>>(
        `${base}/experiments/policies${projectQuery(project)}`,
      ),
    describeDataset: (path: string) =>
      request<DatasetDescription>(`${base}/datasets/describe?path=${encodeURIComponent(path)}`),
    addSamples: () => request<Bundle[]>(`${base}/samples`, json('POST', {})),
    restoreUncertainty: (id: string) =>
      request<Bundle>(`${base}/bundles/${id}/restore-uncertainty`, json('POST', {})),
    deleteBundle: (id: string) => request<void>(`${base}/bundles/${id}`, { method: 'DELETE' }),
    /** つないだ問題の設定 (project は殻で選んだ問題の設定の識別子. 一式の上に候補が無い場合の候補) */
    bundleProblem: (id: string, project: string | null = null) =>
      request<ProblemState>(`${base}/bundles/${id}/problem${projectQuery(project)}`),
    linkProblem: (id: string, path: string | null, project: string | null = null) =>
      request<ProblemState>(
        `${base}/bundles/${id}/problem${projectQuery(project)}`,
        json('PUT', { path }),
      ),
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
    ) => request<Result>(`${base}/bundles/${id}/predict`, json('POST', body)),
    results: (bundle?: string) =>
      request<Result[]>(`${base}/results${bundle ? `?bundle=${encodeURIComponent(bundle)}` : ''}`),
    result: (id: string) => request<Result>(`${base}/results/${id}`),
    compare: (ids: string[]) =>
      request<Comparison>(`${base}/results/compare?ids=${ids.map(encodeURIComponent).join(',')}`),
    rows: (
      id: string,
      params: {
        offset: number
        limit: number
        status?: string
        decision?: string
        group?: number
        /** 群の中心から遠い行 (true) か遠くない行 (false) に絞る */
        far?: boolean
        row?: number
      },
    ) => {
      const search = new URLSearchParams()
      for (const [key, value] of Object.entries(params)) {
        if (value !== undefined && value !== '') search.set(key, String(value))
      }
      return request<Rows>(`${base}/results/${id}/rows?${search.toString()}`)
    },
    representatives: (id: string, count: number) =>
      request<Representatives>(`${base}/results/${id}/representatives?count=${count}`),
    treated: (id: string, offset: number, limit: number) =>
      request<TreatedRows>(`${base}/results/${id}/treated?offset=${offset}&limit=${limit}`),
    prices: (id: string, offset: number, limit: number) =>
      request<PriceRows>(`${base}/results/${id}/prices?offset=${offset}&limit=${limit}`),
    orders: (id: string, offset: number, limit: number) =>
      request<OrderRows>(`${base}/results/${id}/orders?offset=${offset}&limit=${limit}`),
    queue: (id: string) => request<ReviewQueue>(`${base}/results/${id}/queue`),
    reviews: (id: string) => request<{ records: ReviewRecord[] }>(`${base}/results/${id}/reviews`),
    addReview: (id: string, row: string | number, judgment: Judgment, reviewer: string) =>
      request<ReviewRecord>(
        `${base}/results/${id}/reviews`,
        json('POST', { row, judgment, reviewer }),
      ),
    conclude: (id: string, rules: Rules) =>
      request<Result>(`${base}/results/${id}/conclude`, json('POST', { rules })),
    decide: (id: string, decision: Decision | null) =>
      request<Result>(`${base}/results/${id}/decide`, json('POST', { decision })),
    backtest: (id: string) => request<Result>(`${base}/results/${id}/backtest`, json('POST', {})),
    remeasure: (id: string, write: boolean, horizon?: number, time?: string, groups?: string) =>
      request<Remeasure>(
        `${base}/results/${id}/remeasure`,
        json('POST', { write, horizon, time, groups }),
      ),
    explain: (id: string) => request<Result>(`${base}/results/${id}/explain`, json('POST', {})),
    rowReason: (id: string, row: number) =>
      request<RowReason>(`${base}/results/${id}/rows/${row}/reason`, json('POST', {})),
    downloadUrl: (id: string) => `${base}/results/${id}/download`,
    artifact: (id: string, row: number) =>
      request<ArtifactPreview>(`${base}/results/${id}/artifacts/${row}`),
    artifactUrl: (id: string, row: number) => `${base}/results/${id}/artifacts/${row}/file`,
    /** 検索の結果の, 問いの写真 (query) か索引の写真 (item) の縮小画像の場所 */
    searchPhotoUrl: (id: string, side: 'query' | 'item', path: string) =>
      `${base}/results/${id}/search-photo?${new URLSearchParams({ side, path }).toString()}`,
    groupPhotoUrl: (id: string, path: string) =>
      `${base}/results/${id}/group-photo?${new URLSearchParams({ path }).toString()}`,
    deleteResult: (id: string) => request<void>(`${base}/results/${id}`, { method: 'DELETE' }),
    policies: () => request<Policy[]>(`${base}/policies`),
    policy: (id: string) => request<Policy>(`${base}/policies/${id}`),
    registerPolicy: (path: string, name?: string) =>
      request<Policy>(`${base}/policies`, json('POST', { path, name })),
    deletePolicy: (id: string) => request<void>(`${base}/policies/${id}`, { method: 'DELETE' }),
    act: (id: string, observation: unknown) =>
      request<PolicyAct>(`${base}/policies/${id}/act`, json('POST', { observation })),
    policyBatch: (id: string, source: string) =>
      request<PolicyResult>(`${base}/policies/${id}/batch`, json('POST', { source })),
    policyResults: (policy?: string) =>
      request<PolicyResult[]>(
        `${base}/policy-results${policy ? `?policy=${encodeURIComponent(policy)}` : ''}`,
      ),
    policyResult: (id: string) => request<PolicyResult>(`${base}/policy-results/${id}`),
    policyResultRows: (id: string, params: { offset: number; limit: number; status?: string }) => {
      const query = new URLSearchParams({
        offset: String(params.offset),
        limit: String(params.limit),
      })
      if (params.status) query.set('status', params.status)
      return request<PolicyResultRows>(`${base}/policy-results/${id}/rows?${query}`)
    },
    policyResultDownloadUrl: (id: string) => `${base}/policy-results/${id}/download`,
    deletePolicyResult: (id: string) =>
      request<void>(`${base}/policy-results/${id}`, { method: 'DELETE' }),
  }
}

/** 呼び出しの一式の型 */
export type Api = ReturnType<typeof createApi>

/** 単独の頭 `/api` の一式 (根の部品の外の既定. 画面の部品は useApi を使う) */
export const api: Api = createApi('/api')

/** 根の部品が決めた API の頭の呼び出しの一式 */
export const ApiContext = createContext<Api>(api)

/** 根の部品が決めた API の頭の呼び出しの一式を使う */
export function useApi(): Api {
  return useContext(ApiContext)
}
