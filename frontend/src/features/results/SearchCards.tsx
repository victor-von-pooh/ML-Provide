import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import { useApi } from '../../api/client'
import type { Result, SearchResultInfo } from '../../api/types'
import { Button, Card, ErrorBox, Spinner } from '../../components/common'
import { searchPage, type SearchQuery } from '../../lib/search'

/** 1 ページに出す問いの数 (行は問いの数 × 件数. API の 1 回の上限は 500 行) */
const QUERIES_PER_PAGE = 10

/** 待ち時間, 上限と, 骨組みの評価 */
function SearchSummaryCard({ search }: { search: SearchResultInfo }) {
  const evaluation = search.evaluation
  return (
    <Card title="待ち時間と評価">
      <dl className="grid gap-x-6 gap-y-2 text-sm md:grid-cols-2">
        <div>
          <dt className="text-xs text-ink-2">1 件の問いの待ち時間</dt>
          <dd>
            {search.latency
              ? `中央値 ${search.latency.median.toFixed(1)} ms, 最大 ${search.latency.max.toFixed(1)} ms`
              : '並べた問いがありません'}
          </dd>
        </div>
        <div>
          <dt className="text-xs text-ink-2">待ち時間の上限 (守る指標)</dt>
          <dd>
            {search.max_latency_ms === null
              ? '決めていません'
              : `${search.max_latency_ms.toLocaleString()} ms. 超えた問い ${(search.over_latency ?? 0).toLocaleString()} 枚`}
          </dd>
        </div>
        <div>
          <dt className="text-xs text-ink-2">索引の写真</dt>
          <dd>
            {search.items.toLocaleString()} 枚から上位 {search.k} 件
          </dd>
        </div>
        {evaluation ? (
          <>
            <div>
              <dt className="text-xs text-ink-2">
                Recall@K (上位 K 件に入った同じ種類の写真の割合)
              </dt>
              <dd>
                {evaluation.recall_at_k.toFixed(3)}{' '}
                <span className="text-xs text-ink-2">
                  (比べる相手 {evaluation.recall_at_k_baseline.toFixed(3)})
                </span>
              </dd>
            </div>
            <div>
              <dt className="text-xs text-ink-2">
                NDCG@K (上位 K 件の同じ種類の写真の位置で見た並びの良さ)
              </dt>
              <dd>
                {evaluation.ndcg_at_k.toFixed(3)}{' '}
                <span className="text-xs text-ink-2">
                  (比べる相手 {evaluation.ndcg_at_k_baseline.toFixed(3)})
                </span>
              </dd>
            </div>
            <div className="text-xs text-ink-2 md:col-span-2">
              骨組みが検証用の問い {evaluation.queries.toLocaleString()} 枚で数えた評価 (上位{' '}
              {evaluation.k} 件). 比べる相手は{evaluation.baseline ?? '写真を縮めた画素'}です
            </div>
          </>
        ) : (
          <div className="text-xs text-ink-2 md:col-span-2">
            索引の写真に種類が無いので評価はありません. 並んだ写真と近さを見て確かめます
          </div>
        )}
      </dl>
    </Card>
  )
}

/** 問いの写真と, 近い順の索引の写真 */
function QueryRow({ resultId, query }: { resultId: string; query: SearchQuery }) {
  const api = useApi()
  return (
    <li className="flex flex-col gap-2 border-t border-line px-4 py-3 first:border-t-0">
      <div className="flex flex-wrap items-baseline gap-2 text-sm">
        <span className="font-mono text-xs break-all">{query.path}</span>
        {query.latency !== null && (
          <span className="text-xs text-ink-2">待ち時間 {query.latency.toFixed(1)} ms</span>
        )}
      </div>
      {query.status !== 'ok' ? (
        <p className="text-xs text-critical-text">並べられませんでした: {query.reason}</p>
      ) : (
        <div className="flex gap-3 overflow-x-auto pb-1">
          <figure className="flex w-28 shrink-0 flex-col gap-1">
            <img
              src={api.searchPhotoUrl(resultId, 'query', query.path)}
              alt={`問いの写真 ${query.path}`}
              loading="lazy"
              className="size-28 rounded-lg border-2 border-accent object-cover"
            />
            <figcaption className="text-xs text-ink-2">問い</figcaption>
          </figure>
          {query.items.map((item) => (
            <figure key={`${item.rank}:${item.item}`} className="flex w-28 shrink-0 flex-col gap-1">
              <img
                src={api.searchPhotoUrl(resultId, 'item', item.item ?? '')}
                alt={`${item.rank} 位の写真 ${item.item ?? ''}`}
                loading="lazy"
                className="size-28 rounded-lg border border-line object-cover"
              />
              <figcaption className="text-xs">
                {item.rank} 位{item.label ? ` · ${item.label}` : ''}
                <span className="block text-ink-2">
                  近さ {item.similarity === null ? '–' : item.similarity.toFixed(3)}
                </span>
              </figcaption>
            </figure>
          ))}
        </div>
      )}
    </li>
  )
}

/** 検索の結果: 待ち時間と評価, 問いごとの似た写真 */
export function SearchCards({ result, search }: { result: Result; search: SearchResultInfo }) {
  const api = useApi()
  // ページごとの初めの行の位置 (問いの境目で切るので, 読んだページから次の位置が決まる)
  const [starts, setStarts] = useState<number[]>([0])
  const page = starts.length - 1
  const offset = starts[page] ?? 0
  // 1 ページは問いの数 × 件数の行 (並べられない問いは 1 行. 読んだ行は問いの境目で切る)
  const perQuery = Math.max(1, Math.min(search.k, search.items))
  const limit = Math.min(500, QUERIES_PER_PAGE * perQuery)
  const rows = useQuery({
    queryKey: ['search-rows', result.id, offset, limit],
    queryFn: () => api.rows(result.id, { offset, limit }),
    placeholderData: keepPreviousData,
  })
  const total = rows.data?.total ?? 0
  const shown = rows.data ? searchPage(rows.data.rows, offset, total) : null
  const queries = shown?.queries ?? []
  const next = shown?.next ?? null
  return (
    <>
      <SearchSummaryCard search={search} />
      <Card title="似た写真" padded={false}>
        {rows.isPending ? (
          <div className="p-4">
            <Spinner label="写真を読んでいます…" />
          </div>
        ) : rows.error ? (
          <div className="p-4">
            <ErrorBox error={rows.error} live={false} />
          </div>
        ) : (
          <>
            <ul aria-label="問いごとの似た写真">
              {queries.map((query, index) => (
                <QueryRow key={`${index}:${query.path}`} resultId={result.id} query={query} />
              ))}
            </ul>
            {(page > 0 || next !== null) && (
              <div className="flex items-center justify-end gap-2 border-t border-line px-4 py-2 text-sm">
                <Button disabled={page === 0} onClick={() => setStarts(starts.slice(0, -1))}>
                  前へ
                </Button>
                <span aria-live="polite">
                  {page + 1} ページ目 (行 {offset + 1} から, 全 {total.toLocaleString()} 行)
                </span>
                <Button
                  disabled={next === null}
                  onClick={() => next !== null && setStarts([...starts, next])}
                >
                  次へ
                </Button>
              </div>
            )}
          </>
        )}
      </Card>
    </>
  )
}
