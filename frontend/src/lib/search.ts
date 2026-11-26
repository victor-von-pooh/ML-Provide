/** 表の 1 行 (問いと順位ごと) */
export interface SearchRow {
  path: string
  status: string
  reason: string | null
  rank: number | null
  item: string | null
  label: string | null
  similarity: number | null
  latency_ms: number | null
}

/** 問いごとにまとめた行 */
export interface SearchQuery {
  path: string
  status: string
  reason: string | null
  latency: number | null
  items: SearchRow[]
}

/** 表の値を文字か null に */
function text(value: unknown): string | null {
  return value === null || value === undefined || value === '' ? null : String(value)
}

/** 表の値を数か null に */
function number(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

/** 表の行を問いごとにまとめる (行は問いの順, 問いの中は順位の順) */
export function groupQueries(rows: Record<string, unknown>[]): SearchQuery[] {
  const queries: SearchQuery[] = []
  for (const raw of rows) {
    const row: SearchRow = {
      path: text(raw.path) ?? '',
      status: text(raw.status) ?? '',
      reason: text(raw.reason),
      rank: number(raw.rank),
      item: text(raw.item),
      label: text(raw.label),
      similarity: number(raw.similarity),
      latency_ms: number(raw.latency_ms),
    }
    const last = queries.at(-1)
    // 並べられない問いは 1 行, 並べた問いは 1 位から始まる
    if (last && last.path === row.path && row.status === 'ok' && (row.rank ?? 1) > 1) {
      last.items.push(row)
      continue
    }
    queries.push({
      path: row.path,
      status: row.status,
      reason: row.reason,
      latency: row.latency_ms,
      items: row.status === 'ok' ? [row] : [],
    })
  }
  return queries
}

/** 1 ページの問い (問いの境目で切る) と, 次のページの初めの行の位置 */
export interface SearchPage {
  queries: SearchQuery[]
  /** 次のページの初めの行の位置 (最後のページなら null) */
  next: number | null
}

/**
 * offset の行から読んだ行を, 問いの境目で切った 1 ページにする. 後ろに行が残るときは, 最後の問い
 * (途中で切れているかもしれない) を次のページに回す. ただし問いが 1 つしか無いページは, 切れていても
 * そのまま出して先へ進む (件数が 1 ページの行より多い場合)
 */
export function searchPage(
  rows: Record<string, unknown>[],
  offset: number,
  total: number,
): SearchPage {
  const queries = groupQueries(rows)
  const end = offset + rows.length
  if (end >= total || queries.length <= 1) {
    return { queries, next: end >= total ? null : end }
  }
  const last = queries.pop()
  const used = rows.length - (last ? Math.max(1, last.items.length) : 0)
  return { queries, next: offset + used }
}
