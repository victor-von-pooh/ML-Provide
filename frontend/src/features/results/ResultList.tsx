import { Link } from 'react-router'

import type { Result } from '../../api/types'

/** 結果の一覧 (新しい順. 一式の名前を出すかを選べる) */
export function ResultList({ results, showBundle }: { results: Result[]; showBundle: boolean }) {
  if (results.length === 0) return <p className="text-sm text-ink-3">まだ結果がありません.</p>
  return (
    <ul className="flex flex-col divide-y divide-line">
      {results.map((result) => (
        <li key={result.id} className="py-2">
          <Link
            to={`/results/${result.id}`}
            className="text-sm font-medium text-accent-text hover:underline"
          >
            {result.created_at ? new Date(result.created_at).toLocaleString() : result.id}
          </Link>
          {showBundle && result.bundle?.name && (
            <span className="ml-2 text-xs text-ink-3">{result.bundle.name}</span>
          )}
          <div className="truncate text-xs text-ink-3">{result.source}</div>
          <div className="text-xs text-ink-2">{result.sentences.slice(0, 2).join(' / ')}</div>
        </li>
      ))}
    </ul>
  )
}
