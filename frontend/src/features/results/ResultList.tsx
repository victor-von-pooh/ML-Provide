import { Link } from 'react-router'

import type { Result } from '../../api/types'

/** 結果の一覧 (新しい順. 一式の名前を出すかを選べる. onToggle を渡すと比べる結果を選べる) */
export function ResultList({
  results,
  showBundle,
  selected,
  onToggle,
}: {
  results: Result[]
  showBundle: boolean
  selected?: ReadonlySet<string>
  onToggle?: (id: string) => void
}) {
  if (results.length === 0) return <p className="text-sm text-ink-3">まだ結果がありません.</p>
  return (
    <ul className="flex flex-col divide-y divide-line">
      {results.map((result) => {
        const time = result.created_at ? new Date(result.created_at).toLocaleString() : result.id
        return (
          <li key={result.id} className="flex items-start gap-3 py-2">
            {onToggle && (
              <input
                type="checkbox"
                className="mt-1"
                checked={selected?.has(result.id) ?? false}
                onChange={() => onToggle(result.id)}
                aria-label={`${time} の結果を比べる`}
              />
            )}
            <div className="min-w-0 flex-1">
              <Link
                to={`/results/${result.id}`}
                className="text-sm font-medium text-accent-text hover:underline"
              >
                {time}
              </Link>
              {showBundle && result.bundle?.name && (
                <span className="ml-2 text-xs text-ink-3">{result.bundle.name}</span>
              )}
              <div className="truncate text-xs text-ink-3">{result.source}</div>
              <div className="text-xs text-ink-2">{result.sentences.slice(0, 2).join(' / ')}</div>
            </div>
          </li>
        )
      })}
    </ul>
  )
}
