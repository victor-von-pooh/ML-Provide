import { useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { useState } from 'react'
import { Link, useSearchParams } from 'react-router'

import { api } from '../../api/client'
import type { Comparison } from '../../api/types'
import { Card, ErrorBox, PageHeader, Spinner } from '../../components/ui'
import { compareSections } from '../../lib/compare'

// 全ての結果で同じものの名前
const BASIS: Record<string, string> = { bundle: '同じ一式', source: '同じ入力のファイル' }

/** 比べの表 (行は項目, 列は結果. 違う行を目立たせる) */
function CompareTable({ data, onlyDiffers }: { data: Comparison; onlyDiffers: boolean }) {
  const sections = compareSections(data.results)
  return (
    <div className="overflow-auto rounded-xl border border-line bg-surface">
      <table className="w-full text-sm">
        <caption className="sr-only">
          {data.grouping
            ? '結果ごとの結論, 群の大きさ, 結果 1 からの割合の差, 群ごとの列の特徴 (違う行に「違う」の印)'
            : '結果ごとの結論, 判断の内訳, 意思決定 (違う行に「違う」の印)'}
        </caption>
        <thead className="text-xs text-ink-3">
          <tr className="border-b border-line">
            <th scope="col" className="px-4 py-2 text-left font-normal">
              項目
            </th>
            {data.results.map((result, index) => (
              <th key={result.id} scope="col" className="px-4 py-2 text-left font-normal">
                <Link to={`/results/${result.id}`} className="text-accent-text hover:underline">
                  結果 {index + 1}
                </Link>
              </th>
            ))}
          </tr>
        </thead>
        {sections.map((section) => {
          const rows = section.rows.filter((row) => !onlyDiffers || row.differs)
          if (rows.length === 0) return null
          return (
            <tbody key={section.title} className="border-b border-line last:border-0">
              <tr>
                <th
                  scope="colgroup"
                  colSpan={data.results.length + 1}
                  className="bg-surface-2 px-4 py-1.5 text-left text-xs font-semibold text-ink-2"
                >
                  {section.title}
                </th>
              </tr>
              {rows.map((row) => (
                <tr key={row.label} className={clsx('align-top', row.differs && 'bg-warning/10')}>
                  <th scope="row" className="px-4 py-1.5 text-left font-medium">
                    {row.label}
                    {row.differs && (
                      <span className="ml-2 rounded bg-warning/20 px-1.5 text-xs font-medium text-warning-text">
                        違う
                      </span>
                    )}
                  </th>
                  {row.values.map((value, index) => (
                    <td
                      key={data.results[index]?.id ?? index}
                      className="tabular max-w-xs px-4 py-1.5 break-words"
                    >
                      {value}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          )
        })}
      </table>
    </div>
  )
}

/** 選んだ結果どうしの比べ (同じ一式か同じ入力の結果だけ) */
export function ComparePage() {
  const [params] = useSearchParams()
  const ids = (params.get('ids') ?? '').split(',').filter(Boolean)
  const [onlyDiffers, setOnlyDiffers] = useState(false)
  const compare = useQuery({
    queryKey: ['results', 'compare', ids],
    queryFn: () => api.compare(ids),
  })
  const data = compare.data
  return (
    <>
      <PageHeader
        title="結果どうしの比べ"
        subtitle={
          data?.comparable
            ? `${data.basis.map((name) => BASIS[name]).join(', ')}の ${data.results.length} つの結果`
            : '同じ一式か同じ入力の結果を並べます'
        }
        breadcrumb={<Link to="/results">結果</Link>}
      />
      {compare.isPending && <Spinner />}
      {compare.error && <ErrorBox error={compare.error} live={false} />}
      {data && !data.comparable && (
        <Card title="比べられません">
          <ul className="mb-3 flex list-disc flex-col gap-1 pl-5 text-sm">
            {data.reasons.map((reason) => (
              <li key={reason}>{reason}</li>
            ))}
          </ul>
          <ul className="flex flex-col gap-1 text-sm">
            {data.results.map((result, index) => (
              <li key={result.id}>
                <Link to={`/results/${result.id}`} className="text-accent-text hover:underline">
                  結果 {index + 1}
                </Link>
                <span className="ml-2 text-xs text-ink-3">
                  {result.bundle?.name} / {result.source}
                </span>
              </li>
            ))}
          </ul>
        </Card>
      )}
      {data?.comparable && (
        <div className="flex flex-col gap-3">
          <label className="flex items-center gap-2 text-sm text-ink-2">
            <input
              type="checkbox"
              checked={onlyDiffers}
              onChange={(event) => setOnlyDiffers(event.target.checked)}
            />
            違う行だけを出す
          </label>
          {data.grouping && (
            <p className="text-xs text-ink-3">
              同じ群分けの一式の結果の, 群ごとの件数と割合, 学習のときの割合との差, 結果 1
              からの割合の差, 群を作る列の特徴 (数の列は平均, 種類の列は最も多い値と群の中の割合)
              です. 割り当てた行の無い群も 0 件で並べます
            </p>
          )}
          <CompareTable data={data} onlyDiffers={onlyDiffers} />
        </div>
      )}
    </>
  )
}
