import { useQuery } from '@tanstack/react-query'
import { FolderInput } from 'lucide-react'
import { Link } from 'react-router'

import { api } from '../../api/client'
import type { ExperimentRow } from '../../api/types'
import { Dialog } from '../../components/Dialog'
import { Button, ErrorBox, Spinner } from '../../components/ui'

/** 一覧から登録する一式の名前 (問題の設定の名前と実験の場所) */
function experimentName(row: ExperimentRow): string {
  return `${row.problem} (${row.experiment})`
}

/**
 * 問題の設定 (ML-Pathfinder) に記録したプロジェクトの, 推論の一式を持つ実験の一覧.
 * 場所を打たずに, 学習した実験の一式を登録する
 */
export function ExperimentsDialog({
  onClose,
  onRegister,
  pending,
  error,
}: {
  onClose: () => void
  /** 一式の場所と, 登録する名前 (問題の設定の名前と実験の場所) */
  onRegister: (path: string, name: string) => void
  pending: boolean
  /** 登録できなかった理由 */
  error: Error | null
}) {
  // 開くたびに読み直す (学習が終わった実験を拾うため)
  const listing = useQuery({
    queryKey: ['experiments'],
    queryFn: api.experiments,
    staleTime: 0,
  })
  const data = listing.data
  return (
    <Dialog title="実験の一覧から登録" onClose={onClose}>
      <p className="mb-3 text-xs text-ink-3">
        ML-Pathfinder で骨組みを書き出した問題の設定のプロジェクトから, 学習が推論の一式 (inference)
        を作った実験を新しい順に並べます. 一式は写さずに, 場所だけを覚えます.
      </p>
      {listing.isPending && <Spinner label="実験を探しています" />}
      {pending && <Spinner label="一式を確かめています" />}
      {error && (
        <div className="mb-3">
          <ErrorBox error={error} />
        </div>
      )}
      {listing.error && <ErrorBox error={listing.error} />}
      {data && data.experiments.length === 0 && (
        <p role="status" className="text-sm text-ink-2">
          一式を持つ実験がありません ({data.projects} 個のプロジェクトを見ました). ML-Pathfinder
          で骨組みを書き出して学習するか, 「場所で登録」を使います. 問題の設定の置き場:{' '}
          <span className="font-mono text-xs">{data.problems}</span>
        </p>
      )}
      {data && data.experiments.length > 0 && (
        <table className="w-full text-sm">
          <caption className="sr-only">推論の一式を持つ実験</caption>
          <thead className="text-xs text-ink-3">
            <tr className="border-b border-line">
              <th scope="col" className="px-2 py-1 text-left font-normal">
                問題の設定と実験
              </th>
              <th scope="col" className="px-2 py-1 text-left font-normal">
                一式を作った時刻
              </th>
              <th scope="col" className="px-2 py-1 text-right font-normal">
                <span className="sr-only">操作</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {data.experiments.map((row) => (
              <tr key={row.path} className="border-b border-line last:border-0">
                <th scope="row" className="px-2 py-2 text-left font-normal">
                  <span className="font-medium">{row.problem}</span>{' '}
                  <span className="font-mono text-xs">{row.experiment}</span>
                  <span className="block truncate font-mono text-xs text-ink-3" title={row.path}>
                    {row.path}
                  </span>
                </th>
                <td className="px-2 py-2 text-ink-2">
                  {new Date(row.modified).toLocaleString('ja-JP')}
                </td>
                <td className="px-2 py-2 text-right">
                  {row.bundle_id ? (
                    <Link
                      to={`/bundles/${row.bundle_id}`}
                      onClick={onClose}
                      className="text-xs text-accent-text hover:underline"
                    >
                      登録済み
                    </Link>
                  ) : (
                    <Button
                      icon={FolderInput}
                      disabled={pending}
                      aria-label={`${experimentName(row)} を登録`}
                      onClick={() => onRegister(row.path, experimentName(row))}
                    >
                      登録
                    </Button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {data?.truncated && (
        <p className="mt-2 text-xs text-warning-text">
          実験が多いので, 一覧は途中までです. 他の一式は「場所で登録」で選びます.
        </p>
      )}
    </Dialog>
  )
}
