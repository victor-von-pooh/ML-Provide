import { useQuery } from '@tanstack/react-query'
import { FolderInput } from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router'

import { useApi, type Api } from '../../api/client'
import type { ExperimentList, ExperimentRow, PolicyExperimentRow } from '../../api/types'
import { Dialog } from '../../components/Dialog'
import { Button, ErrorBox, Spinner } from '../../components/ui'
import { useStage } from '../../lib/stage'

/** 一覧から登録する一式の名前 (問題の名前と実験の場所) */
function experimentName(row: { problem: string; experiment: string }): string {
  return `${row.problem} (${row.experiment})`
}

/** 一覧の 1 行と, 登録済みなら開く画面の場所 */
interface Row {
  row: ExperimentRow | PolicyExperimentRow
  opened: string | null
}

/** 一式の種類ごとの, 一覧の読み方と文 */
const KINDS = {
  bundle: {
    key: 'experiments',
    load: async (api: Api, project: string | null): Promise<ExperimentList<Row>> => {
      const data = await api.experiments(project)
      return {
        ...data,
        experiments: data.experiments.map((row) => ({
          row,
          opened: row.bundle_id ? `/bundles/${row.bundle_id}` : null,
        })),
      }
    },
    caption: '推論の一式を持つ実験',
    intro: '学習が推論の一式 (inference) を作った実験',
    checking: '一式を確かめています',
  },
  policy: {
    key: 'policy-experiments',
    load: async (api: Api, project: string | null): Promise<ExperimentList<Row>> => {
      const data = await api.policyExperiments(project)
      return {
        ...data,
        experiments: data.experiments.map((row) => ({
          row,
          opened: row.policy_id ? `/policies/${row.policy_id}` : null,
        })),
      }
    },
    caption: '方策の一式を持つ実験',
    intro: '学習が方策の一式 (policy. policy.json を持つ) を書き出した実験',
    checking: '方策の一式を確かめています',
  },
} as const

/**
 * 問題に記録した学習のプロジェクトの, 推論の一式か方策の一式を持つ実験の一覧.
 * 場所を打たずに, 学習した実験の一式を登録する
 */
export function ExperimentsDialog({
  kind = 'bundle',
  onClose,
  onRegister,
  pending,
  error,
}: {
  /** 並べる一式の種類 (推論の一式か方策の一式) */
  kind?: keyof typeof KINDS
  onClose: () => void
  /** 一式の場所と, 登録する名前 (問題の名前と実験の場所) */
  onRegister: (path: string, name: string) => void
  pending: boolean
  /** 登録できなかった理由 */
  error: Error | null
}) {
  const api = useApi()
  // 殻で問題を選んでいれば, その問題が記録した学習のプロジェクトの実験に絞る
  // (上限で打ち切る前に絞るように API に渡す). 外したのはそのときの問題だけで,
  // 殻で選び直すとまた絞る
  const { project } = useStage()
  const [allFor, setAllFor] = useState<string | null>(null)
  const filtering = project !== null && allFor !== project
  const asked = filtering ? project : null
  // 開くたびに読み直す (学習が終わった実験を拾うため)
  const config = KINDS[kind]
  const listing = useQuery({
    queryKey: [config.key, asked],
    queryFn: () => config.load(api, asked),
    staleTime: 0,
  })
  const data = listing.data
  return (
    <Dialog title="実験の一覧から登録" onClose={onClose}>
      <p className="mb-3 text-xs text-ink-3">
        問題の設定の段階で学習のプロジェクトを書き出した問題から, {config.intro}
        を新しい順に並べます. 一式は写さずに, 場所だけを覚えます.
      </p>
      {project !== null && (
        <label className="mb-3 flex items-center gap-2 text-xs text-ink-2">
          <input
            type="checkbox"
            checked={!filtering}
            onChange={(event) => setAllFor(event.target.checked ? project : null)}
          />
          全ての問題の実験を出す (出さなければ, 選んだ問題 {project}
          の実験だけ)
        </label>
      )}
      {listing.isPending && <Spinner label="実験を探しています" />}
      {pending && <Spinner label={config.checking} />}
      {error && (
        <div className="mb-3">
          <ErrorBox error={error} />
        </div>
      )}
      {listing.error && <ErrorBox error={listing.error} />}
      {data && data.experiments.length === 0 && filtering && (
        <p role="status" className="text-sm text-ink-2">
          選んだ問題が記録した学習のプロジェクトに, 一式を持つ実験がありません.
          全ての問題の実験を出すか, 「場所で登録」を使います.
        </p>
      )}
      {data && data.experiments.length === 0 && !filtering && (
        <p role="status" className="text-sm text-ink-2">
          一式を持つ実験がありません ({data.projects} 個の学習のプロジェクトを見ました).
          問題の設定の段階で学習のプロジェクトを書き出して学習するか, 「場所で登録」を使います.
          問題の設定の段階の置き場: <span className="font-mono text-xs">{data.problems}</span>
        </p>
      )}
      {data && data.experiments.length > 0 && (
        <table className="w-full text-sm">
          <caption className="sr-only">{config.caption}</caption>
          <thead className="text-xs text-ink-3">
            <tr className="border-b border-line">
              <th scope="col" className="px-2 py-1 text-left font-normal">
                問題と実験
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
            {data.experiments.map(({ row, opened }) => (
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
                  {opened ? (
                    <Link
                      to={opened}
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
