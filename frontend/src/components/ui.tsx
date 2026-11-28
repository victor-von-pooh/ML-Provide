/**
 * 推論の段階に固有の部品. 共通の部品は common.tsx (殻の正の写し) から使い, ここでは
 * 共通の部品を組み合わせた段階の形 (パンくず, 読み込み中と開けないときの画面,
 * 欄の確かめの誤りの並び) だけを持つ
 */
import { Fragment } from 'react'
import { Link } from 'react-router'

import { problemIds } from '../lib/problems'
import { ErrorBox, FieldError, PageHeader, Spinner } from './common'

/** パンくずの 1 つの項目 (段階の頭の画面より下の画面へのリンク) */
export interface Crumb {
  to: string
  label: string
}

/**
 * パンくず. 頭に段階の名前「推論」で段階の頭の画面 (推論の一式の一覧) へのリンクを置き,
 * その後に items を並べる (殻の段階の並びと同じ名前で, 今どこにいるかが分かるように)
 */
export function Crumbs({ items = [] }: { items?: Crumb[] }) {
  return (
    <>
      <Link to="/">推論</Link>
      {items.map((item) => (
        <Fragment key={item.to}>
          {' / '}
          <Link to={item.to}>{item.label}</Link>
        </Fragment>
      ))}
    </>
  )
}

/**
 * 段階の頭の画面より下の画面の, 読み込み中 (見出しは経路の識別子) か開けないとき
 * (見出しは「<物>を開けません」) の画面. 誤りは画面を開いたときの読み込みの誤りなので
 * 読み上げない
 */
export function PageStatus({
  id,
  thing,
  error,
  crumbs,
}: {
  /** 経路の識別子 (読み込み中の見出し) */
  id: string
  /** 開く物の名前 (「<物>を開けません」) */
  thing: string
  /** 読み込めなかった理由 (無ければ読み込み中) */
  error: Error | null
  crumbs?: Crumb[]
}) {
  return (
    <>
      <PageHeader
        title={error ? `${thing}を開けません` : id}
        breadcrumb={<Crumbs items={crumbs} />}
      />
      {error ? <ErrorBox error={error} live={false} /> : <Spinner />}
    </>
  )
}

/**
 * 欄にまたがる確かめの誤りの並び (要求を送る前に画面で分かる, いくつもの欄の組み合わせの
 * 誤り). FieldError と同じく読み上げの役割を持たず (打つたびに読み上げないため), 送る
 * ボタンの aria-describedby から lib/problems の problemIds で結ぶ. 1 つの欄の誤りは
 * 欄の下に FieldError で出す
 */
export function FieldProblems({
  idBase,
  problems,
  className,
}: {
  idBase: string
  problems: string[]
  className?: string
}) {
  const ids = problemIds(idBase, problems)
  return (
    <div className={className}>
      {problems.map((problem, index) => (
        // 同じ文の誤りが 2 つあっても重ならないように, 順番と組にする
        <FieldError key={`${index}:${problem}`} id={ids[index] ?? ''}>
          {problem}
        </FieldError>
      ))}
    </div>
  )
}
