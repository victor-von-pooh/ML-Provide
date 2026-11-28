import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { useId, useState } from 'react'

import { useApi } from '../../api/client'
import type {
  EffectDistribution,
  QiniCurve,
  Result,
  UpliftEvaluation,
  UpliftMeasured,
  UpliftSummary,
} from '../../api/types'
import { Button, Card, ErrorBox, Spinner } from '../../components/common'
import { EVALUATION_SPLITS } from '../../lib/anomaly'
import { formatNumber, formatValue } from '../../lib/format'
import {
  effectText,
  GROUPS_TEXT,
  liftText,
  PREMISE_TEXT,
  QINI_TEXT,
  resultK,
  topText,
  TREATED_PAGE,
  treatSummary,
  UPLIFT_UNSUPPORTED,
} from '../../lib/uplift'

/** 表の見出しの 1 つ */
function Heading({ children, right = false }: { children: React.ReactNode; right?: boolean }) {
  return (
    <th
      scope="col"
      className={`px-3 py-2 font-normal whitespace-nowrap ${right ? 'text-right' : 'text-left'}`}
    >
      {children}
    </th>
  )
}

/** 数の欄 */
function NumberCell({ children }: { children: React.ReactNode }) {
  return <td className="tabular-nums px-3 py-1.5 text-right">{children}</td>
}

/**
 * 結論の文に初めて出る言葉 (効果, 処置の群と対照の群, Qini の曲線と係数, 上位 K 割) の説明と,
 * 無作為に配った記録を前提にした見積もりという前提
 */
export function UpliftTerms({
  outcome,
  k,
}: {
  outcome: string | null | undefined
  k: number | null
}) {
  return (
    <dl
      aria-label="言葉の説明"
      className="mb-4 grid gap-x-4 gap-y-1 text-xs text-ink-2 md:grid-cols-[10rem_1fr]"
    >
      <dt className="font-medium text-ink">効果</dt>
      <dd>{effectText(outcome)}</dd>
      <dt className="font-medium text-ink">処置の群と対照の群</dt>
      <dd>{GROUPS_TEXT}</dd>
      <dt className="font-medium text-ink">Qini の曲線と係数</dt>
      <dd>{QINI_TEXT}</dd>
      <dt className="font-medium text-ink">{topText(k)} の効果の差</dt>
      <dd>{liftText(k)}</dd>
      <dt className="font-medium text-ink">前提</dt>
      <dd>{PREMISE_TEXT}</dd>
    </dl>
  )
}

/** 曲線の図の大きさと余白 (viewBox の単位) */
const WIDTH = 560
const HEIGHT = 220
const PAD = { left: 48, right: 12, top: 10, bottom: 30 }

/** 縦軸の目盛り (きりのよい 4 つほどの値) */
function ticks(low: number, high: number): number[] {
  const span = high - low || 1
  const raw = span / 4
  const step = 10 ** Math.floor(Math.log10(raw))
  const unit = [1, 2, 5, 10].map((m) => m * step).find((m) => raw <= m) ?? step * 10
  const found: number[] = []
  for (let value = Math.ceil(low / unit) * unit; value <= high + 1e-12; value += unit)
    found.push(Number(value.toFixed(10)))
  return found
}

/**
 * Qini の曲線とでたらめに配る線の図 (1 つの分割). 横は配った行の割合, 縦は増えた結果の積み上げ.
 * 指か矢印のキーで点を選ぶと, その点の値を出す
 */
export function QiniChart({ name, curve }: { name: string; curve: QiniCurve }) {
  const tableId = useId()
  const [focus, setFocus] = useState<number | null>(null)
  const count = curve.fraction.length
  if (count < 2) return <p className="text-sm text-ink-2">{name}の曲線の点が足りません</p>
  const values = [...curve.qini, ...curve.random, 0]
  const low = Math.min(...values)
  const high = Math.max(...values)
  const yTicks = ticks(low, high)
  const bottom = Math.min(low, yTicks[0] ?? low)
  const top = Math.max(high, yTicks[yTicks.length - 1] ?? high)
  const x = (fraction: number) => PAD.left + fraction * (WIDTH - PAD.left - PAD.right)
  const y = (value: number) =>
    PAD.top + (1 - (value - bottom) / (top - bottom || 1)) * (HEIGHT - PAD.top - PAD.bottom)
  const line = (series: number[]) =>
    series
      .map(
        (value, index) =>
          `${index ? 'L' : 'M'}${x(curve.fraction[index] ?? 0).toFixed(1)},${y(value).toFixed(1)}`,
      )
      .join(' ')
  const pick = (clientX: number, box: DOMRect) => {
    const fraction = ((clientX - box.left) / box.width) * WIDTH
    const share = (fraction - PAD.left) / (WIDTH - PAD.left - PAD.right)
    let best = 0
    curve.fraction.forEach((value, index) => {
      if (Math.abs(value - share) < Math.abs((curve.fraction[best] ?? 0) - share)) best = index
    })
    setFocus(best)
  }
  const last = count - 1
  // 選んだ点の値 (選んでいなければ null)
  const point =
    focus === null
      ? null
      : {
          fraction: curve.fraction[focus] ?? 0,
          rows: curve.rows[focus] ?? 0,
          qini: curve.qini[focus] ?? 0,
          random: curve.random[focus] ?? 0,
        }
  return (
    <figure className="flex flex-col gap-2">
      <figcaption className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-2">
        <span className="font-medium text-ink">{name}</span>
        <span className="inline-flex items-center gap-1.5">
          <svg width="20" height="8" aria-hidden>
            <line
              x1="0"
              y1="4"
              x2="20"
              y2="4"
              strokeWidth="2"
              style={{ stroke: 'var(--series-1)' }}
            />
          </svg>
          Qini の曲線 (効果の高い順に配る)
        </span>
        <span className="inline-flex items-center gap-1.5">
          <svg width="20" height="8" aria-hidden>
            <line
              x1="0"
              y1="4"
              x2="20"
              y2="4"
              strokeWidth="1.5"
              strokeDasharray="4 3"
              style={{ stroke: 'var(--text-3)' }}
            />
          </svg>
          でたらめに配る線
        </span>
        <span className="text-ink-3">縦は配ったことで増えた結果の積み上げ</span>
      </figcaption>
      <div className="relative">
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          className="w-full touch-none"
          role="img"
          tabIndex={0}
          aria-label={`${name}の Qini の曲線. 全てに配ったときの増えた結果 ${formatValue(curve.qini[last])}. 値は下の表でも読めます`}
          aria-describedby={tableId}
          onPointerMove={(event) =>
            pick(event.clientX, event.currentTarget.getBoundingClientRect())
          }
          onPointerLeave={() => setFocus(null)}
          onBlur={() => setFocus(null)}
          onKeyDown={(event) => {
            if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return
            event.preventDefault()
            const step = event.key === 'ArrowRight' ? 1 : -1
            setFocus((now) => Math.min(last, Math.max(0, (now ?? (step > 0 ? -1 : count)) + step)))
          }}
        >
          {yTicks.map((value) => (
            <g key={value}>
              <line
                x1={PAD.left}
                x2={WIDTH - PAD.right}
                y1={y(value)}
                y2={y(value)}
                strokeWidth={value === 0 ? 1 : 0.5}
                style={{ stroke: 'var(--border)' }}
              />
              <text
                x={PAD.left - 6}
                y={y(value)}
                textAnchor="end"
                dominantBaseline="middle"
                fontSize="10"
                style={{ fill: 'var(--text-3)' }}
              >
                {formatValue(value)}
              </text>
            </g>
          ))}
          {[0, 0.25, 0.5, 0.75, 1].map((fraction) => (
            <text
              key={fraction}
              x={x(fraction)}
              y={HEIGHT - PAD.bottom + 14}
              textAnchor={fraction === 0 ? 'start' : fraction === 1 ? 'end' : 'middle'}
              fontSize="10"
              style={{ fill: 'var(--text-3)' }}
            >
              {fraction * 100} %
            </text>
          ))}
          <text
            x={(PAD.left + WIDTH - PAD.right) / 2}
            y={HEIGHT - 2}
            textAnchor="middle"
            fontSize="10"
            style={{ fill: 'var(--text-2)' }}
          >
            効果の高い順に配った行の割合
          </text>
          <path
            d={line(curve.random)}
            fill="none"
            strokeWidth="1.5"
            strokeDasharray="4 3"
            style={{ stroke: 'var(--text-3)' }}
          />
          <path
            d={line(curve.qini)}
            fill="none"
            strokeWidth="2"
            strokeLinejoin="round"
            style={{ stroke: 'var(--series-1)' }}
          />
          {point && (
            <g>
              <line
                x1={x(point.fraction)}
                x2={x(point.fraction)}
                y1={PAD.top}
                y2={HEIGHT - PAD.bottom}
                strokeWidth="1"
                style={{ stroke: 'var(--text-3)' }}
              />
              <circle
                cx={x(point.fraction)}
                cy={y(point.random)}
                r="4"
                strokeWidth="2"
                style={{ fill: 'var(--text-3)', stroke: 'var(--surface)' }}
              />
              <circle
                cx={x(point.fraction)}
                cy={y(point.qini)}
                r="4"
                strokeWidth="2"
                style={{ fill: 'var(--series-1)', stroke: 'var(--surface)' }}
              />
            </g>
          )}
        </svg>
        {point && (
          <div
            role="status"
            className="pointer-events-none absolute top-1 rounded-lg border border-line bg-surface px-2 py-1 text-xs shadow-sm"
            style={
              point.fraction > 0.6
                ? { right: `${100 - (x(point.fraction) / WIDTH) * 100 + 2}%` }
                : { left: `${(x(point.fraction) / WIDTH) * 100 + 2}%` }
            }
          >
            <div className="text-ink-2">
              上位 {(point.fraction * 100).toFixed(0)} % ({formatNumber(point.rows)} 行)
            </div>
            <div>
              Qini の曲線{' '}
              <span className="tabular-nums font-semibold">{formatValue(point.qini)}</span>
            </div>
            <div>
              でたらめ <span className="tabular-nums">{formatValue(point.random)}</span>
            </div>
          </div>
        )}
      </div>
      <details className="text-xs text-ink-2">
        <summary className="cursor-pointer">{name}の曲線の値を表で見る</summary>
        <div className="mt-1 max-h-56 overflow-auto">
          <table id={tableId} className="w-full text-xs">
            <caption className="sr-only">{name}の Qini の曲線の値</caption>
            <thead className="sticky top-0 bg-surface text-ink-3">
              <tr className="border-b border-line">
                <Heading right>配った割合</Heading>
                <Heading right>配った行</Heading>
                <Heading right>Qini の曲線</Heading>
                <Heading right>でたらめ</Heading>
              </tr>
            </thead>
            <tbody>
              {curve.fraction.map((fraction, index) => (
                <tr key={index} className="border-b border-line">
                  <NumberCell>{(fraction * 100).toFixed(1)} %</NumberCell>
                  <NumberCell>{formatNumber(curve.rows[index])}</NumberCell>
                  <NumberCell>{formatValue(curve.qini[index])}</NumberCell>
                  <NumberCell>{formatValue(curve.random[index])}</NumberCell>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  )
}

/** 上位 K 割の効果の差 (上位に片方の群が無ければ無し) */
function liftCell(lift: number | null): string {
  return lift === null ? '無し (上位に片方の群が無い)' : formatValue(lift)
}

/**
 * 実験の Qini の係数と上位 K 割の効果の差の表 (分割ごと) と, Qini の曲線の図.
 * 無ければ実験の評価の無い一式と書く
 */
export function UpliftEvaluationsView({
  evaluations,
  curves,
  k,
}: {
  evaluations: Record<string, UpliftEvaluation>
  curves: Record<string, QiniCurve>
  k: number | null
}) {
  const entries = Object.entries(evaluations ?? {})
  const lines = Object.entries(curves ?? {})
  if (entries.length === 0 && lines.length === 0)
    return (
      <p className="text-sm text-ink-2">
        実験の評価が無いので, 効果の分布と上位の行を見て確かめます
      </p>
    )
  return (
    <div className="flex flex-col gap-4">
      {entries.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <caption className="sr-only">
              実験の評価 (分割ごとの Qini の係数と{topText(k)} の効果の差)
            </caption>
            <thead className="text-xs text-ink-3">
              <tr className="border-b border-line">
                <Heading>分割</Heading>
                <Heading right>行</Heading>
                <Heading right>処置の群</Heading>
                <Heading right>対照の群</Heading>
                <Heading right>Qini の係数</Heading>
                <Heading right>{topText(k)} の効果の差</Heading>
                <Heading right>全ての行の結果の差</Heading>
                <Heading right>全ての行の効果 (見積もり) の平均</Heading>
              </tr>
            </thead>
            <tbody>
              {entries.map(([split, item]) => (
                <tr key={split} className="border-b border-line">
                  <th scope="row" className="px-3 py-1.5 text-left font-medium">
                    {EVALUATION_SPLITS[split] ?? split}
                  </th>
                  <NumberCell>{formatNumber(item.rows)}</NumberCell>
                  <NumberCell>{formatNumber(item.treated)} 行</NumberCell>
                  <NumberCell>{formatNumber(item.control)} 行</NumberCell>
                  <NumberCell>{item.qini.toFixed(3)}</NumberCell>
                  <NumberCell>{liftCell(item.uplift_at_k)}</NumberCell>
                  <NumberCell>{formatValue(item.effect)}</NumberCell>
                  <NumberCell>{formatValue(item.estimated)}</NumberCell>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-xs text-ink-3">
            全ての行の結果の差は, 分割の処置の群の結果の平均 − 対照の群の結果の平均
            (誰にでも配ったときの効果) です. {topText(k)} の効果の差がこれより大きいほど,
            効果の高い行を選べています. 全ての行の効果 (見積もり)
            の平均は一式が見積もった効果の分割の平均で,
            全ての行の結果の差に近いほど見積もりの大きさが合っています
          </p>
        </div>
      )}
      {lines.length > 0 && (
        <div className="grid gap-4 lg:grid-cols-2">
          {lines.map(([split, curve]) => (
            <QiniChart key={split} name={EVALUATION_SPLITS[split] ?? split} curve={curve} />
          ))}
        </div>
      )}
    </div>
  )
}

/** 渡した表の処置と結果で数え直した評価 */
function MeasuredView({ measured }: { measured: UpliftMeasured }) {
  return (
    <div className="rounded-lg border border-line p-3 text-sm">
      <p className="font-medium">渡した表の処置と結果で数え直した評価</p>
      <p className="mt-1">
        Qini の係数 <span className="tabular-nums font-semibold">{measured.qini.toFixed(3)}</span>,{' '}
        {topText(measured.k)} の効果の差{' '}
        <span className="tabular-nums font-semibold">{liftCell(measured.uplift_at_k)}</span>,
        全ての行の結果の差 <span className="tabular-nums">{formatValue(measured.effect)}</span>{' '}
        (処置の群 {formatNumber(measured.treated)} 行 / 対照の群 {formatNumber(measured.control)} 行
        {measured.excluded > 0 &&
          `. 処置か結果を読めずに除いた行 ${formatNumber(measured.excluded)} 行`}
        )
      </p>
      <p className="mt-1 text-xs text-ink-3">
        配る相手を無作為に分けた記録でだけ, 効果の確かめになります
      </p>
    </div>
  )
}

/** 効果の分布の表の 1 行 */
function DistributionRow({ label, found }: { label: string; found: EffectDistribution | null }) {
  const keys = ['mean', 'min', 'p10', 'p50', 'p90', 'max'] as const
  return (
    <tr className="border-b border-line">
      <th scope="row" className="px-3 py-1.5 text-left font-medium">
        {label}
      </th>
      {keys.map((key) => (
        <NumberCell key={key}>{found ? formatValue(found[key]) : '–'}</NumberCell>
      ))}
      <NumberCell>{found ? `${(found.positive * 100).toFixed(1)} %` : '–'}</NumberCell>
    </tr>
  )
}

/** 効果の分布 (新しい行と学習用の行) と効果の高い上位 K 割 */
function DistributionCard({ summary, k }: { summary: UpliftSummary; k: number | null }) {
  const top = summary.top_k
  return (
    <Card title="効果の分布">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <caption className="sr-only">
            効果の分布 (新しい行と学習用の行の平均, 最小, 10 % 点, 中央, 90 % 点, 最大,
            効果が正の行)
          </caption>
          <thead className="text-xs text-ink-3">
            <tr className="border-b border-line">
              <Heading>行</Heading>
              <Heading right>平均</Heading>
              <Heading right>最小</Heading>
              <Heading right>10 % 点</Heading>
              <Heading right>中央</Heading>
              <Heading right>90 % 点</Heading>
              <Heading right>最大</Heading>
              <Heading right>効果が正の行</Heading>
            </tr>
          </thead>
          <tbody>
            <DistributionRow
              label={`新しい行 (${formatNumber(summary.predicted)} 行)`}
              found={summary.distribution}
            />
            <DistributionRow
              label={`学習用の行 (${formatNumber(summary.train.rows)} 行)`}
              found={summary.train}
            />
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-sm">
        効果の高い{topText(top?.k ?? k)}
        {top ? (
          <>
            {' '}
            ({formatNumber(top.rows)} 行): 効果の平均{' '}
            <span className="tabular-nums font-semibold">{formatValue(top.mean)}</span>,
            最も低い効果 <span className="tabular-nums">{formatValue(top.lowest)}</span>
          </>
        ) : (
          ': 行がありません'
        )}
      </p>
      <p className="mt-1 text-xs text-ink-3">
        学習用の行の分布と大きく違えば, 新しい行は学習の記録と違う顧客かもしれません
      </p>
    </Card>
  )
}

/** 効果の上位の行 (上位 10 行) */
function TopRowsCard({ summary }: { summary: UpliftSummary }) {
  return (
    <Card title={`効果の上位の行 (${summary.top.length} 行)`} name="効果の上位の行" padded={false}>
      {summary.top.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <caption className="sr-only">
              効果の上位の行 (順位, 行の位置, 行の id, 効果, 処置したときとしないときの結果の予測)
            </caption>
            <thead className="text-xs text-ink-3">
              <tr className="border-b border-line">
                <Heading right>順位</Heading>
                <Heading right>行</Heading>
                <Heading>{summary.id ?? '行の id'}</Heading>
                <Heading right>効果</Heading>
                <Heading right>処置したときの予測</Heading>
                <Heading right>しないときの予測</Heading>
              </tr>
            </thead>
            <tbody>
              {summary.top.map((row, index) => (
                <tr key={row.row} className="border-b border-line">
                  <NumberCell>{index + 1}</NumberCell>
                  <NumberCell>{row.row}</NumberCell>
                  <td className="px-3 py-1.5 font-mono text-xs">
                    {row.id === null ? '–' : String(row.id)}
                  </td>
                  <NumberCell>{formatValue(row.effect)}</NumberCell>
                  <NumberCell>{formatValue(row.treated)}</NumberCell>
                  <NumberCell>{formatValue(row.control)}</NumberCell>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="p-4 text-sm text-ink-2">効果を見積もった行がありません</p>
      )}
    </Card>
  )
}

/** 配る行. 配る判断の文と前提と, 効果の高い順に配る数の上限までの行の一覧 (ページを送る) */
function TreatedCard({ result }: { result: Result }) {
  const api = useApi()
  const treat = treatSummary(result)
  const [offset, setOffset] = useState(0)
  const treated = useQuery({
    queryKey: ['treated', result.id, offset],
    queryFn: () => api.treated(result.id, offset, TREATED_PAGE),
    enabled: !!treat,
    placeholderData: keepPreviousData,
  })
  const rows = treated.data?.rows ?? []
  const total = treated.data?.total ?? 0
  return (
    <Card
      title={`配る行 (${treat ? `${formatNumber(treat.acted)} 行` : '配らない'})`}
      name="配る行"
    >
      <div className="flex flex-col gap-3">
        {result.decision ? (
          <div>
            <ul className="flex flex-col gap-1 text-sm">
              {result.decision.sentences
                .filter((sentence) => !result.decision?.notes.includes(sentence))
                .map((sentence) => (
                  <li key={sentence}>{sentence}</li>
                ))}
            </ul>
            <ul className="mt-1 flex flex-col gap-0.5 text-xs text-ink-3">
              {result.decision.notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          </div>
        ) : (
          <p className="text-sm text-ink-2">配らずに推論した結果なので, 配る行はありません</p>
        )}
        <p className="text-xs text-ink-2">{UPLIFT_UNSUPPORTED}</p>
        {treat && (
          <>
            {treated.error && <ErrorBox error={treated.error} live={false} />}
            {treated.isPending ? (
              <Spinner label="配る行を読んでいます" />
            ) : rows.length > 0 ? (
              <>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <caption className="sr-only">
                      配る行 (効果の高い順. 順位, 行の位置, 行の id, 効果,
                      処置したときとしないときの結果の予測)
                    </caption>
                    <thead className="text-xs text-ink-3">
                      <tr className="border-b border-line">
                        <Heading right>順位</Heading>
                        <Heading right>行</Heading>
                        <Heading>{treated.data?.id ?? '行の id'}</Heading>
                        <Heading right>効果</Heading>
                        <Heading right>処置したときの予測</Heading>
                        <Heading right>しないときの予測</Heading>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((row) => (
                        <tr key={row.row} className="border-b border-line">
                          <NumberCell>{formatNumber(row.rank)}</NumberCell>
                          <NumberCell>{row.row}</NumberCell>
                          <td className="px-3 py-1.5 font-mono text-xs">
                            {row.id === null ? '–' : String(row.id)}
                          </td>
                          <NumberCell>{formatValue(row.effect)}</NumberCell>
                          <NumberCell>{formatValue(row.treated)}</NumberCell>
                          <NumberCell>{formatValue(row.control)}</NumberCell>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {total > TREATED_PAGE && (
                  <div className="flex items-center justify-end gap-2 text-xs text-ink-2">
                    <span role="status">
                      {formatNumber(total)} 行のうち {formatNumber(offset + 1)} から{' '}
                      {formatNumber(Math.min(total, offset + TREATED_PAGE))} 行目
                    </span>
                    <Button
                      disabled={offset === 0}
                      onClick={() => setOffset(Math.max(0, offset - TREATED_PAGE))}
                    >
                      前へ
                    </Button>
                    <Button
                      disabled={offset + TREATED_PAGE >= total}
                      onClick={() => setOffset(offset + TREATED_PAGE)}
                    >
                      次へ
                    </Button>
                  </div>
                )}
              </>
            ) : (
              !treated.error && (
                <p className="text-sm text-ink-2">効果が正の行が無いので, 配る行はありません</p>
              )
            )}
          </>
        )}
      </div>
    </Card>
  )
}

/** 効果の結果の, 効果の分布, 実験の評価と Qini の曲線, 上位の行, 配る行 */
export function UpliftCards({ result, summary }: { result: Result; summary: UpliftSummary }) {
  const k = resultK(result, summary)
  return (
    <>
      <DistributionCard summary={summary} k={k} />
      <Card title="実験の評価">
        <div className="flex flex-col gap-3">
          <UpliftEvaluationsView
            evaluations={summary.evaluations}
            curves={result.uplift?.curves ?? {}}
            k={k}
          />
          {summary.measured && <MeasuredView measured={summary.measured} />}
        </div>
      </Card>
      <TopRowsCard summary={summary} />
      <TreatedCard result={result} />
    </>
  )
}
