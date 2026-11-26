import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import { api } from '../../api/client'
import type {
  ElasticityDistribution,
  PriceDecisionSummary,
  PriceEvaluation,
  PriceSummary,
  Result,
} from '../../api/types'
import { Button, Card, ErrorBox, Spinner } from '../../components/ui'
import { EVALUATION_SPLITS } from '../../lib/anomaly'
import { formatNumber, formatValue } from '../../lib/format'
import {
  candidatesText,
  CHANGE_TEXT,
  changeText,
  costText,
  elasticityText,
  limitText,
  MAX_CHANGE_TEXT,
  PREMISE_TEXT,
  PRICE_TEXT,
  PRICE_UNSUPPORTED,
  priceDecisionSummary,
  PRICES_PAGE,
  PROFIT_TEXT,
  UNITS_TEXT,
} from '../../lib/price'

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
  return <td className="tabular px-3 py-1.5 text-right whitespace-nowrap">{children}</td>
}

/** 弾力性 (%) を書く (無ければ –) */
function elasticityCell(value: number | null | undefined): string {
  return value === null || value === undefined ? '–' : `${formatValue(value)} %`
}

/**
 * 結論の文に初めて出る言葉 (値段, 売れた数, 原価, 利益, 弾力性, 変える割合, 変える幅の上限) の説明と,
 * 試しに値段を変えた記録を前提にした予測という前提
 */
export function PriceTerms({
  cost,
  step,
}: {
  cost: string | null | undefined
  step: number | null | undefined
}) {
  return (
    <dl
      aria-label="言葉の説明"
      className="mb-4 grid gap-x-4 gap-y-1 text-xs text-ink-2 md:grid-cols-[10rem_1fr]"
    >
      <dt className="font-medium text-ink">値段</dt>
      <dd>{PRICE_TEXT}</dd>
      <dt className="font-medium text-ink">売れた数</dt>
      <dd>{UNITS_TEXT}</dd>
      <dt className="font-medium text-ink">原価</dt>
      <dd>{costText(cost)}</dd>
      <dt className="font-medium text-ink">利益 (見込み)</dt>
      <dd>{PROFIT_TEXT}</dd>
      <dt className="font-medium text-ink">弾力性</dt>
      <dd>{elasticityText(step)}</dd>
      <dt className="font-medium text-ink">変える割合</dt>
      <dd>{CHANGE_TEXT}</dd>
      <dt className="font-medium text-ink">変える幅の上限</dt>
      <dd>{MAX_CHANGE_TEXT}</dd>
      <dt className="font-medium text-ink">前提</dt>
      <dd>{PREMISE_TEXT}</dd>
    </dl>
  )
}

/** 利益の比 (無ければ理由) */
function ratioText(ratio: number | null): string {
  return ratio === null ? '無し (今の値段の利益の和が 0 以下)' : `${formatNumber(ratio)} 倍`
}

/** 実験の評価の表 (分割ごと). 無ければ実験の評価の無い一式と書く */
export function PriceEvaluationsView({
  evaluations,
}: {
  evaluations: Record<string, PriceEvaluation>
}) {
  const entries = Object.entries(evaluations ?? {})
  if (entries.length === 0)
    return (
      <p className="text-sm text-ink-2">
        実験の評価が無いので, 予測した売れた数と弾力性を見て確かめます
      </p>
    )
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <caption className="sr-only">
          実験の評価 (分割ごとの売れた数の誤差, 弾力性, 選んだ値段と今の値段の見込みの利益)
        </caption>
        <thead className="text-xs text-ink-3">
          <tr className="border-b border-line">
            <Heading>分割</Heading>
            <Heading right>行</Heading>
            <Heading right>商品</Heading>
            <Heading right>売れた数の誤差 (mae)</Heading>
            <Heading right>rmse</Heading>
            <Heading right>弾力性の平均</Heading>
            <Heading right>選んだ値段の利益</Heading>
            <Heading right>今の値段の利益</Heading>
            <Heading right>利益の差</Heading>
            <Heading right>利益の比</Heading>
            <Heading right>値段を変える商品</Heading>
            <Heading right>変える幅の上限</Heading>
          </tr>
        </thead>
        <tbody>
          {entries.map(([split, item]) => (
            <tr key={split} className="border-b border-line">
              <th scope="row" className="px-3 py-1.5 text-left font-medium whitespace-nowrap">
                {EVALUATION_SPLITS[split] ?? split}
              </th>
              <NumberCell>{formatNumber(item.rows)}</NumberCell>
              <NumberCell>{formatNumber(item.products)}</NumberCell>
              <NumberCell>{formatValue(item.mae)}</NumberCell>
              <NumberCell>{formatValue(item.rmse)}</NumberCell>
              <NumberCell>{elasticityCell(item.elasticity)}</NumberCell>
              <NumberCell>{formatValue(item.profit)}</NumberCell>
              <NumberCell>{formatValue(item.current_profit)}</NumberCell>
              <NumberCell>{formatValue(item.profit_gain)}</NumberCell>
              <NumberCell>{ratioText(item.profit_ratio)}</NumberCell>
              <NumberCell>
                {formatNumber(item.changed)} / {formatNumber(item.products)}
              </NumberCell>
              <NumberCell>{limitText(item.max_change)}</NumberCell>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-ink-3">
        売れた数の誤差は記録の値段での予測の平均絶対誤差 (mae) と二乗平均平方根誤差 (rmse) で,
        小さいほど良い. 利益は商品ごとの 1 行あたりの予測から数えた見込みで, 比が 1
        より大きいほど選んだ値段の方が今の値段より利益が大きい見込みです
      </p>
    </div>
  )
}

/** 選んだ値段と今の値段の見込みの利益の比べ, 値段の判断の文と前提 */
function ProfitCard({ result, decided }: { result: Result; decided: PriceDecisionSummary | null }) {
  return (
    <Card title="利益の比べ">
      {decided ? (
        <div className="flex flex-col gap-3">
          <dl className="grid gap-x-4 gap-y-1 text-sm md:grid-cols-[16rem_1fr]">
            <dt className="text-ink-2">選んだ値段の見込みの利益の和</dt>
            <dd className="tabular font-semibold">{formatValue(decided.total)}</dd>
            <dt className="text-ink-2">今の値段のままの利益の和</dt>
            <dd className="tabular">{formatValue(decided.current_total)}</dd>
            <dt className="text-ink-2">差 (選んだ値段 − 今の値段)</dt>
            <dd className="tabular">{formatValue(decided.gain)}</dd>
            <dt className="text-ink-2">比 (選んだ値段 / 今の値段)</dt>
            <dd className="tabular">{ratioText(decided.ratio)}</dd>
            <dt className="text-ink-2">値段を変える行</dt>
            <dd>
              {formatNumber(decided.changed)} / {formatNumber(decided.rows)} 行 (上げる{' '}
              {formatNumber(decided.raised)} 行, 下げる {formatNumber(decided.lowered)} 行)
            </dd>
            <dt className="text-ink-2">値段の候補 (変える割合)</dt>
            <dd className="tabular">{candidatesText(decided.candidates)}</dd>
            <dt className="text-ink-2">変える幅の上限</dt>
            <dd className="tabular">{limitText(decided.max_change)}</dd>
            <dt className="text-ink-2">上限を超える候補の方が利益の大きい行</dt>
            <dd>
              <span className="tabular font-semibold">{formatNumber(decided.limited)} 行</span>
              <span className="block text-xs text-ink-3">
                上限は値段を急に大きく変えないための決まりなので, 上限を超える候補は選びません.
                多ければ上限を広げたときの利益を推論し直して比べられます
              </span>
            </dd>
            {decided.undecided > 0 && (
              <>
                <dt className="text-ink-2">決めない行</dt>
                <dd>{formatNumber(decided.undecided)} 行 (候補の値段で予測できない)</dd>
              </>
            )}
          </dl>
          {result.decision && (
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
          )}
        </div>
      ) : (
        <p className="text-sm text-ink-2">
          値段を決めずに推論した結果なので, 利益の比べと選んだ値段はありません
        </p>
      )}
      <p className="mt-3 text-xs text-ink-2">{PRICE_UNSUPPORTED}</p>
    </Card>
  )
}

/** 弾力性の分布 (予測した行) と学習用の分割の弾力性の平均 */
function ElasticityCard({
  found,
  summary,
}: {
  found: ElasticityDistribution | null
  summary: PriceSummary
}) {
  const keys = ['mean', 'min', 'p10', 'p50', 'p90', 'max'] as const
  return (
    <Card title="弾力性の分布">
      {found ? (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <caption className="sr-only">
              弾力性の分布 (予測した行の平均, 最小, 10 % 点, 中央, 90 % 点, 最大,
              弾力性が負の行の割合)
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
                <Heading right>弾力性が負の行の割合</Heading>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-line">
                <th scope="row" className="px-3 py-1.5 text-left font-medium whitespace-nowrap">
                  予測した行 ({formatNumber(found.counted)} 行)
                </th>
                {keys.map((key) => (
                  <NumberCell key={key}>{elasticityCell(found[key])}</NumberCell>
                ))}
                <NumberCell>{(found.negative * 100).toFixed(1)} %</NumberCell>
              </tr>
            </tbody>
          </table>
        </div>
      ) : (
        <p className="text-sm text-ink-2">弾力性を数えた行がありません</p>
      )}
      <p className="mt-2 text-sm">
        学習用の分割の弾力性の平均 (実験):{' '}
        <span className="tabular">{elasticityCell(summary.train?.elasticity)}</span>
        {summary.train && (
          <span className="text-ink-2">
            {' '}
            (商品 {formatNumber(summary.train.products)}, 値段を変える商品{' '}
            {formatNumber(summary.train.changed)})
          </span>
        )}
      </p>
      <p className="mt-1 text-xs text-ink-3">
        学習用の分割と大きく違えば, 新しい行は学習の記録と違う商品か値段かもしれません
      </p>
    </Card>
  )
}

/** 選んだ値段. 行ごとの選んだ値段と変える割合の一覧 (見込みの利益の差の大きい順. ページを送る) */
function PricesCard({ result, decided }: { result: Result; decided: PriceDecisionSummary | null }) {
  const [offset, setOffset] = useState(0)
  const prices = useQuery({
    queryKey: ['prices', result.id, offset],
    queryFn: () => api.prices(result.id, offset, PRICES_PAGE),
    enabled: !!decided,
    placeholderData: keepPreviousData,
  })
  const rows = prices.data?.rows ?? []
  const total = prices.data?.total ?? 0
  return (
    <Card
      title={`選んだ値段 (${decided ? `${formatNumber(decided.rows)} 行` : '決めない'})`}
      padded={false}
    >
      {!decided ? (
        <p className="p-4 text-sm text-ink-2">値段を決めずに推論した結果です</p>
      ) : (
        <div className="flex flex-col gap-2">
          {prices.error && <ErrorBox error={prices.error} live={false} />}
          {prices.isPending ? (
            <Spinner label="選んだ値段を読んでいます" />
          ) : rows.length > 0 ? (
            <>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <caption className="sr-only">
                    選んだ値段 (見込みの利益の差の大きい順. 行の位置, 行の id, 商品の id, 今の値段,
                    原価, 変える割合, 選んだ値段, 予測した数, 利益, 弾力性)
                  </caption>
                  <thead className="text-xs text-ink-3">
                    <tr className="border-b border-line">
                      <Heading right>行</Heading>
                      <Heading>{prices.data?.id ?? '行の id'}</Heading>
                      {prices.data?.product && <Heading>{prices.data.product}</Heading>}
                      <Heading right>今の値段</Heading>
                      <Heading right>原価</Heading>
                      <Heading right>変える割合</Heading>
                      <Heading right>選んだ値段</Heading>
                      <Heading right>今の値段での予測した数</Heading>
                      <Heading right>選んだ値段での予測した数</Heading>
                      <Heading right>今の値段の利益</Heading>
                      <Heading right>見込みの利益</Heading>
                      <Heading right>弾力性</Heading>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row) => (
                      <tr key={row.row} className="border-b border-line">
                        <NumberCell>{row.row}</NumberCell>
                        <td className="px-3 py-1.5 font-mono text-xs">
                          {row.id === null ? '–' : String(row.id)}
                        </td>
                        {prices.data?.product && (
                          <td className="px-3 py-1.5 font-mono text-xs">
                            {row.product_id === null ? '–' : String(row.product_id)}
                          </td>
                        )}
                        <NumberCell>{formatValue(row.current_price)}</NumberCell>
                        <NumberCell>{formatValue(row.cost)}</NumberCell>
                        <NumberCell>{changeText(row.change)}</NumberCell>
                        <NumberCell>{formatValue(row.price)}</NumberCell>
                        <NumberCell>{formatValue(row.units)}</NumberCell>
                        <NumberCell>{formatValue(row.price_units)}</NumberCell>
                        <NumberCell>{formatValue(row.current_profit)}</NumberCell>
                        <NumberCell>{formatValue(row.profit)}</NumberCell>
                        <NumberCell>{elasticityCell(row.elasticity)}</NumberCell>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {total > PRICES_PAGE && (
                <div className="flex items-center justify-end gap-2 px-4 pb-3 text-xs text-ink-2">
                  <span role="status">
                    {formatNumber(total)} 行のうち {formatNumber(offset + 1)} から{' '}
                    {formatNumber(Math.min(total, offset + PRICES_PAGE))} 行目
                  </span>
                  <Button
                    disabled={offset === 0}
                    onClick={() => setOffset(Math.max(0, offset - PRICES_PAGE))}
                  >
                    前へ
                  </Button>
                  <Button
                    disabled={offset + PRICES_PAGE >= total}
                    onClick={() => setOffset(offset + PRICES_PAGE)}
                  >
                    次へ
                  </Button>
                </div>
              )}
            </>
          ) : (
            !prices.error && <p className="p-4 text-sm text-ink-2">値段を決めた行がありません</p>
          )}
          <p className="px-4 pb-3 text-xs text-ink-3">
            見込みの利益の差 (見込みの利益 − 今の値段の利益) の大きい順です
          </p>
        </div>
      )}
    </Card>
  )
}

/** 値段の結果の, 利益の比べ, 弾力性の分布, 実験の評価, 選んだ値段 */
export function PriceCards({ result, summary }: { result: Result; summary: PriceSummary }) {
  const decided = priceDecisionSummary(result)
  return (
    <>
      <ProfitCard result={result} decided={decided} />
      <ElasticityCard found={result.price?.elasticity ?? null} summary={summary} />
      <Card title="実験の評価">
        <div className="flex flex-col gap-3">
          <PriceEvaluationsView evaluations={summary.evaluations} />
          {summary.measured && (
            <div className="rounded-lg border border-line p-3 text-sm">
              <p className="font-medium">渡した表の売れた数で数え直した誤差</p>
              <p className="mt-1">
                平均絶対誤差{' '}
                <span className="tabular font-semibold">{formatValue(summary.measured.mae)}</span>,
                二乗平均平方根誤差{' '}
                <span className="tabular">{formatValue(summary.measured.rmse)}</span> (
                {formatNumber(summary.measured.rows)} 行
                {summary.measured.excluded > 0 &&
                  `. 売れた数を読めずに除いた行 ${formatNumber(summary.measured.excluded)} 行`}
                )
              </p>
              <p className="mt-1 text-xs text-ink-3">記録の値段での売れた数の予測の誤差です</p>
            </div>
          )}
        </div>
      </Card>
      <PricesCard result={result} decided={decided} />
    </>
  )
}
