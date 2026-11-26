import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import { api } from '../../api/client'
import type {
  DemandEvaluation,
  DemandQuantiles,
  DemandSummary,
  OrderDecisionSummary,
  Result,
} from '../../api/types'
import { Button, Card, ErrorBox, Spinner } from '../../components/ui'
import { EVALUATION_SPLITS } from '../../lib/anomaly'
import {
  baselineText,
  DEMAND_UNSUPPORTED,
  demandText,
  EXPECTED_COST_TEXT,
  HOLDING_TEXT,
  LEVEL_TEXT,
  levelsText,
  levelText,
  MAX_ORDER_TEXT,
  orderDecisionSummary,
  ORDERS_PAGE,
  PREMISE_TEXT,
  QUANTILE_TEXT,
  SHORTAGE_TEXT,
  stockText,
} from '../../lib/demand'
import { formatNumber, formatValue } from '../../lib/format'

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

/** 損の比 (無ければ理由) */
function ratioText(ratio: number | null): string {
  return ratio === null ? '無し (前の期間と同じ数の発注の損が 0)' : `${formatNumber(ratio)} 倍`
}

/**
 * 結論の文に初めて出る言葉 (需要, 分位点, 品切れの損, 売れ残りの損, 水準 τ, 在庫, 発注の数の上限,
 * 見込みの損, 比べる決め方) の説明と, 記録と見込みの損の前提
 */
export function DemandTerms({
  horizon,
  stock,
}: {
  horizon: number | null | undefined
  stock: string | null | undefined
}) {
  return (
    <dl
      aria-label="言葉の説明"
      className="mb-4 grid gap-x-4 gap-y-1 text-xs text-ink-2 md:grid-cols-[10rem_1fr]"
    >
      <dt className="font-medium text-ink">需要</dt>
      <dd>{demandText(horizon)}</dd>
      <dt className="font-medium text-ink">分位点</dt>
      <dd>{QUANTILE_TEXT}</dd>
      <dt className="font-medium text-ink">品切れの損</dt>
      <dd>{SHORTAGE_TEXT}</dd>
      <dt className="font-medium text-ink">売れ残りの損</dt>
      <dd>{HOLDING_TEXT}</dd>
      <dt className="font-medium text-ink">水準 τ</dt>
      <dd>{LEVEL_TEXT}</dd>
      <dt className="font-medium text-ink">在庫</dt>
      <dd>{stockText(stock)}</dd>
      <dt className="font-medium text-ink">発注の数の上限</dt>
      <dd>{MAX_ORDER_TEXT}</dd>
      <dt className="font-medium text-ink">見込みの損</dt>
      <dd>{EXPECTED_COST_TEXT}</dd>
      <dt className="font-medium text-ink">比べる決め方</dt>
      <dd>{baselineText(horizon)}</dd>
      <dt className="font-medium text-ink">前提</dt>
      <dd>{PREMISE_TEXT}</dd>
    </dl>
  )
}

/** 実験の評価の表 (分割ごと). 無ければ実験の評価の無い一式と書く */
export function DemandEvaluationsView({
  evaluations,
}: {
  evaluations: Record<string, DemandEvaluation>
}) {
  const entries = Object.entries(evaluations ?? {})
  if (entries.length === 0)
    return (
      <p className="text-sm text-ink-2">
        実験の評価が無いので, 需要の分位点の予測と発注の数を見て確かめます
      </p>
    )
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <caption className="sr-only">
          実験の評価 (分割ごとの需要の誤差, 分位点の損失,
          選んだ発注の数と前の期間と同じ数の発注の損)
        </caption>
        <thead className="text-xs text-ink-3">
          <tr className="border-b border-line">
            <Heading>分割</Heading>
            <Heading right>行</Heading>
            <Heading right>商品</Heading>
            <Heading right>需要の誤差 (mae)</Heading>
            <Heading right>分位点の損失</Heading>
            <Heading right>発注の数の損</Heading>
            <Heading right>前の期間と同じ数の損</Heading>
            <Heading right>損の差</Heading>
            <Heading right>損の比</Heading>
            <Heading right>上限で切った行</Heading>
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
              <NumberCell>{formatValue(item.pinball_loss)}</NumberCell>
              <NumberCell>{formatValue(item.order_cost)}</NumberCell>
              <NumberCell>{formatValue(item.baseline_cost)}</NumberCell>
              <NumberCell>{formatValue(item.order_cost_saving)}</NumberCell>
              <NumberCell>{ratioText(item.order_cost_ratio)}</NumberCell>
              <NumberCell>{formatNumber(item.capped)}</NumberCell>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-ink-3">
        需要の誤差は先の期間の売れた数の合計と 0.5 の分位点の平均絶対誤差 (mae) で,
        分位点の損失は学んだ全ての水準の分位点の外れの重み付きの平均です. どちらも小さいほど良い.
        損は実験の分割の実際の売れた数で数えた損の和で, 比が 1
        より小さいほど前の期間と同じ数の発注より損が小さい
      </p>
    </div>
  )
}

/** 見込みの損の比べ (発注の判断のまとめ, 上限に当たった商品, 判断の文と前提) */
function CostCard({ result, decided }: { result: Result; decided: OrderDecisionSummary | null }) {
  return (
    <Card title="見込みの損の比べ">
      {decided ? (
        <div className="flex flex-col gap-3">
          <dl className="grid gap-x-4 gap-y-1 text-sm md:grid-cols-[16rem_1fr]">
            <dt className="text-ink-2">品切れの損と売れ残りの損</dt>
            <dd className="tabular">
              {formatNumber(decided.shortage_cost)} と {formatNumber(decided.holding_cost)} (商品 1
              つあたり)
            </dd>
            <dt className="text-ink-2">水準 τ</dt>
            <dd>
              <span className="tabular font-semibold">{levelText(decided.level)}</span>
              {decided.outside && (
                <span className="block text-xs text-warning-text">
                  学んだ水準の範囲の外なので, 端の水準 {levelText(decided.used_level)}{' '}
                  の分位点を使いました (分布の端は学んでいないので, 見込みの損は少なめに出ます)
                </span>
              )}
            </dd>
            <dt className="text-ink-2">発注の数の上限</dt>
            <dd className="tabular">{formatNumber(decided.max_order)}</dd>
            <dt className="text-ink-2">発注する商品</dt>
            <dd>
              {formatNumber(decided.ordered)} / {formatNumber(decided.rows)} 商品 (発注の数の和{' '}
              <span className="tabular">{formatNumber(decided.total)}</span>)
            </dd>
            <dt className="text-ink-2">見込みの損の和</dt>
            <dd className="tabular font-semibold">{formatValue(decided.expected_cost)}</dd>
            <dt className="text-ink-2">前の期間と同じ数の発注の見込みの損</dt>
            <dd>
              {decided.compared > 0 ? (
                <>
                  <span className="tabular">{formatValue(decided.baseline_cost)}</span>{' '}
                  (この決め方なら{' '}
                  <span className="tabular">{formatValue(decided.compared_cost)}</span>.{' '}
                  {formatNumber(decided.compared)} 商品. 差{' '}
                  <span className="tabular">{formatValue(decided.saving)}</span>, 比{' '}
                  {ratioText(decided.ratio)})
                </>
              ) : (
                '前の期間の記録の揃う商品が無いので, 比べられません'
              )}
            </dd>
            <dt className="text-ink-2">上限に当たった商品</dt>
            <dd>
              <span className="tabular font-semibold">{formatNumber(decided.capped)} 商品</span>
              <span className="block text-xs text-ink-3">
                水準 τ の分位点から在庫を引いた数が上限を超え, 上限で切った商品です.
                上限は置き場所などの決まりなので超えません. 多ければ品切れの損が増えます
              </span>
            </dd>
            {decided.undecided > 0 && (
              <>
                <dt className="text-ink-2">決めない商品</dt>
                <dd>{formatNumber(decided.undecided)} 商品 (分位点を予測できない)</dd>
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
          発注の数を決めずに推論した結果なので, 見込みの損の比べと発注の数はありません
        </p>
      )}
      <p className="mt-3 text-xs text-ink-2">{DEMAND_UNSUPPORTED}</p>
    </Card>
  )
}

/** 需要の分位点 (水準ごとの平均と, 最も小さい水準から最も大きい水準までの幅の分布) */
function QuantileCard({
  found,
  summary,
}: {
  found: DemandQuantiles | null
  summary: DemandSummary
}) {
  const keys = ['mean', 'min', 'p10', 'p50', 'p90', 'max'] as const
  return (
    <Card title="需要の分位点">
      {found?.quantiles && found.width ? (
        <div className="flex flex-col gap-3">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <caption className="sr-only">
                需要の分位点の平均 (予測した商品の, 学んだ水準ごとの先の期間の需要の分位点の平均)
              </caption>
              <thead className="text-xs text-ink-3">
                <tr className="border-b border-line">
                  <Heading>水準</Heading>
                  {found.levels.map((level) => (
                    <Heading key={level} right>
                      {levelText(level)}
                    </Heading>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-line">
                  <th scope="row" className="px-3 py-1.5 text-left font-medium whitespace-nowrap">
                    分位点の平均 ({formatNumber(found.width.counted)} 商品)
                  </th>
                  {found.quantiles.map((value, index) => (
                    <NumberCell key={found.levels[index]}>{formatValue(value)}</NumberCell>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <caption className="sr-only">
                需要の分位点の幅の分布 (商品ごとの最も小さい水準から最も大きい水準までの幅の平均,
                最小, 10 % 点, 中央, 90 % 点, 最大)
              </caption>
              <thead className="text-xs text-ink-3">
                <tr className="border-b border-line">
                  <Heading>幅</Heading>
                  <Heading right>平均</Heading>
                  <Heading right>最小</Heading>
                  <Heading right>10 % 点</Heading>
                  <Heading right>中央</Heading>
                  <Heading right>90 % 点</Heading>
                  <Heading right>最大</Heading>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-line">
                  <th scope="row" className="px-3 py-1.5 text-left font-medium whitespace-nowrap">
                    水準 {levelText(found.low)} から {levelText(found.high)} まで
                  </th>
                  {keys.map((key) => (
                    <NumberCell key={key}>{formatValue(found.width?.[key])}</NumberCell>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
          <p className="text-xs text-ink-3">
            幅は商品ごとの, 先の {found.horizon} 日の需要の見込みの広がりです.
            幅の広い商品ほど需要の読みが難しく, 水準 τ の分位点 (発注の数) が損の比で大きく動きます
          </p>
        </div>
      ) : (
        <p className="text-sm text-ink-2">分位点を予測した商品がありません</p>
      )}
      <p className="mt-2 text-sm">
        学習用の分割の先の期間の需要の平均 (実験):{' '}
        <span className="tabular">{formatValue(summary.train?.demand_mean ?? null)}</span>
        {summary.train && (
          <span className="text-ink-2">
            {' '}
            (商品 {formatNumber(summary.train.products)}, 行 {formatNumber(summary.train.rows)})
          </span>
        )}
      </p>
      {summary.previous && summary.previous.mean !== null && (
        <p className="mt-1 text-sm">
          前の {summary.horizon} 日の売れた数の合計の平均:{' '}
          <span className="tabular">{formatValue(summary.previous.mean)}</span>{' '}
          <span className="text-ink-2">({formatNumber(summary.previous.counted)} 商品)</span>
        </p>
      )}
      <p className="mt-1 text-xs text-ink-3">
        学習用の分割や前の期間と大きく違えば, 新しい記録は学習の記録と違う商品か時期かもしれません
      </p>
    </Card>
  )
}

/** 発注の数. 商品ごとの発注の数の一覧 (発注の数の大きい順. ページを送る) */
function OrdersCard({ result, decided }: { result: Result; decided: OrderDecisionSummary | null }) {
  const [offset, setOffset] = useState(0)
  const orders = useQuery({
    queryKey: ['orders', result.id, offset],
    queryFn: () => api.orders(result.id, offset, ORDERS_PAGE),
    enabled: !!decided,
    placeholderData: keepPreviousData,
  })
  const rows = orders.data?.rows ?? []
  const total = orders.data?.total ?? 0
  const levels = orders.data?.levels ?? []
  return (
    <Card
      title={`発注の数 (${decided ? `${formatNumber(decided.rows)} 商品` : '決めない'})`}
      padded={false}
    >
      {!decided ? (
        <p className="p-4 text-sm text-ink-2">発注の数を決めずに推論した結果です</p>
      ) : (
        <div className="flex flex-col gap-2">
          {orders.error && <ErrorBox error={orders.error} live={false} />}
          {orders.isPending ? (
            <Spinner label="発注の数を読んでいます" />
          ) : rows.length > 0 ? (
            <>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <caption className="sr-only">
                    発注の数 (発注の数の大きい順. 行の位置, 行の id, 商品の id, 発注を決める日,
                    在庫, 前の期間の合計, 需要の分位点, 水準 τ の分位点, 発注の数, 上限, 見込みの損,
                    前の期間と同じ数の発注とその損)
                  </caption>
                  <thead className="text-xs text-ink-3">
                    <tr className="border-b border-line">
                      <Heading right>行</Heading>
                      <Heading>{orders.data?.id ?? '行の id'}</Heading>
                      {orders.data?.product && <Heading>{orders.data.product}</Heading>}
                      <Heading>発注を決める日</Heading>
                      <Heading right>在庫</Heading>
                      <Heading right>前の期間の合計</Heading>
                      <Heading right>需要の分位点 ({levelsText(levels)})</Heading>
                      <Heading right>水準 τ の分位点</Heading>
                      <Heading right>発注の数</Heading>
                      <Heading>上限</Heading>
                      <Heading right>見込みの損</Heading>
                      <Heading right>前の期間と同じ数の発注</Heading>
                      <Heading right>その見込みの損</Heading>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row) => (
                      <tr key={row.row} className="border-b border-line">
                        <NumberCell>{row.row}</NumberCell>
                        <td className="px-3 py-1.5 font-mono text-xs">
                          {row.id === null ? '–' : String(row.id)}
                        </td>
                        {orders.data?.product && (
                          <td className="px-3 py-1.5 font-mono text-xs">
                            {row.product_id === null ? '–' : String(row.product_id)}
                          </td>
                        )}
                        <td className="tabular px-3 py-1.5 whitespace-nowrap">
                          {row.order_date ?? '–'}
                        </td>
                        <NumberCell>{formatValue(row.on_hand)}</NumberCell>
                        <NumberCell>{formatValue(row.previous_demand)}</NumberCell>
                        <NumberCell>
                          {row.quantiles.map((value) => formatValue(value)).join(', ')}
                        </NumberCell>
                        <NumberCell>{formatValue(row.quantile)}</NumberCell>
                        <NumberCell>
                          <span className="font-semibold">{formatNumber(row.order)}</span>
                        </NumberCell>
                        <td className="px-3 py-1.5 text-xs whitespace-nowrap">
                          {row.capped ? (
                            <span className="text-warning-text">上限に当たった</span>
                          ) : (
                            ''
                          )}
                        </td>
                        <NumberCell>{formatValue(row.expected_cost)}</NumberCell>
                        <NumberCell>{formatNumber(row.baseline_order)}</NumberCell>
                        <NumberCell>{formatValue(row.baseline_cost)}</NumberCell>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {total > ORDERS_PAGE && (
                <div className="flex items-center justify-end gap-2 px-4 pb-3 text-xs text-ink-2">
                  <span role="status">
                    {formatNumber(total)} 商品のうち {formatNumber(offset + 1)} から{' '}
                    {formatNumber(Math.min(total, offset + ORDERS_PAGE))} 商品目
                  </span>
                  <Button
                    disabled={offset === 0}
                    onClick={() => setOffset(Math.max(0, offset - ORDERS_PAGE))}
                  >
                    前へ
                  </Button>
                  <Button
                    disabled={offset + ORDERS_PAGE >= total}
                    onClick={() => setOffset(offset + ORDERS_PAGE)}
                  >
                    次へ
                  </Button>
                </div>
              )}
            </>
          ) : (
            !orders.error && (
              <p className="p-4 text-sm text-ink-2">発注の数を決めた商品がありません</p>
            )
          )}
          <p className="px-4 pb-3 text-xs text-ink-3">
            発注の数の大きい順です (上限に当たった商品が上に来ます). 前の期間の合計の無い商品は,
            前の期間と同じ数の発注を数えません
          </p>
        </div>
      )}
    </Card>
  )
}

/** 需要の結果の, 見込みの損の比べ, 需要の分位点, 実験の評価, 発注の数 */
export function DemandCards({ result, summary }: { result: Result; summary: DemandSummary }) {
  const decided = orderDecisionSummary(result)
  return (
    <>
      <CostCard result={result} decided={decided} />
      <QuantileCard found={result.demand ?? null} summary={summary} />
      <Card title="実験の評価">
        <DemandEvaluationsView evaluations={summary.evaluations} />
      </Card>
      <OrdersCard result={result} decided={decided} />
    </>
  )
}
