import type { DemandSummary, OrderDecisionSummary, Result } from '../api/types'

/** 発注の数の一覧の 1 ページの行の数 */
export const ORDERS_PAGE = 100

/** 先の期間の需要の説明 (先の期間の日数を書く) */
export function demandText(horizon: number | null | undefined): string {
  const days = horizon === null || horizon === undefined ? '決めた日数' : `${horizon} 日`
  return `需要は, 発注を決める日の次の日からの先の ${days} (届くまでの日数と発注の周期を合わせた日数) に売れる数の合計です. 商品ごとに, 表の最後の日を発注を決める日とします`
}

/** 分位点の説明 */
export const QUANTILE_TEXT =
  '分位点は, 需要がその数以下になる見込みが水準の割合になる数です (水準 0.75 の分位点なら, 需要は 75 % の見込みでその数以下). モデルは学んだ水準ごとに需要の分位点を予測します'

/** 品切れの損の説明 */
export const SHORTAGE_TEXT =
  '品切れの損は, 1 つ足りないときの損 (売り逃した利益など) です. 商品 1 つあたりで, 売れ残りの損と同じお金の単位で書きます'

/** 売れ残りの損の説明 */
export const HOLDING_TEXT =
  '売れ残りの損は, 1 つ余ったときの損 (捨てる原価か持ち越す費用) です. 商品 1 つあたりで書きます'

/** 水準 τ の説明 */
export const LEVEL_TEXT =
  '水準 τ は 品切れの損 / (品切れの損 + 売れ残りの損) です. 書いた損と需要の分位点の予測が正しいとすれば, 見込みの損が最も小さいのは, 水準 τ の分位点から在庫を引いた数を発注することです'

/** 発注の数の上限の説明 */
export const MAX_ORDER_TEXT =
  '発注の数の上限は, 1 つの商品で 1 度に発注してよい数の限度です (置き場所の限りなど). 上限を超える数は, 見込みの損が小さくなっても発注しません. 上限に当たった商品は印を付けます'

/** 在庫の説明 (在庫の数の列が無ければ 0) */
export function stockText(column: string | null | undefined): string {
  return column
    ? `在庫は, 発注を決める日の在庫の数 (在庫の数の列 ${column}) です. 発注の数は分位点から在庫を引いて決めます`
    : '在庫の数の列が無いので, 在庫を 0 として発注の数を決めます'
}

/** 見込みの損の説明 */
export const EXPECTED_COST_TEXT =
  '見込みの損は, 予測した分位点から決まる需要の分布での, 品切れの損と売れ残りの損の和の見込み (期待値) です. 予測から数えた見込みで, 実際の損ではありません'

/** 前の期間と同じ数の発注の説明 */
export function baselineText(horizon: number | null | undefined): string {
  const days = horizon === null || horizon === undefined ? '同じ日数' : `${horizon} 日`
  return `比べる決め方は, 前の ${days}に売れた数の合計から在庫を引いた数を発注する決め方 (前の期間と同じ数の発注) です. 前の期間の記録の揃う商品だけで比べます`
}

/** 前提の文 */
export const PREMISE_TEXT =
  '需要の分位点は日ごとの売れた数の記録から学んだ予測で, 記録には在庫が 0 の日の売れた数 (需要より少ない) も含むので, 品切れの多い商品では少なめに出ることがあります. 見込みの損と前の期間との比べは同じ予測の分布から数えたもので必ずこの決め方が良く出るので, 実際の損は発注の後の売れた数で確かめてください'

/** 需要の一式で使えないものの文 */
export const DEMAND_UNSUPPORTED =
  '商品ごとに, 水準 τ の分位点から在庫を引いた数を, 0 から上限の間の整数で発注します (切り捨てと切り上げのうち見込みの損の小さい方). 判断の決め方を変えての出し直し, 過去のデータで得と損を確かめること, 予測の外れ方の測り直し, どの列が需要を動かしたかの確かめは使えません'

/** 問題の欄が無い理由 */
export const DEMAND_NO_PROBLEM =
  '損と上限の既定と需要の決め方は一式が実験から写して持つので, 問題はつなぎません'

/** 10 進の数の表記 (指数も) */
const DECIMAL = /^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$/

/** 損の欄を読む (商品 1 つあたりの正の数. 省けない). 誤りがあれば problem */
export function parseCost(
  text: string,
  label: string,
): { value: number | null; problem: string | null } {
  const trimmed = text.trim()
  if (!trimmed) return { value: null, problem: `${label}を書いてください (省けません)` }
  if (!DECIMAL.test(trimmed)) return { value: null, problem: `${label}は数で書いてください` }
  const value = Number(trimmed)
  if (!Number.isFinite(value) || value <= 0)
    return {
      value: null,
      problem: `${label}は 0 より大きい数で書いてください (0 では水準 τ が 0 か 1 になり, 分位点で決まりません)`,
    }
  return { value, problem: null }
}

/** 発注の数の上限の欄を読む (0 以上の整数. 省けない). 誤りがあれば problem */
export function parseMaxOrder(text: string): { value: number | null; problem: string | null } {
  const trimmed = text.trim()
  if (!trimmed) return { value: null, problem: '発注の数の上限を書いてください (省けません)' }
  if (!/^\d+$/.test(trimmed))
    return {
      value: null,
      problem: '発注の数の上限は 0 以上の整数で書いてください (発注の数は整数のため)',
    }
  const value = Number(trimmed)
  if (!Number.isSafeInteger(value)) return { value: null, problem: '発注の数の上限が大きすぎます' }
  return { value, problem: null }
}

/** 水準 τ (どちらかの損が無ければ null) */
export function tauOf(shortage: number | null, holding: number | null): number | null {
  if (shortage === null || holding === null) return null
  return shortage / (shortage + holding)
}

/** 水準を書く (浮動小数の端数を落とす) */
export function levelText(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '–'
  return String(Number(value.toFixed(4)))
}

/** 水準 τ が学んだ水準の範囲の外かと, 使う端の水準 (範囲の中なら null) */
export function outsideLevel(tau: number | null, levels: number[]): number | null {
  if (tau === null || levels.length === 0) return null
  const low = levels[0]!
  const high = levels[levels.length - 1]!
  if (tau < low - 1e-12) return low
  if (tau > high + 1e-12) return high
  return null
}

/** 学んだ水準の並びを書く */
export function levelsText(levels: number[] | null | undefined): string {
  return (levels ?? []).map(levelText).join(', ')
}

/** 需要の結果のまとめ (需要の結果でなければ null) */
export function demandSummary(result: Result): DemandSummary | null {
  const summary = result.summary as DemandSummary | null
  if (result.kind !== 'demand' || !summary || !('levels' in summary)) return null
  return summary
}

/** 発注の判断のまとめ (発注の数を決めていない結果は null) */
export function orderDecisionSummary(result: Result): OrderDecisionSummary | null {
  const decided = result.decision
  // (発注の判断の型は意思決定の欄の型に無いので文字で比べる)
  if (!decided || String(decided.decision?.type) !== 'order') return null
  const summary = decided.summary as unknown as OrderDecisionSummary | null
  if (!summary || typeof summary.capped !== 'number') return null
  return summary
}
