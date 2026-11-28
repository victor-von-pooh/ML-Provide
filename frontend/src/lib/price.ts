import type { PriceDecisionSummary, PriceSummary, Result } from '../api/types'

/** 値段の候補の数の上限 (API の MAX_CANDIDATES と同じ. 候補ごとに全ての行を予測し直すため) */
export const MAX_CANDIDATES = 20

/** 選んだ値段の一覧の 1 ページの行の数 */
export const PRICES_PAGE = 100

/** 実験の今の値段の決め方の日本語の名前 */
export const CURRENT_LABELS: Record<string, string> = {
  median: '商品ごとの記録の値段の中央値',
  row: '行の値段',
}

/** 値段の説明 */
export const PRICE_TEXT =
  '値段は商品 1 つあたりの売り値です. 推論では表の値段の列をその行の今の値段とします'

/** 売れた数の説明 */
export const UNITS_TEXT =
  '売れた数は, その値段で売れる商品の数をモデルが予測した値です (学習の記録では売れた数の列)'

/** 原価の説明 (原価の列が無ければ利益を売上で比べる) */
export function costText(column: string | null | undefined): string {
  return column
    ? `原価は商品 1 つあたりの仕入れや作る費用です (原価の列 ${column})`
    : '原価の列が無いので, 原価を 0 として利益を売上 (値段 × 売れた数) で比べます'
}

/** 利益の説明 */
export const PROFIT_TEXT =
  '利益は (値段 − 原価) × 予測した売れた数です. 予測から数えた見込みで, 売れた実績ではありません'

/** 弾力性の説明 (上げ幅を % で書く) */
export function elasticityText(step: number | null | undefined): string {
  const up = step === null || step === undefined ? '1 %' : `${percentOf(step)} %`
  return `弾力性は, 値段を ${up} 上げたときに予測した売れた数が何 % 変わるかです. 負なら値上げで売れる数が減ります. 今の値段で予測した数が 0 の行は数えません`
}

/** 変える割合の説明 */
export const CHANGE_TEXT =
  '変える割合は今の値段に対する値段の変え方で, 0.1 は 10 % 上げる, -0.05 は 5 % 下げるです. 値段の候補はこの割合の並びで, 今の値段 (0) を含みます'

/** 変える幅の上限の説明 */
export const MAX_CHANGE_TEXT =
  '変える幅の上限は, 選んでよい値上げと値下げの大きさの限度です (0.1 なら 10 % 上げるか 10 % 下げるまで). 値段を急に大きく変えないための決まりなので, 上限を超える候補は, 見込みの利益が大きくても選びません'

/** 前提の文 (試しに値段を変えた記録を前提にした予測と, 見込みの利益) */
export const PREMISE_TEXT =
  '売れた数は値段を入力に入れたモデルの予測で, 試しに値段を変えた記録を前提にしています (売れ行きに合わせて値段を決めた記録から学ぶと偏ります). 選んだ値段の利益は見込みなので, 値段を変える試験で確かめてください'

/** 値段の一式で使えないものの文 */
export const PRICE_UNSUPPORTED =
  '行ごとに, 値段の候補のうち変える幅の上限の中で見込みの利益が最も大きい値段を選びます (同じ利益なら今の値段に近い候補). 判断の決め方を変えての出し直し, 過去のデータで得と損を確かめること, 予測の外れ方の測り直し, どの列が売れた数を動かしたかの確かめは使えません'

/** 問題の欄が無い理由 */
export const PRICE_NO_PROBLEM =
  '値段の候補と上限の既定と値段の決め方は一式が実験から写して持つので, 問題はつなぎません'

/** 割合を % の数にする (浮動小数の端数を落とす) */
function percentOf(value: number): number {
  return Number((value * 100).toFixed(4))
}

/** 変える割合を「+10 %」「-5 %」「0 %」のように書く */
export function changeText(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '–'
  const percent = percentOf(value)
  return `${percent > 0 ? '+' : ''}${percent} %`
}

/** 変える幅の上限を「±10 %」のように書く */
export function limitText(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '–'
  return `±${percentOf(value)} %`
}

/** 候補の並びを % で書く */
export function candidatesText(values: number[] | null | undefined): string {
  return (values ?? []).map(changeText).join(', ')
}

/** 候補の並びを欄に書く文字にする (割合の小数) */
export function candidatesInput(values: number[] | null | undefined): string {
  return (values ?? []).map((value) => String(value)).join(', ')
}

/** 10 進の数の表記 (指数も) */
const DECIMAL = /^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$/

/** 値段の候補の欄を読む (カンマか空白で区切った割合の小数). 誤りがあれば problem */
export function parseCandidates(text: string): { values: number[]; problem: string | null } {
  const parts = text.split(/[\s,、]+/).filter((part) => part !== '')
  if (parts.length === 0)
    return { values: [], problem: '値段の候補を割合の小数で書いてください (例: -0.1, 0, 0.1)' }
  const bad = parts.find((part) => !DECIMAL.test(part))
  if (bad !== undefined) return { values: [], problem: `値段の候補の「${bad}」は数ではありません` }
  const values = parts.map(Number)
  if (values.some((value) => !Number.isFinite(value) || value <= -1))
    return { values, problem: '値段の候補は -1 より大きい割合で書いてください (-1 は値段が 0)' }
  if (new Set(values).size !== values.length)
    return { values, problem: '値段の候補に同じ割合が重なっています' }
  if (!values.includes(0)) return { values, problem: '値段の候補に今の値段 (0) を含めてください' }
  if (values.length > MAX_CANDIDATES)
    return { values, problem: `値段の候補は ${MAX_CANDIDATES} 個までにしてください` }
  return { values, problem: null }
}

/** 変える幅の上限の欄を読む (割合の小数. 省けない). 誤りがあれば problem */
export function parseMaxChange(text: string): { value: number | null; problem: string | null } {
  const trimmed = text.trim()
  if (!trimmed)
    return { value: null, problem: '変える幅の上限を割合の小数で書いてください (省けません)' }
  if (!DECIMAL.test(trimmed)) return { value: null, problem: '変える幅の上限は数で書いてください' }
  const value = Number(trimmed)
  if (!Number.isFinite(value) || value < 0)
    return { value: null, problem: '変える幅の上限は 0 以上の割合で書いてください' }
  return { value, problem: null }
}

/** 上限を超える候補 (選ばれない候補) */
export function outsideCandidates(values: number[], limit: number | null): number[] {
  if (limit === null) return []
  return values.filter((value) => Math.abs(value) > limit + 1e-12)
}

/** 値段の結果のまとめ (値段の結果でなければ null) */
export function priceSummary(result: Result): PriceSummary | null {
  const summary = result.summary as PriceSummary | null
  if (result.kind !== 'price' || !summary || !('evaluations' in summary)) return null
  return summary
}

/** 値段の判断のまとめ (値段を決めていない結果は null) */
export function priceDecisionSummary(result: Result): PriceDecisionSummary | null {
  const decided = result.decision
  // (値段の判断の型は意思決定の欄の型に無いので文字で比べる)
  if (!decided || String(decided.decision?.type) !== 'price') return null
  const summary = decided.summary as unknown as PriceDecisionSummary | null
  if (!summary || typeof summary.limited !== 'number') return null
  return summary
}
