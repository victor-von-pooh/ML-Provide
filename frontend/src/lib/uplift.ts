import type {
  EffectDistribution,
  Result,
  TreatSummary,
  UpliftInfo,
  UpliftSummary,
} from '../api/types'
import { formatNumber, formatValue } from './format'

/** 効果の決め方の日本語の名前 */
export const LEARNER_LABELS: Record<string, string> = {
  t_learner: '処置の群と対照の群の 2 つの予測の差 (T-learner)',
  s_learner: '処置の列を入力の終わりに足した 1 つの予測の差 (S-learner)',
}

/** 効果の決め方ごとの, 処置したときとしないときの予測の出し方 */
export const LEARNER_TEXTS: Record<string, string> = {
  t_learner:
    '処置したときの予測は処置の群の行で学んだ予測, しないときの予測は対照の群の行で学んだ予測です',
  s_learner:
    '処置したときの予測は処置の列を 1 に, しないときの予測は 0 にして入力の終わりに足した 1 つの予測です',
}

/** 効果の決め方の名前と予測の出し方の文 (知らない決め方は名前だけ) */
export function learnerText(learner: string): string {
  const label = LEARNER_LABELS[learner] ?? learner
  const text = LEARNER_TEXTS[learner]
  return text ? `${label}. ${text}` : label
}

/** モデルの入力の幅と出力の文 (s_learner は処置の列を入力の終わりに足し, 出力は結果の予測 1 つ) */
export function modelText(uplift: UpliftInfo): string {
  const columns = uplift.features.length
  const input = uplift.input ? `${uplift.input} ` : ''
  if (uplift.learner === 's_learner') {
    const output = uplift.output ?? 'outcome'
    return `入力 ${input}の幅 ${columns + 1} (見積もりに使う列 ${columns} と処置の列 ${uplift.treatment} の 1. 処置の列には一式が 1 と 0 を入れます). 出力は結果の予測 ${output} の 1 つで, 処置の列を 1 と 0 にして 2 回予測します`
  }
  const treated = uplift.treated ?? 'treated'
  const control = uplift.control ?? 'control'
  return `入力 ${input}の幅 ${columns} (見積もりに使う列. 処置の列は入れません). 出力は処置したときの予測 ${treated} としないときの予測 ${control} の 2 つです`
}

/** 結果の種類の日本語の名前 (ML-Inference の結論の文と同じ言葉) */
export const OUTCOME_LABELS: Record<string, string> = {
  binary: '結果が起きたか (0 と 1)',
  amount: '量',
}

/** 結果の種類ごとの効果の単位 */
export const EFFECT_UNITS: Record<string, string> = {
  binary: '確率の差',
  amount: '値の差',
}

/** 配る行の一覧の 1 ページの行の数 */
export const TREATED_PAGE = 100

/** 効果の説明 (結果の種類に合わせる. 画面で効果が初めて出る所に置く) */
export function effectText(outcome: string | null | undefined): string {
  const unit = EFFECT_UNITS[outcome ?? ''] ?? '確率の差か値の差'
  return `効果は, 処置したとき (クーポンを配るなど) の結果の予測 − しないときの結果の予測です (${unit}). 正なら配ると結果が増え, 0 以下なら配っても増えない見込みです`
}

/** 処置の群と対照の群の説明 */
export const GROUPS_TEXT =
  '処置の群は学習の記録で処置した行 (配った顧客), 対照の群は処置しなかった行 (配らなかった顧客) です. 効果は 2 つの群の結果の違いから学んだ見積もりです'

/** 前提の文 (無作為に配った記録を前提にした見積もり) */
export const PREMISE_TEXT =
  '効果は処置を無作為に配った記録を前提にした見積もりです (配る相手を選んだ記録から学ぶと偏ります). 配った効果は, 配る相手を無作為に分けた試験で確かめてください'

/** Qini の曲線と係数の説明 */
export const QINI_TEXT =
  'Qini の曲線は, 効果の高い順に配る行を増やしたときの, 配ったことで増えた結果の積み上げです (上位の処置の群の結果の和 − 対照の群の結果の和 × 処置の群の行の数 / 対照の群の行の数). 点線はでたらめに配ったときの線です. Qini の係数は, 曲線とでたらめの線の間の面積を理想の並びの面積で割った値で, 1 に近いほど効果の高い行を上に並べられています (0 はでたらめと同じ)'

/** 上位 K 割の効果の差の説明 */
export function liftText(k: number | null): string {
  return `${topText(k)} の効果の差は, 効果の高い${topText(k)} の行のうち, 処置の群の結果の平均 − 対照の群の結果の平均です`
}

/** 効果の一式で使えないものの文 */
export const UPLIFT_UNSUPPORTED =
  '損得のしきい値は決めず, 効果の高い順に配る数の上限まで配ります. 効果が 0 以下の行は上限の中でも配りません. 判断の決め方を変えての出し直し, 過去のデータで得と損を確かめること, 予測の外れ方の測り直し, どの列が効果を動かしたかの確かめは使えません'

/** 問題の欄が無い理由 */
export const UPLIFT_NO_PROBLEM =
  '配る数の上限の既定と効果の決め方は一式が実験から写して持つので, 問題はつなぎません'

/** 上位 K 割を「上位 30 %」のように % で書く (K が分からなければ「上位 – %」) */
export function topText(k: number | null): string {
  return k === null ? '上位 – %' : `上位 ${Number((k * 100).toFixed(4))} %`
}

/** 結果の評価の上位 K 割の K (一式から写した K. 無ければ結論の上位 K 割の K. どちらも無ければ null) */
export function resultK(result: Result, summary: UpliftSummary): number | null {
  return result.uplift?.k ?? summary.top_k?.k ?? null
}

/** 配る数の上限の上限 (API の MAX_BUDGET と同じ. JSON の整数を丸めずに渡せる大きさ) */
export const MAX_BUDGET = 100_000

/** 配る数の上限の欄の誤り (無ければ null) */
export function budgetProblem(text: string): string | null {
  const value = text.trim()
  if (!/^\d+$/.test(value) || Number(value) < 1 || Number(value) > MAX_BUDGET)
    return `配る数の上限は 1 から ${MAX_BUDGET.toLocaleString()} の整数で書いてください`
  return null
}

/** 効果の結果のまとめ (効果の結果でなければ null) */
export function upliftSummary(result: Result): UpliftSummary | null {
  const summary = result.summary as UpliftSummary | null
  if (result.kind !== 'uplift' || !summary || !Array.isArray(summary.top)) return null
  return summary
}

/** 配る判断のまとめ (配っていない結果は null) */
export function treatSummary(result: Result): TreatSummary | null {
  const decided = result.decision
  // (配る判断の型は意思決定の欄の型に無いので文字で比べる)
  if (!decided || String(decided.decision?.type) !== 'treat') return null
  const summary = decided.summary as unknown as TreatSummary | null
  if (!summary || typeof summary.acted !== 'number') return null
  return summary
}

/** 効果の分布の 1 行の文 */
export function distributionText(found: EffectDistribution): string {
  const rows = found.rows === undefined ? '' : `行 ${formatNumber(found.rows)}, `
  return `${rows}平均 ${formatValue(found.mean)}, 最小 ${formatValue(found.min)}, 10 % 点 ${formatValue(found.p10)}, 中央 ${formatValue(found.p50)}, 90 % 点 ${formatValue(found.p90)}, 最大 ${formatValue(found.max)}, 効果が正の行 ${(found.positive * 100).toFixed(1)} %`
}
