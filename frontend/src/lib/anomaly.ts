import type { AnomalyNormal, AnomalySummary, Result, TriageSummary } from '../api/types'
import { formatNumber, formatValue } from './format'

/** 度合いの決め方の日本語の名前 (ML-Inference の結論の文と同じ言葉) */
export const SCORE_LABELS: Record<string, string> = {
  autoencoder: '自己符号化器の再構成の誤差',
  distance: '学習用の正常の行の中心からの距離',
}

/** 人の確かめの判断の日本語の名前 */
export const JUDGMENT_LABELS: Record<string, string> = {
  anomaly: '異常',
  normal: '正常',
}

/** 実験の評価の分割の名前 (ML-Inference の結論の文と同じ言葉) */
export const EVALUATION_SPLITS: Record<string, string> = {
  valid: '検証用',
  test: 'テスト用',
  unlabeled: '正解の欠けた',
}

/** 人の確かめに回す件数の上限の上限 (API の MAX_CAPACITY と同じ. 待ち行列を 1 度に返すため) */
export const MAX_CAPACITY = 1000

/** 異常の度合いの一式で使えないものの文 */
export const ANOMALY_UNSUPPORTED =
  '度合いは確率ではないので, 損得から回すしきい値は決めず, 度合いの高い順に, 人が確かめる件数の上限まで人の確かめに回します. 判断の決め方を変えての出し直し, 過去のデータで得と損を確かめること, 予測の外れ方の測り直し, どの列が度合いを動かしたかの確かめは使えません'

/** 問題の欄が無い理由 */
export const ANOMALY_NO_PROBLEM =
  '件数の上限の既定と度合いの決め方は一式が実験から写して持つので, 問題はつなぎません'

/** 確かめた人の名前を覚えるブラウザの鍵 */
const REVIEWER_KEY = 'ml-provide:reviewer'

/** 件数の上限の欄の誤り (無ければ null) */
export function capacityProblem(text: string): string | null {
  const value = text.trim()
  if (!/^\d+$/.test(value) || Number(value) < 1 || Number(value) > MAX_CAPACITY)
    return `人の確かめに回す件数の上限は 1 から ${MAX_CAPACITY} の整数で書いてください`
  return null
}

/** 異常の度合いの結果のまとめ (異常の度合いの結果でなければ null) */
export function anomalySummary(result: Result): AnomalySummary | null {
  const summary = result.summary as AnomalySummary | null
  if (result.kind !== 'anomaly' || !summary || !Array.isArray(summary.top)) return null
  return summary
}

/** 振り分けのまとめ (振り分けていない結果は null) */
export function triageSummary(result: Result): TriageSummary | null {
  const decided = result.decision
  if (!decided || decided.decision?.type !== 'triage') return null
  const summary = decided.summary as unknown as TriageSummary
  return typeof summary.acted === 'number' ? summary : null
}

/** 覚えた確かめた人の名前 (読めなければ空) */
export function loadReviewer(): string {
  try {
    return window.localStorage.getItem(REVIEWER_KEY) ?? ''
  } catch {
    return ''
  }
}

/** 確かめた人の名前を覚える (書けなければ覚えない) */
export function saveReviewer(name: string): void {
  try {
    window.localStorage.setItem(REVIEWER_KEY, name)
  } catch {
    // 覚えられないブラウザでは, この画面の間だけの名前
  }
}

/** 確かめた人の名前の欄の誤り (無ければ null. 空と改行は ML-Inference も断る) */
export function reviewerProblem(name: string): string | null {
  if (!name.trim()) return '確かめた人の名前を書いてください'
  if (name.trim().length > 80) return '確かめた人の名前は 80 文字までにしてください'
  return null
}

/** 確かめた時刻を手元の時刻で (読めなければそのまま) */
export function reviewedText(when: string | null | undefined): string {
  if (!when) return '–'
  const date = new Date(when)
  return Number.isNaN(date.getTime()) ? when : date.toLocaleString()
}

/** 学習用の正常の行の度合いの分布の 1 行の文 */
export function normalText(normal: AnomalyNormal): string {
  return `行 ${formatNumber(normal.rows)}, 平均 ${formatValue(normal.mean)}, 中央 ${formatValue(normal.p50)}, 90 % 点 ${formatValue(normal.p90)}, 95 % 点 ${formatValue(normal.p95)}, 99 % 点 ${formatValue(normal.p99)}, 最大 ${formatValue(normal.max)}`
}
