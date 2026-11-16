import type { Band, Rules } from '../api/types'

/** 画面で編集する結論の決め方 (空の欄は文字で持つ) */
export interface RulesForm {
  minConfidence: string
  positive: string
  threshold: string
  bands: { label: string; upper: string }[]
  groupBy: string
}

/** 決め方を編集の形にする */
export function toForm(rules: Rules | undefined | null): RulesForm {
  return {
    minConfidence: rules?.min_confidence != null ? String(rules.min_confidence) : '',
    positive: rules?.positive ?? '',
    threshold: rules?.threshold != null ? String(rules.threshold) : '',
    bands: (rules?.bands ?? []).map((band) => ({
      label: band.label,
      upper: band.upper != null ? String(band.upper) : '',
    })),
    groupBy: rules?.group_by ?? '',
  }
}

/** 数の欄を数にする (空は null, 数でなければ NaN) */
function parseNumber(value: string): number | null {
  const text = value.trim()
  if (!text) return null
  return Number(text)
}

/**
 * 編集の形を API の決め方にし, 画面で分かる誤りを返す
 *
 * 細かな確認 (クラスの有無など) はライブラリが行い, 誤りの文を返す
 */
export function fromForm(form: RulesForm, task: string): { rules: Rules; problems: string[] } {
  const rules: Rules = {}
  const problems: string[] = []
  const classify = task === 'classification'
  if (classify) {
    const floor = parseNumber(form.minConfidence)
    if (floor !== null) {
      if (Number.isNaN(floor) || floor < 0 || floor > 1)
        problems.push('確信の下限は 0 から 1 の数にしてください')
      rules.min_confidence = floor
    }
    if (form.positive) {
      const threshold = parseNumber(form.threshold)
      if (threshold === null || Number.isNaN(threshold) || threshold < 0 || threshold > 1)
        problems.push('陽性とする確率は 0 から 1 の数にしてください')
      rules.positive = form.positive
      rules.threshold = threshold
      if (floor !== null) problems.push('陽性のクラスと確信の下限は一緒に使えません')
    }
  } else if (form.bands.length > 0) {
    const bands: Band[] = form.bands.map((band, index) => {
      const last = index === form.bands.length - 1
      const upper = last ? null : parseNumber(band.upper)
      if (!band.label.trim()) problems.push(`帯の ${index + 1} つ目に名前を付けてください`)
      if (!last && (upper === null || Number.isNaN(upper)))
        problems.push(`帯「${band.label || index + 1}」の上限を数で書いてください`)
      return last ? { label: band.label.trim() } : { label: band.label.trim(), upper }
    })
    const uppers = bands.slice(0, -1).map((band) => band.upper ?? Number.NaN)
    if (uppers.some((value, index) => index > 0 && value <= uppers[index - 1]!))
      problems.push('帯は上限の小さい順に並べてください')
    rules.bands = bands
  }
  if (form.groupBy) rules.group_by = form.groupBy
  return { rules, problems }
}
