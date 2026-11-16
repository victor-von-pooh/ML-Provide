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
export function fromForm(
  form: RulesForm,
  task: string,
  classes: string[] = [],
): { rules: Rules; problems: string[] } {
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
    // 陽性は 2 クラスの分類だけ
    if (form.positive && classes.length === 2) {
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
    if (bands.length < 2) problems.push('帯は 2 つ以上にしてください')
    const labels = bands.map((band) => band.label)
    if (new Set(labels).size !== labels.length) problems.push('帯の名前が重なっています')
    rules.bands = bands
  }
  if (form.groupBy.trim()) rules.group_by = form.groupBy.trim()
  return { rules, problems: [...new Set(problems)] }
}

/**
 * 推論のときに送る決め方 (一式の既定値を消した項目は null で送る)
 *
 * ライブラリは一式の既定値に送った決め方を重ね, null の項目を消す
 */
export function withCleared(rules: Rules): Rules {
  return {
    min_confidence: null,
    positive: null,
    threshold: null,
    bands: null,
    group_by: null,
    ...rules,
  }
}
