import type { PolicyAct, PolicySpace } from '../api/types'

/** 守りの結果の名前 */
export const STATUS_LABELS: Record<PolicyAct['status'], string> = {
  ok: 'そのまま',
  clipped: '範囲に収めた',
  fallback: '決めた行動にした',
  stop: '止めた',
  handover: '元の制御に戻した',
}

/** 外れたときの動きの名前 */
export const FALLBACK_LABELS: Record<string, string> = {
  stop: '止める',
  handover: '元の制御に戻す',
  action: '決めた行動にする',
}

/** 空間の要素の数 (discrete は 1 つ) */
export function spaceSize(space: PolicySpace): number {
  if (space.type === 'discrete') return 1
  return space.shape.reduce((total, length) => total * length, 1)
}

/** 空間の短い説明 */
export function spaceLabel(space: PolicySpace): string {
  if (space.type === 'discrete') {
    return `番号 ${space.start} から ${space.start + space.n - 1} (${space.n} 通り)`
  }
  return `数の並び (形 [${space.shape.join(', ')}])`
}

/**
 * 観測の欄の文字を値にする. `[` で始まれば JSON の配列 (入れ子も可), 他は
 * 区切り (カンマか空白) で分けた数の並び. 読めなければ誤りの文
 */
export function parseObservation(text: string): { value: unknown } | { error: string } {
  const trimmed = text.trim()
  if (!trimmed) return { error: '観測を入れてください' }
  if (trimmed.startsWith('[')) {
    try {
      return { value: JSON.parse(trimmed) as unknown }
    } catch {
      return { error: 'JSON の配列として読めません' }
    }
  }
  const tokens = trimmed.split(/[\s,]+/).filter(Boolean)
  const values = tokens.map(Number)
  const wrong = tokens.find((_, index) => !Number.isFinite(values[index]))
  if (wrong !== undefined) return { error: `${wrong} は数として読めません` }
  return { value: values }
}

/** 行動の見せ方 (止めるか元の制御に戻す場合は「行動なし」) */
export function actionLabel(action: PolicyAct['action']): string {
  if (action === null) return '行動なし'
  if (typeof action === 'number') return String(action)
  return JSON.stringify(action)
}
