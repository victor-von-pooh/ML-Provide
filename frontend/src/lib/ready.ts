import type { ReadyLicense, ReadyPreprocess } from '../api/types'

/** 種類の名前を並べて見せる数 (ImageNet の 1000 などを全ては出さない) */
export const CLASS_PREVIEW = 10

/** 既製のモデルの一式で使えない機能 (データセットと外れ方の記録を持たないため) */
export const READY_UNSUPPORTED =
  '既製のモデルの一式はデータセットと外れ方の記録を持たないので, 意思決定, 過去のデータでのたどり直し, 外れ方の確かめ, 列の効き, 何段も先の予測は使えません. 結論の決め方 (確信の下限など) は使えます'

/** 種類の名前の表の要約 (多ければ先頭だけと残りの数) */
export function classesText(classes: string[], limit = CLASS_PREVIEW): string {
  if (classes.length <= limit) return classes.join(', ')
  return `${classes.slice(0, limit).join(', ')} ほか ${(classes.length - limit).toLocaleString()} 種類`
}

/** 前処理の文 (短い辺をそろえ, 真ん中を切り出し, 正規化する) */
export function preprocessText(preprocess: ReadyPreprocess): string {
  const numbers = (values: number[]) => values.map((value) => String(value)).join(', ')
  return (
    `短い辺を ${preprocess.resize} にそろえ (${preprocess.interpolation}), ` +
    `真ん中の ${preprocess.crop} × ${preprocess.crop} を切り出し, ` +
    `平均 [${numbers(preprocess.mean)}] と標準偏差 [${numbers(preprocess.std)}] で正規化`
  )
}

/**
 * ライセンスの警告の文 (警告しないものは null).
 * 警告の印の無い記録は, 商用で使えると分かったもの (commercial) の他を警告する
 */
export function licenseAlert(license: ReadyLicense | null | undefined): string | null {
  if (!license) return 'ライセンスの情報がありません. 使う前に配布元で確かめてください'
  const alert = license.alert ?? license.status !== 'commercial'
  if (!alert) return null
  const message = license.message ?? 'ライセンスを確かめてください'
  return license.source ? `${message} (情報: ${license.source})` : message
}

/** 重みの hash の欄の誤り (空は決めない) */
export function sha256Problem(text: string): string | null {
  const value = text.trim()
  if (!value) return null
  return /^[0-9a-fA-F]{64}$/.test(value)
    ? null
    : '重みの hash は sha256 の 16 進の 64 文字で書いてください'
}
