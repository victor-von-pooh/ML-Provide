// 置き場にまだ作っていない欄の書きかけの下書き (ブラウザの localStorage に欄の種類ごとに残す)

import { useEffect, useRef } from 'react'

// 下書きを置くブラウザの保存場所の名前 (欄の種類ごとの下書きの辞書)
const DRAFT_KEY = 'ml-provide:drafts'
// 下書きを戻せる期間 (それより古いものは戻さずに消す)
const DRAFT_DAYS = 14

/** 下書き (欄の中身と書いた時刻) */
export interface Draft<T> {
  value: T
  time: string
}

/** 下書きの辞書を読む (読めない場合は空. 古い下書きは除く) */
function readDrafts(): Record<string, Draft<unknown>> {
  try {
    const stored = JSON.parse(window.localStorage.getItem(DRAFT_KEY) ?? '{}') as unknown
    if (!stored || typeof stored !== 'object') return {}
    const limit = Date.now() - DRAFT_DAYS * 24 * 60 * 60 * 1000
    const drafts: Record<string, Draft<unknown>> = {}
    for (const [kind, item] of Object.entries(stored as Record<string, unknown>)) {
      if (!item || typeof item !== 'object') continue
      const { value, time } = item as Partial<Draft<unknown>>
      const stamp = Date.parse(String(time))
      if (value !== undefined && Number.isFinite(stamp) && stamp > limit) {
        drafts[kind] = { value, time: String(time) }
      }
    }
    return drafts
  } catch {
    return {}
  }
}

/**
 * 欄の種類の下書きを読む (形の合わないものは戻さない)
 *
 * @param kind - 欄の種類
 * @param restore - 読んだ中身を欄の値にする (形が合わなければ null)
 */
export function readDraft<T>(kind: string, restore: (value: unknown) => T | null): Draft<T> | null {
  const found = readDrafts()[kind]
  if (!found) return null
  const value = restore(found.value)
  return value === null ? null : { value, time: found.time }
}

/** 欄の種類の下書きを書く (書けない場合は古い下書きを残さないように消す) */
export function writeDraft(kind: string, value: unknown): void {
  try {
    window.localStorage.setItem(
      DRAFT_KEY,
      JSON.stringify({ ...readDrafts(), [kind]: { value, time: new Date().toISOString() } }),
    )
  } catch {
    clearDraft(kind)
  }
}

/** 欄の種類の下書きを消す */
export function clearDraft(kind: string): void {
  try {
    const drafts = readDrafts()
    if (!(kind in drafts)) return
    delete drafts[kind]
    window.localStorage.setItem(DRAFT_KEY, JSON.stringify(drafts))
  } catch {
    // 保存場所が使えない場合は何もしない
  }
}

/**
 * 欄の書きかけを, 少し待ってから下書きに書く. 部品が外されるときと `flushKey` が替わる
 * ときは待たずに書く
 *
 * 殻の中で段階を移ると部品が外され, ページを閉じないので `beforeunload` の確かめが出ない.
 * 殻でプロジェクトを替えたときも, そのまま段階を移ることがある. 待つ間の書きかけを黙って
 * 捨てないように, そのときに書く. 書きかけが無くなったら (空の欄に戻したら) 下書きも消す
 *
 * @param kind - 欄の種類
 * @param dirty - 書きかけがあるか
 * @param value - 書く中身 (JSON にできる値)
 * @param flushKey - 替わったときに待たずに書く値 (殻で選んだ問題の設定)
 * @param delay - 待つ時間 (ミリ秒)
 * @returns 書きかけを捨てる関数 (作り終えたときなど. 待っている書き込みも止める)
 */
export function useDraftWriter(
  kind: string,
  dirty: boolean,
  value: unknown,
  flushKey: unknown = null,
  delay = 1000,
): () => void {
  // 中身は描くたびに作り直されるので, 文で比べる
  const text = JSON.stringify(value)
  // 保存場所にある中身 (最初から書きかけなら戻した下書きなので, 同じ中身のうちは
  // 書き直さずに時刻を保つ. 書くたび, 消すたびに置き換える)
  const stored = useRef<string | null>(dirty ? text : null)
  // まだ書いていない中身 (無ければ null) と, 待っている書き込み
  const pending = useRef<string | null>(null)
  const timer = useRef<number | null>(null)
  const stop = () => {
    if (timer.current !== null) window.clearTimeout(timer.current)
    timer.current = null
  }
  // まだ書いていない中身を待たずに書く
  const flush = useRef(() => {
    stop()
    const item = pending.current
    if (item === null) return
    pending.current = null
    writeDraft(kind, JSON.parse(item))
    stored.current = item
  })
  useEffect(() => {
    if (!dirty) {
      stop()
      pending.current = null
      clearDraft(kind)
      stored.current = null
      return
    }
    // 保存場所と同じ中身に戻したら, 待っている書き込み (前の中身) を捨てる
    if (text === stored.current) {
      stop()
      pending.current = null
      return
    }
    pending.current = text
    timer.current = window.setTimeout(() => flush.current(), delay)
    return stop
  }, [kind, dirty, text, delay])
  // flushKey が替わったときは, まだ書いていない中身を待たずに書く (最初は書かない)
  const lastKey = useRef(flushKey)
  useEffect(() => {
    if (Object.is(lastKey.current, flushKey)) return
    lastKey.current = flushKey
    flush.current()
  }, [flushKey])
  // 外されるときも, まだ書いていない中身を待たずに書く
  useEffect(() => {
    const write = flush.current
    return () => write()
  }, [])
  return () => {
    stop()
    pending.current = null
    clearDraft(kind)
    stored.current = null
  }
}
