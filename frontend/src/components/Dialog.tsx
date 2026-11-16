import { useEffect, useRef, useState, type ReactNode } from 'react'

import { Button } from './ui'

/** 画面の上に重ねる対話の枠 (Esc か外を押すと閉じる) */
export function Dialog({
  title,
  onClose,
  children,
}: {
  title: string
  onClose: () => void
  children: ReactNode
}) {
  const ref = useRef<HTMLDialogElement>(null)
  // 開く前に操作していた所 (最初の描画のときに覚え, 開発時の 2 回の実行でも
  // ダイアログの中を覚えない)
  const [before] = useState(() => document.activeElement)
  useEffect(() => {
    ref.current?.showModal()
    return () => {
      // 閉じたら (枠を外したら) そこに戻す. 定義を開いて上の操作を作り直した
      // 場合は, 同じ印 (data-return) の新しいボタンに戻す
      const mark = before instanceof HTMLElement ? before.dataset.return : undefined
      window.requestAnimationFrame(() => {
        if (before instanceof HTMLElement && before.isConnected) before.focus()
        else if (mark) document.querySelector<HTMLElement>(`[data-return="${mark}"]`)?.focus()
      })
    }
  }, [before])
  return (
    <dialog
      ref={ref}
      onClose={onClose}
      onClick={(event) => event.target === ref.current && onClose()}
      aria-label={title}
      className="m-auto w-[min(48rem,92vw)] rounded-lg border border-line bg-surface p-0 text-ink backdrop:bg-black/40"
    >
      <div className="flex items-center justify-between border-b border-line px-4 py-2">
        <h2 className="text-sm font-semibold">{title}</h2>
        <Button variant="ghost" onClick={onClose}>
          閉じる
        </Button>
      </div>
      <div className="max-h-[70vh] overflow-auto p-4">{children}</div>
    </dialog>
  )
}
