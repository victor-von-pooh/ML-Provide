import { keepPreviousData, useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { ArrowUp, File, Folder, Home, Package } from 'lucide-react'
import { useState } from 'react'

import { useApi } from '../api/client'
import { formatBytes } from '../lib/format'
import { Dialog } from './Dialog'
import { Button, ErrorBox, Spinner } from './ui'

/** 手元のファイルかディレクトリを選ぶ対話 (推論の一式のディレクトリには印) */
export function PathPicker({
  title,
  description,
  pickDirs = true,
  pickFiles = true,
  start = '~',
  onClose,
  onPick,
}: {
  title: string
  description?: string
  pickDirs?: boolean
  pickFiles?: boolean
  start?: string
  onClose: () => void
  onPick: (path: string) => void
}) {
  const api = useApi()
  const [dir, setDir] = useState(start)
  const [typed, setTyped] = useState('')
  // 開けないディレクトリの後も, 前の一覧と「上へ」を使えるように
  const listing = useQuery({
    queryKey: ['browse', dir],
    queryFn: () => api.browse(dir),
    placeholderData: keepPreviousData,
  })
  const [picked, setPicked] = useState<string | null>(null)
  const [missing, setMissing] = useState('')
  return (
    <Dialog title={title} onClose={onClose}>
      {description && <p className="mb-3 text-xs text-ink-2">{description}</p>}
      <form
        className="mb-3 flex gap-2"
        onSubmit={(event) => {
          event.preventDefault()
          // 対話は外側のフォームの中に置かれるので, 外側のフォームまで送らない
          event.stopPropagation()
          // ディレクトリなら開き, ファイルならそれを選ぶ
          const value = typed.trim()
          if (!value) return
          setMissing('')
          api.browse(value).then(
            () => setDir(value),
            () =>
              pickFiles
                ? setPicked(value)
                : setMissing(`${value} はディレクトリではないか, 開けません`),
          )
        }}
      >
        <label className="sr-only" htmlFor="picker-path">
          パス
        </label>
        <input
          id="picker-path"
          value={typed}
          onChange={(event) => setTyped(event.target.value)}
          placeholder={listing.data?.path ?? '~/'}
          className="h-8 flex-1 rounded-lg border border-line bg-surface px-2 font-mono text-sm"
        />
        <Button type="submit">移動</Button>
      </form>
      {missing && (
        <p role="alert" className="mb-2 text-xs text-critical-text">
          {missing}
        </p>
      )}
      <div className="mb-2 flex items-center gap-2 text-xs text-ink-3">
        <Button
          variant="ghost"
          icon={ArrowUp}
          disabled={!listing.data?.parent}
          onClick={() => listing.data?.parent && setDir(listing.data.parent)}
        >
          上へ
        </Button>
        <Button variant="ghost" icon={Home} onClick={() => setDir('~')}>
          ホーム
        </Button>
        <span className="truncate font-mono">{listing.data?.path}</span>
        {pickDirs && listing.data && (
          <Button variant="ghost" className="ml-auto" onClick={() => setPicked(listing.data.path)}>
            このディレクトリを選ぶ
          </Button>
        )}
      </div>
      {listing.isPending && <Spinner />}
      {listing.error && <ErrorBox error={listing.error} />}
      <ul
        aria-label="ディレクトリの中身"
        className="mb-4 max-h-72 overflow-auto rounded-lg border border-line"
      >
        {listing.data?.entries.map((entry) => {
          const selectable = entry.is_dir ? pickDirs : pickFiles
          return (
            <li key={entry.path} className="flex items-center">
              <button
                type="button"
                disabled={!selectable && !entry.is_dir}
                onClick={() => (selectable ? setPicked(entry.path) : setDir(entry.path))}
                onDoubleClick={() => entry.is_dir && setDir(entry.path)}
                className={clsx(
                  'flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm hover:bg-surface-2 disabled:opacity-50',
                  picked === entry.path && 'bg-accent-soft',
                )}
              >
                {entry.bundle ? (
                  <Package size={14} className="text-accent" aria-hidden />
                ) : entry.is_dir ? (
                  <Folder size={14} className="text-accent" aria-hidden />
                ) : (
                  <File size={14} className="text-ink-3" aria-hidden />
                )}
                <span className="flex-1 truncate font-mono">{entry.name}</span>
                {entry.bundle && <span className="text-xs text-accent-text">推論の一式</span>}
                {!entry.is_dir && (
                  <span className="text-xs text-ink-3">{formatBytes(entry.size)}</span>
                )}
              </button>
              {entry.is_dir && (
                <Button
                  variant="ghost"
                  className="mr-1 shrink-0"
                  aria-label={`${entry.name} を開く`}
                  onClick={() => setDir(entry.path)}
                >
                  開く
                </Button>
              )}
            </li>
          )
        })}
        {listing.data?.truncated && (
          <li className="px-3 py-2 text-xs text-ink-3">
            多いので先頭の 1000 件だけを出しています.
          </li>
        )}
        {listing.data?.entries.length === 0 && (
          <li className="px-3 py-2 text-sm text-ink-3">空のディレクトリです.</li>
        )}
      </ul>
      <div className="flex items-center gap-2">
        <span className="flex-1 truncate font-mono text-sm" aria-live="polite">
          {picked ?? '選んでいません'}
        </span>
        <Button variant="primary" disabled={!picked} onClick={() => picked && onPick(picked)}>
          選ぶ
        </Button>
      </div>
    </Dialog>
  )
}

/** パスの欄と, 対話で選ぶボタン */
export function PathField({
  id,
  label,
  value,
  onChange,
  pickDirs,
  pickFiles,
  hint,
}: {
  id: string
  label: string
  value: string
  onChange: (value: string) => void
  pickDirs?: boolean
  pickFiles?: boolean
  hint?: string
}) {
  const [open, setOpen] = useState(false)
  return (
    <div>
      <label htmlFor={id} className="mb-1 block text-xs font-medium text-ink-2">
        {label}
      </label>
      <div className="flex gap-2">
        <input
          id={id}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          className="h-8 min-w-0 flex-1 rounded-lg border border-line bg-surface px-2 font-mono text-sm"
        />
        <Button onClick={() => setOpen(true)} data-return={id}>
          選ぶ
        </Button>
      </div>
      {hint && <p className="mt-1 text-xs text-ink-3">{hint}</p>}
      {open && (
        <PathPicker
          title={label}
          pickDirs={pickDirs}
          pickFiles={pickFiles}
          onClose={() => setOpen(false)}
          onPick={(path) => {
            onChange(path)
            setOpen(false)
          }}
        />
      )}
    </div>
  )
}
