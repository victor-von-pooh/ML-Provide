import type { Description } from '../../api/types'
import {
  FORM_LABELS,
  isColumnOutput,
  licenseText,
  ORIGIN_LABELS,
  originNote,
  outputDetails,
  placeText,
  ROLE_LABELS,
  TARGET_KIND_LABELS,
} from '../../lib/outputs'

/** 説明の 1 行 */
function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[9rem_1fr] gap-2 py-1">
      <dt className="text-ink-3">{label}</dt>
      <dd className="min-w-0 break-words">{children}</dd>
    </div>
  )
}

/**
 * 正解の出どころの注意と学習済みの重みの警告 (必ず出す).
 * 一式の画面と結果の画面で使う
 */
export function OutputNotices({
  target,
  warnings,
}: {
  target: Description['target_info']
  warnings: string[] | undefined
}) {
  const origin = originNote(target)
  if (!origin && !warnings?.length) return null
  return (
    <div role="note" aria-label="一式の注意" className="flex flex-col gap-1">
      {origin && (
        <p className="rounded-lg border border-warning-text/40 bg-surface p-2 text-sm text-warning-text">
          {origin}
        </p>
      )}
      {warnings?.map((warning, i) => (
        <p
          key={`${i}:${warning}`}
          className="rounded-lg border border-critical-text/40 bg-surface p-2 text-sm text-critical-text"
        >
          学習済みの重みの警告: {warning}
        </p>
      ))}
    </div>
  )
}

/** 一式の受け持ち, 出力の形, 正解, 手順, 学習済みの重みの置き方 */
export function OutputView({
  description,
  modelLoaded,
}: {
  description: Description
  modelLoaded: boolean
}) {
  const form = description.output?.form
  const column = isColumnOutput(description)
  const details = outputDetails(description.output)
  const target = description.target_info
  const procedure = description.procedure
  const pretrained = description.pretrained ?? []
  const sourced = pretrained.some((record) => record.place === 'source')
  return (
    <dl className="text-sm">
      <Row label="一式の版">{description.format_version ?? '–'}</Row>
      <Row label="受け持ち">{ROLE_LABELS[description.role ?? 'predict'] ?? description.role}</Row>
      <Row label="出力の形">
        {form ? (FORM_LABELS[form] ?? form) : '–'}
        {details.length > 0 && <span className="text-ink-2"> ({details.join('. ')})</span>}
        {!column && (
          <p className="text-xs text-ink-2">
            列の出力でないので, 結論は件数だけです (結論の決め方と意思決定は使えません). 1
            件ごとの出力は結果の画面で見られます
          </p>
        )}
      </Row>
      <Row label="正解">
        {target ? (
          <>
            {TARGET_KIND_LABELS[target.kind] ?? target.kind}
            {target.origin && (
              <>
                {' '}
                / 出どころ: {ORIGIN_LABELS[target.origin.type] ?? target.origin.type}
                {target.origin.name && ` (${target.origin.name})`}
              </>
            )}
          </>
        ) : (
          '不明 (正解の無いデータセットか, 書いていない)'
        )}
      </Row>
      <Row label="繰り返しの手順">
        {procedure ? (
          <span>
            <span className="font-mono">{procedure.name}</span>, {procedure.steps} 回, 乱数の種{' '}
            {procedure.seed}
            {procedure.options && Object.keys(procedure.options).length > 0 && (
              <span className="font-mono text-xs"> {JSON.stringify(procedure.options)}</span>
            )}
          </span>
        ) : (
          'なし'
        )}
      </Row>
      <Row label="学習済みの重み">
        {pretrained.length === 0 ? (
          'なし'
        ) : (
          <ul aria-label="学習済みの重みの置き方" className="flex flex-col gap-1">
            {pretrained.map((record) => {
              const license = licenseText(record)
              return (
                <li key={record.node}>
                  <span className="font-mono text-xs">{record.node}</span>: {placeText(record)}.
                  ライセンス{' '}
                  <span className={license.warn ? 'text-warning-text' : undefined}>
                    {license.text}
                  </span>
                  {record.source && <span className="text-ink-3"> (情報: {record.source})</span>}
                </li>
              )
            })}
          </ul>
        )}
        {sourced && !modelLoaded && (
          <p className="text-xs text-ink-3">
            出どころから読む重みは推論のときに読み, 記録と同じかを確かめます (違えば警告を出します)
          </p>
        )}
      </Row>
    </dl>
  )
}
