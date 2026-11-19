import { useQuery } from '@tanstack/react-query'
import { Download } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { api } from '../../api/client'
import type { ArtifactPreview, Result } from '../../api/types'
import { Card, ErrorBox, Spinner } from '../../components/ui'
import { gridPixels, imagePixels, labelColor, projectPoints } from '../../lib/artifacts'
import { formatNumber } from '../../lib/format'
import { FORM_LABELS } from '../../lib/outputs'

// 点群の散布図の 1 辺
const PLOT = 320

// 次元の名前
const AXES = ['x', 'y', 'z']

/** RGBA の画素をキャンバスに描く (間引いた絵を大きく見せる) */
function PixelCanvas({
  pixels,
  label,
}: {
  pixels: { width: number; height: number; data: Uint8ClampedArray }
  label: string
}) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const canvas = ref.current
    // テストの環境など, 描けない場合は何もしない
    const context = canvas?.getContext?.('2d')
    if (!canvas || !context || pixels.width === 0) return
    canvas.width = pixels.width
    canvas.height = pixels.height
    const image = context.createImageData(pixels.width, pixels.height)
    image.data.set(pixels.data)
    context.putImageData(image, 0, 0)
  }, [pixels])
  const scale = Math.max(1, Math.floor(320 / Math.max(pixels.width, pixels.height, 1)))
  return (
    <canvas
      ref={ref}
      role="img"
      aria-label={label}
      className="border border-line [image-rendering:pixelated]"
      style={{ width: pixels.width * scale, height: pixels.height * scale }}
    />
  )
}

/** 点群の 2 つの次元の散布図 */
function PointsPlot({ preview }: { preview: ArtifactPreview }) {
  const points = preview.points ?? []
  const dims = points[0]?.length ?? 0
  const [xChosen, setXAxis] = useState(0)
  const [yChosen, setYAxis] = useState(1)
  // 次元が少ない点群でも選べる範囲に収める
  const last = Math.max(0, dims - 1)
  const xAxis = Math.min(xChosen, last)
  const yAxis = Math.min(yChosen, last)
  const placed = projectPoints(points, xAxis, yAxis, PLOT)
  return (
    <div className="flex flex-col gap-2">
      <div className={dims < 2 ? 'hidden' : 'flex gap-3 text-xs'}>
        {(
          [
            ['横', xAxis, setXAxis],
            ['縦', yAxis, setYAxis],
          ] as const
        ).map(([name, value, set]) => (
          <label key={name} className="flex items-center gap-1">
            {name}の次元
            <select
              value={value}
              onChange={(event) => set(Number(event.target.value))}
              className="h-7 rounded border border-line bg-surface px-1"
            >
              {AXES.slice(0, dims).map((axis, i) => (
                <option key={axis} value={i}>
                  {axis}
                </option>
              ))}
            </select>
          </label>
        ))}
      </div>
      <svg
        role="img"
        aria-label={`点群の散布図 (${points.length.toLocaleString()} 点)`}
        width={PLOT}
        height={PLOT}
        className="rounded border border-line bg-surface"
      >
        {placed.map((point, i) => (
          <circle key={i} cx={point.x} cy={point.y} r={1.5} className="fill-accent" />
        ))}
      </svg>
      <p className="text-xs text-ink-3">
        {(preview.total ?? points.length).toLocaleString()} 点のうち{' '}
        {points.length.toLocaleString()} 点を間引いて描いています
      </p>
    </div>
  )
}

/** 1 件の配列の見せ方 (形ごと) */
export function ArtifactBody({ preview, names }: { preview: ArtifactPreview; names: string[] }) {
  // 描く画素は見本が変わったときだけ作り直す
  const gridImage = useMemo(() => (preview.grid ? gridPixels(preview.grid) : null), [preview.grid])
  const photo = useMemo(() => (preview.image ? imagePixels(preview.image) : null), [preview.image])
  const shape = preview.shape.join(' × ')
  return (
    <div className="flex flex-col gap-3 text-sm">
      <p className="text-ink-2">
        形 ({preview.shape.join(', ')}), 型 {preview.dtype}
        {preview.view === 'values' && `, 長さ (L2) ${formatNumber(preview.norm)}`}
      </p>
      {preview.view === 'grid' && gridImage && (
        <>
          <PixelCanvas pixels={gridImage} label={`領域 (${preview.row} 行目, ${shape})`} />
          <ul aria-label="領域の番号ごとの画素" className="flex flex-wrap gap-3 text-xs">
            {(preview.labels ?? []).map((item) => (
              <li key={item.label} className="flex items-center gap-1">
                <span
                  aria-hidden
                  className="inline-block size-3 rounded-sm"
                  style={{ backgroundColor: `rgb(${labelColor(item.label).join(',')})` }}
                />
                {item.label}
                {names[item.label] ? ` (${names[item.label]})` : ''}: {item.count.toLocaleString()}{' '}
                画素
              </li>
            ))}
          </ul>
        </>
      )}
      {preview.view === 'image' && photo && (
        <PixelCanvas pixels={photo} label={`画像 (${preview.row} 行目, ${shape})`} />
      )}
      {preview.view === 'points' && <PointsPlot preview={preview} />}
      {preview.view === 'values' && (
        <div>
          <p className="mb-1 text-xs text-ink-3">
            最初の {Math.min(preview.values.length, preview.size)} 個の値 (全 {preview.size} 個)
          </p>
          <p aria-label="最初の値" className="font-mono text-xs break-words">
            {preview.values.map((value) => formatNumber(value)).join(', ')}
          </p>
        </div>
      )}
    </div>
  )
}

/** 配列の出力の結果の, 選んだ行の配列 */
export function ArtifactCard({
  result,
  row,
  onRow,
  names,
}: {
  result: Result
  row: number | null
  onRow: (row: number) => void
  names: string[]
}) {
  // 行を選んでいなければ, 推論した最初の行
  const first = useQuery({
    queryKey: ['rows', result.id, 'first-ok'],
    queryFn: () => api.rows(result.id, { offset: 0, limit: 1, status: 'ok' }),
    enabled: row === null,
  })
  const firstRow = first.data?.rows[0]?.index
  const shown = row ?? (typeof firstRow === 'number' ? firstRow : null)
  const preview = useQuery({
    queryKey: ['artifact', result.id, shown],
    queryFn: () => api.artifact(result.id, shown ?? 0),
    enabled: shown !== null,
    retry: false,
  })
  const form = result.artifacts?.form ?? result.output_form ?? ''
  const total = result.summary?.rows ?? 0
  return (
    <Card
      title={`1 件の出力 (${FORM_LABELS[form] ?? form})`}
      actions={
        shown !== null && (
          <a
            href={api.artifactUrl(result.id, shown)}
            className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-line bg-surface px-3 text-sm font-medium hover:bg-surface-2"
          >
            <Download className="size-4" aria-hidden />
            .npy
          </a>
        )
      }
    >
      <div className="mb-3 flex items-center gap-2 text-sm">
        <label htmlFor="artifact-row">行 (#)</label>
        <input
          id="artifact-row"
          type="number"
          min={0}
          max={Math.max(0, total - 1)}
          value={shown ?? ''}
          onChange={(event) => {
            // 空にした途中は 0 に飛ばさない
            if (event.target.value === '') return
            const value = Number(event.target.value)
            if (Number.isInteger(value) && value >= 0) onRow(value)
          }}
          className="h-8 w-24 rounded-lg border border-line bg-surface px-2"
        />
        <span className="text-xs text-ink-3">予測の表の「見る」でも選べます</span>
      </div>
      {first.isPending && row === null && <Spinner />}
      {first.error && row === null && <ErrorBox error={first.error} live={false} />}
      {shown === null && first.isSuccess && (
        <p className="text-sm text-ink-2">推論した行がありません</p>
      )}
      {preview.isPending && shown !== null && <Spinner />}
      {preview.error && <ErrorBox error={preview.error} live={false} />}
      {preview.data && <ArtifactBody preview={preview.data} names={names} />}
    </Card>
  )
}
