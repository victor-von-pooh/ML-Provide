/** 領域のクラスの色 (0 は背景として灰) */
const PALETTE: [number, number, number][] = [
  [229, 231, 235],
  [37, 99, 235],
  [220, 38, 38],
  [22, 163, 74],
  [217, 119, 6],
  [147, 51, 234],
  [8, 145, 178],
  [219, 39, 119],
]

/** 領域の番号の色 (番号が多ければ繰り返す) */
export function labelColor(label: number): [number, number, number] {
  const index = label <= 0 ? 0 : 1 + ((label - 1) % (PALETTE.length - 1))
  return PALETTE[index] ?? [0, 0, 0]
}

/** 領域の格子を RGBA の画素にする */
export function gridPixels(grid: number[][]): {
  width: number
  height: number
  data: Uint8ClampedArray
} {
  const height = grid.length
  const width = grid[0]?.length ?? 0
  const data = new Uint8ClampedArray(width * height * 4)
  grid.forEach((row, y) =>
    row.forEach((label, x) => {
      const [r, g, b] = labelColor(label)
      const at = (y * width + x) * 4
      data[at] = r
      data[at + 1] = g
      data[at + 2] = b
      data[at + 3] = 255
    }),
  )
  return { width, height, data }
}

/** 0 から 255 の画像の格子 ((高さ, 幅) か (高さ, 幅, 3)) を RGBA の画素にする */
export function imagePixels(image: (number | number[])[][]): {
  width: number
  height: number
  data: Uint8ClampedArray
} {
  const height = image.length
  const width = image[0]?.length ?? 0
  const data = new Uint8ClampedArray(width * height * 4)
  image.forEach((row, y) =>
    row.forEach((pixel, x) => {
      const [r = 0, g = 0, b = 0] = typeof pixel === 'number' ? [pixel, pixel, pixel] : pixel
      const at = (y * width + x) * 4
      data[at] = r
      data[at + 1] = g
      data[at + 2] = b
      data[at + 3] = 255
    }),
  )
  return { width, height, data }
}

/** 点群の 2 つの次元を, 縦横の比を保って描く場所に移す (縦は上が大きい値) */
export function projectPoints(
  points: number[][],
  xAxis: number,
  yAxis: number,
  size: number,
  margin = 8,
): { x: number; y: number }[] {
  if (points.length === 0) return []
  const xs = points.map((point) => point[xAxis] ?? 0)
  const ys = points.map((point) => point[yAxis] ?? 0)
  const [xLow, xHigh] = [Math.min(...xs), Math.max(...xs)]
  const [yLow, yHigh] = [Math.min(...ys), Math.max(...ys)]
  const span = Math.max(xHigh - xLow, yHigh - yLow) || 1
  const scale = (size - margin * 2) / span
  // 狭い方の向きは真ん中に寄せる
  const xPad = (size - margin * 2 - (xHigh - xLow) * scale) / 2
  const yPad = (size - margin * 2 - (yHigh - yLow) * scale) / 2
  return xs.map((x, i) => ({
    x: margin + xPad + (x - xLow) * scale,
    y: size - margin - yPad - ((ys[i] ?? 0) - yLow) * scale,
  }))
}
