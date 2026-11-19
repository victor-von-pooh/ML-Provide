import { gridPixels, imagePixels, labelColor, projectPoints } from './artifacts'

describe('配列の見せ方', () => {
  it('領域の番号ごとに色を付ける (0 は背景)', () => {
    const { width, height, data } = gridPixels([
      [0, 1],
      [2, 1],
    ])
    expect([width, height]).toEqual([2, 2])
    expect(Array.from(data.slice(0, 4))).toEqual([...labelColor(0), 255])
    expect(Array.from(data.slice(4, 8))).toEqual([...labelColor(1), 255])
    expect(labelColor(1)).not.toEqual(labelColor(2))
    expect(labelColor(8)).toEqual(labelColor(1))
  })

  it('灰色と色の画像を RGBA にする', () => {
    expect(Array.from(imagePixels([[128]]).data)).toEqual([128, 128, 128, 255])
    expect(Array.from(imagePixels([[[1, 2, 3]]]).data)).toEqual([1, 2, 3, 255])
    expect(imagePixels([]).width).toBe(0)
  })

  it('点群の 2 つの次元を縦横の比を保って移す', () => {
    const placed = projectPoints(
      [
        [0, 0, 5],
        [2, 1, 5],
      ],
      0,
      1,
      100,
      0,
    )
    // 横の幅 2 が 100 に, 縦は真ん中に寄せ, 上が大きい値
    expect(placed[0]).toEqual({ x: 0, y: 75 })
    expect(placed[1]).toEqual({ x: 100, y: 25 })
    expect(projectPoints([], 0, 1, 100)).toEqual([])
  })
})
