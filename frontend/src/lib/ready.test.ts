import { classesText, licenseAlert, preprocessText, sha256Problem } from './ready'

describe('既製のモデルの一式の文', () => {
  it('種類の名前は多ければ先頭と残りの数にする', () => {
    expect(classesText(['cat', 'dog'])).toBe('cat, dog')
    const many = Array.from({ length: 1000 }, (_, i) => `c${i}`)
    expect(classesText(many)).toBe('c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 ほか 990 種類')
  })

  it('前処理を大きさ, 切り出し, 正規化の文にする', () => {
    const text = preprocessText({
      resize: 256,
      crop: 224,
      interpolation: 'bilinear',
      mean: [0.485, 0.456, 0.406],
      std: [0.229, 0.224, 0.225],
    })
    expect(text).toContain('短い辺を 256 にそろえ (bilinear)')
    expect(text).toContain('224 × 224')
    expect(text).toContain('[0.485, 0.456, 0.406]')
  })

  it('ライセンスは警告するものだけ文を返し, 情報が無ければ確かめるよう言う', () => {
    const base = {
      license: 'apache-2.0',
      status: 'ok',
      redistributable: true,
      gated: false,
      alert: false,
      message: null,
      source: null,
      url: null,
    }
    expect(licenseAlert(base)).toBeNull()
    expect(
      licenseAlert({ ...base, alert: true, message: 'ライセンス: 不明', source: 'torchvision' }),
    ).toBe('ライセンス: 不明 (情報: torchvision)')
    expect(licenseAlert(null)).toMatch(/確かめて/)
    expect(licenseAlert({ ...base, alert: null, status: 'unknown' })).toMatch(/確かめて/)
    expect(licenseAlert({ ...base, alert: null, status: 'commercial' })).toBeNull()
  })

  it('重みのハッシュ値は空か 16 進の 64 文字', () => {
    expect(sha256Problem('')).toBeNull()
    expect(sha256Problem(` ${'A'.repeat(64)} `)).toBeNull()
    expect(sha256Problem('abc')).toMatch(/64 文字/)
    expect(sha256Problem('g'.repeat(64))).toMatch(/64 文字/)
  })
})
