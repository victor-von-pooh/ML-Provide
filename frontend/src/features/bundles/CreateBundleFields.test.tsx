import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { AutoDetails } from './CreateBundleFields'

/** 欄の details 要素 */
function details(): HTMLDetailsElement {
  return screen.getByText('入力の欄').closest('details') as HTMLDetailsElement
}

/** 開き閉じを利用者が切り替える (onToggle は details の toggle の知らせ) */
function toggle(open: boolean) {
  details().open = open
  fireEvent(details(), new Event('toggle'))
}

const view = (wantOpen: boolean, problems: number) => (
  <AutoDetails summary="入力の欄" wantOpen={wantOpen} problems={problems}>
    <input aria-label="入力の名前" />
  </AutoDetails>
)

describe('AutoDetails', () => {
  it('誤りが 0 から 1 以上になると開き, 中の入力が見え, 誤りが消えても閉じない', () => {
    const { rerender } = render(view(false, 0))
    expect(details().open).toBe(false)
    rerender(view(false, 1))
    expect(details().open).toBe(true)
    expect(screen.getByLabelText('入力の名前')).toBeVisible()
    rerender(view(false, 0))
    expect(details().open).toBe(true)
  })

  it('利用者の開き閉じを写し, 誤りが続く間は開き直さない', () => {
    const { rerender } = render(view(false, 1))
    toggle(false)
    rerender(view(false, 2))
    expect(details().open).toBe(false)
    // 一度消えてから出直すと開く
    rerender(view(false, 0))
    rerender(view(false, 1))
    expect(details().open).toBe(true)
  })

  it('wantOpen が真になると開き, 偽になっても閉じない', () => {
    const { rerender } = render(view(false, 0))
    rerender(view(true, 0))
    expect(details().open).toBe(true)
    rerender(view(false, 0))
    expect(details().open).toBe(true)
  })
})
