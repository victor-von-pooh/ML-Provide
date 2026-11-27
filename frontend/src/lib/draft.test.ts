import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { readDraft, useDraftWriter, writeDraft } from './draft'

/** 下書きの名前の欄だけを読む */
function storedName(): string | null {
  const found = readDraft('form', (value) =>
    value && typeof value === 'object' && 'name' in value ? String(value.name) : null,
  )
  return found?.value ?? null
}

/** フックに渡す欄 */
interface Props {
  name: string
  flushKey?: unknown
}

/** 名前の欄を持つ欄の下書きを書く部品のフック (空の名前は書きかけ無し) */
function useNameDraft({ name, flushKey = null }: Props) {
  return useDraftWriter('form', name !== '', { name }, flushKey)
}

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
})

describe('useDraftWriter', () => {
  it('待ってから書き, 外されるときと flushKey が替わるときは待たずに書く', () => {
    const hook = renderHook(useNameDraft, {
      initialProps: { name: 'A' } as Props,
    })
    hook.rerender({ name: 'AB' })
    expect(storedName()).toBeNull()
    act(() => vi.advanceTimersByTime(1000))
    expect(storedName()).toBe('AB')
    hook.rerender({ name: 'ABC', flushKey: 'p' })
    expect(storedName()).toBe('ABC')
    hook.rerender({ name: 'ABCD', flushKey: 'p' })
    hook.unmount()
    expect(storedName()).toBe('ABCD')
  })

  it('戻した下書きは触るまで書き直さない (時刻を保つ)', () => {
    writeDraft('form', { name: 'A' })
    const time = readDraft('form', (value) => value)?.time
    act(() => vi.advanceTimersByTime(5000))
    const hook = renderHook(useNameDraft, {
      initialProps: { name: 'A' } as Props,
    })
    act(() => vi.advanceTimersByTime(1000))
    hook.unmount()
    expect(readDraft('form', (value) => value)?.time).toBe(time)
  })

  it('戻した値に打ち直して待つ前に外すと, 前の中身でなく戻した値のまま', () => {
    writeDraft('form', { name: 'A' })
    const hook = renderHook(useNameDraft, {
      initialProps: { name: 'A' } as Props,
    })
    hook.rerender({ name: 'AB' })
    act(() => vi.advanceTimersByTime(500))
    hook.rerender({ name: 'A' })
    hook.unmount()
    expect(storedName()).toBe('A')
  })

  it('書いた後に戻した値に打ち直して外すと, 最後に見ていた値を書く', () => {
    writeDraft('form', { name: 'A' })
    const hook = renderHook(useNameDraft, {
      initialProps: { name: 'A' } as Props,
    })
    hook.rerender({ name: 'AB' })
    act(() => vi.advanceTimersByTime(1000))
    expect(storedName()).toBe('AB')
    hook.rerender({ name: 'A' })
    hook.unmount()
    expect(storedName()).toBe('A')
  })

  it('空の欄に戻すと下書きを消し, 捨てると待っている書き込みも止める', () => {
    const hook = renderHook(useNameDraft, {
      initialProps: { name: 'A' } as Props,
    })
    hook.rerender({ name: 'AB' })
    act(() => vi.advanceTimersByTime(1000))
    hook.rerender({ name: '' })
    expect(storedName()).toBeNull()
    hook.rerender({ name: 'X' })
    act(() => hook.result.current())
    act(() => vi.advanceTimersByTime(1000))
    hook.unmount()
    expect(storedName()).toBeNull()
  })
})
