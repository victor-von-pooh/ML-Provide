import { useEffect, useState } from 'react'

/** 値が変わってから少し待った後の値 (打つたびに重い API を呼ばないため) */
export function useDebounced<T>(value: T, delay = 300): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(value), delay)
    return () => window.clearTimeout(timer)
  }, [value, delay])
  return debounced
}
