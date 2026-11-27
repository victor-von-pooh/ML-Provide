import '@testing-library/jest-dom/vitest'

// 下書きなどブラウザの保存場所に残したものを, 次のテストに持ち越さない
beforeEach(() => {
  window.localStorage.clear()
})
