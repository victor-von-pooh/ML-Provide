import { defineConfig } from '@playwright/test'

// 画面の通しのテスト. e2e/serve.sh が作業用の置き場でアプリを起動する
const port = Number(process.env.E2E_PORT ?? 8794)

export default defineConfig({
  testDir: 'e2e',
  workers: 1,
  timeout: 60_000,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    // 起動ごとのトークンの代わりに決まったトークンを送る
    extraHTTPHeaders: { 'x-provide-token': 'e2e-token' },
    // 手元では入っている Chrome を使い, CI では Playwright の Chromium を使う
    channel: process.env.CI ? undefined : 'chrome',
    viewport: { width: 1600, height: 900 },
    trace: 'retain-on-failure',
  },
  webServer: {
    // 手元では古い画面で確かめないように, 先に画面をビルド (CI はビルド済み)
    command: process.env.CI ? 'bash e2e/serve.sh' : 'pnpm build && bash e2e/serve.sh',
    url: `http://127.0.0.1:${port}/api/info`,
    env: { E2E_PORT: String(port) },
    timeout: 180_000,
    reuseExistingServer: false,
  },
})
