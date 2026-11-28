/// <reference lib="dom" />
/// <reference lib="dom.iterable" />
// README と操作ガイドの画面の例 (docs/images/*.png) を撮る (CI では使わない).
// 先に pnpm build をして, 別の端末で bash e2e/screens.sh を起動しておき,
// frontend で node e2e/screens.ts を実行する
import { resolve } from 'node:path'

import { chromium, expect, type Locator, type Page } from '@playwright/test'

const port = Number(process.env.SCREENS_PORT ?? 8797)
const base = `http://127.0.0.1:${port}`
const home = resolve('e2e/.screens')
const out = resolve('../docs/images')

/** 画面に出る作業用の置き場 (ホーム) の場所を ~ で示す (手元の場所を写さない) */
function hideHome(prefix: string) {
  const swap = (text: string) => text.split(prefix).join('~')
  const run = () => {
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT)
    for (let node = walker.nextNode(); node; node = walker.nextNode())
      if (node.nodeValue?.includes(prefix)) node.nodeValue = swap(node.nodeValue)
    for (const input of document.querySelectorAll('input'))
      if (input.value.includes(prefix)) input.value = swap(input.value)
  }
  document.addEventListener('DOMContentLoaded', () => {
    new MutationObserver(run).observe(document.body, {
      subtree: true,
      childList: true,
      characterData: true,
    })
    setInterval(run, 100)
  })
}

/** 要素を本文の上の端に寄せる */
async function scrollTo(target: Locator) {
  // 本文の欄だけを送る (scrollIntoView はページ全体も送り, 下に空白が出る)
  await target.evaluate((element) => {
    const main = document.querySelector('main')!
    const top = element.getBoundingClientRect().top - main.getBoundingClientRect().top
    main.scrollBy(0, top - 16)
  })
}

async function shot(page: Page, name: string) {
  // 場所の置き換えと描画が済むのを待つ
  await page.waitForTimeout(800)
  await page.mouse.move(0, 0)
  await page.screenshot({ path: `${out}/${name}.png` })
}

/** 一式を開き, 見本のデータ (新しいか過去) で推論して結果の画面に移る */
async function infer(page: Page, bundle: string, sample: RegExp, ahead = false) {
  await page.goto(`${base}/`)
  await page.getByRole('link', { name: bundle }).first().click()
  await page.getByRole('button', { name: sample }).click()
  if (ahead) await page.getByLabel(/何段も先まで予測する/).check()
  await page.getByRole('button', { name: '推論する' }).click()
  await expect(page.getByText(/件のうち/).first()).toBeVisible({ timeout: 60_000 })
}

const churn = '見本: 解約の予測 (表の分類)'
const visits = '見本: 来店数の予測 (表の回帰)'
const sales = '見本: 日々の売上の予測 (時系列, 先の予測)'

const browser = await chromium.launch({ channel: process.env.CI ? undefined : 'chrome' })
const context = await browser.newContext({
  viewport: { width: 1600, height: 900 },
  extraHTTPHeaders: { 'x-provide-token': 'e2e-token' },
})
await context.addInitScript(hideHome, home)
const page = await context.newPage()

// 一覧: 見本の一式を入れる (その場で学習するので長めに待つ)
await page.goto(`${base}/`)
await page.getByRole('button', { name: '見本の一式を入れる' }).click()
await expect(page.getByRole('link', { name: churn })).toBeVisible({ timeout: 180_000 })
await expect(page.getByRole('link', { name: sales })).toBeVisible({ timeout: 180_000 })
await shot(page, 'bundles')

// 学習の結果から作る
await page.getByRole('button', { name: '学習の結果から作る' }).click()
await page.getByLabel('名前', { exact: true }).fill('解約の予測')
await shot(page, 'create')

// 一式の画面
await page.goto(`${base}/`)
await page.getByRole('link', { name: churn }).click()
await expect(page.getByText('一式での評価')).toBeVisible()
await page.getByRole('button', { name: /見本の新しいデータ/ }).click()
await shot(page, 'bundle')

// 意思決定の欄
await scrollTo(page.getByRole('region', { name: '意思決定' }))
await shot(page, 'decision-form')

// 結果: 結論と意思決定 (新しいデータ 50 行)
await page.getByRole('button', { name: '推論する' }).click()
await expect(page.getByText(/件のうち/).first()).toBeVisible({ timeout: 60_000 })
await shot(page, 'result')

// 予測の表
await scrollTo(page.getByRole('table', { name: '予測と判断の表' }))
await shot(page, 'predictions')

// 決め方を変えて出し直し, 保留で絞る
await page.getByLabel('陽性のクラス').selectOption('')
await page.getByLabel(/確信の下限/).fill('0.8')
await page.getByRole('button', { name: '出し直す' }).click()
await expect(page.getByRole('list', { name: '判断の内訳' })).toContainText('保留')
await page.getByLabel('判断で絞る').selectOption('保留')
await scrollTo(page.getByRole('heading', { name: '予測の表' }))
await shot(page, 'reconclude')

// 過去のデータ 300 行でたどり直す
await infer(page, churn, /見本の過去のデータ/)
await page.getByRole('button', { name: '過去のデータでたどり直す' }).click()
const replay = page.getByRole('region', { name: 'たどり直し' })
await expect(replay).toContainText('この方針', { timeout: 60_000 })
await scrollTo(page.getByRole('heading', { name: /^意思決定/ }))
await shot(page, 'decision-result')

// 量を決める: 日々の売上の先の 7 日
await infer(page, sales, /見本の新しいデータ/, true)
await scrollTo(page.getByRole('heading', { name: /^意思決定/ }))
await shot(page, 'decision-quantity')

// 回帰: 値の帯と店ごと
await infer(page, visits, /見本の新しいデータ/)
await expect(page.getByRole('heading', { name: 'shop ごと' })).toBeVisible()
await shot(page, 'regression')

// 列の効き
await page.getByRole('button', { name: '求める', exact: true }).click()
const effects = page.getByRole('list', { name: '列の効き' })
await expect(effects).toContainText('shop', { timeout: 60_000 })
await scrollTo(page.getByRole('heading', { name: /^列の効き/ }))
await shot(page, 'explain')

// 外れ方の確かめ: 来店数の過去のデータ
await infer(page, visits, /見本の過去のデータ/)
await page.getByRole('button', { name: '確かめる' }).click()
await expect(page.getByRole('list', { name: '外れ方の確かめの結果' })).toContainText('300 行', {
  timeout: 60_000,
})
await scrollTo(page.getByRole('heading', { name: /^外れ方の確かめ/ }))
await shot(page, 'remeasure')

// 方策の一式: 実験の一覧から登録して行動を決める
await page.goto(`${base}/policies`)
await page.getByRole('button', { name: '実験の一覧から登録' }).click()
await page
  .getByRole('dialog', { name: '実験の一覧から登録' })
  .getByRole('button', { name: /cartpole.* を登録/ })
  .click()
await page
  .getByLabel(/^観測 \(/)
  .fill('0.5478467492858172, -0.9208531449445188, -0.18361059042552214, -1.9338894578858836')
await page.getByRole('button', { name: '行動を決める' }).click()
await expect(page.getByRole('status').filter({ hasText: '行動' })).toContainText('そのまま')
await shot(page, 'policy')

await browser.close()
