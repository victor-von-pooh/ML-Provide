import { expect, test } from '@playwright/test'

test.describe.configure({ mode: 'serial' })

test('見本の一式を入れ, 見本の新しいデータを推論して結論を出し直す', async ({ page }) => {
  // 見本はその場で学習するので, 遅い CI でも足りるように長めに待つ
  test.setTimeout(150_000)
  await page.goto('/')
  await page.getByRole('button', { name: '見本の一式を入れる' }).click()
  const churn = page.getByRole('link', { name: '見本: 解約の予測 (表の分類)' })
  await expect(churn).toBeVisible({ timeout: 120_000 })

  // 一式の説明と, 見本の新しいデータでの推論
  await churn.click()
  await expect(page.getByText('一式での評価')).toBeVisible()
  await page.getByRole('button', { name: /見本の新しいデータ/ }).click()
  // 既定の決め方は陽性のクラス (解約) と確率 0.4
  await expect(page.getByLabel('陽性のクラス')).toHaveValue('解約')
  await page.getByRole('button', { name: '推論する' }).click()

  // 結論の文と内訳
  await expect(page.getByText('50 件のうち 50 件を予測')).toBeVisible({ timeout: 30_000 })
  await expect(page.getByRole('list', { name: '判断の内訳' })).toContainText('解約')

  // 確信の下限に変えて出し直すと, 保留が出る
  await page.getByLabel('陽性のクラス').selectOption('')
  await page.getByLabel(/確信の下限/).fill('0.99')
  await page.getByRole('button', { name: '出し直す' }).click()
  await expect(page.getByRole('list', { name: '判断の内訳' })).toContainText('保留')
  await expect(page.getByRole('heading', { name: '結論の決め方' })).toHaveCount(1)

  // 判断で絞った表
  await page.getByLabel('判断で絞る').selectOption('保留')
  await expect(page.getByRole('table', { name: '予測と判断の表' })).toContainText('保留')

  // 列の効き (前処理で落とす id は 0)
  await page.getByRole('button', { name: '求める' }).click()
  const effects = page.getByRole('list', { name: '列の効き' })
  await expect(effects).toContainText('months', { timeout: 30_000 })
  await expect(effects.getByRole('listitem').last()).toContainText('id')
})

test('文の見本に文を書いて推論する', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('link', { name: '見本: レビューの分類 (文)' }).click()
  await page.getByLabel('文を書く').check()
  await page.getByLabel('推論する文 (1 行に 1 つ)').fill('本体は最高でした。\n配送はひどい。')
  await page.getByRole('button', { name: '推論する' }).click()
  await expect(page.getByText('2 件のうち 2 件を予測')).toBeVisible({ timeout: 30_000 })
  const table = page.getByRole('table', { name: '予測と判断の表' })
  await expect(table).toContainText('本体は最高でした。')
  await expect(table.getByRole('row').nth(1)).toContainText('良い')
  await expect(table.getByRole('row').nth(2)).toContainText('悪い')
})

test('回帰の見本は値の帯と店ごとにまとめる', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('link', { name: '見本: 来店数の予測 (表の回帰)' }).click()
  await page.getByRole('button', { name: /見本の新しいデータ/ }).click()
  await page.getByRole('button', { name: '推論する' }).click()
  await expect(page.getByText(/予測の合計/)).toBeVisible({ timeout: 30_000 })
  await expect(page.getByRole('heading', { name: 'shop ごと' })).toBeVisible()
  await expect(page.getByRole('table', { name: /まとめる列の値ごと/ })).toContainText('本店')
})
