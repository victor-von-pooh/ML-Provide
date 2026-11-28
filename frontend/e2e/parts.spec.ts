import { expect, test } from '@playwright/test'

// 殻の共通の部品 (R12): 見出し, 欄の区域, 対話の枠の focus と閉じ方, 開けない物, 知らない経路
test('画面の頭の見出しは段階の名前で, 対話の枠は欄に focus を置いて閉じると戻る', async ({
  page,
}) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('推論')
  await expect(page.getByRole('region', { name: '推論の一式' })).toBeVisible()

  // 開くとパスの欄に focus があり, Esc で閉じると押したボタンに戻る
  const byPath = page.getByRole('button', { name: '場所で登録' })
  await byPath.click()
  const picker = page.getByRole('dialog', { name: '一式を場所で登録' })
  await expect(picker.getByLabel('パス', { exact: true })).toBeFocused()
  await page.keyboard.press('Escape')
  await expect(picker).toHaveCount(0)
  await expect(byPath).toBeFocused()

  // 一式を作る対話は名前の欄に focus を置き, 枠の外 (背景) を押しても閉じる
  const create = page.getByRole('button', { name: '学習の結果から作る' })
  await create.click()
  const dialog = page.getByRole('dialog', { name: '学習の結果から一式を作る' })
  await expect(dialog.getByLabel('名前', { exact: true })).toBeFocused()
  await page.mouse.click(5, 5)
  await expect(dialog).toHaveCount(0)
  await expect(create).toBeFocused()

  // 下の画面はパンくずの頭に段階の名前のリンクを置く
  await page.goto('/results')
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('結果')
  const crumbs = page.getByRole('navigation', { name: 'パンくず' })
  await crumbs.getByRole('link', { name: '推論' }).click()
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('推論')

  // 開けない一式は「一式を開けません」と, API の見つからない物の文
  await page.goto('/bundles/000000000000')
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('一式を開けません')
  await expect(page.getByText('一式 000000000000 が見つかりません')).toBeVisible()

  // 知らない経路は「ページが見つかりません」と段階の頭へのリンク
  await page.goto('/nowhere')
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('推論')
  await expect(page.getByRole('heading', { name: 'ページが見つかりません' })).toBeVisible()
  await page.getByRole('link', { name: '推論の一式の一覧へ戻る' }).click()
  await expect(page.getByRole('region', { name: '推論の一式' })).toBeVisible()
})
