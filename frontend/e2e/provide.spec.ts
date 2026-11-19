import path from 'node:path'

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
  await expect(effects.getByRole('listitem').filter({ hasText: /^id/ })).toContainText('0.000')
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

test('時系列の見本で何段も先まで予測する', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('link', { name: '見本: 日々の売上の予測 (時系列, 先の予測)' }).click()
  await page.getByRole('button', { name: /見本の新しいデータ/ }).click()
  await page.getByLabel(/何段も先まで予測する/).check()
  await page.getByRole('button', { name: '推論する' }).click()
  await expect(page.getByText('14 件のうち 14 件を予測')).toBeVisible({ timeout: 30_000 })
  await expect(page.getByText(/何段も先の予測/)).toBeVisible()
  const table = page.getByRole('table', { name: '予測と判断の表' })
  await expect(table.getByRole('columnheader', { name: 'step' })).toBeVisible()
})

test('解約の見本で, 過去のデータを既定の意思決定で推論しかたどり直す', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('link', { name: '見本: 解約の予測 (表の分類)' }).click()
  // 既定の意思決定 (対象を選ぶ) が入っている
  await expect(page.getByLabel('型', { exact: true })).toHaveValue('select')
  await page.getByRole('button', { name: /見本の過去のデータ/ }).click()
  await page.getByRole('button', { name: '推論する' }).click()
  await expect(page.getByText(/300 件のうち 10 件/)).toBeVisible({ timeout: 30_000 })
  await expect(page.getByText(/効果が当たりの確率に比例することは仮定/)).toBeVisible()

  // たどり直すと, 方針ごとの実際の得が出る
  await page.getByRole('button', { name: '過去のデータでたどり直す' }).click()
  await expect(page.getByRole('region', { name: 'たどり直し' })).toContainText('この方針', {
    timeout: 30_000,
  })

  // 振り分けに決め直すと, たどり直しは消える
  await page.getByLabel('型', { exact: true }).selectOption('triage')
  await page.getByLabel('当たりのクラス (行動の対象)').selectOption('解約')
  await page.getByLabel('当たりを見逃す損').fill('9')
  await page.getByLabel('外れを回す損 (空振り)').fill('1')
  await page.getByRole('button', { name: '決め直す' }).click()
  await expect(page.getByText(/以上を回すこと/)).toBeVisible({ timeout: 30_000 })
  await expect(page.getByRole('region', { name: 'たどり直し' })).toHaveCount(0)
})

test('解約の見本の過去のデータで, 外れ方を確かめて一式に書き戻し, 元に戻す', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('link', { name: '見本: 解約の予測 (表の分類)' }).click()
  await expect(page.getByText(/検証用の分割 の [\d,]+ 行で測った/)).toBeVisible()
  await page.getByRole('button', { name: /見本の過去のデータ/ }).click()
  await page.getByRole('button', { name: '推論する' }).click()
  await expect(page.getByText(/300 件のうち 10 件/)).toBeVisible({ timeout: 30_000 })

  // 確かめるだけでは一式は変わらない
  const status = page.getByRole('status', { name: '外れ方の確かめの状態' })
  await page.getByRole('button', { name: '確かめる' }).click()
  const found = page.getByRole('list', { name: '外れ方の確かめの結果' })
  await expect(found).toContainText(
    /300 行で, 確率と実際に当たった割合のずれは平均 [\d.]+ ポイント/,
    {
      timeout: 30_000,
    },
  )
  await expect(status).toContainText('一式は変えていません')

  // 確認して書き戻すと, もう一度確かめた結果に替わる
  page.once('dialog', (dialog) => dialog.accept())
  await page.getByRole('button', { name: '一式に書き戻す' }).click()
  await expect(status).toContainText('一式に書き戻しました', { timeout: 30_000 })
  await page.getByRole('button', { name: '確かめる' }).click()
  await expect(status).toContainText('一式は変えていません', { timeout: 30_000 })
  await expect(found).toContainText('記録したときは')

  // 一式の画面で, 作ったときの記録に戻す
  await page.getByRole('link', { name: '見本: 解約の予測 (表の分類)' }).first().click()
  await expect(page.getByText(/測り直した実績 \(.+\) の 300 行で測った/)).toBeVisible()
  page.once('dialog', (dialog) => dialog.accept())
  await page.getByRole('button', { name: '外れ方を作ったときの記録に戻す' }).click()
  await expect(page.getByText(/検証用の分割 の [\d,]+ 行で測った/)).toBeVisible({ timeout: 30_000 })
  await expect(page.getByRole('button', { name: '外れ方を作ったときの記録に戻す' })).toHaveCount(0)
})

test('日々の売上の見本の過去のデータで, 7 段先までの外れ方を確かめる', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('link', { name: '見本: 日々の売上の予測 (時系列, 先の予測)' }).click()
  await page.getByRole('button', { name: /見本の過去のデータ/ }).click()
  await page.getByRole('button', { name: '推論する' }).click()
  await expect(page.getByRole('heading', { name: /外れ方の確かめ/ })).toBeVisible({
    timeout: 30_000,
  })

  // 段の数だけ書いて時刻の列を外すと, 確かめられない
  await page.getByLabel(/何段先まで/).fill('7')
  await expect(page.getByLabel('予測を始める時刻の列')).toHaveValue('date')
  await page.getByLabel('予測を始める時刻の列').selectOption('')
  await expect(page.getByText('何段も先の外れ方には, 時刻の列を選んでください')).toBeVisible()
  await page.getByLabel('予測を始める時刻の列').selectOption('date')
  await page.getByRole('button', { name: '確かめる' }).click()
  await expect(page.getByRole('list', { name: '外れ方の確かめの結果' })).toContainText(
    /何段も先の外れ方 \(予測を始める時刻 \d+ 個\): 80 % の幅は 1 段目/,
    { timeout: 60_000 },
  )

  // 時刻の列を変えると前の結果は消え, 書き戻せない. 日時に読めない列は誤り
  await page.getByLabel('予測を始める時刻の列').selectOption('shop')
  await expect(page.getByRole('list', { name: '外れ方の確かめの結果' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: '一式に書き戻す' })).toHaveCount(0)
  await page.getByRole('button', { name: '確かめる' }).click()
  await expect(page.getByText(/日時として読めません/)).toBeVisible({ timeout: 60_000 })
  await page.getByLabel('予測を始める時刻の列').selectOption('date')
  await page.getByRole('button', { name: '確かめる' }).click()
  await expect(page.getByRole('list', { name: '外れ方の確かめの結果' })).toContainText(
    '何段も先の外れ方',
    { timeout: 60_000 },
  )

  // 書き戻すと, 一式の画面に段ごとに測ったことが出る
  page.once('dialog', (dialog) => dialog.accept())
  await page.getByRole('button', { name: '一式に書き戻す' }).click()
  await expect(page.getByRole('status', { name: '外れ方の確かめの状態' })).toContainText(
    '一式に書き戻しました',
    { timeout: 60_000 },
  )
  await page
    .getByRole('link', { name: '見本: 日々の売上の予測 (時系列, 先の予測)' })
    .first()
    .click()
  await expect(page.getByText(/何段も先の外れ方は 7 段目まで段ごと/)).toBeVisible()
})

test('日々の売上の見本に問題の設定をつなぐと, 損得と根拠が既定になる', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('link', { name: '見本: 日々の売上の予測 (時系列, 先の予測)' }).click()
  // 一式の既定値 (欠品 3, 余り 1)
  const under = page.getByLabel('1 単位足りないときの損 (欠品の利益など)')
  await expect(under).toHaveValue('3')

  // つなぐと, 問題の設定の損得と根拠
  await page
    .getByLabel('問題の設定 (problem.yaml か, それを持つディレクトリ)')
    .fill(path.resolve(import.meta.dirname, '.work', 'problem'))
  await page.getByRole('button', { name: 'つなぐ', exact: true }).click()
  await expect(page.getByRole('heading', { name: '問題の設定: 総菜の作る量' })).toBeVisible()
  await expect(page.getByText('損得の根拠: 欠品は粗利 200 円, 廃棄は原価 150 円')).toBeVisible()
  await expect(under).toHaveValue('200')
  await expect(page.getByText(/問題の設定「総菜の作る量」の損得/)).toBeVisible()

  // その値で推論できる
  await page.getByRole('button', { name: /見本の過去のデータ/ }).click()
  await page.getByRole('button', { name: '推論する' }).click()
  await expect(page.getByText(/件のうち/).first()).toBeVisible({ timeout: 30_000 })

  // 外すと一式の既定値に戻る
  await page
    .getByRole('link', { name: '見本: 日々の売上の予測 (時系列, 先の予測)' })
    .first()
    .click()
  await page.getByRole('button', { name: 'つなぎを外す' }).click()
  await expect(page.getByRole('heading', { name: '問題の設定', exact: true })).toBeVisible()
  await expect(under).toHaveValue('3')
})

test('受け持ちが列の予測でない問題の設定は, つなぐとまだ扱わないと出し, 既定は一式のまま', async ({
  page,
}) => {
  await page.goto('/')
  await page.getByRole('link', { name: '見本: 日々の売上の予測 (時系列, 先の予測)' }).click()
  // 前のテストがつないだままでも, 外してから始める (テストの順に依らない)
  const unlink = page.getByRole('button', { name: 'つなぎを外す' })
  const field = page.getByLabel('問題の設定 (problem.yaml か, それを持つディレクトリ)')
  await expect(unlink.or(field)).toBeVisible()
  if (await unlink.isVisible()) await unlink.click()
  const under = page.getByLabel('1 単位足りないときの損 (欠品の利益など)')
  await expect(under).toHaveValue('3')

  await field.fill(path.resolve(import.meta.dirname, '.work', 'problem-generate'))
  await page.getByRole('button', { name: 'つなぐ', exact: true }).click()
  await expect(page.getByRole('heading', { name: '問題の設定: 総菜の作る量' })).toBeVisible()
  await expect(page.getByText('受け持ち: 作る, 正解の種類は画像')).toBeVisible()
  await expect(page.getByText(/推論の結論はまだ扱いません/).first()).toBeVisible()
  await expect(under).toHaveValue('3')

  await page.getByRole('button', { name: 'つなぎを外す' }).click()
  await expect(page.getByRole('heading', { name: '問題の設定', exact: true })).toBeVisible()
})

test('効果を確率の差で渡すと, 向きの欄が出て, 無作為の試験で確かめると書く', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('link', { name: '見本: 解約の予測 (表の分類)' }).click()
  await expect(page.getByLabel(/当たりをどうしたいか/)).toHaveCount(0)
  await page.getByLabel(/効果の渡し方/).selectOption('uplift')
  await page.getByLabel(/^効果 \(/).fill('-0.1')
  await page.getByLabel(/当たりをどうしたいか/).selectOption('reduce')
  await page.getByRole('button', { name: /見本の新しいデータ/ }).click()
  await page.getByRole('button', { name: '推論する' }).click()
  await expect(page.getByText(/しないときの当たりの確率の見積もり/)).toBeVisible({
    timeout: 30_000,
  })
  await expect(page.getByText(/行動を無作為に割り当てた試験で確かめる/)).toBeVisible()
})
