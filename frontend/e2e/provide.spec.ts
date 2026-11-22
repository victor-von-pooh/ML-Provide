import fs from 'node:fs'
import path from 'node:path'

import { expect, test, type APIRequestContext } from '@playwright/test'

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
  await page.getByRole('button', { name: '求める', exact: true }).click()
  const effects = page.getByRole('list', { name: '列の効き' })
  await expect(effects).toContainText('months', { timeout: 30_000 })
  await expect(effects.getByRole('listitem').filter({ hasText: /^id/ })).toContainText('0.000')

  // 表で選んだ 1 行の理由 (前処理で落とす id は効かない)
  await page.getByLabel('判断で絞る').selectOption('')
  await page.getByRole('button', { name: '0 行目の予測の理由', exact: true }).click()
  const reason = page.getByRole('list', { name: '列ごとの効き' })
  await expect(reason).toContainText('months', { timeout: 30_000 })
  await expect(reason.getByRole('listitem').filter({ hasText: /^id/ })).toContainText('0.000')
  await expect(page.getByLabel('行の番号')).toHaveValue('0')
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

test('図形の領域の見本で, 見本の新しい画像の領域を値の名前で見る', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('link', { name: '見本: 図形の領域 (画像の領域の分割)' }).click()
  await expect(page.getByText(/値ごとの名前 0: 背景, 1: 円, 2: 四角/)).toBeVisible()
  await expect(
    page.getByText(/この一式の出力は領域なので, 結論は推論した件数だけです/),
  ).toBeVisible()
  await expect(page.getByRole('button', { name: /見本の過去のデータ/ })).toHaveCount(0)
  await page
    .getByRole('button', { name: '見本の新しいデータ (画像を並べたディレクトリ) を使う' })
    .click()
  await page.getByRole('button', { name: '推論する' }).click()
  await expect(page.getByText('6 件のうち 6 件を推論')).toBeVisible({ timeout: 30_000 })
  await expect(page.getByRole('heading', { name: '1 件の出力 (領域)' })).toBeVisible()
  await expect(page.getByRole('img', { name: /^領域/ })).toBeVisible()
  await expect(page.getByRole('list', { name: '領域の番号ごとの画素' })).toContainText(
    /\((背景|円|四角)\)/,
  )
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

  // 分類の一式は外れ方を種類ごとに分けない (確率の当たり具合だけ)
  await expect(page.getByLabel(/外れ方を分ける列/)).toHaveCount(0)

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

test('日々の売上の見本に問題の設定をつなぐと, 損得, 根拠, 今のやり方が既定になる', async ({
  page,
}) => {
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
  // 今のやり方は列を足さずに, 問題の設定の決まり (7 期前) で比べる
  await expect(
    page.getByText('空なので, 問題の設定の今のやり方 (7 期前の値と同じ量) で比べます.'),
  ).toBeVisible()

  // その値で推論し, 今のやり方の列の無い過去のデータでたどり直せる
  await page.getByRole('button', { name: /見本の過去のデータ/ }).click()
  await page.getByRole('button', { name: '推論する' }).click()
  await expect(page.getByText(/件のうち/).first()).toBeVisible({ timeout: 30_000 })
  await page.getByRole('button', { name: '過去のデータでたどり直す' }).click()
  const replay = page.getByRole('region', { name: 'たどり直し' })
  await expect(replay).toContainText('今のやり方', { timeout: 30_000 })
  await expect(replay).toContainText('7 期前の値と同じ量')

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

test('版 3 の領域の一式は, 結論の決め方を出さずに件数と 1 件の領域を見せる', async ({ page }) => {
  const root = path.resolve(import.meta.dirname, '.work', 'v3')
  const bundle = { id: await registerFresh(page.request, path.join(root, 'mask')) }
  try {
    // 一式の画面: 受け持ちと出力, 正解の出どころの注意, 決め方と意思決定の欄は無い
    await page.goto(`/bundles/${bundle.id}`)
    await expect(page.getByText(/番号ごとの名前 0: 背景, 1: 物/)).toBeVisible()
    await expect(page.getByText('今の処理の出力 (今の判定)', { exact: false })).toBeVisible()
    await expect(page.getByRole('note', { name: '一式の注意' })).toContainText(
      'その処理を超えられません',
    )
    await expect(
      page.getByText(/この一式の出力は領域なので, 結論は推論した件数だけです/),
    ).toBeVisible()
    await expect(page.getByRole('region', { name: '結論の決め方' })).toHaveCount(0)
    await expect(page.getByRole('region', { name: '意思決定' })).toHaveCount(0)

    // 推論すると件数だけの結論と, 1 件の領域
    await page.getByLabel(/推論するデータ/).fill(path.join(root, 'new.csv'))
    await page.getByRole('button', { name: '推論する' }).click()
    await expect(page.getByText('6 件のうち 5 件を推論 (1 件は推論できず)')).toBeVisible({
      timeout: 30_000,
    })
    await expect(page.getByRole('note', { name: '一式の注意' })).toContainText('今の判定 の出力')
    await expect(page.getByRole('heading', { name: '結論の決め方' })).toHaveCount(0)
    await expect(page.getByRole('heading', { name: '意思決定' })).toHaveCount(0)
    await expect(page.getByRole('heading', { name: '1 件の出力 (領域)' })).toBeVisible()
    await expect(page.getByRole('img', { name: /^領域/ })).toBeVisible()
    await expect(page.getByRole('list', { name: '領域の番号ごとの画素' })).toContainText('(物)')
    await expect(page.getByLabel('行 (#)')).toHaveValue('0')

    // 表の「見る」で行を選ぶ (推論しなかった行には無い)
    const table = page.getByRole('table', { name: '予測と判断の表' })
    await expect(table.getByRole('row').nth(3).getByRole('button', { name: '見る' })).toHaveCount(0)
    await table.getByRole('row').nth(4).getByRole('button', { name: '見る' }).click()
    await expect(page.getByLabel('行 (#)')).toHaveValue('3')
    await expect(page.getByRole('link', { name: '.npy' })).toHaveAttribute(
      'href',
      /artifacts\/3\/file$/,
    )
  } finally {
    await page.request.delete(`/api/bundles/${bundle.id}`)
  }
})

test('版 3 のベクトルの一式は最初の値を見せ, 登録簿に無い手順は起動の設定を案内する', async ({
  page,
}) => {
  const root = path.resolve(import.meta.dirname, '.work', 'v3')
  const id = await registerFresh(page.request, path.join(root, 'vector'))
  const file = path.join(root, 'later', 'inference.yaml')
  const original = fs.readFileSync(file)
  let laterId = ''
  try {
    await page.goto(`/bundles/${id}`)
    await expect(page.getByText('まとめる', { exact: true })).toBeVisible()
    await page.getByLabel(/推論するデータ/).fill(path.join(root, 'new.csv'))
    await page.getByRole('button', { name: '推論する' }).click()
    await expect(page.getByRole('heading', { name: '1 件の出力 (ベクトル)' })).toBeVisible({
      timeout: 30_000,
    })
    await expect(page.getByText(/形 \(5\)/)).toBeVisible()
    await expect(page.getByLabel('最初の値')).toBeVisible()

    // 登録した後に手順が登録簿に無いものに変わった一式
    laterId = await registerFresh(page.request, path.join(root, 'later'))
    fs.copyFileSync(path.join(root, 'later-inference.yaml'), file)
    await page.goto(`/bundles/${laterId}`)
    await expect(page.getByText(/手順 my_sampler は登録簿にありません/)).toBeVisible()
    await expect(page.getByText(/ML_PROVIDE_PROCEDURES/)).toBeVisible()
  } finally {
    // 書き換えた inference.yaml と登録を元に戻す (やり直しても同じに動くように)
    fs.writeFileSync(file, original)
    await page.request.delete(`/api/bundles/${id}`)
    if (laterId) await page.request.delete(`/api/bundles/${laterId}`)
  }
})

test('起動の設定で読んだ自作の部品の一式を推論し, 登録されていない部品は起動の設定を案内する', async ({
  page,
}) => {
  const root = path.resolve(import.meta.dirname, '.work', 'v3')

  // 起動の設定 (ML_PROVIDE_UNITS) で読んだモジュールと部品
  const info = (await (await page.request.get('/api/info')).json()) as {
    unit_modules: { name: string; units: string[] }[]
  }
  expect(info.unit_modules).toEqual([
    { name: 'e2e_units', units: ['e2e.Twice'] },
    { name: 'e2e_ready', units: ['e2e.ReadyColors'] },
  ])

  const id = await registerFresh(page.request, path.join(root, 'unit'))
  const file = path.join(root, 'unit', 'model_spec.yaml')
  const original = fs.readFileSync(file)
  try {
    await page.goto(`/bundles/${id}`)
    await page.getByLabel(/推論するデータ/).fill(path.join(root, 'new.csv'))
    await page.getByRole('button', { name: '推論する' }).click()
    await expect(page.getByRole('heading', { name: '1 件の出力 (ベクトル)' })).toBeVisible({
      timeout: 30_000,
    })
    await expect(page.getByText(/形 \(4\)/)).toBeVisible()

    // 部品を登録されていない名前に書き換えると, 一式の画面に名前と起動の設定が出る
    fs.copyFileSync(path.join(root, 'unit-missing-spec.yaml'), file)
    await page.goto(`/bundles/${id}`)
    await expect(page.getByText(/e2e_missing\.Block/)).toBeVisible()
    await expect(page.getByText(/ML_PROVIDE_UNITS/)).toBeVisible()

    // 場所での登録も同じ案内で断る
    const response = await page.request.post('/api/bundles', {
      data: { path: path.join(root, 'unit'), name: 'もう一度' },
    })
    expect(response.status()).toBe(400)
    expect(((await response.json()) as { detail: string }).detail).toContain('--units')
  } finally {
    // 書き換えた定義と登録を元に戻す (やり直しても同じに動くように)
    fs.writeFileSync(file, original)
    await page.request.delete(`/api/bundles/${id}`)
  }
})

test('実験の一覧から一式を登録し, 写しが指す元の問題の設定をつなぐと, 元を直した損得が既定になる', async ({
  page,
}) => {
  const work = path.resolve(import.meta.dirname, '.work')
  const run = path.join(work, 'project', 'outputs', 'sales', '2026-01-02', '09-00-00')
  const original = path.join(work, 'pathfinder', 'problems', 'p1', 'problem.yaml')
  await page.goto('/')
  await page.getByRole('button', { name: '実験の一覧から登録' }).click()
  const dialog = page.getByRole('dialog', { name: '実験の一覧から登録' })
  const table = dialog.getByRole('table', { name: '推論の一式を持つ実験' })
  await expect(table).toContainText('sales/2026-01-02/09-00-00')
  await expect(table).toContainText(path.join(run, 'inference'))
  await dialog
    .getByRole('button', { name: '総菜の作る量 (sales/2026-01-02/09-00-00) を登録' })
    .click()

  // 一式の画面に移り, 候補は実験の写しでなく ML-Pathfinder の置き場の元
  await expect(
    page.getByRole('heading', { name: '総菜の作る量 (sales/2026-01-02/09-00-00)' }),
  ).toBeVisible({ timeout: 30_000 })
  await expect(page.locator('#problem-hint')).toHaveText(original)
  await expect(page.getByText(/が指す元の問題の設定です/)).toBeVisible()
  await page.getByRole('button', { name: 'これをつなぐ' }).click()
  await expect(page.getByRole('heading', { name: '問題の設定: 総菜の作る量' })).toBeVisible()
  const under = page.getByLabel('1 単位足りないときの損 (欠品の利益など)')
  await expect(under).toHaveValue('200')

  // 元を直すと (ML-Pathfinder で損得を変えたように), 開き直したときの既定に入る
  const before = fs.readFileSync(original, 'utf-8')
  try {
    fs.writeFileSync(original, before.replace('under_cost: 200', 'under_cost: 260'))
    await page.reload()
    await expect(under).toHaveValue('260')
  } finally {
    fs.writeFileSync(original, before)
  }

  // 一覧では登録済みと出る
  await page.goto('/')
  await page.getByRole('button', { name: '実験の一覧から登録' }).click()
  await expect(
    page
      .getByRole('dialog', { name: '実験の一覧から登録' })
      .getByRole('link', { name: '登録済み' }),
  ).toBeVisible()
})

test('領域の値が飛ぶ (0, 3, 7) データセットの一式で, 凡例の名前が値と合う', async ({ page }) => {
  const root = path.resolve(import.meta.dirname, '.work', 'items')
  await page.goto('/')
  await page.getByRole('button', { name: '学習の結果から作る' }).click()
  const dialog = page.getByRole('dialog', { name: '学習の結果から一式を作る' })
  await dialog.getByLabel('名前', { exact: true }).fill('値の飛ぶ領域')
  await dialog.getByLabel('モデルの定義 (YAML か JSON)').fill(path.join(root, 'mask_spec.yaml'))
  await dialog.getByLabel(/^実験のディレクトリ/).fill(path.join(root, 'runs', 'mask'))
  await dialog.getByLabel(/^データセット/).fill(path.join(root, 'masks_gap'))
  const summary = dialog.getByRole('note', { name: '版 2 のデータセット' })
  await expect(summary).toContainText('3: circle')
  await expect(summary).toContainText('7: square')
  await dialog.getByRole('button', { name: '作る' }).click()

  await expect(page.getByRole('heading', { name: '値の飛ぶ領域' })).toBeVisible({ timeout: 30_000 })
  await expect(page.getByText(/値ごとの名前 0: background, 3: circle, 7: square/)).toBeVisible()
  const id = new URL(page.url()).pathname.split('/').pop() ?? ''
  try {
    await page
      .getByLabel(/推論するデータ \(画像のファイル/)
      .fill(path.join(root, 'raw_gap', 'images'))
    await page.getByRole('button', { name: '推論する' }).click()
    await expect(page.getByText('8 件のうち 8 件を推論')).toBeVisible({ timeout: 30_000 })
    // 凡例の値は 0, 3, 7 のどれかで, 名前はその値のもの (番号 1, 2 の名前でない)
    const legend = page.getByRole('list', { name: '領域の番号ごとの画素' }).getByRole('listitem')
    await expect(legend.first()).toBeVisible()
    for (const text of await legend.allTextContents())
      expect(text).toMatch(/^(0 \(background\)|3 \(circle\)|7 \(square\)): /)
  } finally {
    await page.request.delete(`/api/bundles/${id}`)
  }
})

test('版 2 の画像と領域のデータセットと実験から一式を作り, 新しい画像の領域を値の名前で見る', async ({
  page,
}) => {
  const root = path.resolve(import.meta.dirname, '.work', 'items')
  await page.goto('/')
  await page.getByRole('button', { name: '学習の結果から作る' }).click()
  const dialog = page.getByRole('dialog', { name: '学習の結果から一式を作る' })
  await dialog.getByLabel('名前', { exact: true }).fill('版 2 の領域')
  await dialog.getByLabel('モデルの定義 (YAML か JSON)').fill(path.join(root, 'mask_spec.yaml'))
  await dialog.getByLabel(/^データセット/).fill(path.join(root, 'masks'))
  // 版 2 のデータセットは項目と値の名前を見せ, 列のデータセットの欄は出さない
  const summary = dialog.getByRole('note', { name: '版 2 のデータセット' })
  await expect(summary).toContainText('image (画像, 入力)')
  await expect(summary).toContainText('1: circle')
  await expect(dialog.getByLabel('入力から選ぶ点の数 (要る)')).toHaveCount(0)
  await dialog.getByText('出力が複数のモデル (あれば)').click()
  await expect(dialog.getByLabel('受け持ち')).toHaveCount(0)
  await dialog.getByLabel(/^実験のディレクトリ/).fill(path.join(root, 'runs', 'mask'))
  await dialog.getByRole('button', { name: '作る' }).click()

  // 一式の画面: 版 4, 1 件の扱い, 値ごとの名前, IoU
  await expect(page.getByRole('heading', { name: '版 2 の領域' })).toBeVisible({ timeout: 30_000 })
  await expect(page.getByText('画素を 0 から 1 にする')).toBeVisible()
  await expect(page.getByText(/値ごとの名前 0: background, 1: circle, 2: square/)).toBeVisible()
  await expect(page.getByText(/IoU \(背景を除くクラスの平均\)/)).toBeVisible()
  const id = new URL(page.url()).pathname.split('/').pop() ?? ''
  try {
    // 新しい画像のディレクトリを推論すると, 件数と 1 件の領域 (凡例は値の名前)
    await page
      .getByLabel(/推論するデータ \(画像のファイル/)
      .fill(path.join(root, 'raw_m', 'images'))
    await page.getByRole('button', { name: '推論する' }).click()
    await expect(page.getByText('8 件のうち 8 件を推論')).toBeVisible({ timeout: 30_000 })
    await expect(page.getByRole('heading', { name: '1 件の出力 (領域)' })).toBeVisible()
    await expect(page.getByRole('img', { name: /^領域/ })).toBeVisible()
    await expect(page.getByRole('list', { name: '領域の番号ごとの画素' })).toContainText(
      /\((background|circle|square)\)/,
    )
  } finally {
    await page.request.delete(`/api/bundles/${id}`)
  }
})

test('版 2 の撮影と点群のデータセットは選ぶ点の数を求め, 新しい撮影の点群を元の座標で見る', async ({
  page,
}) => {
  const root = path.resolve(import.meta.dirname, '.work', 'items')
  await page.goto('/')
  await page.getByRole('button', { name: '学習の結果から作る' }).click()
  const dialog = page.getByRole('dialog', { name: '学習の結果から一式を作る' })
  await dialog.getByLabel('名前', { exact: true }).fill('版 2 の点群')
  await dialog.getByLabel('モデルの定義 (YAML か JSON)').fill(path.join(root, 'cloud_spec.yaml'))
  await dialog.getByLabel(/^データセット/).fill(path.join(root, 'clouds'))
  await dialog.getByLabel(/^実験のディレクトリ/).fill(path.join(root, 'runs', 'cloud'))
  await expect(dialog.getByRole('note', { name: '版 2 のデータセット' })).toContainText(
    'capture (撮影, 入力)',
  )

  // 選ぶ点の数が無いと作れない
  await expect(dialog.getByText(/選ぶ点の数 \(学習と同じ数\)/)).toBeVisible()
  await expect(dialog.getByRole('button', { name: '作る' })).toBeDisabled()
  await dialog.getByLabel('入力から選ぶ点の数 (要る)').fill('512')
  await dialog.getByLabel(/確かめで正解から選ぶ点の数/).fill('64')
  await dialog.getByRole('button', { name: '作る' }).click()

  // 一式の画面: 1 件の扱いと元の座標, Chamfer 距離
  await expect(page.getByRole('heading', { name: '版 2 の点群' })).toBeVisible({ timeout: 30_000 })
  await expect(page.getByText(/深度から点に戻して決まった数を選ぶ/)).toBeVisible()
  await expect(page.getByText(/points: 512, target_points: 64/)).toBeVisible()
  await expect(page.getByText(/^元の座標/)).toBeVisible()
  await expect(page.getByText(/Chamfer 距離/)).toBeVisible()
  const id = new URL(page.url()).pathname.split('/').pop() ?? ''
  try {
    // 新しい撮影のディレクトリを推論すると, 1 件の点群の散布図 (元の座標)
    await page
      .getByLabel(/推論するデータ \(撮影のディレクトリ/)
      .fill(path.join(root, 'raw_c', 'captures'))
    await page.getByRole('button', { name: '推論する' }).click()
    await expect(page.getByText('6 件のうち 6 件を推論')).toBeVisible({ timeout: 30_000 })
    await expect(page.getByRole('heading', { name: '1 件の出力 (点群)' })).toBeVisible()
    await expect(page.getByRole('img', { name: /点群の散布図 \(256 点\)/ })).toBeVisible()
    await expect(page.getByText(/点は元の座標/)).toBeVisible()
  } finally {
    await page.request.delete(`/api/bundles/${id}`)
  }
})
test('版 3 のデータセットは中身の手順を見せ, 大きさの違う画像も手順をかけて推論する', async ({
  page,
}) => {
  const root = path.resolve(import.meta.dirname, '.work', 'items')
  await page.goto('/')
  await page.getByRole('button', { name: '学習の結果から作る' }).click()
  const dialog = page.getByRole('dialog', { name: '学習の結果から一式を作る' })
  await dialog.getByLabel('名前', { exact: true }).fill('版 3 の領域')
  await dialog.getByLabel('モデルの定義 (YAML か JSON)').fill(path.join(root, 'mask_spec.yaml'))
  await dialog.getByLabel(/^データセット/).fill(path.join(root, 'masks_v3'))
  // 版 2 と同じ要約に, 書き出しでかけた中身の手順
  const summary = dialog.getByRole('note', { name: '版 3 のデータセット' })
  await expect(summary).toContainText('image (画像, 入力)')
  await expect(summary).toContainText('中身の手順 (image): resize (width 16, height 16)')
  await expect(summary).toContainText('中身の手順 (mask): resize_mask')
  await dialog.getByLabel(/^実験のディレクトリ/).fill(path.join(root, 'runs', 'mask'))
  await dialog.getByRole('button', { name: '作る' }).click()

  await expect(page.getByRole('heading', { name: '版 3 の領域' })).toBeVisible({ timeout: 30_000 })
  const id = new URL(page.url()).pathname.split('/').pop() ?? ''
  try {
    // 24 x 24 の元の画像は, 一式が手順をかけて推論する
    await page
      .getByLabel(/推論するデータ \(画像のファイル/)
      .fill(path.join(root, 'raw_m3', 'images'))
    await page.getByRole('button', { name: '推論する' }).click()
    await expect(page.getByText('8 件のうち 8 件を推論')).toBeVisible({ timeout: 30_000 })
    await expect(page.getByRole('img', { name: /^領域/ })).toBeVisible()
  } finally {
    await page.request.delete(`/api/bundles/${id}`)
  }
})

test('拡散のモデルの実験から, データセットを使わずに雑音だけの一式を画面で作り, 件数と種で生成する', async ({
  page,
}) => {
  const root = path.resolve(import.meta.dirname, '.work', 'noise')
  await page.goto('/')
  await page.getByRole('button', { name: '学習の結果から作る' }).click()
  const dialog = page.getByRole('dialog', { name: '学習の結果から一式を作る' })
  await dialog.getByLabel('名前', { exact: true }).fill('拡散の生成')
  await dialog.getByLabel('モデルの定義 (YAML か JSON)').fill(path.join(root, 'denoiser.yaml'))
  // データセットを使わないと, データセットの欄が消え, 手順, 出力の形, 入力の欄が開く
  // 選んでからやめると, 自動で足した入力の行を残さない
  const noData = dialog.getByLabel('雑音だけから作る (データセットを使わない)')
  await noData.check()
  await expect(dialog.getByLabel('入力の名前')).toHaveCount(2)
  await noData.uncheck()
  await expect(dialog.getByLabel('入力の名前')).toHaveCount(0)
  await noData.check()
  await expect(dialog.getByLabel(/^データセット \(/)).toHaveCount(0)
  await dialog.getByLabel(/^実験のディレクトリ/).fill(path.join(root, 'runs', 'diffusion'))
  await expect(dialog.getByRole('button', { name: '作る' })).toBeDisabled()
  const procedure = dialog.getByLabel(/^手順 \(登録簿の名前/)
  // 雑音から始まる手順だけを選べる
  await expect(procedure.locator('option', { hasText: 'greedy' })).toHaveCount(0)
  await procedure.selectOption('ddim')
  await dialog.getByLabel(/^回数/).fill('20')
  await dialog.getByLabel('eta', { exact: true }).fill('0')
  await dialog.getByLabel('出力の形 (要る)').selectOption('vector')
  await dialog.getByLabel(/^1 件の形/).fill('16')
  // 入力の名前が空なら送らずに知らせ, 手順の設定は既定値を見せる
  await expect(
    dialog.getByText(/^「モデルの入力ごとの出どころ」: モデルの入力の名前 \(モデルの定義/),
  ).toBeVisible()
  await expect(dialog.getByLabel('train_steps', { exact: true })).toHaveAttribute(
    'placeholder',
    '空なら既定 1000',
  )
  await dialog.getByLabel('入力の名前').nth(0).fill('x')
  await dialog.getByLabel('入力の名前').nth(1).fill('t')
  await expect(dialog.getByLabel('出どころ').nth(1)).toHaveValue('time')
  await dialog.getByRole('button', { name: '作る' }).click()

  // 一式の画面: 雑音だけ, 手順, 推論は件数と種
  await expect(page.getByRole('heading', { name: '拡散の生成' })).toBeVisible({ timeout: 30_000 })
  await expect(page.getByText(/雑音だけ \(データセットなし/)).toBeVisible()
  await expect(page.getByText(/t ← 手順の時刻, x ← 手順の中身/)).toBeVisible()
  // 手順の設定は読める文で, 雑音だけの一式には問題の設定の欄を出さない
  await expect(page.getByText('(設定: eta 0)')).toBeVisible()
  await expect(page.getByRole('heading', { name: /^問題の設定/ })).toHaveCount(0)
  const id = new URL(page.url()).pathname.split('/').pop() ?? ''
  try {
    await expect(page.getByLabel(/^推論するデータ/)).toHaveCount(0)
    await page.getByLabel(/^作る件数/).fill('0')
    await expect(page.getByRole('button', { name: '推論する' })).toBeDisabled()
    await page.getByLabel(/^作る件数/).fill('5')
    await page.getByLabel(/^乱数の種/).fill('7')
    await page.getByRole('button', { name: '推論する' }).click()
    await expect(page.getByText('5 件のうち 5 件を推論')).toBeVisible({ timeout: 30_000 })
    await expect(page.getByText('件数 5, 種 7')).toBeVisible()
    await expect(page.getByRole('heading', { name: '1 件の出力 (ベクトル)' })).toBeVisible()
    await expect(page.getByLabel('最初の値')).toBeVisible()
  } finally {
    await page.request.delete(`/api/bundles/${id}`)
  }
})

test('雑音だけをやめても手順の中身の入力が残れば, 分かる言葉で止める', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: '学習の結果から作る' }).click()
  const dialog = page.getByRole('dialog', { name: '学習の結果から一式を作る' })
  const noData = dialog.getByLabel('雑音だけから作る (データセットを使わない)')
  await noData.check()
  // 雑音だけの行には先頭の番号の欄が無い
  await expect(dialog.getByLabel(/^先頭に足した番号/)).toHaveCount(0)
  await dialog.getByLabel('入力の名前').nth(0).fill('x')
  await noData.uncheck()
  // 書いた行は残り, 手順の無い一式では手順の中身を使えないと出る
  await expect(dialog.getByLabel('入力の名前')).toHaveCount(1)
  await expect(dialog.getByText(/繰り返しの手順を選んだ一式だけで使えます/)).toBeVisible()
  await expect(dialog.getByRole('button', { name: '作る' })).toBeDisabled()

  // 畳んだ欄の誤りは欄の名前を添え, 誤りが出ると欄を開く
  await dialog.getByLabel('入力の名前').nth(0).fill('toString')
  await dialog.getByRole('button', { name: '入力を足す' }).click()
  await dialog.getByLabel('入力の名前').nth(1).fill('toString')
  await expect(
    dialog.getByText('「モデルの入力ごとの出どころ」: 入力の名前 toString が重なっています'),
  ).toBeVisible()
})

test('骨組みの学習が実験の中に作った版 4 の一式を, 実験の一覧から登録する', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: '実験の一覧から登録' }).click()
  const dialog = page.getByRole('dialog', { name: '実験の一覧から登録' })
  const table = dialog.getByRole('table', { name: '推論の一式を持つ実験' })
  await expect(table).toContainText('masks/2026-01-03/09-00-00')
  await dialog
    .getByRole('button', { name: '総菜の作る量 (masks/2026-01-03/09-00-00) を登録' })
    .click()
  await expect(
    page.getByRole('heading', { name: '総菜の作る量 (masks/2026-01-03/09-00-00)' }),
  ).toBeVisible({ timeout: 30_000 })
  await expect(page.getByText(/値ごとの名前 0: background/)).toBeVisible()
  await expect(page.getByText('画素を 0 から 1 にする')).toBeVisible()
})

test('骨組みの学習が実験の中に書き出した方策の一式を, 実験の一覧から登録して行動を決める', async ({
  page,
}) => {
  const run = path.resolve(
    import.meta.dirname,
    '.work',
    'policy-project',
    'outputs',
    'cartpole',
    '2026-01-04',
    '09-00-00',
  )
  await page.goto('/policies')
  await page.getByRole('button', { name: '実験の一覧から登録' }).click()
  const dialog = page.getByRole('dialog', { name: '実験の一覧から登録' })
  const table = dialog.getByRole('table', { name: '方策の一式を持つ実験' })
  await expect(table).toContainText('cartpole/2026-01-04/09-00-00')
  await expect(table).toContainText(path.join(run, 'policy'))
  // 推論の一式の実験は並べない
  await expect(table).not.toContainText('sales/2026-01-02/09-00-00')
  await dialog
    .getByRole('button', { name: '棒を立てる制御 (cartpole/2026-01-04/09-00-00) を登録' })
    .click()

  // 方策の画面に移り, 一式に写した定義から組み立てた方策で行動を決める
  await expect(
    page.getByRole('heading', { name: '棒を立てる制御 (cartpole/2026-01-04/09-00-00)' }),
  ).toBeVisible({ timeout: 30_000 })
  await page
    .getByLabel(/^観測 \(/)
    .fill('0.5478467492858172, -0.9208531449445188, -0.18361059042552214, -1.9338894578858836')
  await page.getByRole('button', { name: '行動を決める' }).click()
  const result = page.getByRole('status').filter({ hasText: '行動' })
  await expect(result).toContainText('1')
  await expect(result).toContainText('そのまま')

  // 観測の CSV でまとめて決め, 結果の一覧から開き直す (範囲の外の 1 件は元の制御に戻す)
  const observations = path.resolve(import.meta.dirname, '.work', 'observations.csv')
  fs.writeFileSync(
    observations,
    'x,v,theta,omega\n0.5478467492858172,-0.9208531449445188,-0.18361059042552214,-1.9338894578858836\n2.5,0,0,0\n',
  )
  await page.getByLabel(/^観測のファイル/).fill(observations)
  await page.getByRole('button', { name: 'まとめて決める' }).click()
  const bars = page.getByRole('list', { name: '守りの結果ごとの件数' })
  await expect(bars).toContainText('そのまま1 件', { timeout: 30_000 })
  await expect(bars).toContainText('元の制御に戻した1 件')
  await page.goto('/results')
  await page
    .locator('section', { has: page.getByRole('heading', { name: '方策の結果' }) })
    .getByRole('link')
    .first()
    .click()
  await expect(page.getByRole('table', { name: '観測ごとの行動と守りの結果の表' })).toContainText(
    '元の制御に戻した',
  )

  // 一覧では登録済みと出る
  await page.goto('/policies')
  await page.getByRole('button', { name: '実験の一覧から登録' }).click()
  await expect(
    page
      .getByRole('dialog', { name: '実験の一覧から登録' })
      .getByRole('link', { name: '登録済み' }),
  ).toBeVisible()
})

/** 同じ場所の登録を外してから, 場所で一式を登録して id を返す */
async function registerFresh(request: APIRequestContext, place: string): Promise<string> {
  const listed = (await (await request.get('/api/bundles')).json()) as {
    id: string
    path: string
  }[]
  for (const bundle of listed.filter((item) => item.path === place)) {
    await request.delete(`/api/bundles/${bundle.id}`)
  }
  const response = await request.post('/api/bundles', { data: { path: place } })
  expect(response.status()).toBe(201)
  return ((await response.json()) as { id: string }).id
}

test('既製のモデルの一式を下見して作り, ライセンスの警告と重みの置き場を見て, 画像の種類を推論する', async ({
  page,
}) => {
  const images = path.resolve(import.meta.dirname, '.work', 'ready-images')
  await page.goto('/')
  await page.getByRole('button', { name: '既製のモデルから作る' }).click()
  const dialog = page.getByRole('dialog', { name: '既製のモデルから作る' })
  // 起動の設定で足した作り物の出どころ (ネットワークを使わない)
  await dialog.getByLabel('出どころ').selectOption('e2e')
  await dialog.getByLabel('モデルの名前').fill('colors')
  await dialog.getByLabel('重みの名前 (版)').fill('V9')
  await expect(dialog.getByText(/colors の重みに V9 がありません/)).toBeVisible()
  await expect(dialog.getByRole('button', { name: '作って登録' })).toBeDisabled()
  await dialog.getByLabel('重みの名前 (版)').fill('V1')
  const preview = dialog.getByRole('region', { name: '下見' })
  await expect(preview).toContainText('3 種類')
  await expect(preview).toContainText('赤いもの, 緑のもの, 青いもの')
  await expect(preview).toContainText('真ん中の 16 × 16 を切り出し')
  await expect(preview.getByRole('alert')).toContainText('ライセンスの情報を読めませんでした')
  await dialog.getByLabel('一式の名前').fill('色の分類')
  await dialog.getByRole('button', { name: '作って登録' }).click()

  // 一式の画面: ライセンスの警告, 出どころ, 重みの置き場. 意思決定の欄は出さず,
  // 問題の設定は既製のモデルの問題の設定をつなげると書く
  await expect(page.getByRole('heading', { name: '色の分類' })).toBeVisible({ timeout: 30_000 })
  const id = new URL(page.url()).pathname.split('/').pop() ?? ''
  try {
    await expect(page.getByRole('note', { name: 'ライセンスの警告' })).toBeVisible()
    await expect(page.getByText(/e2e の colors \(重み V1\)/)).toBeVisible()
    await expect(page.getByText(/e2e-colors-0123abcd\.pth/)).toBeVisible()
    await expect(page.getByText(/置き場にありません/)).toBeVisible()
    await expect(page.getByRole('region', { name: '意思決定' })).toHaveCount(0)
    await expect(page.getByText(/作り方が既製のモデルをそのままの問題の設定/)).toBeVisible()

    // 画像のディレクトリを推論し, 種類の名前と確信度を見る (読めない画像は飛ばす)
    await page.getByLabel(/推論する画像/).fill(images)
    await page.getByRole('button', { name: '推論する' }).click()
    await expect(page.getByText('4 件のうち 3 件を予測')).toBeVisible({ timeout: 30_000 })
    await expect(page.getByText(/意思決定, 過去のデータでのたどり直し/)).toBeVisible()
    await expect(page.getByRole('heading', { name: /^意思決定/ })).toHaveCount(0)
    const table = page.getByRole('table', { name: '予測と判断の表' })
    await expect(table.getByRole('row').filter({ hasText: 'a_red.png' })).toContainText('赤いもの')
    await expect(table.getByRole('row').filter({ hasText: 'c_blue.png' })).toContainText('青いもの')
    await expect(table.getByRole('columnheader', { name: 'confidence' })).toBeVisible()
  } finally {
    await page.request.delete(`/api/bundles/${id}`)
  }
})
