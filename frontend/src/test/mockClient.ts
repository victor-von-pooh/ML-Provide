// テストで api/client を差し替えるときの形 (画面の部品は useApi で呼び出しの一式を取る)

/**
 * 差し替えた呼び出しの一式を, `api` と `useApi` の両方で返すモジュールにする
 *
 * @param module - 差し替えたモジュールの中身 (`api` は呼び出しの一式)
 */
export function mockClient<T extends { api: object }>(module: T) {
  return { ...module, useApi: () => module.api }
}
