import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { BrowserRouter } from 'react-router'

import { App } from './App'
import { ApiContext, ApiError, createApi } from './api/client'
import { StageContext, type StageValue } from './lib/stage'
import { ThemeProvider } from './lib/theme'

/** 推論の段階の画面の根の部品の props (殻と単独で同じ形) */
export interface StageAppProps {
  /** 画面の経路の頭 (例 `/infer`). 単独では `""` */
  basename: string
  /** API の頭 (例 `/api/infer`). 単独では `/api` */
  apiBase: string
  /** 殻で選んだ問題の識別子. null は選んでいない (実験と一式の一覧を絞らない) */
  project: string | null
  /** 一式に置き場の別の問題の設定をつないだときに, その識別子で呼ぶ */
  onProjectChange: (project: string | null) => void
  /** 問題の設定の一覧が変わったときに呼ぶ (このアプリは一覧を変えないので呼ばない) */
  onProjectsChange: () => void
  /** 殻の下に載せたか (アプリの名前と明暗の切り替えを出さず, 明暗は読むだけにする) */
  embedded: boolean
}

/** 部品ごとの TanStack Query の client */
function makeQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        refetchOnWindowFocus: false,
        // 誤りの応答などの利用者側のエラーは繰り返さない
        retry: (count, error) => !(error instanceof ApiError && error.status < 500) && count < 1,
      },
    },
  })
}

/**
 * 推論の段階の画面の根の部品
 *
 * 自分のルーター (basename の下) と TanStack Query の client を持つ. 単独では main.tsx が
 * 既定の値 (basename `""`, apiBase `/api`, project null, embedded false) で描く
 */
export function StageApp({ basename, apiBase, project, onProjectChange, embedded }: StageAppProps) {
  const [queryClient] = useState(makeQueryClient)
  const api = useMemo(() => createApi(apiBase), [apiBase])
  const stage = useMemo<StageValue>(
    () => ({ project, onProjectChange, embedded }),
    [project, onProjectChange, embedded],
  )
  return (
    <ThemeProvider readOnly={embedded}>
      <QueryClientProvider client={queryClient}>
        <ApiContext.Provider value={api}>
          <StageContext.Provider value={stage}>
            <BrowserRouter basename={basename}>
              {/* 部品の範囲の印 (殻の画面で押したキーを拾わないため. 箱は作らない) */}
              <div data-stage-root className="contents">
                <App />
              </div>
            </BrowserRouter>
          </StageContext.Provider>
        </ApiContext.Provider>
      </QueryClientProvider>
    </ThemeProvider>
  )
}
