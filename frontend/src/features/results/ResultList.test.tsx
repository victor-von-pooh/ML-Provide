import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'

import type { Result } from '../../api/types'
import { ResultList } from './ResultList'

// 2 つの結果 (時刻の無い結果は id を出す)
const results: Result[] = ['r1', 'r2'].map((id) => ({
  id,
  bundle: null,
  bundle_id: null,
  source: 'new.csv',
  created_at: null,
  task: 'classification',
  summary: null,
  sentences: [],
}))

describe('結果の一覧で比べる結果を選ぶ', () => {
  it('上限まで選ぶと, 選んでいない結果は選べない', () => {
    render(
      <MemoryRouter>
        <ResultList
          results={results}
          showBundle={false}
          selected={new Set(['r1'])}
          onToggle={() => {}}
          full
        />
      </MemoryRouter>,
    )
    expect(screen.getByLabelText('r1 の結果を比べる')).toBeEnabled()
    expect(screen.getByLabelText('r2 の結果を比べる')).toBeDisabled()
  })
})
