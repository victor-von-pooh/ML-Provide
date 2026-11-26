import { fireEvent, render, screen, within } from '@testing-library/react'

import { ClusterRepresentativesTable } from './ClusterCards'

// 既製の埋め込みの群分けの一式 (版 10) の代表の行 (値は写真の場所と幅と高さ)
const groups = [
  {
    group: 0,
    rows: [
      { row: 1, distance: 0.1, values: { path: 'red/a.png', width: 40, height: 30 } },
      { row: 0, distance: 0.2, values: { path: 'red/b.png', width: 32, height: 24 } },
    ],
  },
  { group: 1, rows: [] },
]
const summary = [
  { group: 0, rows: 2, share: 1, stability: 0.9, width: 36, height: 27 },
  { group: 1, rows: 0, share: 0, stability: 0.8, width: null, height: null },
  { group: '全体', rows: 2, share: 1, stability: null, width: 36, height: 27 },
]

describe('群の代表の行の表', () => {
  it('群を作る列に無い写真の場所を, 群を作る列の前に出す', () => {
    render(
      <ClusterRepresentativesTable
        groups={groups}
        columns={['width', 'height']}
        summary={summary}
        caption="代表の行"
      />,
    )
    const table = screen.getByRole('table', { name: '代表の行' })
    const headings = within(table)
      .getAllByRole('columnheader')
      .map((cell) => cell.textContent)
    expect(headings).toEqual(['群', '順位', '#', '距離', 'path', 'width', 'height'])
    const rows = within(table).getAllByRole('row')
    expect(rows[1]).toHaveTextContent('red/a.png')
    // 行の無い群の 1 行は写真の場所の列の分も横に伸ばす
    expect(within(rows[3]!).getByText('割り当てた行はありません')).toHaveAttribute('colspan', '6')
  })

  it('写真を群に分けた結果は, 群の列の次に代表の写真の縮小画像を出し, 読めない写真は文で出す', () => {
    render(
      <ClusterRepresentativesTable
        groups={groups}
        columns={['width', 'height']}
        summary={summary}
        photoUrl={(path) => `/photo?path=${path}`}
        caption="代表の行"
      />,
    )
    const table = screen.getByRole('table', { name: '代表の行' })
    const headings = within(table)
      .getAllByRole('columnheader')
      .map((cell) => cell.textContent)
    expect(headings).toEqual(['群', '写真', '順位', '#', '距離', 'path', 'width', 'height'])
    const image = screen.getByRole('img', { name: '群 0 の 1 番目の代表の写真 red/a.png' })
    expect(image).toHaveAttribute('src', '/photo?path=red/a.png')
    expect(image).toHaveAttribute('loading', 'lazy')
    expect(screen.getAllByRole('img')).toHaveLength(2)
    // 読めない写真は画像の代わりに「読めません」
    fireEvent.error(image)
    expect(
      screen.getByRole('img', { name: '群 0 の 1 番目の代表の写真 red/a.png (読めません)' }),
    ).toHaveTextContent('読めません')
    // 行の無い群の 1 行は写真の列の分も横に伸ばす
    const rows = within(table).getAllByRole('row')
    expect(within(rows[3]!).getByText('割り当てた行はありません')).toHaveAttribute('colspan', '7')
  })
})
