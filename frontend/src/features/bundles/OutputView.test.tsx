import { render, screen } from '@testing-library/react'

import type { ArtifactPreview, Description } from '../../api/types'
import { ArtifactBody } from '../results/ArtifactCard'
import { OutputNotices, OutputView } from './OutputView'

const base: Description = {
  path: '/b',
  name: 'mask',
  created_at: null,
  kind: 'table',
  task: 'none',
  target: null,
  classes: [],
  features: ['a'],
  text_column: null,
  reading: { loader: null, options: {} },
  steps: [],
  item_steps: [],
  model: { name: null, inputs: {}, output: null, kind: 'logits' },
  run: null,
  conclusion: {},
  check: null,
  notes: [],
}

describe('列の出力でない一式の画面', () => {
  it('受け持ち, 出力の形, 正解, 手順, 学習済みの重みを出す', () => {
    render(
      <OutputView
        description={{
          ...base,
          format_version: 3,
          role: 'generate',
          output: { form: 'vector', shape: [16] },
          column_output: false,
          target_info: { kind: 'mask', origin: { type: 'process', name: '今の判定' } },
          procedure: { name: 'ddim', steps: 50, seed: 0, options: { eta: 0 } },
          pretrained: [
            {
              node: 'backbone',
              hub_id: 'org/model',
              pinned_revision: 'abc',
              license: 'other',
              redistributable: false,
              place: 'source',
            },
          ],
        }}
        modelLoaded={false}
      />,
    )
    expect(screen.getByText('作る')).toBeInTheDocument()
    expect(screen.getByText(/ベクトル/)).toBeInTheDocument()
    expect(screen.getByText(/結論は件数だけです/)).toBeInTheDocument()
    expect(screen.getByText(/今の処理の出力/)).toBeInTheDocument()
    expect(screen.getByText('ddim')).toBeInTheDocument()
    expect(screen.getByText(/50 回, 乱数の種/)).toBeInTheDocument()
    expect(screen.getByRole('list', { name: '学習済みの重みの置き方' })).toHaveTextContent(
      'org/model @ abc',
    )
    expect(screen.getByText(/再配布できない/)).toBeInTheDocument()
    expect(screen.getByText(/推論のときに読み/)).toBeInTheDocument()
  })

  it('表の出力の一式は列の名前と型, 種類の数を出し, 作った表を CSV で持ち出せると書く', () => {
    render(
      <OutputView
        description={{
          ...base,
          kind: 'noise',
          format_version: 7,
          role: 'generate',
          output: {
            form: 'table',
            shape: [5],
            columns: [
              { name: 'price', type: 'number', mean: 100, std: 20, integer: false },
              { name: 'count', type: 'number', mean: 3, std: 1.5, integer: true },
              { name: 'color', type: 'category', categories: ['blue', 'green', 'red'] },
            ],
          },
          column_output: false,
        }}
        modelLoaded={false}
      />,
    )
    expect(screen.getByText('(列 3, 符号化した行の幅 5)')).toBeInTheDocument()
    expect(screen.getByText(/CSV で持ち出せます/)).toBeInTheDocument()
    const list = screen.getByRole('list', { name: '表の列の名前と型' })
    expect(list).toHaveTextContent('price: 数 (平均 100, 標準偏差 20)')
    expect(list).toHaveTextContent('count: 整数 (平均 3, 標準偏差 1.5)')
    expect(list).toHaveTextContent('color: 種類 (3 種類: blue, green, red)')
  })

  it('版 2 の一式は予測するとクラスで, 件数だけとは書かない', () => {
    render(
      <OutputView
        description={{ ...base, format_version: 2, role: 'predict', output: { form: 'class' } }}
        modelLoaded={false}
      />,
    )
    expect(screen.getByText('予測する')).toBeInTheDocument()
    expect(screen.queryByText(/結論は件数だけです/)).not.toBeInTheDocument()
    expect(screen.getAllByText('なし')).toHaveLength(2)
  })

  it('正解の出どころの注意と重みの警告を必ず出す', () => {
    const { container } = render(
      <OutputNotices
        target={{ kind: 'mask', origin: { type: 'process', name: '今の判定' } }}
        warnings={['backbone の重みが記録と違います']}
      />,
    )
    expect(screen.getByRole('note')).toHaveTextContent('その処理を超えられません')
    expect(screen.getByRole('note')).toHaveTextContent('学習済みの重みの警告: backbone')
    container.remove()
    const { container: empty } = render(<OutputNotices target={null} warnings={[]} />)
    expect(empty).toBeEmptyDOMElement()
  })

  it('配列の形ごとに見せる', () => {
    const preview: ArtifactPreview = {
      row: 0,
      file: '000000.npy',
      form: 'mask',
      shape: [2, 2],
      dtype: 'int64',
      view: 'grid',
      values: [0, 1, 1, 0],
      size: 4,
      norm: 1.41,
      grid: [
        [0, 1],
        [1, 0],
      ],
      labels: [
        { label: 0, count: 2 },
        { label: 1, count: 2 },
      ],
    }
    const { unmount } = render(<ArtifactBody preview={preview} names={['背景', '物']} />)
    expect(screen.getByRole('img', { name: /^領域/ })).toBeInTheDocument()
    expect(screen.getByRole('list', { name: '領域の番号ごとの画素' })).toHaveTextContent(
      '1 (物): 2',
    )
    unmount()
    render(
      <ArtifactBody
        preview={{ ...preview, form: 'vector', shape: [4], view: 'values', grid: undefined }}
        names={[]}
      />,
    )
    expect(screen.getByLabelText('最初の値')).toHaveTextContent('0, 1, 1, 0')
    expect(screen.getByText(/長さ \(L2\)/)).toBeInTheDocument()
  })

  it('点群は散布図にする', () => {
    render(
      <ArtifactBody
        preview={{
          row: 0,
          file: '000000.npy',
          form: 'point_cloud',
          shape: [3, 3],
          dtype: 'float32',
          view: 'points',
          values: [],
          size: 9,
          norm: 1,
          points: [
            [0, 0, 0],
            [1, 1, 1],
            [2, 0, 1],
          ],
          total: 3,
        }}
        names={[]}
      />,
    )
    expect(screen.getByRole('img', { name: /点群の散布図 \(3 点\)/ })).toBeInTheDocument()
    expect(screen.getByLabelText('縦の次元')).toBeInTheDocument()
  })
})
