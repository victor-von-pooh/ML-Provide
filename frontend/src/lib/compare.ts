import type { Result } from '../api/types'
import { DECISION_TYPES } from './decision'
import { formatPercent, formatValue, TASK_LABELS } from './format'

/** 比べの表の 1 行 (結果ごとの値と, 結果の間で違うか) */
export interface CompareRow {
  label: string
  values: string[]
  /** 結果の間で値が違うか (時刻のように違って当然の行は偽) */
  differs: boolean
}

/** 比べの表のまとまり (見出しと行) */
export interface CompareSection {
  title: string
  rows: CompareRow[]
}

// 値の無いことを表す文字
const NONE = '–'

/** 行を作る (全ての値が同じでなければ違う行) */
function row(label: string, values: string[], plain = false): CompareRow {
  return { label, values, differs: !plain && new Set(values).size > 1 }
}

/** 決め方や意思決定の値を文字にする */
function show(value: unknown): string {
  if (value === null || value === undefined || value === '') return NONE
  if (typeof value === 'number') return formatValue(value)
  if (typeof value === 'boolean') return value ? 'はい' : 'いいえ'
  if (typeof value === 'string') return value
  return JSON.stringify(value)
}

/** 欄の説明の頭 (括弧の前) */
function short(label: string): string {
  return label.split(' (')[0] ?? label
}

/** 何を比べたか (一式, 入力, 時刻) */
function heads(results: Result[]): CompareSection {
  return {
    title: '何を比べたか',
    rows: [
      row(
        '一式',
        results.map((r) => r.bundle?.name ?? r.bundle?.path ?? NONE),
      ),
      row(
        'タスク',
        results.map((r) => TASK_LABELS[r.task ?? ''] ?? show(r.task)),
      ),
      row(
        '入力',
        results.map((r) => show(r.source)),
      ),
      row(
        '何段も先の予測',
        results.map((r) => (r.forecast ? 'はい' : 'いいえ')),
      ),
      row(
        '時刻',
        results.map((r) => (r.created_at ? new Date(r.created_at).toLocaleString() : NONE)),
        true,
      ),
    ],
  }
}

/** 結論の決め方 */
function rules(results: Result[]): CompareSection {
  const bands = (r: Result) =>
    (r.rules?.bands ?? [])
      .map((band) => (band.upper != null ? `${band.label} (${band.upper} まで)` : band.label))
      .join(', ')
  return {
    title: '結論の決め方',
    rows: [
      row(
        '保留にする確信の下限',
        results.map((r) => show(r.rules?.min_confidence)),
      ),
      row(
        '陽性のクラス',
        results.map((r) => show(r.rules?.positive)),
      ),
      row(
        '陽性とする確率',
        results.map((r) => show(r.rules?.threshold)),
      ),
      row(
        '値の帯',
        results.map((r) => show(bands(r))),
      ),
      row(
        'まとめる列',
        results.map((r) => show(r.rules?.group_by)),
      ),
    ],
  }
}

/** 結論 (全体の文と件数) と判断の内訳 */
function conclusions(results: Result[]): CompareSection[] {
  const regression = results.some((r) => r.task !== 'classification')
  const counts: CompareRow[] = [
    row(
      '結論の文',
      results.map((r) => r.sentences.slice(0, r.overall_count ?? r.sentences.length).join(' ')),
    ),
    row(
      '行',
      results.map((r) => show(r.summary?.rows)),
    ),
    row(
      '予測した行',
      results.map((r) => show(r.summary?.predicted)),
    ),
    row(
      '予測しなかった行',
      results.map((r) => show(r.summary?.skipped)),
    ),
  ]
  if (regression) {
    counts.push(
      row(
        '予測の合計',
        results.map((r) => show(r.summary?.values?.sum)),
      ),
      row(
        '予測の平均',
        results.map((r) => show(r.summary?.values?.mean)),
      ),
    )
  }
  // 判断の名前は全ての結果の和 (無い結果は 0 件)
  const names = [
    ...new Set(results.flatMap((r) => (r.summary?.decisions ?? []).map((d) => d.decision))),
  ]
  const breakdown = names.map((name) =>
    row(
      name,
      results.map((r) => {
        const found = r.summary?.decisions?.find((d) => d.decision === name)
        return found ? `${found.count.toLocaleString()} (${formatPercent(found.share)})` : '0'
      }),
    ),
  )
  return [
    { title: '結論', rows: counts },
    { title: '判断の内訳', rows: breakdown },
  ]
}

/** 意思決定 (型, 決め方の値, 行動の件数, 見込みの得と損, たどり直し) */
function decisions(results: Result[]): CompareSection | null {
  if (!results.some((r) => r.decision)) return null
  const types = [...new Set(results.flatMap((r) => (r.decision ? [r.decision.decision.type] : [])))]
  const fields = [
    ...new Map(
      types.flatMap((type) =>
        (DECISION_TYPES[type]?.fields ?? []).map((field) => [field.key, field] as const),
      ),
    ).values(),
  ]
  // 意思決定の決め方の値は, 型の欄に無いもの (陽性のクラスなど) も出す
  const known = new Set(['type', ...fields.map((field) => field.key)])
  const extra = [
    ...new Set(
      results.flatMap((r) => Object.keys(r.decision?.decision ?? {}).filter((k) => !known.has(k))),
    ),
  ]
  const summary = (r: Result, key: string) => show(r.decision?.summary[key])
  const baselines = [
    ...new Set(results.flatMap((r) => Object.keys(r.decision?.summary.baselines ?? {}))),
  ]
  const policies = [...new Set(results.flatMap((r) => Object.keys(r.backtest?.policies ?? {})))]
  const money = (value: number | undefined) =>
    value === undefined
      ? NONE
      : value < 0
        ? `損 ${formatValue(-value)}`
        : `得 ${formatValue(value)}`
  return {
    title: '意思決定',
    rows: [
      row(
        '型',
        results.map((r) =>
          r.decision ? short(DECISION_TYPES[r.decision.decision.type]?.label ?? '') : 'しない',
        ),
      ),
      ...fields.map((field) =>
        row(
          short(field.label),
          results.map((r) => show(r.decision?.decision[field.key])),
        ),
      ),
      ...extra.map((key) =>
        row(
          key,
          results.map((r) => show(r.decision?.decision[key])),
        ),
      ),
      row(
        '行動する件数',
        results.map((r) => summary(r, 'acting')),
      ),
      row(
        '量か費用の合計',
        results.map((r) => summary(r, 'amount')),
      ),
      row(
        '見込みの得',
        results.map((r) => summary(r, 'expected_gain')),
      ),
      row(
        '見込みの損',
        results.map((r) => summary(r, 'expected_cost')),
      ),
      row(
        '予測の値どおりに置いた損',
        results.map((r) => summary(r, 'baseline_cost')),
      ),
      ...baselines.map((name) =>
        row(
          `${name}の見込み`,
          results.map((r) => show(r.decision?.summary.baselines?.[name])),
        ),
      ),
      ...policies.map((name) =>
        row(
          `たどり直し: ${name}`,
          results.map((r) => money(r.backtest?.policies[name])),
        ),
      ),
    ],
  }
}

/** 予測の表の列のうち, 全ての結果には無い列 (一式で違う列) */
function columns(results: Result[]): CompareSection | null {
  const all = results.map((r) => new Set(r.columns ?? []))
  const own = (r: Result, index: number) =>
    (r.columns ?? []).filter((name) => all.some((set, i) => i !== index && !set.has(name)))
  const values = results.map((r, index) => own(r, index).join(', ') || NONE)
  if (values.every((value) => value === NONE)) return null
  return { title: '予測の表の列', rows: [row('他の結果に無い列', values)] }
}

/** 比べの表のまとまりを作る (どの結果にも値の無い行と, 行の無いまとまりは除く) */
export function compareSections(results: Result[]): CompareSection[] {
  return [
    heads(results),
    rules(results),
    ...conclusions(results),
    decisions(results),
    columns(results),
  ]
    .filter((section): section is CompareSection => !!section)
    .map((section) => ({
      ...section,
      rows: section.rows.filter((item) => item.values.some((value) => value !== NONE)),
    }))
    .filter((section) => section.rows.length > 0)
}
