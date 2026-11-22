/**
 * 外れ方を種類ごとに分けられる列 (回帰と予測の一式だけ. 分類は空).
 * 予測と判断の列 (own), 種類ごとの確率の列 (p:<種類>), 目的の列は除く
 */
export function groupColumns(
  task: string | null,
  columns: string[],
  target: string | null | undefined,
  own: ReadonlySet<string>,
): string[] {
  if (task === 'classification') return []
  return columns.filter((name) => !own.has(name) && !name.startsWith('p:') && name !== target)
}
