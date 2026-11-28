/**
 * 欄にまたがる確かめの誤りの並び (components/ui の FieldProblems) の, 誤りごとの id
 * (送るボタンの aria-describedby から結び, 押せない理由として読ませるため)
 *
 * @param idBase - 並びの id の頭
 * @param problems - 誤りの文の並び
 */
export function problemIds(idBase: string, problems: string[]): string[] {
  return problems.map((_problem, index) => `${idBase}-problem-${index}`)
}
