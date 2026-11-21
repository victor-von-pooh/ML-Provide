import { useMutation, useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import { api } from '../../api/client'
import type { Bundle, DatasetDescription } from '../../api/types'
import { Dialog } from '../../components/Dialog'
import { PathField } from '../../components/PathPicker'
import { Button, ErrorBox } from '../../components/ui'
import {
  ARRAY_FORMS,
  buildInputs,
  buildOutputOptions,
  buildProcedure,
  EMPTY_OUTPUT,
  EMPTY_PROCEDURE,
  newInputRow,
  type InputRow,
  type OutputForm,
  type ProcedureForm,
} from '../../lib/bundleForm'
import {
  ENTRY_KIND_LABELS,
  FORM_LABELS,
  ORIGIN_LABELS,
  ROLE_LABELS,
  stepsText,
} from '../../lib/outputs'
import { useDebounced } from '../../lib/useDebounced'
import { InputFields, OutputFields, ProcedureFields } from './CreateBundleFields'

const field = 'h-8 rounded-lg border border-line bg-surface px-2 text-sm'

/** 空でなく, 0 以上の整数でない欄か */
function notCount(value: string, positive = false): boolean {
  if (!value.trim()) return false
  const number = Number(value)
  return !Number.isInteger(number) || number < (positive ? 1 : 0)
}

/** ファイルの組のデータセットの版 (版 3 は版 2 に項目ごとの中身の手順を足した版) */
const ITEMS_VERSIONS = [2, 3]

/** ファイルの組のデータセットの要約 (受け持ち, 正解, 項目, 中身の手順, 分割, 領域の値の名前) */
function ItemsSummary({ described }: { described: DatasetDescription }) {
  const target = described.target
  const origin = target?.origin?.type
  const splits = Object.entries(described.splits ?? {})
  const classes = described.classes ?? []
  const stepped = (described.entries ?? []).filter((entry) => (entry.steps ?? []).length > 0)
  return (
    <div
      role="note"
      aria-label={`版 ${described.format_version} のデータセット`}
      className="rounded-lg border border-line p-2 text-xs text-ink-2"
    >
      <p>
        版 {described.format_version} のデータセット (ファイルの組):{' '}
        {ROLE_LABELS[described.role ?? ''] ?? described.role}, 正解{' '}
        {FORM_LABELS[target?.kind ?? ''] ?? target?.kind}
        {origin && ` (出どころ: ${ORIGIN_LABELS[origin] ?? origin})`}
      </p>
      <p>
        項目:{' '}
        {(described.entries ?? [])
          .map(
            (entry) =>
              `${entry.name} (${ENTRY_KIND_LABELS[entry.kind] ?? entry.kind}, ${
                entry.role === 'input' ? '入力' : entry.role === 'target' ? '正解' : entry.role
              })`,
          )
          .join(', ')}
      </p>
      {stepped.map((entry) => (
        <p key={entry.name}>
          中身の手順 ({entry.name}): {stepsText(entry.steps ?? [])}
        </p>
      ))}
      {splits.length > 0 && (
        <p>件数: {splits.map(([name, count]) => `${name} ${count ?? '–'}`).join(', ')}</p>
      )}
      {classes.length > 0 && (
        <p>
          領域の値の名前:{' '}
          {classes.map((item) => `${item.value}: ${item.name ?? '(名前なし)'}`).join(', ')}
        </p>
      )}
    </div>
  )
}

/** 学習の結果 (モデルの定義, データセット, 実験) から推論の一式を作る */
export function CreateBundleDialog({
  onClose,
  onCreated,
}: {
  onClose: () => void
  onCreated: (bundle: Bundle) => void
}) {
  const [name, setName] = useState('')
  const [spec, setSpec] = useState('')
  // データセットを使わない雑音だけの一式 (拡散やフローの生成のモデル)
  const [noData, setNoData] = useState(false)
  const [dataset, setDataset] = useState('')
  const [weights, setWeights] = useState('')
  const [inputRows, setInputRows] = useState<InputRow[]>([])
  const [output, setOutput] = useState('')
  const [center, setCenter] = useState('')
  const [scale, setScale] = useState('')
  const [trainScale, setTrainScale] = useState(false)
  const [copyWeights, setCopyWeights] = useState(true)
  // 受け持ちと出力の形 (空はタスクから. 正解の無いデータセットと雑音だけの一式は出力の形が要る)
  const [role, setRole] = useState('')
  const [form, setForm] = useState('')
  const [outputFields, setOutputFields] = useState<OutputForm>(EMPTY_OUTPUT)
  // 繰り返しの手順 (登録簿から選ぶ)
  const [procedureForm, setProcedureForm] = useState<ProcedureForm>(EMPTY_PROCEDURE)
  const info = useQuery({ queryKey: ['info'], queryFn: api.info, staleTime: Infinity })
  const procedures = info.data?.procedures ?? []
  const procedureEntry = procedures.find((entry) => entry.name === procedureForm.name)
  // ファイルの組のデータセットは, 選んだときに項目を読み, 1 件の扱いの欄を出す
  const datasetPath = useDebounced(noData ? '' : dataset.trim())
  const described = useQuery({
    queryKey: ['dataset', datasetPath],
    queryFn: () => api.describeDataset(datasetPath),
    enabled: !!datasetPath,
    retry: false,
  })
  const itemsData =
    !noData && described.data && ITEMS_VERSIONS.includes(described.data.format_version)
      ? described.data
      : null
  // 説明で止めたデータセット (読めない, 一式を作れない版) は作れない
  const describeFailed = !noData && !!described.error && datasetPath === dataset.trim()
  const itemInputs = (itemsData?.entries ?? []).filter((entry) => entry.role === 'input')
  const [itemInput, setItemInput] = useState('')
  const chosenInput = itemInputs.find((entry) => entry.name === itemInput) ?? itemInputs[0]
  const needsPoints = !!chosenInput?.points
  const [points, setPoints] = useState('')
  const [targetPoints, setTargetPoints] = useState('')
  const [seed, setSeed] = useState('')
  // 列のデータセットと雑音だけの一式の欄 (ファイルの組のデータセットは 1 件の扱いの欄だけ)
  const built = buildInputs(inputRows, noData, !!procedureForm.name)
  const builtProcedure = buildProcedure(procedureForm, procedureEntry)
  const builtOutput = buildOutputOptions(form, outputFields, procedureEntry?.start === 'noise')
  const chooseNoData = (checked: boolean) => {
    setNoData(checked)
    if (checked) {
      // 雑音だけの一式は手順の中身と時刻を渡すことが多いので, 2 つの入力の行を用意する
      if (inputRows.length === 0)
        setInputRows([newInputRow('state', true), newInputRow('time', true)])
      // 候補から外れる手順と出力の形は空にする (欄の表示と送る値を食い違わせない).
      // 入力の出どころは, 書いた名前を消さないように問題として知らせる
      if (procedureEntry && procedureEntry.start !== 'noise') setProcedureForm(EMPTY_PROCEDURE)
      if (form && !ARRAY_FORMS.includes(form)) setForm('')
    } else {
      // 列のデータセットに戻すと畳む欄なので, 雑音だけの一式のために選んだ値を残さない
      setInputRows(inputRows.filter((row) => !row.auto))
      setProcedureForm(EMPTY_PROCEDURE)
      setRole('')
      setForm('')
      setOutputFields(EMPTY_OUTPUT)
    }
  }
  const create = useMutation({
    // 送ったときの欄の印 (欄を変えたら前の誤りを出さないため)
    mutationFn: (_key: string) => {
      if (itemsData) {
        // 省いた欄 (正解の項目, 扱い方, 領域の値, 戻し方) はデータセットから
        const options: Record<string, number> = {}
        if (needsPoints) {
          options.points = Number(points)
          if (targetPoints.trim()) options.target_points = Number(targetPoints)
          if (seed.trim()) options.seed = Number(seed)
        }
        const items = {
          ...(itemInputs.length > 1 && chosenInput ? { input: chosenInput.name } : {}),
          ...(needsPoints ? { options } : {}),
        }
        return api.createBundle({
          name: name.trim(),
          spec: spec.trim(),
          dataset: dataset.trim(),
          weights: weights.trim(),
          output: output.trim() || null,
          copy_weights: copyWeights,
          items: Object.keys(items).length > 0 ? items : null,
        })
      }
      const targetScale = noData
        ? null
        : trainScale
          ? ('train' as const)
          : center.trim() && scale.trim()
            ? { center: Number(center), scale: Number(scale) }
            : null
      return api.createBundle({
        name: name.trim(),
        spec: spec.trim(),
        dataset: noData ? null : dataset.trim(),
        weights: weights.trim(),
        inputs: built.inputs,
        output: output.trim() || null,
        target_scale: targetScale,
        copy_weights: copyWeights,
        role: role || null,
        form: form || null,
        output_options: builtOutput.options,
        procedure: builtProcedure.procedure,
      })
    },
    onSuccess: onCreated,
  })
  // 片方だけの欄と数でない値は, 送る前に知らせる (ファイルの組のデータセットは点の欄だけ)
  const problems: string[] = []
  if (itemsData) {
    if (needsPoints && (!points.trim() || notCount(points, true)))
      problems.push('入力が撮影か点群なので, 選ぶ点の数 (学習と同じ数) を正の整数で書いてください')
    if (needsPoints && notCount(targetPoints, true))
      problems.push('正解から選ぶ点の数は正の整数で書いてください')
    if (needsPoints && notCount(seed))
      problems.push('点を選ぶ乱数の種は 0 以上の整数で書いてください')
  } else {
    problems.push(...built.problems, ...builtProcedure.problems, ...builtOutput.problems)
    if (noData && !procedureForm.name)
      problems.push('雑音だけの一式は, 雑音から始まる手順 (ddim, euler など) を選んでください')
    if (!noData && !trainScale && !!center.trim() !== !!scale.trim())
      problems.push('目的の値の中心と幅は両方を書いてください')
    if (!noData && !trainScale && scale.trim() && !(Number(scale) > 0))
      problems.push('幅は正の数にしてください')
  }
  // 欄の中身の印 (送った後に欄を変えたら, 前の誤りを消す)
  const formKey = JSON.stringify([
    name,
    spec,
    noData,
    dataset,
    weights,
    inputRows,
    output,
    center,
    scale,
    trainScale,
    copyWeights,
    role,
    form,
    outputFields,
    procedureForm,
    itemInput,
    points,
    targetPoints,
    seed,
  ])
  const ready =
    name.trim() &&
    spec.trim() &&
    (noData || dataset.trim()) &&
    weights.trim() &&
    problems.length === 0 &&
    !describeFailed
  return (
    <Dialog title="学習の結果から一式を作る" onClose={onClose}>
      <form
        className="flex flex-col gap-3"
        onSubmit={(event) => {
          event.preventDefault()
          if (ready) create.mutate(formKey)
        }}
      >
        <p className="text-xs text-ink-2">
          モデルの定義, ML-EDA (ML-Fathom の画面) で書き出したデータセット (列のデータセットか,
          画像と領域, 撮影と点群の版 2 か版 3 のデータセット), ML-Found の実験 (か重みのファイル)
          を選びます. 雑音だけから作る生成のモデル (拡散やフロー) はデータセットを使いません.
          学習用の見本 (雑音だけの一式は手順で作った 2 件) を推論できることを確かめてから,
          このアプリの置き場に作ります (元のファイルは変えません).
        </p>
        <div>
          <label htmlFor="bundle-name" className="mb-1 block text-xs font-medium text-ink-2">
            名前
          </label>
          <input
            id="bundle-name"
            value={name}
            onChange={(event) => setName(event.target.value)}
            maxLength={80}
            className={`${field} w-full`}
          />
        </div>
        <PathField
          id="bundle-spec"
          label="モデルの定義 (YAML か JSON)"
          value={spec}
          onChange={setSpec}
          pickDirs={false}
        />
        <div className="flex items-start gap-2 text-sm">
          <input
            id="bundle-no-data"
            type="checkbox"
            className="mt-1"
            checked={noData}
            aria-describedby="bundle-no-data-note"
            onChange={(event) => chooseNoData(event.target.checked)}
          />
          <div>
            <label htmlFor="bundle-no-data">雑音だけから作る (データセットを使わない)</label>
            <p id="bundle-no-data-note" className="text-xs text-ink-3">
              条件の無い生成のモデル (拡散, フロー) の一式です. 推論では件数と種から作ります.
              雑音から始まる手順, 配列の出力の形と 1 件の形, 入力の出どころ (手順の中身と時刻)
              を下の欄で選びます
            </p>
          </div>
        </div>
        {!noData && (
          <PathField
            id="bundle-dataset"
            label="データセット (dataset.yaml を持つディレクトリ)"
            value={dataset}
            onChange={setDataset}
            pickFiles={false}
          />
        )}
        {itemsData && <ItemsSummary described={itemsData} />}
        {describeFailed && <p className="text-xs text-warning-text">{described.error?.message}</p>}
        {itemsData && (itemInputs.length > 1 || needsPoints) && (
          <fieldset className="grid gap-3 rounded-lg border border-line p-3 md:grid-cols-2">
            <legend className="px-1 text-xs font-semibold text-ink-2">
              1 件の扱い (学習のスクリプトと同じ値)
            </legend>
            {itemInputs.length > 1 && (
              <div>
                <label htmlFor="bundle-item-input" className="mb-1 block text-xs text-ink-2">
                  モデルに渡す入力の項目
                </label>
                <select
                  id="bundle-item-input"
                  value={chosenInput?.name ?? ''}
                  onChange={(event) => setItemInput(event.target.value)}
                  className={`${field} w-full`}
                >
                  {itemInputs.map((entry) => (
                    <option key={entry.name} value={entry.name}>
                      {entry.name} ({ENTRY_KIND_LABELS[entry.kind] ?? entry.kind})
                    </option>
                  ))}
                </select>
              </div>
            )}
            {needsPoints && (
              <>
                <div>
                  <label htmlFor="bundle-points" className="mb-1 block text-xs text-ink-2">
                    入力から選ぶ点の数 (要る)
                  </label>
                  <input
                    id="bundle-points"
                    type="number"
                    min={1}
                    value={points}
                    placeholder="512"
                    onChange={(event) => setPoints(event.target.value)}
                    className={`${field} w-full`}
                  />
                </div>
                <div>
                  <label htmlFor="bundle-target-points" className="mb-1 block text-xs text-ink-2">
                    確かめで正解から選ぶ点の数 (空なら全ての点)
                  </label>
                  <input
                    id="bundle-target-points"
                    type="number"
                    min={1}
                    value={targetPoints}
                    onChange={(event) => setTargetPoints(event.target.value)}
                    className={`${field} w-full`}
                  />
                </div>
                <div>
                  <label htmlFor="bundle-seed" className="mb-1 block text-xs text-ink-2">
                    点を選ぶ乱数の種 (空なら 0)
                  </label>
                  <input
                    id="bundle-seed"
                    type="number"
                    min={0}
                    value={seed}
                    onChange={(event) => setSeed(event.target.value)}
                    className={`${field} w-full`}
                  />
                </div>
                <p className="text-xs text-ink-3 md:col-span-2">
                  ML-Quest の骨組みで学習した実験なら, config の data.input_points と
                  data.target_points, 種は data.seed + 1 (検証用の分割の種) を書きます.
                  違う数だと黙って違う予測になります
                </p>
              </>
            )}
          </fieldset>
        )}
        <PathField
          id="bundle-weights"
          label="実験のディレクトリか重みのファイル"
          value={weights}
          onChange={setWeights}
          hint="実験のディレクトリなら final_model.pt を使い, 状態とベストの値を記録します"
        />
        {!itemsData && (
          <>
            <details className="rounded-lg border border-line p-3" open={noData}>
              <summary className="cursor-pointer text-xs font-semibold text-ink-2">
                {noData ? '繰り返しの手順 (要る)' : '繰り返しの手順 (生成のモデル. あれば)'}
              </summary>
              <div className="mt-3">
                {info.error ? (
                  <ErrorBox error={info.error} />
                ) : (
                  <ProcedureFields
                    form={procedureForm}
                    onChange={setProcedureForm}
                    procedures={procedures}
                    noData={noData}
                  />
                )}
              </div>
            </details>
            <details className="rounded-lg border border-line p-3" open={noData}>
              <summary className="cursor-pointer text-xs font-semibold text-ink-2">
                {noData ? '出力の形 (要る)' : '受け持ちと出力の形 (あれば)'}
              </summary>
              <div className="mt-3">
                <OutputFields
                  role={role}
                  onRole={setRole}
                  form={form}
                  onForm={setForm}
                  fields={outputFields}
                  onFields={setOutputFields}
                  noData={noData}
                />
              </div>
            </details>
            <details className="rounded-lg border border-line p-3" open={noData}>
              <summary className="cursor-pointer text-xs font-semibold text-ink-2">
                {noData
                  ? 'モデルの入力ごとの出どころ'
                  : 'モデルの入力ごとの出どころ (入力が複数か, 先頭に番号を足した場合)'}
              </summary>
              <div className="mt-3">
                <InputFields rows={inputRows} onChange={setInputRows} noData={noData} />
              </div>
            </details>
          </>
        )}
        <details className="rounded-lg border border-line p-3">
          <summary className="cursor-pointer text-xs font-semibold text-ink-2">
            {itemsData || noData
              ? '出力が複数のモデル (あれば)'
              : '学習のスクリプトが前処理の外でしたこと (あれば)'}
          </summary>
          <div className="mt-3 grid gap-3 md:grid-cols-2">
            <div>
              <label htmlFor="bundle-output" className="mb-1 block text-xs text-ink-2">
                使う出力の名前 (出力が複数のモデル)
              </label>
              <input
                id="bundle-output"
                value={output}
                placeholder="logits"
                onChange={(event) => setOutput(event.target.value)}
                className={`${field} w-full font-mono`}
              />
            </div>
            {!itemsData && !noData && (
              <fieldset className="md:col-span-2">
                <legend className="mb-1 text-xs text-ink-2">
                  目的の値を標準化して学習した場合 (回帰と予測)
                </legend>
                <label className="mb-2 flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={trainScale}
                    onChange={(event) => setTrainScale(event.target.checked)}
                  />
                  学習用の分割の平均と標準偏差で標準化した (中心と幅を求める)
                </label>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-ink-3">または (y - 中心) / 幅 の</span>
                  <input
                    aria-label="中心"
                    type="number"
                    value={center}
                    disabled={trainScale}
                    placeholder="中心"
                    onChange={(event) => setCenter(event.target.value)}
                    className={`${field} w-28`}
                  />
                  <input
                    aria-label="幅"
                    type="number"
                    value={scale}
                    disabled={trainScale}
                    placeholder="幅"
                    onChange={(event) => setScale(event.target.value)}
                    className={`${field} w-28`}
                  />
                </div>
              </fieldset>
            )}
          </div>
        </details>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={copyWeights}
            onChange={(event) => setCopyWeights(event.target.checked)}
          />
          重みを一式に写す (外すと実験の重みを指す. 実験を消すと推論できなくなる)
        </label>
        {problems.map((problem) => (
          <p key={problem} className="text-xs text-critical-text">
            {problem}
          </p>
        ))}
        {create.error && create.variables === formKey && <ErrorBox error={create.error} />}
        <div className="flex justify-end">
          <Button type="submit" variant="primary" disabled={!ready || create.isPending}>
            {create.isPending ? '確かめています…' : '作る'}
          </Button>
        </div>
      </form>
    </Dialog>
  )
}
