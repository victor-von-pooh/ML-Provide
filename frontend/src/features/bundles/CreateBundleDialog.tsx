import { useMutation, useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import { useApi } from '../../api/client'
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
  MAX_PROCEDURE_STEPS,
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
import { readDraft, useDraftWriter } from '../../lib/draft'
import { useStage } from '../../lib/stage'
import { useDebounced } from '../../lib/useDebounced'
import { AutoDetails, InputFields, OutputFields, ProcedureFields } from './CreateBundleFields'

const field = 'h-8 rounded-lg border border-line bg-surface px-2 text-sm'

/** 畳んだ欄の名前 (受け持ちと出力の形の欄は, 雑音だけかで名前が変わる) */
const SECTION_PROCEDURE = '繰り返しの手順'
const SECTION_INPUTS = 'モデルの入力ごとの出どころ'

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

/** 欄の書きかけの下書きの種類 */
const DRAFT_KIND = 'create-bundle'

/** 下書きに残す欄の中身 (入力の行の印は戻すときに付け直す) */
interface CreateDraft {
  name: string
  spec: string
  noData: boolean
  dataset: string
  weights: string
  inputRows: Omit<InputRow, 'id'>[]
  output: string
  center: string
  scale: string
  trainScale: boolean
  copyWeights: boolean
  role: string
  form: string
  outputFields: OutputForm
  procedureForm: ProcedureForm
  itemInput: string
  points: string
  targetPoints: string
  seed: string
}

/** 空の欄 (書きかけが無い形) */
const EMPTY_DRAFT: CreateDraft = {
  name: '',
  spec: '',
  noData: false,
  dataset: '',
  weights: '',
  inputRows: [],
  output: '',
  center: '',
  scale: '',
  trainScale: false,
  copyWeights: true,
  role: '',
  form: '',
  outputFields: EMPTY_OUTPUT,
  procedureForm: EMPTY_PROCEDURE,
  itemInput: '',
  points: '',
  targetPoints: '',
  seed: '',
}

/** 値が文字だけの辞書か */
function isTextRecord(value: unknown): value is Record<string, string> {
  return (
    !!value &&
    typeof value === 'object' &&
    !Array.isArray(value) &&
    Object.values(value).every((item) => typeof item === 'string')
  )
}

/** 読んだ下書きを欄の中身にする (形の合わない欄は空の値. 辞書でなければ戻さない) */
function restoreDraft(stored: unknown): CreateDraft | null {
  if (!stored || typeof stored !== 'object' || Array.isArray(stored)) return null
  const value = stored as Record<string, unknown>
  const draft: Record<string, unknown> = { ...EMPTY_DRAFT }
  for (const [key, empty] of Object.entries(EMPTY_DRAFT)) {
    const item = value[key]
    if (typeof empty === 'string' || typeof empty === 'boolean') {
      if (typeof item === typeof empty) draft[key] = item
    }
  }
  // 入力の行は, 出どころを持つものだけ (auto は雑音だけの一式が足した行の印)
  if (Array.isArray(value.inputRows)) {
    draft.inputRows = value.inputRows.flatMap((row: unknown) => {
      if (!row || typeof row !== 'object') return []
      const item = row as Record<string, unknown>
      if (typeof item.source !== 'string') return []
      return [
        {
          name: typeof item.name === 'string' ? item.name : '',
          source: item.source,
          prepend: typeof item.prepend === 'string' ? item.prepend : '',
          ...(item.auto === true ? { auto: true } : {}),
        },
      ]
    })
  }
  const fields = value.outputFields
  if (isTextRecord(fields)) draft.outputFields = { ...EMPTY_OUTPUT, ...fields }
  const procedure = value.procedureForm
  if (procedure && typeof procedure === 'object' && !Array.isArray(procedure)) {
    const item = procedure as Record<string, unknown>
    draft.procedureForm = {
      name: typeof item.name === 'string' ? item.name : '',
      steps: typeof item.steps === 'string' ? item.steps : '',
      seed: typeof item.seed === 'string' ? item.seed : '',
      options: isTextRecord(item.options) ? item.options : {},
      extra: typeof item.extra === 'string' ? item.extra : '',
    }
  }
  return draft as unknown as CreateDraft
}

/**
 * 学習の結果 (モデルの定義, データセット, 実験) から推論の一式を作る
 *
 * 欄の書きかけは下書きに残し, 次に開いたときに戻す (殻の中で段階を移ると部品が外され,
 * 書きかけを確かめずに捨てるため)
 */
export function CreateBundleDialog({
  onClose,
  onCreated,
}: {
  onClose: () => void
  onCreated: (bundle: Bundle) => void
}) {
  // 開いたときの下書き (捨てたら空の欄で作り直す)
  const [restored, setRestored] = useState(() => readDraft(DRAFT_KIND, restoreDraft))
  const [round, setRound] = useState(0)
  return (
    <Dialog title="学習の結果から一式を作る" onClose={onClose}>
      <CreateBundleForm
        key={round}
        initial={restored?.value ?? EMPTY_DRAFT}
        restoredAt={restored?.time ?? null}
        onDiscard={() => {
          setRestored(null)
          setRound((value) => value + 1)
        }}
        onCreated={onCreated}
      />
    </Dialog>
  )
}

/** 一式を作る欄 (initial は戻した下書きか空の欄) */
function CreateBundleForm({
  initial,
  restoredAt,
  onDiscard,
  onCreated,
}: {
  initial: CreateDraft
  restoredAt: string | null
  onDiscard: () => void
  onCreated: (bundle: Bundle) => void
}) {
  const api = useApi()
  const [name, setName] = useState(initial.name)
  const [spec, setSpec] = useState(initial.spec)
  // データセットを使わない雑音だけの一式 (拡散やフローの生成のモデル)
  const [noData, setNoData] = useState(initial.noData)
  const [dataset, setDataset] = useState(initial.dataset)
  const [weights, setWeights] = useState(initial.weights)
  const [inputRows, setInputRows] = useState<InputRow[]>(() =>
    initial.inputRows.map((row) => ({ ...newInputRow(row.source, row.auto), ...row })),
  )
  const [output, setOutput] = useState(initial.output)
  const [center, setCenter] = useState(initial.center)
  const [scale, setScale] = useState(initial.scale)
  const [trainScale, setTrainScale] = useState(initial.trainScale)
  const [copyWeights, setCopyWeights] = useState(initial.copyWeights)
  // 受け持ちと出力の形 (空はタスクから. 正解の無いデータセットと雑音だけの一式は出力の形が要る)
  const [role, setRole] = useState(initial.role)
  const [form, setForm] = useState(initial.form)
  const [outputFields, setOutputFields] = useState<OutputForm>(initial.outputFields)
  // 繰り返しの手順 (登録簿から選ぶ)
  const [procedureForm, setProcedureForm] = useState<ProcedureForm>(initial.procedureForm)
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
  // 説明を読んでいる間と, 欄が読んだ場所と違う間 (打ってから待つ間) は作れない
  // (前のデータセットの説明で欄を決めて送らないため)
  const describing =
    !noData && !!dataset.trim() && (datasetPath !== dataset.trim() || described.isFetching)
  const itemInputs = (itemsData?.entries ?? []).filter((entry) => entry.role === 'input')
  const [itemInput, setItemInput] = useState(initial.itemInput)
  const chosenInput = itemInputs.find((entry) => entry.name === itemInput) ?? itemInputs[0]
  const needsPoints = !!chosenInput?.points
  const [points, setPoints] = useState(initial.points)
  const [targetPoints, setTargetPoints] = useState(initial.targetPoints)
  const [seed, setSeed] = useState(initial.seed)
  // 列のデータセットと雑音だけの一式の欄 (ファイルの組のデータセットは 1 件の扱いの欄だけ)
  const built = buildInputs(inputRows, noData, !!procedureForm.name)
  const maxSteps = info.data?.max_procedure_steps ?? MAX_PROCEDURE_STEPS
  const builtProcedure = buildProcedure(procedureForm, procedureEntry, maxSteps)
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
  // 殻で選んだ問題を, 作る一式につなぐ既定にする. 外したのはそのときの問題だけで,
  // 殻で選び直すとまたつなぐ (一覧の絞りと同じ)
  // 推論に渡せない問題と置き場に無い問題は, 欄を出さずに理由を添える
  const { project } = useStage()
  const [unlinked, setUnlinked] = useState(false)
  const [unlinkedFor, setUnlinkedFor] = useState(project)
  if (unlinkedFor !== project) {
    // 殻で問題を替えたら外した状態を消す (A → B → A で外したままにしない)
    setUnlinkedFor(project)
    setUnlinked(false)
  }
  const projectProblem = useQuery({
    queryKey: ['project-problem', project],
    queryFn: () => api.projectProblem(project ?? ''),
    enabled: project !== null,
  })
  // 読み終えるまでは欄も理由も出さない (読む間だけ出て消える見た目にしないため)
  const usable = projectProblem.data?.usable === true
  const linking = project !== null && usable && !unlinked
  const problem = linking ? project : null
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
          problem,
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
        problem,
      })
    },
    onSuccess: (bundle) => {
      // 作り終えた欄の書きかけは要らない
      discardDraft()
      onCreated(bundle)
    },
  })
  // 欄の書きかけを下書きに残す (殻で選んだ問題が替わるときは待たずに書く)
  const drafted: CreateDraft = {
    name,
    spec,
    noData,
    dataset,
    weights,
    inputRows: inputRows.map(({ id: _id, ...row }) => row),
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
  }
  const discardDraft = useDraftWriter(
    DRAFT_KIND,
    JSON.stringify(drafted) !== JSON.stringify(EMPTY_DRAFT),
    drafted,
    project,
  )
  // 畳んだ欄の名前 (見出しと誤りに添える名前を同じにする)
  const sectionOutput = noData ? '出力の形' : '受け持ちと出力の形'
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
    // 畳んだ欄の誤りは, どの欄かを添える (欄も開く)
    problems.push(
      ...builtProcedure.problems.map((problem) => `「${SECTION_PROCEDURE}」: ${problem}`),
      ...builtOutput.problems.map((problem) => `「${sectionOutput}」: ${problem}`),
      ...built.problems.map((problem) => `「${SECTION_INPUTS}」: ${problem}`),
    )
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
    problem,
  ])
  const ready =
    name.trim() &&
    spec.trim() &&
    (noData || dataset.trim()) &&
    weights.trim() &&
    problems.length === 0 &&
    !describeFailed &&
    !describing
  return (
    <>
      {restoredAt && (
        <div
          role="status"
          className="mb-3 flex items-center justify-between gap-2 rounded-lg border border-line p-2 text-xs text-ink-2"
        >
          <span>
            作っていない書きかけ ({new Date(restoredAt).toLocaleString('ja-JP')}) を欄に戻しました.
          </span>
          <Button
            variant="ghost"
            onClick={() => {
              // 待っている書き込みも止めてから, 空の欄で作り直す
              discardDraft()
              onDiscard()
            }}
            disabled={create.isPending}
          >
            書きかけを捨てる
          </Button>
        </div>
      )}
      <form
        className="flex flex-col gap-3"
        onSubmit={(event) => {
          event.preventDefault()
          if (ready) create.mutate(formKey)
        }}
      >
        <p className="text-xs text-ink-2">
          モデルの定義, データの分析の段階で書き出したデータセット (列のデータセットか, 画像と領域,
          撮影と点群の版 2 か版 3 のデータセット), 学習の実験 (か重みのファイル) を選びます.
          雑音だけから作る生成のモデル (拡散やフロー) はデータセットを使いません. 学習用の見本
          (雑音だけの一式は手順で作った 2 件) を推論できることを確かめてから,
          推論の段階の置き場に作ります (元のファイルは変えません).
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
        {describing && (
          <p role="status" className="text-xs text-ink-2">
            データセットの説明を読み込んでいます
          </p>
        )}
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
                  骨組みの学習のプロジェクトで学習した実験なら, config の data.input_points と
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
            <AutoDetails
              summary={
                noData
                  ? `${SECTION_PROCEDURE} (要る)`
                  : `${SECTION_PROCEDURE} (生成のモデル. あれば)`
              }
              wantOpen={noData}
              problems={builtProcedure.problems.length}
            >
              {info.error ? (
                <ErrorBox error={info.error} />
              ) : (
                <ProcedureFields
                  form={procedureForm}
                  onChange={setProcedureForm}
                  procedures={procedures}
                  noData={noData}
                  maxSteps={maxSteps}
                />
              )}
            </AutoDetails>
            <AutoDetails
              summary={noData ? `${sectionOutput} (要る)` : `${sectionOutput} (あれば)`}
              wantOpen={noData}
              problems={builtOutput.problems.length}
            >
              <OutputFields
                role={role}
                onRole={setRole}
                form={form}
                onForm={setForm}
                fields={outputFields}
                onFields={setOutputFields}
                noData={noData}
              />
            </AutoDetails>
            <AutoDetails
              summary={
                noData ? SECTION_INPUTS : `${SECTION_INPUTS} (入力が複数か, 先頭に番号を足した場合)`
              }
              wantOpen={noData}
              problems={built.problems.length}
            >
              <InputFields rows={inputRows} onChange={setInputRows} noData={noData} />
            </AutoDetails>
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
        {project !== null && usable && (
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={linking}
              onChange={(event) => setUnlinked(!event.target.checked)}
            />
            選んだ問題 {project} をつなぐ (問題を使わない一式は, つながずに作る)
          </label>
        )}
        {project !== null && projectProblem.data?.usable === false && (
          <p className="text-xs text-ink-2">
            選んだ問題 {project} はつながずに作ります: {projectProblem.data?.reason}
          </p>
        )}
        {/* 誤りは打つたびに変わるので, alert でなく polite で読み上げる */}
        <div aria-live="polite" className="flex flex-col gap-1">
          {problems.map((problem) => (
            <p key={problem} className="text-xs text-critical-text">
              {problem}
            </p>
          ))}
        </div>
        {create.error && create.variables === formKey && <ErrorBox error={create.error} />}
        <div className="flex justify-end">
          <Button type="submit" variant="primary" disabled={!ready || create.isPending}>
            {create.isPending ? '確かめています…' : '作る'}
          </Button>
        </div>
      </form>
    </>
  )
}
