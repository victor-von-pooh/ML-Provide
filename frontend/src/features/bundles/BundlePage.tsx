import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Play, Trash2, Undo2 } from 'lucide-react'
import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router'

import { api } from '../../api/client'
import type { Bundle, Description, LinkedProblem, UncertaintySummary } from '../../api/types'
import { PathField } from '../../components/PathPicker'
import { Button, Card, ErrorBox, PageHeader, Spinner } from '../../components/ui'
import { SEED_RANGE_PROBLEM, SHORT_SOURCE_LABELS } from '../../lib/bundleForm'
import { CLUSTER_NO_PROBLEM, CLUSTER_UNSUPPORTED } from '../../lib/clusters'
import { formatNumber, KIND_LABELS, TASK_LABELS } from '../../lib/format'
import { fromDecisionForm, toDecisionForm, type DecisionForm } from '../../lib/decision'
import {
  ENTRY_KIND_LABELS,
  FORM_LABELS,
  isColumnOutput,
  METRIC_LABELS,
  optionsText,
  PREPARE_LABELS,
  sourceLabel,
} from '../../lib/outputs'
import {
  MAX_RANK_K,
  RANK_NO_PROBLEM,
  RANK_UNSUPPORTED,
  rankColumnsText,
  rankKProblem,
  rankUsers,
  reactionRuleText,
} from '../../lib/ranking'
import { classesText, licenseAlert, preprocessText, READY_UNSUPPORTED } from '../../lib/ready'
import { fromForm, toForm, withCleared, type RulesForm } from '../../lib/rules'
import { ClusterStabilityView, ClusterSummaryTable } from '../results/ClusterCards'
import { CompareResultsCard, MAX_COMPARE } from '../results/CompareResultsCard'
import { DecisionEditor } from '../results/DecisionEditor'
import { RulesEditor } from '../results/RulesEditor'
import { OutputNotices, OutputView } from './OutputView'
import { ProblemCard } from './ProblemCard'

/** 説明の 1 行 */
function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[9rem_1fr] gap-2 py-1">
      <dt className="text-ink-3">{label}</dt>
      <dd className="min-w-0 break-words">{children}</dd>
    </div>
  )
}

/** 版 2 のデータセットの一式 (版 4) の 1 件の扱い */
function ItemsRows({ items }: { items: NonNullable<Description['items']> }) {
  const kind = items.input_kind ?? ''
  const options = Object.entries(items.options)
  return (
    <>
      <Row label="データ">版 2 のデータセット (ファイルの組)</Row>
      <Row label="1 件の扱い">
        <span className="font-mono text-xs">{items.input}</span> ({ENTRY_KIND_LABELS[kind] ?? kind})
        → {PREPARE_LABELS[items.prepare] ?? items.prepare}
        {options.length > 0 && (
          <span className="font-mono text-xs">
            {' '}
            ({options.map(([key, value]) => `${key}: ${String(value)}`).join(', ')})
          </span>
        )}
      </Row>
      {items.target && (
        <Row label="正解の項目">
          <span className="font-mono text-xs">{items.target}</span>
        </Row>
      )}
      {items.restore === 'shift' && (
        <Row label="出力の座標">元の座標 (正規化の平行移動を戻した, 撮影の世界の座標)</Row>
      )}
    </>
  )
}

/** 既製のモデルの一式 (版 6) の出どころ, 前処理, 重みの置き場 */
function ReadyRows({ ready }: { ready: NonNullable<Description['ready']> }) {
  return (
    <>
      <Row label="データ">画像 (既製のモデル. データセットも学習の結果も無い)</Row>
      <Row label="出どころ">
        {ready.library} の <span className="font-mono text-xs">{ready.model}</span> (重み{' '}
        <span className="font-mono text-xs">{ready.weights}</span>)
      </Row>
      <Row label="前処理">{preprocessText(ready.preprocess)}</Row>
      <Row label="重みのファイル">
        <span className="font-mono text-xs break-all">{ready.weights_file}</span>
        <span className="block text-xs text-ink-2">
          {ready.weights_cached
            ? '出どころから読んで置き場にあります (次からはこれを読み, ハッシュ値で確かめます)'
            : '置き場にありません. 推論のときに出どころから取りに行きます (ネットワークを使えない場所では, このファイルを置きます)'}
        </span>
      </Row>
    </>
  )
}

/** 並べるの一式 (版 8) の番号の表と索引, 件数の既定, 反応の表の列 */
function RankRows({ rank }: { rank: NonNullable<Description['rank']> }) {
  return (
    <>
      <Row label="データ">
        利用者と品物の番号の表と, 品物の埋め込みの索引 (データセットなし. 推論は利用者から並べる)
      </Row>
      <Row label="利用者と品物">
        学習用の反応にいた利用者 {rank.users.toLocaleString()} 人と品物{' '}
        {rank.items.toLocaleString()} 件 (埋め込みの幅 {rank.width})
      </Row>
      <Row label="並べる件数の既定">{rank.k} 件 (問題の設定の件数)</Row>
      <Row label="反応の表の列">
        <span className="font-mono text-xs">{rankColumnsText(rank.columns)}</span>
        <span className="block text-xs text-ink-2">
          前に反応した品物の表と正解の反応の表の列です. {reactionRuleText(rank.columns)}
        </span>
      </Row>
    </>
  )
}

/** 群分けの一式 (版 9) の列の符号化と群の中心, 群を作る列, 群の数 */
function GroupRows({ group }: { group: NonNullable<Description['group']> }) {
  return (
    <>
      <Row label="データ">
        列の符号化と群の中心 (データセットなし. 推論は表の行を最も近い群に割り当てる)
      </Row>
      <Row label={`群を作る列 (${group.columns.length})`}>
        <span className="font-mono text-xs">{group.columns.join(', ')}</span>
        <span className="block text-xs text-ink-2">
          推論する表はこの列を持ちます. 数の列は学習のときの平均と標準偏差で,
          種類の列は学習のときの値ごとに符号化します (学習に無い値はどの値でもない扱い)
        </span>
      </Row>
      <Row label="群の数">
        {group.k} 個 (学習で {group.rows.toLocaleString()} 行を分けた. 符号化した行の幅{' '}
        {group.features}, 埋め込みの幅 {group.width})
      </Row>
      {group.check && (
        <Row label="割り当て直しの一致">
          学習の表 {group.check.rows.toLocaleString()} 行を一式で割り当て直し,{' '}
          {(group.check.agreement * 100).toFixed(1)} % が実験と同じ群
        </Row>
      )}
    </>
  )
}

/** 群分けの一式の, 学習のときの群の要約と群の安定性 */
function GroupLearnedCard({ group }: { group: NonNullable<Description['group']> }) {
  return (
    <Card title="学習のときの群">
      <div className="flex flex-col gap-4">
        <section aria-label="学習の群の安定性">
          <h3 className="mb-2 text-xs font-semibold text-ink-2">群の安定性</h3>
          <ClusterStabilityView stability={group.stability} />
        </section>
        <section aria-label="学習のときの群の要約">
          <h3 className="mb-2 text-xs font-semibold text-ink-2">
            群の要約 (学習で分けた {group.rows.toLocaleString()} 行)
          </h3>
          <ClusterSummaryTable
            rows={group.summary}
            columns={group.columns}
            caption="学習のときの群の要約 (群ごとの件数, 割合, 安定性, 列の特徴と全体)"
          />
        </section>
      </div>
    </Card>
  )
}

/** 既製のモデルの一式の注意 (ライセンスの警告は推論の前に気付けるように上に出す) */
function ReadyNotices({ ready }: { ready: NonNullable<Description['ready']> }) {
  const alert = licenseAlert(ready.license)
  if (!alert) return null
  return (
    <p
      role="note"
      aria-label="ライセンスの警告"
      className="rounded-lg border border-warning-text/40 bg-surface p-2 text-sm text-warning-text"
    >
      ライセンスの警告: {alert}
    </p>
  )
}

/** 一式の説明 (何を入れると何が出るか, どの学習の結果か) */
function DescriptionView({ description }: { description: Description }) {
  const run = description.run
  const inputs = Object.entries(description.model.inputs)
  const items = description.items
  // 雑音だけの一式はデータセットを持たず, 件数と種から作る
  const noise = description.kind === 'noise'
  // 既製のモデルの一式は読み込みと前処理の代わりに出どころと重みの前処理
  const ready = description.ready
  // 並べるの一式は番号の表と索引 (モデルの入力は利用者と品物の番号で, 列の出どころを持たない)
  const rank = description.rank
  // 群分けの一式は列の符号化と群の中心 (モデルの入力は群を作る列を一式の式で符号化したもの)
  const group = description.group
  const unit = items ? '件' : '行'
  return (
    <dl className="text-sm">
      {items ? (
        <ItemsRows items={items} />
      ) : ready ? (
        <ReadyRows ready={ready} />
      ) : rank ? (
        <RankRows rank={rank} />
      ) : group ? (
        <GroupRows group={group} />
      ) : noise ? (
        <Row label="データ">雑音だけ (データセットなし. 推論は件数と種から作る)</Row>
      ) : (
        <Row label="データとタスク">
          {KIND_LABELS[description.kind] ?? description.kind} /{' '}
          {TASK_LABELS[description.task] ?? description.task}
          {description.target && ` (目的の列 ${description.target})`}
        </Row>
      )}
      {description.classes.length > 0 && (
        <Row label={`クラス (${description.classes.length.toLocaleString()})`}>
          {classesText(description.classes)}
        </Row>
      )}
      {description.features.length > 0 && (
        <Row label={`学習に使う列 (${description.features.length})`}>
          <span className="font-mono text-xs">{description.features.join(', ')}</span>
        </Row>
      )}
      {description.text_column && <Row label="文の列">{description.text_column}</Row>}
      {!items && !noise && !ready && !rank && !group && (
        <>
          <Row label="読み込み">
            <span className="font-mono text-xs">
              {description.reading.loader ?? '自動'}{' '}
              {Object.keys(description.reading.options).length > 0 &&
                `(${optionsText(description.reading.options)})`}
            </span>
          </Row>
          <Row label="前処理">
            <span className="font-mono text-xs">
              {[...description.steps, ...description.item_steps].join(' → ') || 'なし'}
            </span>
          </Row>
        </>
      )}
      {!rank && !group && (
        <Row label="モデルの入力">
          <span className="font-mono text-xs">
            {inputs
              .map(
                ([name, plan]) =>
                  `${name} ← ${SHORT_SOURCE_LABELS[plan.source] ?? plan.source}${
                    plan.prepend.length > 0 ? ` (先頭に ${plan.prepend.join(', ')})` : ''
                  }`,
              )
              .join(', ')}
          </span>
        </Row>
      )}
      {run && (
        <Row label="学習の結果">
          {run.status ?? '状態不明'}
          {run.monitor && run.best_value != null && (
            <>
              , {run.monitor} {formatNumber(run.best_value)} (エポック {run.best_epoch ?? '–'})
            </>
          )}
        </Row>
      )}
      {description.check && (
        <Row label="一式での評価">
          {description.check.split} の {description.check.rows.toLocaleString()} {unit}で{' '}
          {METRIC_LABELS[description.check.metric] ?? description.check.metric}{' '}
          {formatNumber(description.check.value)}
          {description.check.recorded != null && (
            <>
              {' '}
              (実験の記録 {formatNumber(description.check.recorded)}.{' '}
              {description.check.matches ? (
                <span className="text-good-text">合う</span>
              ) : (
                <span className="text-critical-text">合わない</span>
              )}
              )
            </>
          )}
        </Row>
      )}
      {description.uncertainty && (
        <Row label="外れ方の記録">
          <UncertaintyText summary={description.uncertainty} />
        </Row>
      )}
      {description.notes.map((note) => (
        <p key={note} className="mt-1 text-xs text-warning-text">
          {note}
        </p>
      ))}
    </dl>
  )
}

/**
 * 推論するデータと結論の決め方を選んで推論する.
 * 問題の設定をつないだ一式は, 意思決定と結論の決め方の既定を問題の設定の値にする
 */
function PredictCard({
  bundle,
  description,
  linked,
}: {
  bundle: Bundle
  description: Description
  linked: LinkedProblem | null
}) {
  const navigate = useNavigate()
  const client = useQueryClient()
  const textual = description.kind === 'text' && !!description.text_column
  // 雑音だけの一式はファイルと文の代わりに件数と種 (種は空なら一式の手順の種)
  const noise = description.kind === 'noise'
  // 既製のモデルの一式は画像だけで, 意思決定を使えない (結論の決め方は使える)
  const readyModel = description.kind === 'ready'
  // 群分けの一式は群を作る列を持つ表のファイルだけで, 結論は群の要約
  const group = description.group
  const info = useQuery({ queryKey: ['info'], queryFn: api.info, staleTime: Infinity })
  const maxSamples = info.data?.max_samples ?? 1000
  const [count, setCount] = useState('4')
  const [noiseSeed, setNoiseSeed] = useState('')
  // 列の出力でない一式 (領域, 点群, ベクトル, 画像, 文) は結論が件数だけ
  const column = isColumnOutput(description)
  const outputForm = description.output?.form ?? ''
  const [mode, setMode] = useState<'file' | 'texts'>('file')
  const [source, setSource] = useState('')
  const [texts, setTexts] = useState('')
  // 目的の列の過去の値を使う回帰と予測の一式は, 何段も先まで予測できる
  const canForecast = column && !!description.forecastable
  const [ahead, setAhead] = useState(false)
  // 推論の結論を扱わない受け持ちの問題の設定は, 一式の既定値のまま
  const defaults =
    linked && linked.handled
      ? { conclusion: linked.conclusion, decision: linked.decision }
      : { conclusion: description.conclusion, decision: description.decision }
  const [form, setForm] = useState<RulesForm>(() => toForm(defaults.conclusion))
  const { rules, problems: ruleProblems } = fromForm(form, description.task, description.classes)
  const [decisionForm, setDecisionForm] = useState<DecisionForm>(() =>
    toDecisionForm(defaults.decision),
  )
  // つないだ問題の設定が変わったら (つなぐ, 外す, 中身を直す), 既定を読み直す
  const defaultsKey = linked ? linked.version : 'bundle'
  const [seenKey, setSeenKey] = useState(defaultsKey)
  if (seenKey !== defaultsKey) {
    setSeenKey(defaultsKey)
    setForm(toForm(defaults.conclusion))
    setDecisionForm(toDecisionForm(defaults.decision))
  }
  const { decision, problems: decisionProblems } = fromDecisionForm(
    decisionForm,
    description.classes,
  )
  const problems = column ? [...ruleProblems, ...(readyModel ? [] : decisionProblems)] : []
  if (noise) {
    const number = Number(count)
    if (!/^\d+$/.test(count.trim()) || number < 1 || number > maxSamples)
      problems.push(`件数は 1 から ${maxSamples} の整数で書いてください`)
    if (noiseSeed.trim() && !/^-?\d+$/.test(noiseSeed.trim()))
      problems.push('種は整数で書いてください')
    else if (noiseSeed.trim() && !Number.isSafeInteger(Number(noiseSeed)))
      problems.push(SEED_RANGE_PROBLEM)
  }
  const predict = useMutation({
    // 送ったときの欄の印 (欄を変えたら前の誤りを出さないため)
    mutationFn: (_key: string) =>
      api.predict(
        bundle.id,
        noise
          ? {
              count: Number(count),
              ...(noiseSeed.trim() ? { seed: Number(noiseSeed) } : {}),
            }
          : !column
            ? mode === 'texts'
              ? { texts: texts.split('\n') }
              : { source: source.trim() }
            : readyModel
              ? { source: source.trim(), rules: withCleared(rules) }
              : mode === 'texts'
                ? { texts: texts.split('\n'), rules: withCleared(rules), decision: decision ?? {} }
                : {
                    source: source.trim(),
                    rules: withCleared(rules),
                    forecast: canForecast && ahead,
                    // 意思決定しない場合は空 (省くと一式の既定値になる)
                    decision: decision ?? {},
                  },
      ),
    onSuccess: (result) => {
      client.invalidateQueries({ queryKey: ['results'] })
      navigate(`/results/${result.id}`)
    },
  })
  // 欄の中身の印 (送った後に欄を変えたら, 前の誤りを消す)
  const formKey = JSON.stringify([mode, source, texts, count, noiseSeed, ahead, form, decisionForm])
  const ready =
    (noise || (mode === 'texts' ? texts.trim() : source.trim())) && problems.length === 0
  return (
    <Card title="推論する">
      <form
        className="flex flex-col gap-4"
        onSubmit={(event) => {
          event.preventDefault()
          if (ready) predict.mutate(formKey)
        }}
      >
        {textual && (
          <div role="radiogroup" aria-label="推論するデータ" className="flex gap-4 text-sm">
            <label className="flex items-center gap-1">
              <input type="radio" checked={mode === 'file'} onChange={() => setMode('file')} />
              ファイル
            </label>
            <label className="flex items-center gap-1">
              <input type="radio" checked={mode === 'texts'} onChange={() => setMode('texts')} />
              文を書く
            </label>
          </div>
        )}
        {noise ? (
          <fieldset className="grid gap-3 md:grid-cols-2">
            <legend className="sr-only">作る件数と種</legend>
            <div>
              <label htmlFor="predict-count" className="mb-1 block text-xs font-medium text-ink-2">
                作る件数 ({maxSamples} 件まで)
              </label>
              <input
                id="predict-count"
                type="number"
                min={1}
                max={maxSamples}
                value={count}
                onChange={(event) => setCount(event.target.value)}
                className="h-8 w-full rounded-lg border border-line bg-surface px-2 text-sm"
              />
            </div>
            <div>
              <label htmlFor="predict-seed" className="mb-1 block text-xs font-medium text-ink-2">
                乱数の種 (空なら一式の種 {description.procedure?.seed ?? 0})
              </label>
              <input
                id="predict-seed"
                type="number"
                value={noiseSeed}
                onChange={(event) => setNoiseSeed(event.target.value)}
                className="h-8 w-full rounded-lg border border-line bg-surface px-2 text-sm"
              />
            </div>
            <p className="text-xs text-ink-3 md:col-span-2">
              同じ件数と種なら同じ出力になります. 始まりの雑音を全ての件の分を一度に作るので, 1
              件の形が大きいと受け付ける件数が減ります
            </p>
            {problems.map((problem) => (
              <p key={problem} role="alert" className="text-xs text-critical-text md:col-span-2">
                {problem}
              </p>
            ))}
          </fieldset>
        ) : mode === 'file' ? (
          <div>
            <PathField
              id="predict-source"
              label={
                readyModel
                  ? '推論する画像 (画像のファイルか, 画像を並べたディレクトリ)'
                  : group
                    ? `群に割り当てる行の表 (群を作る列 ${group.columns.join(', ')} を持つファイル)`
                    : sourceLabel(description.items?.input_kind, description.kind)
              }
              value={source}
              onChange={setSource}
              hint={
                readyModel
                  ? '重みの説明の前処理 (大きさ, 切り出し, 正規化) をかけ, 種類の名前と確信度を出します. 読めない画像は理由を付けて飛ばします'
                  : group
                    ? '行ごとに学習のときと同じ式で列を符号化し, 最も近い群の中心に割り当てます. 数の列を数にできない行は理由を付けて飛ばします'
                    : description.items
                      ? '学習のときと同じ 1 件の扱いをかけます (撮影は同じ正規化も). 読めない件は理由を付けて飛ばします'
                      : '目的の列は有っても無くてもかまいません. 学習のときと同じ読み込みで読みます'
              }
            />
            {bundle.sample_input && (
              <p className="mt-1 text-xs text-ink-2">
                見本の一式です.{' '}
                <button
                  type="button"
                  className="text-accent-text underline"
                  onClick={() => {
                    // 時系列の見本は先の行を持つので, 先まで予測するのが既定
                    setSource(bundle.sample_input ?? '')
                    if (canForecast) setAhead(true)
                  }}
                >
                  {description.items
                    ? '見本の新しいデータ (画像を並べたディレクトリ) を使う'
                    : '見本の新しいデータ (目的の列の無い表) を使う'}
                </button>
                {bundle.sample_past && (
                  <>
                    {' '}
                    ·{' '}
                    <button
                      type="button"
                      className="text-accent-text underline"
                      onClick={() => {
                        // 過去のデータは目的の列があるので, 1 つ先の予測でたどり直せる
                        setSource(bundle.sample_past ?? '')
                        setAhead(false)
                      }}
                    >
                      見本の過去のデータ (目的の列あり. たどり直せる) を使う
                    </button>
                  </>
                )}
              </p>
            )}
          </div>
        ) : (
          <div>
            <label htmlFor="predict-texts" className="mb-1 block text-xs font-medium text-ink-2">
              推論する文 (1 行に 1 つ)
            </label>
            <textarea
              id="predict-texts"
              value={texts}
              onChange={(event) => setTexts(event.target.value)}
              rows={6}
              className="w-full rounded-lg border border-line bg-surface p-2 text-sm"
            />
          </div>
        )}
        {canForecast && mode === 'file' && (
          <div className="flex items-start gap-2 text-sm">
            <input
              id="predict-ahead"
              type="checkbox"
              className="mt-1"
              checked={ahead}
              aria-describedby="predict-ahead-note"
              onChange={(event) => setAhead(event.target.checked)}
            />
            <div>
              <label htmlFor="predict-ahead">何段も先まで予測する</label>
              {linked?.horizon != null && linked.horizon > 1 && (
                <p className="text-xs text-ink-2">
                  問題の設定では {linked.horizon} 期先まで予測します
                </p>
              )}
              <p id="predict-ahead-note" className="text-xs text-ink-3">
                目的の列が空の先の行を,
                予測した値を次の行の過去の値に使いながら系列ごとに順に予測します.
                結果は先の行だけで, 何段目かを step に出します (先の段ほど誤りが積み重なります)
              </p>
            </div>
          </div>
        )}
        {group && (
          <p className="text-xs text-ink-2">
            {CLUSTER_UNSUPPORTED}. {CLUSTER_NO_PROBLEM}
          </p>
        )}
        {!column && !group && (
          <p className="text-xs text-ink-2">
            この一式の出力は{FORM_LABELS[outputForm] ?? outputForm}なので,
            結論は推論した件数だけです. 結論の決め方と意思決定は使えません
          </p>
        )}
        {readyModel && <p className="text-xs text-ink-2">{READY_UNSUPPORTED}</p>}
        {column && !readyModel && (
          <section aria-label="意思決定">
            <h3 className="mb-2 text-xs font-semibold text-ink-2">
              意思決定 (任意. だから何をするか. 結果の画面で変えて決め直せます)
            </h3>
            {linked && (
              <p className="mb-2 text-xs text-ink-3">
                {!linked.handled
                  ? `「${linked.name}」: ${linked.text}`
                  : linked.decision
                    ? `問題の設定「${linked.name}」の損得と結論の決め方を既定にしています`
                    : `問題の設定「${linked.name}」は意思決定しない (予測を見て人が決める) ので, 意思決定は空にしています`}
              </p>
            )}
            <DecisionEditor
              form={decisionForm}
              onChange={setDecisionForm}
              task={description.task}
              classes={description.classes}
              idBase="predict-decision"
            />
          </section>
        )}
        {column && (
          <section aria-label="結論の決め方">
            <h3 className="mb-2 text-xs font-semibold text-ink-2">
              結論の決め方 (結果の画面で変えて出し直せます)
            </h3>
            <RulesEditor
              form={form}
              onChange={setForm}
              task={description.task}
              classes={description.classes}
              columns={[]}
              suggestions={[
                ...(description.text_column ? [description.text_column] : []),
                ...description.features,
              ]}
              idBase="predict"
            />
            {problems.map((problem) => (
              <p key={problem} role="alert" className="mt-1 text-xs text-critical-text">
                {problem}
              </p>
            ))}
          </section>
        )}
        {predict.error && predict.variables === formKey && <ErrorBox error={predict.error} />}
        <div className="flex justify-end">
          <Button
            type="submit"
            variant="primary"
            icon={Play}
            disabled={!ready || predict.isPending}
          >
            {predict.isPending ? '推論しています…' : '推論する'}
          </Button>
        </div>
      </form>
    </Card>
  )
}

/**
 * 並べるの一式で, 利用者 (書くか表のファイル), 件数 K, 前に反応した品物を除くか,
 * 正解の反応の表 (任意) を選んで並べる. 結論の決め方と意思決定は使えない
 */
function RankPredictCard({
  bundle,
  rank,
}: {
  bundle: Bundle
  rank: NonNullable<Description['rank']>
}) {
  const navigate = useNavigate()
  const client = useQueryClient()
  const [mode, setMode] = useState<'users' | 'file'>('users')
  const [users, setUsers] = useState('')
  const [source, setSource] = useState('')
  const [k, setK] = useState(String(rank.k))
  const [exclude, setExclude] = useState(false)
  const [seen, setSeen] = useState('')
  const [truth, setTruth] = useState('')
  const written = rankUsers(users)
  const problems: string[] = []
  const kProblem = rankKProblem(k)
  if (kProblem) problems.push(kProblem)
  if (exclude && !seen.trim())
    problems.push('前に反応した品物を除くときは, 前に反応した品物の表を選んでください')
  const columns = rankColumnsText(rank.columns)
  const predict = useMutation({
    // 送ったときの欄の印 (欄を変えたら前の誤りを出さないため)
    mutationFn: (_key: string) =>
      api.predict(bundle.id, {
        ...(mode === 'users' ? { users: written } : { source: source.trim() }),
        ...(k.trim() ? { k: Number(k) } : {}),
        ...(exclude ? { seen: seen.trim() } : {}),
        ...(truth.trim() ? { truth: truth.trim() } : {}),
      }),
    onSuccess: (result) => {
      client.invalidateQueries({ queryKey: ['results'] })
      navigate(`/results/${result.id}`)
    },
  })
  const formKey = JSON.stringify([mode, users, source, k, exclude, seen, truth])
  const ready =
    (mode === 'users' ? written.length > 0 : source.trim() !== '') && problems.length === 0
  return (
    <Card title="利用者ごとに並べる">
      <form
        className="flex flex-col gap-4"
        onSubmit={(event) => {
          event.preventDefault()
          if (ready) predict.mutate(formKey)
        }}
      >
        <div role="radiogroup" aria-label="並べる利用者" className="flex gap-4 text-sm">
          <label className="flex items-center gap-1">
            <input type="radio" checked={mode === 'users'} onChange={() => setMode('users')} />
            利用者を書く
          </label>
          <label className="flex items-center gap-1">
            <input type="radio" checked={mode === 'file'} onChange={() => setMode('file')} />
            利用者の表のファイル
          </label>
        </div>
        {mode === 'users' ? (
          <div>
            <label htmlFor="rank-users" className="mb-1 block text-xs font-medium text-ink-2">
              並べる利用者 (1 行に 1 人. 学習用の反応にいた利用者の値)
            </label>
            <textarea
              id="rank-users"
              value={users}
              onChange={(event) => setUsers(event.target.value)}
              rows={5}
              className="w-full rounded-lg border border-line bg-surface p-2 font-mono text-sm"
            />
          </div>
        ) : (
          <PathField
            id="rank-source"
            label={`利用者の表 (利用者の列 ${rank.columns.user} を持つ CSV, TSV, Parquet)`}
            value={source}
            onChange={setSource}
            pickFiles
            hint="表の上の行の利用者から順に並べ, 結果の表もその順になります. 値は文字のまま読みます (007 は 7 になりません)"
          />
        )}
        <div className="max-w-xs">
          <label htmlFor="rank-k" className="mb-1 block text-xs font-medium text-ink-2">
            並べる件数 K (既定は一式の {rank.k} 件. {MAX_RANK_K} 件まで)
          </label>
          <input
            id="rank-k"
            type="number"
            min={1}
            max={MAX_RANK_K}
            value={k}
            onChange={(event) => setK(event.target.value)}
            className="h-8 w-full rounded-lg border border-line bg-surface px-2 text-sm"
          />
        </div>
        <div className="flex flex-col gap-2">
          <div className="flex items-start gap-2 text-sm">
            <input
              id="rank-exclude"
              type="checkbox"
              className="mt-1"
              checked={exclude}
              aria-describedby="rank-exclude-note"
              onChange={(event) => setExclude(event.target.checked)}
            />
            <div>
              <label htmlFor="rank-exclude">前に反応した品物を除く</label>
              <p id="rank-exclude-note" className="text-xs text-ink-3">
                利用者ごとに, 前に反応した品物の表にある品物を除いて並べます.
                除くと品物が残らない利用者は, 理由を付けて並べません
              </p>
            </div>
          </div>
          {exclude && (
            <PathField
              id="rank-seen"
              label={`前に反応した品物の表 (列 ${columns})`}
              value={seen}
              onChange={setSeen}
              pickFiles
              hint={reactionRuleText(rank.columns)}
            />
          )}
        </div>
        <PathField
          id="rank-truth"
          label={`正解の反応の表 (任意. 並べた後の反応. 列 ${columns})`}
          value={truth}
          onChange={setTruth}
          pickFiles
          hint="渡すと, 上位 K 件に入った正解から Recall@K と NDCG を数えます (前に反応した品物を除くときは, その品物を正解から除きます)"
        />
        <p className="text-xs text-ink-2">
          {RANK_UNSUPPORTED}. {RANK_NO_PROBLEM}
        </p>
        {problems.map((problem) => (
          <p key={problem} role="alert" className="text-xs text-critical-text">
            {problem}
          </p>
        ))}
        {predict.error && predict.variables === formKey && <ErrorBox error={predict.error} />}
        <div className="flex justify-end">
          <Button
            type="submit"
            variant="primary"
            icon={Play}
            disabled={!ready || predict.isPending}
          >
            {predict.isPending ? '並べています…' : '並べる'}
          </Button>
        </div>
      </form>
    </Card>
  )
}

/** 一式の説明, 推論, 過去の結果 */
export function BundlePage() {
  const { id = '' } = useParams()
  const navigate = useNavigate()
  const client = useQueryClient()
  const bundle = useQuery({ queryKey: ['bundles', id], queryFn: () => api.bundle(id) })
  const results = useQuery({ queryKey: ['results', id], queryFn: () => api.results(id) })
  // 雑音だけの一式, 並べるの一式, 群分けの一式は, 問題の設定の結論と意思決定を扱わないので問題の設定を読まない
  // (既製のモデルの一式は読み, 結論の決め方だけを既定にする).
  // 種類は一式か一覧の読んだ値から. 分からない間は一式と並べて読む (推論の欄を 1 往復早く出すため.
  // 読まない種類の一式なら, サーバーは問題の設定を読まずに空の状態を返す)
  const listed = client.getQueryData<Bundle[]>(['bundles'])?.find((item) => item.id === id)
  const kind = bundle.data?.description?.kind ?? listed?.description?.kind
  const noise = kind === 'noise' || kind === 'rank' || kind === 'group'
  const problem = useQuery({
    queryKey: ['bundle-problem', id],
    queryFn: () => api.bundleProblem(id),
    enabled: !noise,
  })
  const remove = useMutation({
    mutationFn: () => api.deleteBundle(id),
    onSuccess: () => {
      // 消した一式は取り直さずに外し, 一覧だけを読み直す
      client.removeQueries({ queryKey: ['bundles', id] })
      client.removeQueries({ queryKey: ['bundle-problem', id] })
      client.invalidateQueries({ queryKey: ['bundles'], exact: true })
      navigate('/')
    },
  })
  if (bundle.isPending) return <Spinner />
  if (bundle.error) return <ErrorBox error={bundle.error} />
  const data = bundle.data
  return (
    <>
      <PageHeader
        title={data.name}
        subtitle={
          <span className="font-mono">
            {data.path}
            {data.copied_from && (
              <span className="font-sans"> (写した元の学習の実験: {data.copied_from})</span>
            )}
          </span>
        }
        breadcrumb={<Link to="/">推論の一式</Link>}
        actions={
          <Button
            variant="danger"
            icon={Trash2}
            onClick={() => {
              const message = data.copied_from
                ? 'この一式を消しますか? (学習の実験から写した一式なので, 写しを消します. 写した元の実験の一式は残ります. 結果は残ります)'
                : data.managed
                  ? 'この一式を消しますか? (このアプリが作った一式なので, ファイルも消します. 結果は残ります)'
                  : '登録を外しますか? (一式のファイルは残ります)'
              if (window.confirm(message)) remove.mutate()
            }}
          >
            {data.managed ? '消す' : '登録を外す'}
          </Button>
        }
      />
      {remove.error && <ErrorBox error={remove.error} />}
      {!data.ok || !data.description ? (
        <ErrorBox error={`一式を読めません: ${data.error ?? ''}`} />
      ) : (
        <div className="grid gap-4 xl:grid-cols-[1fr_22rem]">
          {/* 実験の記録と合わない一式は, 推論の前に気付けるように上に出す */}
          {data.description.check?.matches === false && (
            <div className="xl:col-span-2">
              <ErrorBox error="この一式で推論した検証用の評価が, 実験の記録と合いません. 予測が誤っている可能性があります (下の「一式の説明」を参照)" />
            </div>
          )}
          <div className="flex min-w-0 flex-col gap-4">
            {data.description.ready && <ReadyNotices ready={data.description.ready} />}
            <OutputNotices
              target={data.description.target_info}
              warnings={data.description.pretrained_warnings}
            />
            {!noise && problem.isPending && (
              <p role="status" className="text-xs text-ink-3">
                問題の設定を読んでいます…
              </p>
            )}
            {!noise && problem.error && (
              <ErrorBox error={problem.error} live={false}>
                <p className="mt-1 text-xs text-ink-2">
                  推論の既定は一式の既定値です.{' '}
                  <button
                    type="button"
                    className="text-accent-text underline"
                    onClick={() => void problem.refetch()}
                  >
                    読み直す
                  </button>
                </p>
              </ErrorBox>
            )}
            {!noise && problem.data && (
              <ProblemCard
                bundleId={data.id}
                state={problem.data}
                readyModel={data.description.kind === 'ready'}
              />
            )}
            {/* 問題の設定を読み終えてから, その値を既定にして出す */}
            {data.description.rank ? (
              <RankPredictCard bundle={data} rank={data.description.rank} />
            ) : (
              (noise || !problem.isPending) && (
                <PredictCard
                  bundle={data}
                  description={data.description}
                  linked={noise ? null : (problem.data?.linked ?? null)}
                />
              )
            )}
            <Card title="一式の説明">
              <DescriptionView description={data.description} />
              {data.description.uncertainty?.restorable && <RestoreButton bundle={data} />}
            </Card>
            {data.description.group && <GroupLearnedCard group={data.description.group} />}
            {/* 版 3 からの項目 (版 2 までの一式は前と同じ画面) */}
            {(data.description.format_version ?? 0) >= 3 && (
              <Card title="受け持ちと出力">
                <OutputView description={data.description} modelLoaded={!!data.model_loaded} />
              </Card>
            )}
          </div>
          <CompareResultsCard
            title="この一式の結果"
            results={results}
            showBundle={false}
            hint={`結果を 2 つから ${MAX_COMPARE} つ選ぶと, 結論, 判断の内訳, 意思決定を並べて比べます.`}
          />
        </div>
      )}
    </>
  )
}

/** 分割の名前 (recent は測り直した実績) */
const SPLIT_LABELS: Record<string, string> = {
  valid: '検証用の分割',
  train: '学習用の分割',
  recent: '測り直した実績',
}

/** 外れ方の記録の要約の文 */
function UncertaintyText({ summary }: { summary: UncertaintySummary }) {
  const where = SPLIT_LABELS[summary.split ?? ''] ?? summary.split ?? '–'
  const when = summary.measured_at ? ` (${summary.measured_at.slice(0, 16).replace('T', ' ')})` : ''
  return (
    <>
      {where}
      {when}
      {summary.rows != null && ` の ${summary.rows.toLocaleString()} 行`}で測った
      {summary.kind && `. ${summary.kind === 'relative' ? '比' : '差'}の形`}
      {summary.coverage &&
        `. 80 % の幅がテスト用の分割を覆った割合 ${(summary.coverage['80'] * 100).toFixed(1)} %`}
      {summary.ece != null && `. 確率のずれ ${(summary.ece * 100).toFixed(1)} ポイント`}
      {!!summary.steps_measured && `. 何段も先の外れ方は ${summary.steps_measured} 段目まで段ごと`}
      {!!summary.horizon &&
        !summary.steps_measured &&
        `. 何段も先の外れ方は ${summary.horizon} 段先まで測ろうとしたが, 誤りが少なく測れなかった`}
      {summary.groups &&
        `. ${summary.groups.column ?? '–'} ごとに ${summary.groups.count} 種類で測った (20 行より少ない種類は全ての行の外れ方)`}
    </>
  )
}

/** 測り直した外れ方を, 一式を作ったときの記録に戻すボタン */
function RestoreButton({ bundle }: { bundle: Bundle }) {
  const client = useQueryClient()
  const restore = useMutation({
    mutationFn: () => api.restoreUncertainty(bundle.id),
    onSuccess: () => client.invalidateQueries({ queryKey: ['bundles'] }),
  })
  return (
    <div className="mt-2">
      <Button
        icon={Undo2}
        disabled={restore.isPending}
        onClick={() =>
          window.confirm('外れ方の記録を, 一式を作ったときの記録に戻しますか?') && restore.mutate()
        }
      >
        {restore.isPending ? '戻しています…' : '外れ方を作ったときの記録に戻す'}
      </Button>
      {restore.error && <ErrorBox error={restore.error} />}
    </div>
  )
}
