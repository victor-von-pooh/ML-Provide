import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Play, Trash2, Undo2 } from 'lucide-react'
import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router'

import { api } from '../../api/client'
import type { Bundle, Description, LinkedProblem, UncertaintySummary } from '../../api/types'
import { PathField } from '../../components/PathPicker'
import { Button, Card, ErrorBox, PageHeader, Spinner } from '../../components/ui'
import { SEED_RANGE_PROBLEM, SHORT_SOURCE_LABELS } from '../../lib/bundleForm'
import {
  ANOMALY_NO_PROBLEM,
  ANOMALY_UNSUPPORTED,
  capacityProblem,
  MAX_CAPACITY,
  normalText,
  SCORE_LABELS,
} from '../../lib/anomaly'
import { CLUSTER_NO_PROBLEM, CLUSTER_UNSUPPORTED } from '../../lib/clusters'
import {
  budgetProblem,
  MAX_BUDGET,
  distributionText,
  effectText,
  GROUPS_TEXT,
  LEARNER_LABELS,
  OUTCOME_LABELS,
  PREMISE_TEXT,
  topText,
  UPLIFT_NO_PROBLEM,
  UPLIFT_UNSUPPORTED,
} from '../../lib/uplift'
import {
  candidatesInput,
  candidatesText,
  CHANGE_TEXT,
  changeText,
  costText,
  CURRENT_LABELS,
  elasticityText,
  limitText,
  MAX_CANDIDATES,
  MAX_CHANGE_TEXT,
  outsideCandidates,
  parseCandidates,
  parseMaxChange,
  PRICE_NO_PROBLEM,
  PRICE_TEXT,
  PRICE_UNSUPPORTED,
  UNITS_TEXT,
} from '../../lib/price'
import {
  baselineText,
  DEMAND_NO_PROBLEM,
  DEMAND_UNSUPPORTED,
  demandText,
  HOLDING_TEXT,
  levelsText,
  levelText,
  MAX_ORDER_TEXT,
  outsideLevel,
  parseCost,
  parseMaxOrder,
  QUANTILE_TEXT,
  SHORTAGE_TEXT,
  stockText,
  tauOf,
} from '../../lib/demand'
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
import { AnomalyEvaluationsTable } from '../results/AnomalyCards'
import { UpliftEvaluationsView } from '../results/UpliftCards'
import { PriceEvaluationsView, PriceTerms } from '../results/PriceCards'
import { DemandEvaluationsView, DemandTerms } from '../results/DemandCards'
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

/** 群分けの一式 (版 9 と 10) の列の符号化と群の中心, 群を作る列 (版 10 は群の要約の列), 群の数 */
function GroupRows({ group }: { group: NonNullable<Description['group']> }) {
  // 既製の埋め込みの群分けの一式 (版 10) は列を符号化しないので, 列は群の要約の列だけ
  const ready = group.features === null
  return (
    <>
      {/* 版 10 のデータの行は既製のモデルの行 (ReadyRows) */}
      {!ready && (
        <Row label="データ">
          列の符号化と群の中心 (データセットなし. 推論は表の行を最も近い群に割り当てる)
        </Row>
      )}
      <Row label={`${ready ? '群の要約の列' : '群を作る列'} (${group.columns.length})`}>
        <span className="font-mono text-xs">{group.columns.join(', ')}</span>
        <span className="block text-xs text-ink-2">
          {ready
            ? '群の要約で群ごとの特徴を見る写真の列です. 群は既製のモデルの埋め込みで分け, この列は符号化しません'
            : '推論する表はこの列を持ちます. 数の列は学習のときの平均と標準偏差で, 種類の列は学習のときの値ごとに符号化します (学習に無い値はどの値でもない扱い)'}
        </span>
      </Row>
      <Row label="群の数">
        {group.k} 個 (学習で {group.rows.toLocaleString()} 行を分けた.{' '}
        {group.features !== null && `符号化した行の幅 ${group.features}, `}埋め込みの幅{' '}
        {group.width})
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

/** 異常の度合いの一式 (版 11) の度合いの決め方, 列, 前処理, 件数の上限, 正常の行の分布, 確かめ */
function AnomalyRows({
  anomaly,
  steps,
}: {
  anomaly: NonNullable<Description['anomaly']>
  steps: string[]
}) {
  return (
    <>
      <Row label="データ">
        度合いの決め方と前処理のレシピ (データセットなし. 推論は表の行ごとに異常の度合いを出す)
      </Row>
      <Row label="度合いの決め方">
        {SCORE_LABELS[anomaly.score] ?? anomaly.score}
        <span className="block text-xs text-ink-2">大きいほど異常で, 確率ではありません</span>
      </Row>
      <Row label={`度合いに使う列 (${anomaly.features.length})`}>
        <span className="font-mono text-xs">{anomaly.features.join(', ')}</span>
        <span className="block text-xs text-ink-2">
          推論する表に学習のときと同じ前処理をかけた後の列です
        </span>
      </Row>
      <Row label="行の id の列">
        <span className="font-mono text-xs">{anomaly.id}</span>
        <span className="block text-xs text-ink-2">人の確認の記録が行を指す列です</span>
      </Row>
      <Row label="前処理">
        <span className="font-mono text-xs">{steps.join(' → ') || 'なし'}</span>
      </Row>
      <Row label="人の確認に回す件数の上限">
        1 回に {anomaly.capacity.toLocaleString()} 件 (実験の評価の上位 K 件は{' '}
        {anomaly.k.toLocaleString()} 件)
      </Row>
      <Row label="正常の行の度合い">{normalText(anomaly.normal)}</Row>
      {anomaly.check && (
        <Row label="元の表での確かめ">
          学習の元の表 {anomaly.check.rows.toLocaleString()} 行の度合いを一式で出し直し,{' '}
          {(anomaly.check.agreement * 100).toFixed(1)} % が実験の度合いと合う (差の最大{' '}
          {formatNumber(anomaly.check.max_diff)})
        </Row>
      )}
    </>
  )
}

/** 効果の一式 (版 12) の効果の決め方, 結果の種類, 列, 前処理, 配る数の上限, 学習用の行の効果の分布, 確かめ */
function UpliftRows({
  uplift,
  steps,
}: {
  uplift: NonNullable<Description['uplift']>
  steps: string[]
}) {
  return (
    <>
      <Row label="データ">
        効果の決め方と前処理のレシピ (データセットなし. 推論は表の行ごとに効果を見積もる)
      </Row>
      <Row label="効果の決め方">
        {LEARNER_LABELS[uplift.learner] ?? uplift.learner}
        <span className="block text-xs text-ink-2">{effectText(uplift.outcome)}</span>
        <span className="block text-xs text-ink-2">{GROUPS_TEXT}</span>
        <span className="block text-xs text-ink-2">{PREMISE_TEXT}</span>
      </Row>
      <Row label="結果の種類">{OUTCOME_LABELS[uplift.outcome] ?? uplift.outcome}</Row>
      <Row label={`見積もりに使う列 (${uplift.features.length})`}>
        <span className="font-mono text-xs">{uplift.features.join(', ')}</span>
        <span className="block text-xs text-ink-2">
          推論する表に学習のときと同じ前処理をかけた後の列です
        </span>
      </Row>
      <Row label="行の id の列">
        <span className="font-mono text-xs">{uplift.id}</span>
      </Row>
      <Row label="処置の列">
        <span className="font-mono text-xs">{uplift.treatment}</span>
        <span className="block text-xs text-ink-2">
          推論する表には要りません. あれば結果の列と合わせて評価を数え直します
        </span>
      </Row>
      <Row label="前処理">
        <span className="font-mono text-xs">{steps.join(' → ') || 'なし'}</span>
      </Row>
      <Row label="配る数の上限">
        {uplift.budget === null
          ? 'なし (推論の欄で決める)'
          : `${uplift.budget.toLocaleString()} 件`}{' '}
        (実験の評価は{topText(uplift.k)})
      </Row>
      <Row label="学習用の行の効果">{distributionText(uplift.train)}</Row>
      {uplift.check && (
        <Row label="元の表での確かめ">
          学習の元の表 {uplift.check.rows.toLocaleString()} 行の効果を一式で出し直し,{' '}
          {(uplift.check.agreement * 100).toFixed(1)} % が実験の効果と合う (差の最大{' '}
          {formatNumber(uplift.check.max_diff)})
        </Row>
      )}
    </>
  )
}

/** 値段の一式 (版 13) の列, 前処理, 値段の候補と変える幅の上限, 学習用の分割の要約, 確かめ */
function PriceRows({
  price,
  steps,
}: {
  price: NonNullable<Description['price']>
  steps: string[]
}) {
  return (
    <>
      <Row label="データ">
        値段の決め方と前処理のレシピ (データセットなし.
        推論は表の行ごとに候補の値段で売れた数を予測し, 値段を選ぶ)
      </Row>
      <Row label={`予測に使う列 (${price.features.length})`}>
        <span className="font-mono text-xs">{price.features.join(', ')}</span>
        <span className="block text-xs text-ink-2">
          推論する表に学習のときと同じ前処理をかけた後の列です. 最後が値段の列です
        </span>
      </Row>
      <Row label="値段の列">
        <span className="font-mono text-xs">{price.price}</span>
        <span className="block text-xs text-ink-2">{PRICE_TEXT}</span>
      </Row>
      <Row label="売れた数の列">
        <span className="font-mono text-xs">{price.units}</span>
        <span className="block text-xs text-ink-2">
          {UNITS_TEXT}. 推論する表には要りません. あれば記録の値段での予測の誤差を数え直します
        </span>
      </Row>
      <Row label="原価の列">
        <span className="font-mono text-xs">{price.cost ?? 'なし'}</span>
        <span className="block text-xs text-ink-2">{costText(price.cost)}</span>
      </Row>
      {price.product && (
        <Row label="商品の id の列">
          <span className="font-mono text-xs">{price.product}</span>
        </Row>
      )}
      <Row label="行の id の列">
        <span className="font-mono text-xs">{price.id}</span>
      </Row>
      <Row label="前処理">
        <span className="font-mono text-xs">{steps.join(' → ') || 'なし'}</span>
      </Row>
      <Row label="値段の候補 (変える割合)">
        <span className="tabular">{candidatesText(price.candidates)}</span>
        <span className="block text-xs text-ink-2">{CHANGE_TEXT}</span>
      </Row>
      <Row label="変える幅の上限">
        <span className="tabular">{limitText(price.max_change)}</span>
        <span className="block text-xs text-ink-2">{MAX_CHANGE_TEXT}</span>
      </Row>
      <Row label="実験の今の値段">
        {CURRENT_LABELS[price.current] ?? price.current}
        <span className="block text-xs text-ink-2">
          実験で弾力性と利益を数えるときの今の値段 (商品ごとの記録の値段の中央値, または行の値段)
          です
        </span>
      </Row>
      <Row label="学習用の分割">
        商品 {formatNumber(price.train.products)}, 値段を変える商品{' '}
        {formatNumber(price.train.changed)}, 弾力性の平均{' '}
        {price.train.elasticity === null ? '–' : `${formatNumber(price.train.elasticity)} %`}
        <span className="block text-xs text-ink-2">{elasticityText(price.elasticity_step)}</span>
      </Row>
      {price.check && (
        <Row label="元の表での確かめ">
          学習の元の表の {price.check.rows.toLocaleString()}{' '}
          {price.product ? '商品 (商品の id の列で照らした)' : '行 (行の id の列で照らした)'}{' '}
          で一式が値段を選び直し, {(price.check.agreement * 100).toFixed(1)} %
          が実験の選んだ割合と合う (1 行あたりの見込みの利益の差の最大{' '}
          {formatNumber(price.check.max_diff)})
        </Row>
      )}
    </>
  )
}

/** 需要の一式 (版 14) の列, 前処理, 先の期間と学んだ水準, 発注の損と上限, 学習用の分割の要約, 確かめ */
function DemandRows({
  demand,
  steps,
}: {
  demand: NonNullable<Description['demand']>
  steps: string[]
}) {
  return (
    <>
      <Row label="データ">
        需要の決め方と前処理のレシピ (データセットなし.
        推論は日ごとの記録の表の商品ごとに先の期間の需要の分位点を予測し, 発注の数を決める)
      </Row>
      <Row label={`予測に使う列 (${demand.features.length})`}>
        <span className="font-mono text-xs">{demand.features.join(', ')}</span>
        <span className="block text-xs text-ink-2">
          推論する表に学習のときと同じ前処理 (過去の売れた数の列を日ごとにずらすなど)
          をかけた後の列です
        </span>
      </Row>
      <Row label="商品の id の列">
        <span className="font-mono text-xs">{demand.product}</span>
      </Row>
      <Row label="日付の列">
        <span className="font-mono text-xs">{demand.date}</span>
        <span className="block text-xs text-ink-2">
          日ごとの記録です. 商品ごとに最も遅い日を発注を決める日とします
        </span>
      </Row>
      <Row label="売れた数の列">
        <span className="font-mono text-xs">{demand.units}</span>
        <span className="block text-xs text-ink-2">
          日ごとの売れた数です. 過去の売れた数の列と前の期間の合計に使います
          (発注を決める日の値は要りません)
        </span>
      </Row>
      <Row label="在庫の数の列">
        <span className="font-mono text-xs">{demand.stock ?? 'なし'}</span>
        <span className="block text-xs text-ink-2">{stockText(demand.stock)}</span>
      </Row>
      <Row label="行の id の列">
        <span className="font-mono text-xs">{demand.id}</span>
      </Row>
      <Row label="前処理">
        <span className="font-mono text-xs">{steps.join(' → ') || 'なし'}</span>
      </Row>
      <Row label="先の期間">
        <span className="tabular">{demand.horizon} 日</span>
        <span className="block text-xs text-ink-2">{demandText(demand.horizon)}</span>
      </Row>
      <Row label={`学んだ水準 (${demand.levels.length})`}>
        <span className="tabular">{levelsText(demand.levels)}</span>
        <span className="block text-xs text-ink-2">{QUANTILE_TEXT}</span>
      </Row>
      <Row label="品切れの損と売れ残りの損">
        <span className="tabular">
          {formatNumber(demand.shortage_cost)} と {formatNumber(demand.holding_cost)} (水準 τ{' '}
          {levelText(tauOf(demand.shortage_cost, demand.holding_cost))})
        </span>
        <span className="block text-xs text-ink-2">
          {SHORTAGE_TEXT}. {HOLDING_TEXT}
        </span>
      </Row>
      <Row label="発注の数の上限">
        <span className="tabular">{formatNumber(demand.max_order)}</span>
        <span className="block text-xs text-ink-2">{MAX_ORDER_TEXT}</span>
      </Row>
      {demand.train && (
        <Row label="学習用の分割">
          行 {formatNumber(demand.train.rows)}, 商品 {formatNumber(demand.train.products)},
          先の期間の需要の平均{' '}
          {demand.train.demand_mean === null ? '–' : formatNumber(demand.train.demand_mean)}
        </Row>
      )}
      {demand.check && (
        <Row label="元の表での確かめ">
          学習の元の表の {demand.check.rows.toLocaleString()} 行 (商品と発注を決める日で照らした)
          で一式が発注の数を決め直し, {(demand.check.agreement * 100).toFixed(1)} %
          が実験の発注の数と合う (水準 τ の分位点の差の最大 {formatNumber(demand.check.max_diff)}
          {demand.check.pinball_loss !== null &&
            `. 分位点の損失 ${formatNumber(demand.check.pinball_loss)}`}
          )
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
  // 異常の度合いの一式は度合いの決め方と前処理のレシピ (入力は度合いに使う列を前処理したもの)
  const anomaly = description.anomaly
  // 効果の一式は効果の決め方と前処理のレシピ (入力は見積もりに使う列を前処理したもの)
  const uplift = description.uplift
  // 値段の一式は値段の決め方と前処理のレシピ (入力は予測に使う列を前処理したもの)
  const price = description.price
  // 需要の一式は需要の決め方と前処理のレシピ (入力は予測に使う列を前処理したもの)
  const demand = description.demand
  const unit = items ? '件' : '行'
  return (
    <dl className="text-sm">
      {items ? (
        <ItemsRows items={items} />
      ) : ready ? (
        <>
          <ReadyRows ready={ready} />
          {/* 既製の埋め込みの群分けの一式 (版 10) は群の行も */}
          {group && <GroupRows group={group} />}
        </>
      ) : rank ? (
        <RankRows rank={rank} />
      ) : group ? (
        <GroupRows group={group} />
      ) : anomaly ? (
        <AnomalyRows anomaly={anomaly} steps={description.steps} />
      ) : uplift ? (
        <UpliftRows uplift={uplift} steps={description.steps} />
      ) : price ? (
        <PriceRows price={price} steps={description.steps} />
      ) : demand ? (
        <DemandRows demand={demand} steps={description.steps} />
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
      {description.features.length > 0 && !anomaly && !uplift && !price && !demand && (
        <Row label={`学習に使う列 (${description.features.length})`}>
          <span className="font-mono text-xs">{description.features.join(', ')}</span>
        </Row>
      )}
      {description.text_column && <Row label="文の列">{description.text_column}</Row>}
      {!items &&
        !noise &&
        !ready &&
        !rank &&
        !group &&
        !anomaly &&
        !uplift &&
        !price &&
        !demand && (
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
      {!rank && !group && !anomaly && !uplift && !price && !demand && (
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
  // 異常の度合いの一式は行の表のファイルと, 人の確認に回す件数の上限 (振り分けだけ)
  const anomaly = description.anomaly
  const [capacity, setCapacity] = useState(() =>
    String(description.decision?.capacity ?? anomaly?.capacity ?? ''),
  )
  // 効果の一式は行の表のファイルと, 配る数の上限 (効果の上位から配る判断だけ)
  const uplift = description.uplift
  const [budget, setBudget] = useState(() =>
    String(description.decision?.budget ?? uplift?.budget ?? ''),
  )
  // 値段の一式は行の表のファイルと, 値段の候補と変える幅の上限 (値段の判断だけ. 欄は割合の小数)
  const price = description.price
  const priceDefaults = String(description.decision?.type) === 'price' ? description.decision : null
  const [candidates, setCandidates] = useState(() =>
    candidatesInput((priceDefaults?.candidates as number[] | undefined) ?? price?.candidates ?? []),
  )
  const [maxChange, setMaxChange] = useState(() =>
    String(priceDefaults?.max_change ?? price?.max_change ?? ''),
  )
  const parsedCandidates = parseCandidates(candidates)
  const parsedMaxChange = parseMaxChange(maxChange)
  // 需要の一式は日ごとの記録の表のファイルと, 品切れの損, 売れ残りの損, 発注の数の上限 (発注の判断だけ)
  const demand = description.demand
  const orderDefaults = String(description.decision?.type) === 'order' ? description.decision : null
  const [shortage, setShortage] = useState(() =>
    String(orderDefaults?.shortage_cost ?? demand?.shortage_cost ?? ''),
  )
  const [holding, setHolding] = useState(() =>
    String(orderDefaults?.holding_cost ?? demand?.holding_cost ?? ''),
  )
  const [maxOrder, setMaxOrder] = useState(() =>
    String(orderDefaults?.max_order ?? demand?.max_order ?? ''),
  )
  const parsedShortage = parseCost(shortage, '品切れの損')
  const parsedHolding = parseCost(holding, '売れ残りの損')
  const parsedMaxOrder = parseMaxOrder(maxOrder)
  const tau = tauOf(parsedShortage.value, parsedHolding.value)
  const edge = outsideLevel(tau, demand?.levels ?? [])
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
  const capacityError = anomaly ? capacityProblem(capacity) : null
  if (capacityError) problems.push(capacityError)
  const budgetError = uplift ? budgetProblem(budget) : null
  if (budgetError) problems.push(budgetError)
  if (price) {
    if (parsedCandidates.problem) problems.push(parsedCandidates.problem)
    if (parsedMaxChange.problem) problems.push(parsedMaxChange.problem)
  }
  if (demand) {
    for (const parsed of [parsedShortage, parsedHolding, parsedMaxOrder])
      if (parsed.problem) problems.push(parsed.problem)
  }
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
          : anomaly
            ? { source: source.trim(), decision: { type: 'triage', capacity: Number(capacity) } }
            : uplift
              ? { source: source.trim(), decision: { type: 'treat', budget: Number(budget) } }
              : price
                ? {
                    source: source.trim(),
                    decision: {
                      type: 'price',
                      candidates: parsedCandidates.values,
                      max_change: parsedMaxChange.value ?? 0,
                    },
                  }
                : demand
                  ? {
                      source: source.trim(),
                      decision: {
                        type: 'order',
                        shortage_cost: parsedShortage.value ?? 0,
                        holding_cost: parsedHolding.value ?? 0,
                        max_order: parsedMaxOrder.value ?? 0,
                      },
                    }
                  : !column
                    ? mode === 'texts'
                      ? { texts: texts.split('\n') }
                      : { source: source.trim() }
                    : readyModel
                      ? { source: source.trim(), rules: withCleared(rules) }
                      : mode === 'texts'
                        ? {
                            texts: texts.split('\n'),
                            rules: withCleared(rules),
                            decision: decision ?? {},
                          }
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
  const formKey = JSON.stringify([
    mode,
    source,
    texts,
    count,
    noiseSeed,
    ahead,
    form,
    decisionForm,
    capacity,
    budget,
    candidates,
    maxChange,
    shortage,
    holding,
    maxOrder,
  ])
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
        {/* 値段の一式は, 推論の欄が値段の言葉の初めて出る所なので, 欄の上に言葉の説明と前提 */}
        {price && <PriceTerms cost={price.cost} step={price.elasticity_step} />}
        {/* 需要の一式も同じく, 欄の上に言葉の説明と前提 */}
        {demand && <DemandTerms horizon={demand.horizon} stock={demand.stock} />}
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
                    : anomaly
                      ? `度合いを出す行の表 (前処理に要る元の列と行の id の列 ${anomaly.id} を持つファイル)`
                      : uplift
                        ? `効果を見積もる行の表 (前処理に要る元の列と行の id の列 ${uplift.id} を持つファイル)`
                        : price
                          ? `値段を選ぶ商品の表 (前処理に要る元の列, 値段の列 ${price.price}, 行の id の列 ${price.id} を持つファイル)`
                          : demand
                            ? `発注の数を決める日ごとの記録の表 (商品の id の列 ${demand.product}, 日付の列 ${demand.date}, 売れた数の列 ${demand.units}, 行の id の列 ${demand.id} を持つファイル)`
                            : sourceLabel(description.items?.input_kind, description.kind)
              }
              value={source}
              onChange={setSource}
              hint={
                readyModel
                  ? '重みの説明の前処理 (大きさ, 切り出し, 正規化) をかけ, 種類の名前と確信度を出します. 読めない画像は理由を付けて飛ばします'
                  : group
                    ? '行ごとに学習のときと同じ式で列を符号化し, 最も近い群の中心に割り当てます. 数の列を数にできない行は理由を付けて飛ばします'
                    : anomaly
                      ? '学習のときと同じ前処理をかけ, 行ごとに異常の度合いを出します. 前処理の後に欠けた値のある行は理由を付けて飛ばします'
                      : uplift
                        ? `学習のときと同じ前処理をかけ, 行ごとに効果を見積もります. 処置の列 ${uplift.treatment} と結果の列は要りません (あれば評価を数え直します). 前処理の後に欠けた値のある行は理由を付けて飛ばします`
                        : price
                          ? `1 行を 1 つの商品の今の状態とし, 値段の列を今の値段にして, 候補の値段ごとに売れた数を予測します. 売れた数の列 ${price.units} は要りません (あれば誤差を数え直します). 前処理の後に欠けた値のある行と, 値段か原価の読めない行は理由を付けて飛ばします`
                          : demand
                            ? `学習のときと同じ形の日ごとの記録です (前処理に要る元の列${demand.stock ? `, 在庫の数の列 ${demand.stock}` : ''}). 商品ごとに最も遅い日を発注を決める日とし, その日の売れた数は要りません. 過去の売れた数の列を作るため, 前の日までの記録も入れます. 前処理の後に欠けた値のある商品と, 在庫の数の読めない商品は理由を付けて飛ばします`
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
        {anomaly && (
          <section aria-label="人の確認">
            <div className="max-w-xs">
              <label
                htmlFor="predict-capacity"
                className="mb-1 block text-xs font-medium text-ink-2"
              >
                人の確認に回す件数の上限 (既定は一式の {anomaly.capacity} 件. {MAX_CAPACITY} 件まで)
              </label>
              <input
                id="predict-capacity"
                type="number"
                min={1}
                max={MAX_CAPACITY}
                value={capacity}
                onChange={(event) => setCapacity(event.target.value)}
                className="h-8 w-full rounded-lg border border-line bg-surface px-2 text-sm"
              />
            </div>
            <p className="mt-2 text-xs text-ink-2">
              {ANOMALY_UNSUPPORTED}. {ANOMALY_NO_PROBLEM}
            </p>
            {problems.map((problem) => (
              <p key={problem} role="alert" className="mt-1 text-xs text-critical-text">
                {problem}
              </p>
            ))}
          </section>
        )}
        {uplift && (
          <section aria-label="配る判断">
            <div className="max-w-xs">
              <label htmlFor="predict-budget" className="mb-1 block text-xs font-medium text-ink-2">
                配る数の上限 (
                {uplift.budget === null ? '一式に既定なし' : `既定は一式の ${uplift.budget} 件`}.{' '}
                {MAX_BUDGET.toLocaleString()} 件まで)
              </label>
              <input
                id="predict-budget"
                type="number"
                min={1}
                max={MAX_BUDGET}
                value={budget}
                onChange={(event) => setBudget(event.target.value)}
                className="h-8 w-full rounded-lg border border-line bg-surface px-2 text-sm"
              />
            </div>
            <p className="mt-2 text-xs text-ink-2">
              {UPLIFT_UNSUPPORTED}. {UPLIFT_NO_PROBLEM}
            </p>
            <p className="mt-1 text-xs text-ink-3">{PREMISE_TEXT}</p>
            {problems.map((problem) => (
              <p key={problem} role="alert" className="mt-1 text-xs text-critical-text">
                {problem}
              </p>
            ))}
          </section>
        )}
        {price && (
          <section aria-label="値段の判断">
            <div className="grid gap-3 md:grid-cols-2">
              <div>
                <label
                  htmlFor="predict-candidates"
                  className="mb-1 block text-xs font-medium text-ink-2"
                >
                  値段の候補 (今の値段に対する変える割合の小数をカンマで区切る. 0 を含む.{' '}
                  {MAX_CANDIDATES} 個まで)
                </label>
                <input
                  id="predict-candidates"
                  value={candidates}
                  onChange={(event) => setCandidates(event.target.value)}
                  aria-describedby="predict-candidates-note"
                  className="h-8 w-full rounded-lg border border-line bg-surface px-2 font-mono text-sm"
                />
                <p id="predict-candidates-note" className="mt-1 text-xs text-ink-3">
                  {parsedCandidates.problem
                    ? `一式の既定は ${candidatesText(price.candidates)}`
                    : `% では ${candidatesText(parsedCandidates.values)} (既定は一式の ${candidatesText(price.candidates)})`}
                </p>
              </div>
              <div>
                <label
                  htmlFor="predict-max-change"
                  className="mb-1 block text-xs font-medium text-ink-2"
                >
                  変える幅の上限 (割合の小数. 省けません)
                </label>
                <input
                  id="predict-max-change"
                  type="number"
                  min={0}
                  step="any"
                  value={maxChange}
                  onChange={(event) => setMaxChange(event.target.value)}
                  aria-describedby="predict-max-change-note"
                  className="h-8 w-full rounded-lg border border-line bg-surface px-2 text-sm"
                />
                <p id="predict-max-change-note" className="mt-1 text-xs text-ink-3">
                  {parsedMaxChange.value === null
                    ? `一式の既定は ${limitText(price.max_change)} (省けません)`
                    : `% では ${limitText(parsedMaxChange.value)} (既定は一式の ${limitText(price.max_change)})`}
                </p>
              </div>
            </div>
            {!parsedCandidates.problem &&
              outsideCandidates(parsedCandidates.values, parsedMaxChange.value).length > 0 && (
                <p className="mt-2 text-xs text-warning-text">
                  上限を超える候補{' '}
                  {outsideCandidates(parsedCandidates.values, parsedMaxChange.value)
                    .map(changeText)
                    .join(', ')}{' '}
                  は選びません (利益が大きければ,
                  結果の画面に上限を超える候補の方が利益の大きい行として数えます)
                </p>
              )}
            <p className="mt-2 text-xs text-ink-2">
              {PRICE_UNSUPPORTED}. {PRICE_NO_PROBLEM}
            </p>
            {problems.map((problem) => (
              <p key={problem} role="alert" className="mt-1 text-xs text-critical-text">
                {problem}
              </p>
            ))}
          </section>
        )}
        {demand && (
          <section aria-label="発注の判断">
            <div className="grid gap-3 md:grid-cols-3">
              <div>
                <label
                  htmlFor="predict-shortage"
                  className="mb-1 block text-xs font-medium text-ink-2"
                >
                  品切れの損 (1 つ足りないときの損. 正の数)
                </label>
                <input
                  id="predict-shortage"
                  type="number"
                  min={0}
                  step="any"
                  value={shortage}
                  onChange={(event) => setShortage(event.target.value)}
                  aria-describedby="predict-shortage-note"
                  className="h-8 w-full rounded-lg border border-line bg-surface px-2 text-sm"
                />
                <p id="predict-shortage-note" className="mt-1 text-xs text-ink-3">
                  一式の既定は {formatNumber(demand.shortage_cost)}
                </p>
              </div>
              <div>
                <label
                  htmlFor="predict-holding"
                  className="mb-1 block text-xs font-medium text-ink-2"
                >
                  売れ残りの損 (1 つ余ったときの損. 正の数)
                </label>
                <input
                  id="predict-holding"
                  type="number"
                  min={0}
                  step="any"
                  value={holding}
                  onChange={(event) => setHolding(event.target.value)}
                  aria-describedby="predict-holding-note"
                  className="h-8 w-full rounded-lg border border-line bg-surface px-2 text-sm"
                />
                <p id="predict-holding-note" className="mt-1 text-xs text-ink-3">
                  一式の既定は {formatNumber(demand.holding_cost)}
                </p>
              </div>
              <div>
                <label
                  htmlFor="predict-max-order"
                  className="mb-1 block text-xs font-medium text-ink-2"
                >
                  発注の数の上限 (0 以上の整数. 省けません)
                </label>
                <input
                  id="predict-max-order"
                  type="number"
                  min={0}
                  step={1}
                  value={maxOrder}
                  onChange={(event) => setMaxOrder(event.target.value)}
                  aria-describedby="predict-max-order-note"
                  className="h-8 w-full rounded-lg border border-line bg-surface px-2 text-sm"
                />
                <p id="predict-max-order-note" className="mt-1 text-xs text-ink-3">
                  一式の既定は {formatNumber(demand.max_order)}
                </p>
              </div>
            </div>
            <p className="mt-2 text-xs text-ink-2" role="status">
              {tau === null
                ? '損を 2 つとも書くと, 水準 τ が決まります'
                : `水準 τ = ${levelText(tau)} (品切れの損 / (品切れの損 + 売れ残りの損)). 学んだ水準は ${levelsText(demand.levels)}`}
            </p>
            {edge !== null && (
              <p className="mt-1 text-xs text-warning-text">
                水準 τ が学んだ水準の範囲の外なので, 端の水準 {levelText(edge)} の分位点を使います
                (分布の端は学んでいないので, 見込みの損は少なめに出ます)
              </p>
            )}
            {parsedMaxOrder.value === 0 && (
              <p className="mt-1 text-xs text-warning-text">
                上限が 0 なので, どの商品も発注しません
              </p>
            )}
            <p className="mt-2 text-xs text-ink-2">
              {baselineText(demand.horizon)}. {DEMAND_UNSUPPORTED}. {DEMAND_NO_PROBLEM}
            </p>
            {problems.map((problem) => (
              <p key={problem} role="alert" className="mt-1 text-xs text-critical-text">
                {problem}
              </p>
            ))}
          </section>
        )}
        {!column && !group && !anomaly && !uplift && !price && !demand && (
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
  // 雑音だけの一式, 並べるの一式, 群分けの一式, 異常の度合いの一式, 効果の一式, 値段の一式, 需要の一式は, 問題の設定の結論と意思決定を扱わないので問題の設定を読まない
  // (既製のモデルの一式は読み, 結論の決め方だけを既定にする).
  // 種類は一式か一覧の読んだ値から. 分からない間は一式と並べて読む (推論の欄を 1 往復早く出すため.
  // 読まない種類の一式なら, サーバーは問題の設定を読まずに空の状態を返す)
  const listed = client.getQueryData<Bundle[]>(['bundles'])?.find((item) => item.id === id)
  const kind = bundle.data?.description?.kind ?? listed?.description?.kind
  const noise =
    kind === 'noise' ||
    kind === 'rank' ||
    kind === 'group' ||
    kind === 'anomaly' ||
    kind === 'uplift' ||
    kind === 'price' ||
    kind === 'demand'
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
            {data.description.anomaly && (
              <Card title="実験の評価">
                <AnomalyEvaluationsTable evaluations={data.description.anomaly.evaluations} />
              </Card>
            )}
            {data.description.price && (
              <Card title="実験の評価">
                <PriceEvaluationsView evaluations={data.description.price.evaluations} />
              </Card>
            )}
            {data.description.demand && (
              <Card title="実験の評価">
                <DemandEvaluationsView evaluations={data.description.demand.evaluations} />
              </Card>
            )}
            {data.description.uplift && (
              <Card title="実験の評価">
                <UpliftEvaluationsView
                  evaluations={data.description.uplift.evaluations}
                  curves={data.description.uplift.curves}
                  k={data.description.uplift.k}
                />
              </Card>
            )}
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
