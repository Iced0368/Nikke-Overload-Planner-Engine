import { useEffect, useState } from "react";
import { type OverloadBudgetOptimizationSummary } from "../../../lib/overloadBudgetOptimizer.ts";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { type MonteCarloSimulationSummary } from "../../../lib/overloadMonteCarlo.ts";
import { type OverloadOptionTarget } from "../../../lib/overloadOptions";
import {
  type OverloadForcedLockAlternative,
  type OverloadOptimizationProgress,
  type OverloadPolicyOptimizationResult,
  type OverloadStateValue,
} from "../../../lib/overloadPolicyOptimizer.ts";
import { getOptionIndex, getOptionName } from "../model/model";
import { type PlannerMode } from "./PlannerModeSection";

const CUMULATIVE_DISTRIBUTION_TRIM_SHARE = 0.985;
const TERMINAL_PIE_GROUP_LIMIT = 6;
const TERMINAL_PIE_COLORS = ["#ffc178", "#83b0ff", "#7ec48f", "#ff8f70", "#d2b4ff", "#7adad1", "#8d96a9"];
const FORCED_LOCK_ALTERNATIVE_MAX_RATIO = 0.05;
const FORCED_LOCK_ALTERNATIVE_MAX_SCORE_DELTA = 5;

type PlannerResultPanelProps = {
  mode: PlannerMode;
  result: OverloadPolicyOptimizationResult | null;
  isStale: boolean;
  isOptimizing: boolean;
  optimizationProgress: OverloadOptimizationProgress | null;
  currentStateValue: OverloadStateValue | null;
  forcedLockAlternatives: OverloadForcedLockAlternative[];
  moduleBudget: number;
  budgetOptimizationResult: OverloadBudgetOptimizationSummary | null;
  budgetOptimizationError: string | null;
  isBudgetOptimizing: boolean;
  isBudgetStale: boolean;
  detailedSimulationResult: MonteCarloSimulationSummary | null;
  simulationError: string | null;
  isSimulationRunning: boolean;
  onRunDetailedSimulation: () => void;
  targetGrades: OverloadOptionTarget[];
};

type TerminalPieDatum = {
  key: string;
  outcome: "success" | "failure" | "mixed";
  label: string;
  secondaryLabel: string | null;
  count: number;
  share: number;
  color: string;
};

type SimulationComparisonCard = {
  key: string;
  label: string;
  expected: number;
  simulated: number;
  accentClassName: string;
  valueType?: "number" | "percent";
  expectedLabel?: string;
  simulatedLabel?: string;
};

type SimulationMetaItem = {
  key: string;
  label: string;
  value: string;
};

type CumulativeMetricKey = "weighted" | "module" | "lockKey";

const CUMULATIVE_METRIC_CONFIG: Record<
  CumulativeMetricKey,
  {
    label: string;
    dataKey: "cumulativePercent";
    distributionKey:
      | "cumulativeCostDistribution"
      | "cumulativeModuleCostDistribution"
      | "cumulativeLockKeyCostDistribution";
    color: string;
    className: string;
    gradientId: string;
    caption: string;
  }
> = {
  weighted: {
    label: "가중 비용",
    dataKey: "cumulativePercent",
    distributionKey: "cumulativeCostDistribution",
    color: "#ffc178",
    className: "is-weighted",
    gradientId: "cumulativeWeightedAreaGradient",
    caption: "모듈과 락키 가치가 반영된 내부 점수 기준입니다.",
  },
  module: {
    label: "순수 모듈",
    dataKey: "cumulativePercent",
    distributionKey: "cumulativeModuleCostDistribution",
    color: "#83b0ff",
    className: "is-module",
    gradientId: "cumulativeModuleAreaGradient",
    caption: "실제 모듈 소모량 누적 분포입니다.",
  },
  lockKey: {
    label: "순수 락키",
    dataKey: "cumulativePercent",
    distributionKey: "cumulativeLockKeyCostDistribution",
    color: "#7ec48f",
    className: "is-lockkey",
    gradientId: "cumulativeLockKeyAreaGradient",
    caption: "락키 개수 누적 분포입니다.",
  },
};

function buildTargetGradeMap(targetGrades: OverloadOptionTarget[]) {
  return new Map(targetGrades.map((target) => [getOptionIndex(target.id), target.grade]));
}

function buildTerminalGroupLabel(
  optionIndexes: number[],
  emptyCount: number,
  targetGradeMap: Map<number, number>,
  outcome: "success" | "failure",
) {
  const parts = optionIndexes.map((optionIndex) => getOptionName(optionIndex));
  for (let index = 0; index < emptyCount; index++) {
    parts.push("빈 슬롯");
  }

  const gradeParts = optionIndexes.map((optionIndex) => String((targetGradeMap.get(optionIndex) ?? 0) + 1));
  for (let index = 0; index < emptyCount; index++) {
    gradeParts.push("-");
  }

  return {
    label: `${outcome === "success" ? "성공" : "실패"} · ${parts.join(" / ")}`,
    secondaryLabel: `${outcome === "success" ? "목표 달성 종료" : "예산 소진 종료"} · 등급 기준: ${gradeParts.join("/")}`,
  };
}

function buildGroupedTerminalPieData(
  terminalStateDistribution: MonteCarloSimulationSummary["terminalStateDistribution"],
  targetGrades: OverloadOptionTarget[],
) {
  const targetGradeMap = buildTargetGradeMap(targetGrades);
  const grouped = new Map<string, Omit<TerminalPieDatum, "color">>();

  for (const entry of terminalStateDistribution) {
    const optionIndexes = entry.state
      .slice(0, 3)
      .filter((value): value is number => value !== 0)
      .sort((left, right) => left - right);
    const emptyCount = 3 - optionIndexes.length;
    const key = `${entry.outcome}|${optionIndexes.join(",")}|${emptyCount}`;
    const existing = grouped.get(key);

    if (existing) {
      existing.count += entry.count;
      existing.share += entry.share;
      continue;
    }

    const { label, secondaryLabel } = buildTerminalGroupLabel(optionIndexes, emptyCount, targetGradeMap, entry.outcome);
    grouped.set(key, {
      key,
      outcome: entry.outcome,
      label,
      secondaryLabel,
      count: entry.count,
      share: entry.share,
    });
  }

  const sorted = Array.from(grouped.values()).sort((left, right) => right.count - left.count);
  const topGroups = sorted.slice(0, TERMINAL_PIE_GROUP_LIMIT);
  const otherGroups = sorted.slice(TERMINAL_PIE_GROUP_LIMIT);
  const pieData: TerminalPieDatum[] = topGroups.map((group, index) => ({
    ...group,
    color: TERMINAL_PIE_COLORS[index % TERMINAL_PIE_COLORS.length]!,
  }));

  if (otherGroups.length > 0) {
    pieData.push({
      key: "others",
      outcome: "mixed",
      label: `기타 ${otherGroups.length}개 그룹`,
      secondaryLabel: null,
      count: otherGroups.reduce((sum, group) => sum + group.count, 0),
      share: otherGroups.reduce((sum, group) => sum + group.share, 0),
      color: TERMINAL_PIE_COLORS[TERMINAL_PIE_COLORS.length - 1]!,
    });
  }

  return {
    pieData,
    groupedStateCount: sorted.length,
  };
}

function getDisplayedCumulativeDistribution(distribution: MonteCarloSimulationSummary["cumulativeCostDistribution"]) {
  const trimStartIndex = distribution.findIndex(
    (bucket) => bucket.cumulativeShare >= CUMULATIVE_DISTRIBUTION_TRIM_SHARE,
  );

  if (trimStartIndex === -1 || trimStartIndex >= distribution.length - 1) {
    return {
      buckets: distribution,
      isTrimmed: false,
      trimmedShare: null,
    };
  }

  return {
    buckets: distribution.slice(0, trimStartIndex + 1),
    isTrimmed: true,
    trimmedShare: distribution[trimStartIndex]!.cumulativeShare,
  };
}

function buildCumulativeChartData(buckets: MonteCarloSimulationSummary["cumulativeCostDistribution"]) {
  if (buckets.length === 0) {
    return null;
  }

  return {
    chartData: buckets.map((bucket) => ({
      costUpperBound: bucket.upperBound,
      cumulativePercent: Number((bucket.cumulativeShare * 100).toFixed(2)),
    })),
    firstUpperBound: buckets[0]!.upperBound,
    lastUpperBound: buckets[buckets.length - 1]!.upperBound,
    lastShare: buckets[buckets.length - 1]!.cumulativeShare,
  };
}

function renderTerminalPieTooltip(active?: boolean, payload?: ReadonlyArray<{ payload: TerminalPieDatum }>) {
  const entry = active ? payload?.[0]?.payload : undefined;

  if (!entry) {
    return null;
  }

  return (
    <div
      style={{
        display: "grid",
        gap: 6,
        minWidth: 180,
        padding: "0.75rem 0.85rem",
        borderRadius: 14,
        border: "1px solid rgba(255,255,255,0.08)",
        background: "rgba(14, 19, 30, 0.96)",
        boxShadow: "0 16px 32px rgba(7, 12, 22, 0.28)",
      }}
    >
      <strong style={{ color: "#edf2fa", fontSize: "0.9rem", lineHeight: 1.35 }}>{entry.label}</strong>
      {entry.secondaryLabel ? (
        <span style={{ color: "rgba(233, 238, 247, 0.66)", fontSize: "0.78rem", lineHeight: 1.4 }}>
          {entry.secondaryLabel}
        </span>
      ) : null}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
          gap: "0.45rem 0.85rem",
          marginTop: 2,
        }}
      >
        <div style={{ display: "grid", gap: 2 }}>
          <span style={{ color: "rgba(233, 238, 247, 0.62)", fontSize: "0.72rem" }}>비율</span>
          <strong style={{ color: "#f4f7fb", fontSize: "0.88rem" }}>{(entry.share * 100).toFixed(1)}%</strong>
        </div>
        <div style={{ display: "grid", gap: 2 }}>
          <span style={{ color: "rgba(233, 238, 247, 0.62)", fontSize: "0.72rem" }}>횟수</span>
          <strong style={{ color: "#f4f7fb", fontSize: "0.88rem" }}>{entry.count.toLocaleString()}회</strong>
        </div>
      </div>
    </div>
  );
}

function formatComparisonDelta(value: number) {
  const sign = value > 0 ? "+" : value < 0 ? "-" : "";
  return `${sign}${Math.abs(value).toFixed(2)}`;
}

function formatComparisonValue(value: number, valueType: "number" | "percent") {
  if (valueType === "percent") {
    return `${value.toFixed(2)}%`;
  }

  return value.toFixed(2);
}

function getOptimizationPhaseLabel(progress: OverloadOptimizationProgress) {
  switch (progress.phase) {
    case "policy":
      return "정책 수렴 계산";
    case "expectation":
      return "기대 소모량 평가";
    case "done":
      return "계산 완료";
  }
}

function getActionTitle(currentStateValue: OverloadStateValue) {
  if (currentStateValue.action.type === "done") {
    return "목표 달성";
  }

  return currentStateValue.action.type === "option" ? "효과 변경" : "수치 재설정";
}

function getActionTitleFromAction(action: OverloadStateValue["action"] | null) {
  if (!action || action.type === "done") {
    return action ? "목표 달성" : "예산 부족";
  }

  return action.type === "option" ? "효과 변경" : "수치 재설정";
}

function collectLockedSlots(lockState: [boolean, boolean, boolean]) {
  return lockState
    .map((locked, index) => (locked ? `${index + 1}번 슬롯` : null))
    .filter((slot): slot is string => slot !== null);
}

function buildLockDescription(
  moduleLock: [boolean, boolean, boolean],
  keyLock: [boolean, boolean, boolean],
  noLockText: string,
  suffix: string,
) {
  const moduleLockedSlots = collectLockedSlots(moduleLock);
  const keyLockedSlots = collectLockedSlots(keyLock);

  if (moduleLockedSlots.length === 0 && keyLockedSlots.length === 0) {
    return noLockText;
  }

  const parts: string[] = [];
  if (moduleLockedSlots.length > 0) {
    parts.push(`${moduleLockedSlots.join(", ")}은 모듈 잠금`);
  }
  if (keyLockedSlots.length > 0) {
    parts.push(`${keyLockedSlots.join(", ")}은 락키 잠금`);
  }

  return `${parts.join(", ")}${suffix}`;
}

function getActionDescription(currentStateValue: OverloadStateValue) {
  if (currentStateValue.action.type === "done") {
    return "현재 시작 상태가 이미 목표 조건을 만족합니다.";
  }

  return buildLockDescription(
    currentStateValue.action.moduleLock,
    currentStateValue.action.keyLock,
    "잠금 없이 진행하는 것이 가장 유리합니다.",
    "으로 진행하는 것이 가장 유리합니다.",
  );
}

function getActionDescriptionFromAction(action: OverloadStateValue["action"] | null) {
  if (!action) {
    return "입력한 모듈 예산 안에서는 목표를 달성할 수 있는 정책이 확인되지 않았습니다.";
  }

  if (action.type === "done") {
    return "현재 시작 상태가 이미 목표 조건을 만족합니다.";
  }

  return buildLockDescription(
    action.moduleLock,
    action.keyLock,
    "예산 안에서는 잠금 없이 진행하는 것이 가장 유리합니다.",
    "으로 진행하는 것이 가장 유리합니다.",
  );
}

function formatProtectedMaskLabel(protectedMask: [boolean, boolean, boolean]) {
  const lockedSlots = collectLockedSlots(protectedMask);
  if (lockedSlots.length === 0) {
    return "잠금 없음";
  }

  return lockedSlots.join(" · ");
}

function getAlternativeActionSummary(alternative: OverloadForcedLockAlternative) {
  return alternative.action.type === "option" ? "효과 변경" : "수치 재설정";
}

function getForcedLockAlternativeTitle(alternative: OverloadForcedLockAlternative) {
  return getAlternativeActionSummary(alternative);
}

export function PlannerResultPanel({
  mode,
  result,
  isStale,
  isOptimizing,
  optimizationProgress,
  currentStateValue,
  forcedLockAlternatives,
  moduleBudget,
  budgetOptimizationResult,
  budgetOptimizationError,
  isBudgetOptimizing,
  isBudgetStale,
  detailedSimulationResult,
  simulationError,
  isSimulationRunning,
  onRunDetailedSimulation,
  targetGrades,
}: PlannerResultPanelProps) {
  const [isSimulationExpanded, setIsSimulationExpanded] = useState(false);
  const [activeCumulativeMetric, setActiveCumulativeMetric] = useState<CumulativeMetricKey>("weighted");
  const [activeTerminalPieKey, setActiveTerminalPieKey] = useState<string | null>(null);
  const isClassicMode = mode === "classic";
  const isBudgetMode = mode === "budget";
  const hasBudgetResult = budgetOptimizationResult !== null && !isBudgetStale;
  const hasClassicResult = currentStateValue !== null && !isStale;
  const hasActiveResult = isClassicMode ? hasClassicResult : hasBudgetResult;

  useEffect(() => {
    if (detailedSimulationResult) {
      setIsSimulationExpanded(true);
    }
  }, [detailedSimulationResult]);

  useEffect(() => {
    setActiveTerminalPieKey(null);
  }, [detailedSimulationResult]);

  const activeCumulativeConfig = CUMULATIVE_METRIC_CONFIG[activeCumulativeMetric];
  const simulationComparisonCards: SimulationComparisonCard[] | null = detailedSimulationResult
    ? isClassicMode && currentStateValue
      ? [
          {
            key: "module",
            label: "모듈",
            expected: detailedSimulationResult.estimatedModuleCost,
            simulated: detailedSimulationResult.sampleMeanModuleCost,
            accentClassName: "is-module",
          },
          {
            key: "lockKey",
            label: "락키",
            expected: detailedSimulationResult.estimatedLockKeyCost,
            simulated: detailedSimulationResult.sampleMeanLockKeyCost,
            accentClassName: "is-lockkey",
          },
          {
            key: "weighted",
            label: "가중 평균",
            expected: currentStateValue.cost,
            simulated: detailedSimulationResult.sampleMean,
            accentClassName: "is-weighted",
          },
        ]
      : isBudgetMode && budgetOptimizationResult
        ? [
            {
              key: "success",
              label: "성공 확률",
              expected: budgetOptimizationResult.current.successProbability * 100,
              simulated: detailedSimulationResult.sampleSuccessRate * 100,
              accentClassName: "is-probability",
              valueType: "percent",
            },
            {
              key: "lockKey",
              label: "락키",
              expected: budgetOptimizationResult.current.expectedLockKeyCost,
              simulated: detailedSimulationResult.sampleMeanLockKeyCost,
              accentClassName: "is-lockkey",
            },
            {
              key: "module",
              label: "모듈 예산",
              expected: moduleBudget,
              simulated: detailedSimulationResult.sampleMeanModuleCost,
              accentClassName: "is-module",
              expectedLabel: "예산",
              simulatedLabel: "시뮬 평균",
            },
          ]
        : null
    : null;
  const simulationMetaItems: SimulationMetaItem[] | null = detailedSimulationResult
    ? isClassicMode
      ? [
          {
            key: "standard-error",
            label: "표준 오차",
            value: detailedSimulationResult.standardError.toFixed(3),
          },
          {
            key: "trial-count",
            label: "시뮬레이션 횟수",
            value: detailedSimulationResult.trialCount.toLocaleString(),
          },
        ]
      : [
          {
            key: "success-count",
            label: "성공 횟수",
            value: detailedSimulationResult.successCount.toLocaleString(),
          },
          {
            key: "failure-count",
            label: "실패 횟수",
            value: detailedSimulationResult.failureCount.toLocaleString(),
          },
          {
            key: "success-error",
            label: "성공률 표준 오차",
            value: `${(detailedSimulationResult.successRateStandardError * 100).toFixed(2)}%`,
          },
          {
            key: "trial-count",
            label: "시뮬레이션 횟수",
            value: detailedSimulationResult.trialCount.toLocaleString(),
          },
        ]
    : null;
  const activeCumulativeDistribution = detailedSimulationResult
    ? detailedSimulationResult[activeCumulativeConfig.distributionKey]
    : null;
  const displayedCumulativeDistribution = activeCumulativeDistribution
    ? getDisplayedCumulativeDistribution(activeCumulativeDistribution)
    : null;
  const cumulativeChart = displayedCumulativeDistribution
    ? buildCumulativeChartData(displayedCumulativeDistribution.buckets)
    : null;
  const terminalPie = detailedSimulationResult
    ? buildGroupedTerminalPieData(detailedSimulationResult.terminalStateDistribution, targetGrades)
    : null;
  const budgetChartData = budgetOptimizationResult?.curve.map((point) => ({
    moduleBudget: point.moduleBudget,
    successPercent: Number((point.successProbability * 100).toFixed(2)),
  }));
  const visibleForcedLockAlternatives = (() => {
    const rankedAlternatives = forcedLockAlternatives.filter((alternative) => !alternative.isCurrentOptimal);
    return rankedAlternatives.filter((alternative) => {
      const optimalCost = Math.max(alternative.cost - alternative.deltaFromOptimal, 1e-9);
      return (
        alternative.deltaFromOptimal < FORCED_LOCK_ALTERNATIVE_MAX_SCORE_DELTA &&
        alternative.deltaFromOptimal / optimalCost <= FORCED_LOCK_ALTERNATIVE_MAX_RATIO
      );
    });
  })();
  const hasDetailedSimulationResult = detailedSimulationResult !== null;
  const activeAction = isClassicMode
    ? (currentStateValue?.action ?? null)
    : (budgetOptimizationResult?.current.action ?? null);
  const actionTitle =
    isClassicMode && currentStateValue ? getActionTitle(currentStateValue) : getActionTitleFromAction(activeAction);
  const actionDescription =
    isClassicMode && currentStateValue
      ? getActionDescription(currentStateValue)
      : getActionDescriptionFromAction(activeAction);
  const resultBadge = isClassicMode
    ? isStale
      ? "재실행 필요"
      : result
        ? `${result.iterationsRun}회 계산`
        : "아직 계산 전"
    : isBudgetStale
      ? "재실행 필요"
      : budgetOptimizationResult
        ? `${moduleBudget.toLocaleString()} 모듈 기준`
        : "아직 계산 전";
  const heroTitle = isClassicMode
    ? "현재 상태 기대 소모량"
    : `모듈 ${moduleBudget.toLocaleString()}개 안에서의 최대 달성 확률`;
  const heroCaption = isClassicMode
    ? currentStateValue
      ? `내부 가중 점수 ${currentStateValue.cost.toFixed(2)}`
      : ""
    : budgetOptimizationResult
      ? `같은 확률이면 기대 락키 소모 ${budgetOptimizationResult.current.expectedLockKeyCost.toFixed(2)}개 정책을 우선합니다.`
      : "";

  const handleDetailTrigger = () => {
    if (!hasDetailedSimulationResult) {
      onRunDetailedSimulation();
      return;
    }

    setIsSimulationExpanded((current) => !current);
  };

  const detailToggleLabel = isSimulationRunning
    ? "시뮬레이션 실행 중..."
    : hasDetailedSimulationResult
      ? isSimulationExpanded
        ? "상세 결과 접기"
        : "상세 결과 펼치기"
      : "상세 결과 계산";
  const detailToggleCaption = hasDetailedSimulationResult
    ? isSimulationExpanded
      ? "분포 그래프와 도달 상태 요약을 숨깁니다."
      : "분포 그래프와 도달 상태 요약을 다시 표시합니다."
    : isClassicMode
      ? "몬테카를로 시뮬레이션을 실행해 평균 소모 분포를 계산합니다."
      : "몬테카를로 시뮬레이션을 실행해 예산 내 실제 달성률과 분포를 계산합니다.";

  return (
    <div className="panel result-panel">
      <div className="panel-header">
        <h2>결과 확인</h2>
        <span className="result-badge">{resultBadge}</span>
      </div>

      {isClassicMode && isOptimizing && optimizationProgress ? (
        <div className="optimization-progress-card">
          <div className="optimization-progress-copy">
            <span className="result-label">계산 진행도</span>
            <strong>{getOptimizationPhaseLabel(optimizationProgress)}</strong>
            <span className="section-caption">
              {optimizationProgress.completedIterations.toLocaleString()} /{" "}
              {optimizationProgress.totalIterations.toLocaleString()} · {optimizationProgress.percent.toFixed(1)}%
            </span>
          </div>
          <div className="optimization-progress-bar-shell" aria-hidden="true">
            <div
              className="optimization-progress-bar-fill"
              style={{ width: `${Math.max(4, optimizationProgress.percent)}%` }}
            />
          </div>
        </div>
      ) : null}

      {isBudgetMode && isBudgetOptimizing ? (
        <div className="optimization-progress-card budget-progress-card">
          <div className="optimization-progress-copy">
            <span className="result-label">예산 최적화</span>
            <strong>모듈 {moduleBudget.toLocaleString()}개 기준 성공 확률 계산 중</strong>
            <span className="section-caption">달성 확률이 최대인 정책과 예산별 확률 곡선을 계산합니다.</span>
          </div>
        </div>
      ) : null}

      {isBudgetMode && budgetOptimizationError ? (
        <div className="error-banner simulation-error-banner">{budgetOptimizationError}</div>
      ) : null}

      {(isClassicMode && isStale) || (isBudgetMode && isBudgetStale) ? (
        <div className={isBudgetMode ? "stale-banner budget-stale-banner" : "stale-banner"}>
          <p>
            {isClassicMode
              ? "목표 상태 또는 목표 등급이 변경되어 이전 최적화 결과가 더 이상 유효하지 않습니다."
              : "시작 상태, 목표 조건 또는 모듈 예산이 변경되어 이전 예산 확률 결과가 더 이상 유효하지 않습니다."}
          </p>
          <p>
            {isClassicMode
              ? "현재 목표 조건으로 다시 최적화 실행을 눌러 주세요."
              : "현재 조건으로 다시 예산 기반 실행을 눌러 주세요."}
          </p>
        </div>
      ) : hasActiveResult ? (
        <>
          <div className={isBudgetMode ? "result-hero is-budget" : "result-hero"}>
            <div>
              <p className="result-label">{heroTitle}</p>
              {isClassicMode && currentStateValue ? (
                <h3 className="result-hero-amount">
                  <span className="result-hero-unit">모듈</span>
                  <span className="result-hero-number">{currentStateValue.expectedCosts.module.toFixed(2)}</span>
                  <span className="result-hero-unit">개</span>
                  <span className="result-hero-separator">/</span>
                  <span className="result-hero-unit">락키</span>
                  <span className="result-hero-number">{currentStateValue.expectedCosts.lockKey.toFixed(2)}</span>
                  <span className="result-hero-unit">개</span>
                </h3>
              ) : (
                <h3 className="result-hero-amount result-hero-percent">
                  {((budgetOptimizationResult?.current.successProbability ?? 0) * 100).toFixed(2)}%
                </h3>
              )}
              <p className="section-caption">{heroCaption}</p>
            </div>
          </div>

          <div className="result-action-card is-featured">
            <div className="result-action-copy">
              <div className="result-action-header">
                <span className="result-action-kicker">Recommend</span>
                <span className="result-action-badge">현재 추천</span>
              </div>
              <h3>{actionTitle}</h3>
              <p>{actionDescription}</p>
            </div>
            <div className="result-action-visual is-featured">
              {activeAction ? (
                activeAction.type === "done" ? (
                  <div className="action-done-badge">완료</div>
                ) : (
                  <>
                    <div className={`action-type-badge action-type-${activeAction.type}`}>{actionTitle}</div>
                    <div className="lock-visual-grid">
                      {[0, 1, 2].map((index) => {
                        const isModuleLocked = activeAction.moduleLock[index];
                        const isKeyLocked = activeAction.keyLock[index];
                        const className = isModuleLocked
                          ? "lock-visual-slot is-module-locked"
                          : isKeyLocked
                            ? "lock-visual-slot is-key-locked"
                            : "lock-visual-slot";

                        return (
                          <div className={className} key={`lock-${mode}-${index}`}>
                            <span className="lock-visual-index">{index + 1}</span>
                            <span className="lock-visual-state">
                              {isModuleLocked ? "모듈" : isKeyLocked ? "락키" : "개방"}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </>
                )
              ) : (
                <div className="budget-no-path">달성 가능 정책 없음</div>
              )}
            </div>
          </div>

          {isClassicMode && visibleForcedLockAlternatives.length > 0 ? (
            <div className="forced-lock-card">
              <div className="forced-lock-list">
                {visibleForcedLockAlternatives.map((alternative, index) => {
                  const alternativeTitle = getForcedLockAlternativeTitle(alternative);

                  return (
                    <div
                      className="forced-lock-row"
                      key={`${index}-${alternative.action.type}-${formatProtectedMaskLabel(alternative.protectedMask)}`}
                    >
                      <div className="result-action-copy forced-lock-action-copy">
                        <div className="forced-lock-header">
                          <span className="result-action-kicker">Alternative</span>
                        </div>
                        <h3>{alternativeTitle}</h3>
                        <div className="forced-lock-hero">
                          <h3 className="result-hero-amount forced-lock-amount">
                            <span className="result-hero-unit">모듈</span>
                            <span className="result-hero-number">{alternative.expectedCosts.module.toFixed(2)}</span>
                            <span className="result-hero-unit">개</span>
                            <span className="result-hero-separator">/</span>
                            <span className="result-hero-unit">락키</span>
                            <span className="result-hero-number">{alternative.expectedCosts.lockKey.toFixed(2)}</span>
                            <span className="result-hero-unit">개</span>
                          </h3>
                          <p className="section-caption">내부 가중 점수 {alternative.cost.toFixed(2)}</p>
                        </div>
                      </div>
                      <div className="result-action-visual forced-lock-action-visual">
                        <div className={`action-type-badge action-type-${alternative.action.type}`}>
                          {getAlternativeActionSummary(alternative)}
                        </div>
                        <div className="lock-visual-grid">
                          {[0, 1, 2].map((slotIndex) => {
                            const isModuleLocked = alternative.action.moduleLock[slotIndex];
                            const isKeyLocked = alternative.action.keyLock[slotIndex];
                            const className = isModuleLocked
                              ? "lock-visual-slot is-module-locked"
                              : isKeyLocked
                                ? "lock-visual-slot is-key-locked"
                                : "lock-visual-slot";

                            return (
                              <div className={className} key={`forced-lock-${index}-${slotIndex}`}>
                                <span className="lock-visual-index">{slotIndex + 1}</span>
                                <span className="lock-visual-state">
                                  {isModuleLocked ? "모듈" : isKeyLocked ? "락키" : "개방"}
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : null}

          {isBudgetMode && budgetChartData ? (
            <div className="budget-curve-card">
              <div className="simulation-block-header">
                <h3>예산별 달성 확률</h3>
                <span className="section-caption">
                  현재 시작 상태에서 모듈 예산을 늘릴 때 최대 달성 확률이 어떻게 바뀌는지 보여줍니다.
                </span>
              </div>
              <div className="budget-curve-shell">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={budgetChartData} margin={{ top: 18, right: 22, left: 8, bottom: 8 }}>
                    <defs>
                      <linearGradient id="budgetSuccessCurveGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#ff8f70" stopOpacity={0.34} />
                        <stop offset="100%" stopColor="#ff8f70" stopOpacity={0.05} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke="rgba(255,255,255,0.08)" strokeDasharray="3 3" vertical={false} />
                    <XAxis
                      dataKey="moduleBudget"
                      tickLine={false}
                      axisLine={false}
                      minTickGap={22}
                      tick={{ fill: "rgba(233, 238, 247, 0.62)", fontSize: 12 }}
                    />
                    <YAxis
                      type="number"
                      domain={[0, 100]}
                      tickCount={5}
                      tickFormatter={(value) => `${value}%`}
                      tickLine={false}
                      axisLine={false}
                      width={46}
                      tick={{ fill: "rgba(233, 238, 247, 0.62)", fontSize: 12 }}
                    />
                    <Tooltip
                      formatter={(value) => [`${Number(value).toFixed(2)}%`, "최대 달성 확률"]}
                      labelFormatter={(label) => `모듈 ${Number(label).toFixed(0)}개`}
                      contentStyle={{
                        borderRadius: 14,
                        border: "1px solid rgba(255,255,255,0.08)",
                        background: "rgba(14, 19, 30, 0.96)",
                        boxShadow: "0 16px 32px rgba(7, 12, 22, 0.28)",
                      }}
                      labelStyle={{ color: "#edf2fa", fontWeight: 700, marginBottom: 4 }}
                      itemStyle={{ color: "#edf2fa", padding: 0 }}
                    />
                    <Area
                      type="monotone"
                      dataKey="successPercent"
                      name="최대 달성 확률"
                      stroke="#ff8f70"
                      strokeWidth={2.4}
                      fill="url(#budgetSuccessCurveGradient)"
                      dot={false}
                      activeDot={{ r: 4, stroke: "#171b28", strokeWidth: 2, fill: "#ff8f70" }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          ) : null}

          <div className="detail-divider-wrap">
            <button
              className={hasDetailedSimulationResult ? "detail-divider is-toggle" : "detail-divider is-action"}
              onClick={handleDetailTrigger}
              disabled={isSimulationRunning}
              aria-expanded={hasDetailedSimulationResult ? isSimulationExpanded : undefined}
            >
              <span className="detail-divider-line" aria-hidden="true" />
              <span className="detail-divider-copy">
                <strong>
                  <span className="button-content">
                    <span className="button-icon" aria-hidden="true">
                      <svg viewBox="0 0 20 20" focusable="false">
                        <path d="M4.75 14.5h10.5" />
                        <path d="M6.25 12V8.75" />
                        <path d="M10 12V5.5" />
                        <path d="M13.75 12V7" />
                      </svg>
                    </span>
                    <span>{detailToggleLabel}</span>
                  </span>
                </strong>
                <span>{detailToggleCaption}</span>
              </span>
              <span
                className={
                  hasDetailedSimulationResult && isSimulationExpanded
                    ? "detail-divider-icon is-expanded"
                    : "detail-divider-icon"
                }
                aria-hidden="true"
              >
                ▾
              </span>
              <span className="detail-divider-line" aria-hidden="true" />
            </button>
            {hasDetailedSimulationResult ? (
              <button className="detail-refresh-button" onClick={onRunDetailedSimulation}>
                <span className="detail-refresh-icon" aria-hidden="true">
                  <svg viewBox="0 0 20 20" focusable="false">
                    <path d="M15.25 8.25A5.5 5.5 0 1 0 16 11" />
                    <path d="M15.25 4.75v3.5h-3.5" />
                  </svg>
                </span>
                <span className="detail-refresh-copy">
                  <strong>다시 계산</strong>
                  <span>같은 조건으로 시뮬레이션을 새로 실행합니다.</span>
                </span>
              </button>
            ) : null}
          </div>

          {simulationError ? <div className="error-banner simulation-error-banner">{simulationError}</div> : null}

          {detailedSimulationResult && isSimulationExpanded ? (
            <div className="simulation-panel">
              <div className="simulation-summary-grid">
                {simulationComparisonCards?.map((card) => {
                  const delta = card.simulated - card.expected;
                  const valueType = card.valueType ?? "number";
                  const deltaText =
                    valueType === "percent" ? `${formatComparisonDelta(delta)}%` : formatComparisonDelta(delta);

                  return (
                    <div
                      className={`simulation-stat-card simulation-compare-card ${card.accentClassName}`}
                      key={card.key}
                    >
                      <div className="simulation-compare-header">
                        <span className="result-label">{card.label}</span>
                        <strong>{deltaText}</strong>
                      </div>
                      <div className="simulation-compare-values">
                        <div>
                          <span className="result-label">{card.expectedLabel ?? "기대값"}</span>
                          <strong>{formatComparisonValue(card.expected, valueType)}</strong>
                        </div>
                        <div>
                          <span className="result-label">{card.simulatedLabel ?? "시뮬레이션"}</span>
                          <strong>{formatComparisonValue(card.simulated, valueType)}</strong>
                        </div>
                      </div>
                    </div>
                  );
                })}

                {simulationMetaItems ? (
                  <div className="simulation-stat-card simulation-meta-card">
                    {simulationMetaItems.map((item) => (
                      <div key={item.key}>
                        <span className="result-label">{item.label}</span>
                        <strong>{item.value}</strong>
                      </div>
                    ))}
                  </div>
                ) : null}
              </div>

              <div className="simulation-distribution-grid">
                {isClassicMode ? (
                  <div className="simulation-block">
                    <div className="simulation-block-header">
                      <h3>누적 비용 분포</h3>
                    </div>
                    {cumulativeChart ? (
                      <div className="cumulative-chart-panel">
                        <div className="cumulative-chart-legend">
                          {(
                            Object.entries(CUMULATIVE_METRIC_CONFIG) as Array<
                              [CumulativeMetricKey, (typeof CUMULATIVE_METRIC_CONFIG)[CumulativeMetricKey]]
                            >
                          ).map(([metricKey, config]) => (
                            <button
                              key={metricKey}
                              type="button"
                              className={
                                activeCumulativeMetric === metricKey
                                  ? `cumulative-legend-item is-button is-active ${config.className}`
                                  : `cumulative-legend-item is-button ${config.className}`
                              }
                              onClick={() => setActiveCumulativeMetric(metricKey)}
                            >
                              {config.label}
                            </button>
                          ))}
                        </div>
                        <div className="cumulative-chart-summary">
                          <div>
                            <span className="result-label">선택 지표</span>
                            <strong>{activeCumulativeConfig.label}</strong>
                          </div>
                          <div>
                            <span className="result-label">표시 구간</span>
                            <strong>{`${cumulativeChart.firstUpperBound.toFixed(0)} ~ ${cumulativeChart.lastUpperBound.toFixed(0)}`}</strong>
                          </div>
                          <div>
                            <span className="result-label">마지막 누적 비율</span>
                            <strong>{(cumulativeChart.lastShare * 100).toFixed(1)}%</strong>
                          </div>
                          <div>
                            <span className="result-label">설명</span>
                            <strong>{activeCumulativeConfig.caption}</strong>
                          </div>
                        </div>
                        <div className="cumulative-chart-shell">
                          <div className="cumulative-chart-stage">
                            <div className="cumulative-chart-svg" aria-label="누적 소모량 분포 그래프">
                              <ResponsiveContainer width="100%" height="100%">
                                <AreaChart
                                  data={cumulativeChart.chartData}
                                  margin={{ top: 18, right: 24, left: 18, bottom: 12 }}
                                >
                                  <defs>
                                    <linearGradient id="cumulativeWeightedAreaGradient" x1="0" y1="0" x2="0" y2="1">
                                      <stop offset="0%" stopColor="#ffc178" stopOpacity={0.34} />
                                      <stop offset="100%" stopColor="#ffc178" stopOpacity={0.03} />
                                    </linearGradient>
                                    <linearGradient id="cumulativeModuleAreaGradient" x1="0" y1="0" x2="0" y2="1">
                                      <stop offset="0%" stopColor="#83b0ff" stopOpacity={0.18} />
                                      <stop offset="100%" stopColor="#83b0ff" stopOpacity={0.02} />
                                    </linearGradient>
                                    <linearGradient id="cumulativeLockKeyAreaGradient" x1="0" y1="0" x2="0" y2="1">
                                      <stop offset="0%" stopColor="#7ec48f" stopOpacity={0.18} />
                                      <stop offset="100%" stopColor="#7ec48f" stopOpacity={0.02} />
                                    </linearGradient>
                                  </defs>
                                  <CartesianGrid
                                    stroke="rgba(255,255,255,0.08)"
                                    strokeDasharray="3 3"
                                    vertical={false}
                                  />
                                  <XAxis
                                    dataKey="costUpperBound"
                                    tickLine={false}
                                    axisLine={false}
                                    minTickGap={24}
                                    padding={{ left: 8, right: 8 }}
                                    tick={{ fill: "rgba(233, 238, 247, 0.62)", fontSize: 12 }}
                                  />
                                  <YAxis
                                    type="number"
                                    domain={[0, 100]}
                                    tickCount={3}
                                    tickFormatter={(value) => `${value}%`}
                                    tickLine={false}
                                    axisLine={false}
                                    width={44}
                                    padding={{ top: 8, bottom: 8 }}
                                    tick={{ fill: "rgba(233, 238, 247, 0.62)", fontSize: 12 }}
                                  />
                                  <Tooltip
                                    formatter={(value) => [
                                      `${Number(value).toFixed(1)}%`,
                                      activeCumulativeConfig.label,
                                    ]}
                                    labelFormatter={(label) => `≤ ${Number(label).toFixed(0)}`}
                                    contentStyle={{
                                      borderRadius: 14,
                                      border: "1px solid rgba(255,255,255,0.08)",
                                      background: "rgba(14, 19, 30, 0.96)",
                                      boxShadow: "0 16px 32px rgba(7, 12, 22, 0.28)",
                                    }}
                                    labelStyle={{ color: "#edf2fa", fontWeight: 700, marginBottom: 4 }}
                                    itemStyle={{ color: "#edf2fa", padding: 0 }}
                                    cursor={{ stroke: "rgba(255, 193, 120, 0.35)", strokeWidth: 1 }}
                                  />
                                  <Area
                                    type="monotone"
                                    name={activeCumulativeConfig.label}
                                    dataKey={activeCumulativeConfig.dataKey}
                                    stroke={activeCumulativeConfig.color}
                                    strokeWidth={2.5}
                                    fill={`url(#${activeCumulativeConfig.gradientId})`}
                                    dot={false}
                                    activeDot={{
                                      r: 4,
                                      stroke: "#171b28",
                                      strokeWidth: 2,
                                      fill: activeCumulativeConfig.color,
                                    }}
                                  />
                                </AreaChart>
                              </ResponsiveContainer>
                            </div>
                          </div>
                        </div>
                        {displayedCumulativeDistribution?.isTrimmed ? (
                          <p className="cumulative-chart-caption">
                            누적 {(displayedCumulativeDistribution.trimmedShare! * 100).toFixed(1)}% 이후 꼬리 구간은
                            생략했습니다.
                          </p>
                        ) : null}
                      </div>
                    ) : null}
                  </div>
                ) : null}

                <div className="simulation-block">
                  <div className="simulation-block-header">
                    <h3>{isClassicMode ? "도달 상태 분포" : "종결 상태 분포"}</h3>
                    <span className="section-caption">
                      순서 통합 후 {terminalPie?.groupedStateCount ?? 0}개 그룹, 원본{" "}
                      {detailedSimulationResult.terminalStateCount}종
                    </span>
                  </div>
                  {terminalPie ? (
                    <div className="terminal-pie-layout">
                      <div className="terminal-pie-chart-shell">
                        <ResponsiveContainer width="100%" height="100%">
                          <PieChart>
                            <Tooltip
                              content={({ active, payload }) =>
                                renderTerminalPieTooltip(
                                  active,
                                  payload as ReadonlyArray<{ payload: TerminalPieDatum }> | undefined,
                                )
                              }
                            />
                            <Pie
                              data={terminalPie.pieData}
                              dataKey="share"
                              nameKey="label"
                              innerRadius="52%"
                              outerRadius="84%"
                              paddingAngle={2}
                              stroke="rgba(12, 17, 28, 0.85)"
                              strokeWidth={2}
                              onMouseEnter={(_, index) => {
                                const entry = terminalPie.pieData[index];
                                setActiveTerminalPieKey(entry?.key ?? null);
                              }}
                              onMouseLeave={() => setActiveTerminalPieKey(null)}
                            >
                              {terminalPie.pieData.map((entry) => (
                                <Cell
                                  key={entry.key}
                                  fill={entry.color}
                                  fillOpacity={
                                    activeTerminalPieKey === null || activeTerminalPieKey === entry.key ? 1 : 0.42
                                  }
                                  strokeWidth={activeTerminalPieKey === entry.key ? 4 : 2}
                                />
                              ))}
                            </Pie>
                          </PieChart>
                        </ResponsiveContainer>
                      </div>
                      <div className="terminal-pie-legend">
                        {terminalPie.pieData.map((entry) => (
                          <div
                            className={`${
                              entry.outcome === "success"
                                ? "terminal-pie-legend-row is-success"
                                : entry.outcome === "failure"
                                  ? "terminal-pie-legend-row is-failure"
                                  : "terminal-pie-legend-row"
                            }${
                              activeTerminalPieKey === null
                                ? ""
                                : activeTerminalPieKey === entry.key
                                  ? " is-active"
                                  : " is-dimmed"
                            }`}
                            key={`legend-${entry.key}`}
                            onMouseEnter={() => setActiveTerminalPieKey(entry.key)}
                            onMouseLeave={() => setActiveTerminalPieKey(null)}
                          >
                            <span className="terminal-pie-swatch" style={{ backgroundColor: entry.color }} />
                            <div className="terminal-pie-copy">
                              <strong>{entry.label}</strong>
                              {entry.secondaryLabel ? <span>{entry.secondaryLabel}</span> : null}
                            </div>
                            <div className="terminal-pie-stats">
                              <strong>{(entry.share * 100).toFixed(1)}%</strong>
                              <span>{`${entry.count.toLocaleString()}회`}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
            </div>
          ) : null}
        </>
      ) : (
        <div className="empty-state">
          <p>
            {isClassicMode
              ? "왼쪽에서 시작 상태와 목표 조건을 정한 뒤 최적화 실행을 누르면 결과가 여기에 표시됩니다."
              : "왼쪽에서 모듈 예산과 목표 조건을 정한 뒤 예산 기반 실행을 누르면 결과가 여기에 표시됩니다."}
          </p>
        </div>
      )}
    </div>
  );
}
