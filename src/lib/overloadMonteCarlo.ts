import {
  defaultCostWeights,
  lockCosts,
  lockKeyCosts,
  overloadGradeProbabilities,
  overloadOptions,
  rerollCosts,
  slotOptionProbabilities,
  type OverloadCostWeights,
  type OverloadOptionTarget,
} from "./overloadOptions";
import { type OverloadBudgetOptimizationResult } from "./overloadBudgetOptimizer";
import { type OverloadPolicyOptimizationResult, type OverloadState } from "./overloadPolicyOptimizer";

const DEFAULT_TRIAL_COUNT = 5000;
const DEFAULT_MAX_STEPS = 10000;
const DEFAULT_COST_BUCKET_SIZE = 5;

type SimulationEpisodeResult = {
  totalCost: number;
  totalModuleCost: number;
  totalLockKeyCost: number;
  terminalState: OverloadState;
  outcome: "success" | "failure";
};

export type MonteCarloCostBucket = {
  upperBound: number;
  cumulativeCount: number;
  cumulativeShare: number;
};

export type MonteCarloTerminalStateStat = {
  state: OverloadState;
  count: number;
  share: number;
  outcome: "success" | "failure";
};

export type MonteCarloSimulationSummary = {
  trialCount: number;
  estimatedCost: number;
  estimatedModuleCost: number;
  estimatedLockKeyCost: number;
  estimatedConvertedLockKeyCost: number;
  estimatedSuccessProbability: number;
  sampleMean: number;
  sampleMeanModuleCost: number;
  sampleMeanLockKeyCost: number;
  sampleMeanConvertedLockKeyCost: number;
  sampleSuccessRate: number;
  successCount: number;
  failureCount: number;
  standardError: number;
  successRateStandardError: number;
  cumulativeCostDistribution: MonteCarloCostBucket[];
  cumulativeModuleCostDistribution: MonteCarloCostBucket[];
  cumulativeLockKeyCostDistribution: MonteCarloCostBucket[];
  cumulativeConvertedLockKeyCostDistribution: MonteCarloCostBucket[];
  terminalStateDistribution: MonteCarloTerminalStateStat[];
  terminalStateCount: number;
};

type MonteCarloSimulationOptions = {
  trialCount?: number;
  maxSteps?: number;
  costBucketSize?: number;
  costWeights?: OverloadCostWeights;
};

type FlatPolicyTransition = {
  isTerminal: boolean;
  weightedCost: number;
  moduleCost: number;
  lockKeyCost: number;
  nextStateIndexes: Int32Array;
  nextStateCumulativeProbabilities: Float64Array;
};

type FlatBudgetTransition = {
  terminalOutcome: "success" | "failure" | null;
  weightedCost: number;
  moduleCost: number;
  lockKeyCost: number;
  nextRemainingBudget: number;
  nextStateIndexes: Int32Array;
  nextStateCumulativeProbabilities: Float64Array;
};

const ACTION_DONE = 0;
const ACTION_GRADE = 2;
const OPTION_RADIX = overloadOptions.length;

function* iterateOverloadStates() {
  for (let o1 = 0; o1 < overloadOptions.length; o1++) {
    for (let o2 = 0; o2 < overloadOptions.length; o2++) {
      if (o1 === o2 && o1 !== 0) continue;
      for (let o3 = 0; o3 < overloadOptions.length; o3++) {
        if ((o1 === o3 || o2 === o3) && o3 !== 0) continue;
        for (let g1 = 0; g1 <= (o1 ? 1 : 0); g1++) {
          for (let g2 = 0; g2 <= (o2 ? 1 : 0); g2++) {
            for (let g3 = 0; g3 <= (o3 ? 1 : 0); g3++) {
              for (let m1 = 0; m1 <= (o1 ? 1 : 0); m1++) {
                for (let m2 = 0; m2 <= (o2 ? 1 : 0); m2++) {
                  for (let m3 = 0; m3 <= (o3 ? 1 : 0); m3++) {
                    if (m1 + m2 + m3 > 2) continue;
                    yield [o1, o2, o3, g1, g2, g3, m1, m2, m3] as OverloadState;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

function countBits(mask: number) {
  return Number(Boolean(mask & 1)) + Number(Boolean(mask & 2)) + Number(Boolean(mask & 4));
}

function buildMask([b1, b2, b3]: [number, number, number]) {
  return b1 | (b2 << 1) | (b3 << 2);
}

function actionModuleMask(
  action: Exclude<
    OverloadPolicyOptimizationResult["stateValues"][number][number][number][number][number][number][number][number][number]["action"],
    { type: "done" }
  >,
) {
  return buildMask([Number(action.moduleLock[0]), Number(action.moduleLock[1]), Number(action.moduleLock[2])]);
}

function actionKeyMask(
  action: Exclude<
    OverloadPolicyOptimizationResult["stateValues"][number][number][number][number][number][number][number][number][number]["action"],
    { type: "done" }
  >,
) {
  return buildMask([Number(action.keyLock[0]), Number(action.keyLock[1]), Number(action.keyLock[2])]);
}

function getStateValue(
  stateValues: OverloadPolicyOptimizationResult["stateValues"],
  [o1, o2, o3, g1, g2, g3, m1, m2, m3]: OverloadState,
) {
  return stateValues[o1][o2][o3][g1][g2][g3][m1][m2][m3];
}

function buildSimulationData(targetGrades: OverloadOptionTarget[]) {
  const optionIndexById = new Map<string, number>();
  const gradeTailProbabilityByThreshold = Array<number>(overloadGradeProbabilities.length + 1).fill(0);
  const optionProbabilityByIndex = overloadOptions.map((option) => option?.probability ?? 0);

  for (let index = 1; index < overloadOptions.length; index++) {
    const option = overloadOptions[index];
    if (option) {
      optionIndexById.set(option.id, index);
    }
  }

  for (let grade = overloadGradeProbabilities.length - 1; grade >= 0; grade--) {
    gradeTailProbabilityByThreshold[grade] =
      gradeTailProbabilityByThreshold[grade + 1]! + overloadGradeProbabilities[grade]!;
  }

  const requiredGradeByOption = Array(overloadOptions.length).fill(0);
  for (const target of targetGrades) {
    const optionIndex = optionIndexById.get(target.id);
    if (optionIndex !== undefined) {
      requiredGradeByOption[optionIndex] = target.grade;
    }
  }

  const successProbabilityByOption = requiredGradeByOption.map(
    (requiredGrade) => gradeTailProbabilityByThreshold[requiredGrade]!,
  );

  return {
    optionProbabilityByIndex,
    successProbabilityByOption,
  };
}

function buildActionCosts(currentModuleMask: number, nextModuleMask: number, keyMask: number, protectedCount: number) {
  const keptModuleCount = countBits(currentModuleMask & nextModuleMask);
  const nextModuleCount = countBits(nextModuleMask);
  const moduleCost = rerollCosts[protectedCount]! + lockCosts[nextModuleCount]! - lockCosts[keptModuleCount]!;
  const lockKeyCost = lockKeyCosts[countBits(keyMask)]!;
  return {
    moduleCost,
    lockKeyCost,
  };
}

function createStateKey([o1, o2, o3, g1, g2, g3, m1, m2, m3]: OverloadState) {
  return `${o1},${o2},${o3},${g1},${g2},${g3},${m1},${m2},${m3}`;
}

function encodeStateKey([o1, o2, o3, g1, g2, g3, m1, m2, m3]: OverloadState) {
  let key = o1;
  key = key * OPTION_RADIX + o2;
  key = key * OPTION_RADIX + o3;
  key = key * 2 + g1;
  key = key * 2 + g2;
  key = key * 2 + g3;
  key = key * 2 + m1;
  key = key * 2 + m2;
  key = key * 2 + m3;
  return key;
}

function buildGradeTransitionDistribution(
  state: OverloadState,
  nextModuleMask: number,
  protectedMask: number,
  successProbabilityByOption: number[],
  stateIndexByKey: Int32Array,
) {
  const distribution = new Map<number, number>();

  for (let gradeMask = 0; gradeMask < 8; gradeMask++) {
    const nextState = [...state] as OverloadState;
    let probability = 1;

    for (let slot = 0; slot < 3; slot++) {
      const option = state[slot]!;
      if (option === 0) {
        nextState[slot + 3] = 0;
        continue;
      }

      if ((protectedMask >> slot) & 1) {
        nextState[slot + 3] = state[slot + 3]!;
        continue;
      }

      const nextGrade = (gradeMask >> slot) & 1;
      nextState[slot + 3] = nextGrade;
      probability *= nextGrade ? successProbabilityByOption[option]! : 1 - successProbabilityByOption[option]!;
    }

    nextState[6] = nextModuleMask & 1 ? 1 : 0;
    nextState[7] = nextModuleMask & 2 ? 1 : 0;
    nextState[8] = nextModuleMask & 4 ? 1 : 0;

    if (probability <= 0) {
      continue;
    }

    const nextStateIndex = stateIndexByKey[encodeStateKey(nextState)];
    distribution.set(nextStateIndex, (distribution.get(nextStateIndex) ?? 0) + probability);
  }

  return distribution;
}

function buildOptionTransitionDistribution(
  state: OverloadState,
  nextModuleMask: number,
  protectedMask: number,
  states: OverloadState[],
  successProbabilityByOption: number[],
  optionProbabilityByIndex: number[],
  stateIndexByKey: Int32Array,
) {
  const distribution = new Map<number, number>();

  candidateLoop: for (const candidate of states) {
    for (let slot = 0; slot < 3; slot++) {
      if (candidate[slot + 6] !== ((nextModuleMask >> slot) & 1 ? 1 : 0)) {
        continue candidateLoop;
      }
    }

    let probability = 1;
    for (let slot = 0; slot < 3; slot++) {
      const currentOption = state[slot]!;
      const currentGrade = state[slot + 3]!;
      const nextOption = candidate[slot]!;
      const nextGrade = candidate[slot + 3]!;

      if ((protectedMask >> slot) & 1 && (currentOption !== nextOption || currentGrade !== nextGrade)) {
        continue candidateLoop;
      }

      if (nextOption === 0) {
        probability *= 1 - slotOptionProbabilities[slot]!;
        continue;
      }

      const successProbability = successProbabilityByOption[nextOption]!;
      probability *=
        slotOptionProbabilities[slot]! *
        optionProbabilityByIndex[nextOption]! *
        (nextGrade ? successProbability : 1 - successProbability);
    }

    if (probability <= 0) {
      continue;
    }

    const nextStateIndex = stateIndexByKey[encodeStateKey(candidate)];
    distribution.set(nextStateIndex, (distribution.get(nextStateIndex) ?? 0) + probability);
  }

  return distribution;
}

function buildBudgetTableIndex(result: OverloadBudgetOptimizationResult, remainingBudget: number, stateIndex: number) {
  return remainingBudget * result.stateCount + stateIndex;
}

function getBudgetStateIndex(result: OverloadBudgetOptimizationResult, state: OverloadState) {
  return result.stateIndexByKey[encodeStateKey(state)];
}

function readBudgetStateValue(result: OverloadBudgetOptimizationResult, remainingBudget: number, state: OverloadState) {
  const stateIndex = getBudgetStateIndex(result, state);
  if (stateIndex === -1) {
    return null;
  }

  const tableIndex = buildBudgetTableIndex(result, remainingBudget, stateIndex);
  return {
    stateIndex,
    tableIndex,
    successProbability: result.successProbabilityTable[tableIndex] ?? 0,
    expectedLockKeyCost: result.expectedLockKeyCostTable[tableIndex] ?? 0,
    actionType: result.actionTypeTable[tableIndex] ?? -1,
    actionModuleMask: result.actionModuleMaskTable[tableIndex] ?? -1,
    actionKeyMask: result.actionKeyMaskTable[tableIndex] ?? -1,
  };
}

function isBetterBudgetState(
  leftProbability: number,
  leftExpectedLockKeyCost: number,
  rightProbability: number,
  rightExpectedLockKeyCost: number,
) {
  if (leftProbability > rightProbability + 1e-12) {
    return true;
  }

  if (Math.abs(leftProbability - rightProbability) <= 1e-12) {
    return leftExpectedLockKeyCost + 1e-12 < rightExpectedLockKeyCost;
  }

  return false;
}

function normalizeTransitionDistribution(distribution: Map<number, number>) {
  const entries = Array.from(distribution.entries()).sort(([left], [right]) => left - right);
  const totalProbability = entries.reduce((sum, [, probability]) => sum + probability, 0);
  if (totalProbability <= 0) {
    throw new Error("유효한 전이 확률을 만들지 못했습니다.");
  }

  let cumulative = 0;
  return entries.map(([stateIndex, probability], entryIndex) => {
    cumulative += probability / totalProbability;
    return {
      stateIndex,
      cumulativeProbability: entryIndex === entries.length - 1 ? 1 : cumulative,
    };
  });
}

function buildStateIndexByKey(states: OverloadState[]) {
  const stateIndexByKey = new Int32Array(OPTION_RADIX * OPTION_RADIX * OPTION_RADIX * 2 * 2 * 2 * 2 * 2 * 2).fill(-1);

  for (let stateIndex = 0; stateIndex < states.length; stateIndex++) {
    stateIndexByKey[encodeStateKey(states[stateIndex]!)] = stateIndex;
  }

  return stateIndexByKey;
}

function buildFlatPolicyTransition(
  stateIndex: number,
  result: OverloadPolicyOptimizationResult,
  stateIndexByKey: Int32Array,
  successProbabilityByOption: number[],
  optionProbabilityByIndex: number[],
  costWeights: OverloadCostWeights,
): FlatPolicyTransition {
  const state = result.states[stateIndex]!;
  const currentValue = getStateValue(result.stateValues, state);
  if (currentValue.action.type === "done") {
    return {
      isTerminal: true,
      weightedCost: 0,
      moduleCost: 0,
      lockKeyCost: 0,
      nextStateIndexes: new Int32Array(0),
      nextStateCumulativeProbabilities: new Float64Array(0),
    };
  }

  const nextModuleMask = actionModuleMask(currentValue.action);
  const keyMask = actionKeyMask(currentValue.action);
  const protectedMask = nextModuleMask | keyMask;
  const currentModuleMask = buildMask([state[6]!, state[7]!, state[8]!]);
  const { moduleCost, lockKeyCost } = buildActionCosts(
    currentModuleMask,
    nextModuleMask,
    keyMask,
    countBits(protectedMask),
  );
  const sampledDistribution =
    currentValue.action.type === "grade"
      ? buildGradeTransitionDistribution(
          state,
          nextModuleMask,
          protectedMask,
          successProbabilityByOption,
          stateIndexByKey,
        )
      : buildOptionTransitionDistribution(
          state,
          nextModuleMask,
          protectedMask,
          result.states,
          successProbabilityByOption,
          optionProbabilityByIndex,
          stateIndexByKey,
        );

  const effectiveDistribution = new Map<number, number>();
  for (const [sampledStateIndex, probability] of sampledDistribution) {
    const sampledState = result.states[sampledStateIndex]!;
    const sampledValue = getStateValue(result.stateValues, sampledState);
    const nextStateIndex = sampledValue.cost < currentValue.cost - 1e-9 ? sampledStateIndex : stateIndex;
    effectiveDistribution.set(nextStateIndex, (effectiveDistribution.get(nextStateIndex) ?? 0) + probability);
  }

  const normalizedTransitions = normalizeTransitionDistribution(effectiveDistribution);
  return {
    isTerminal: false,
    weightedCost: costWeights.module * moduleCost + costWeights.lockKey * lockKeyCost,
    moduleCost,
    lockKeyCost,
    nextStateIndexes: Int32Array.from(normalizedTransitions.map((transition) => transition.stateIndex)),
    nextStateCumulativeProbabilities: Float64Array.from(
      normalizedTransitions.map((transition) => transition.cumulativeProbability),
    ),
  };
}

function buildPrebuiltPolicyTransitions(
  startStateIndex: number,
  result: OverloadPolicyOptimizationResult,
  stateIndexByKey: Int32Array,
  successProbabilityByOption: number[],
  optionProbabilityByIndex: number[],
  costWeights: OverloadCostWeights,
) {
  const transitions = Array<FlatPolicyTransition | undefined>(result.states.length);
  const queued = new Uint8Array(result.states.length);
  const queue = [startStateIndex];
  queued[startStateIndex] = 1;

  while (queue.length > 0) {
    const stateIndex = queue.pop()!;
    if (transitions[stateIndex]) {
      continue;
    }

    const transition = buildFlatPolicyTransition(
      stateIndex,
      result,
      stateIndexByKey,
      successProbabilityByOption,
      optionProbabilityByIndex,
      costWeights,
    );
    transitions[stateIndex] = transition;

    if (transition.isTerminal) {
      continue;
    }

    for (const nextStateIndex of transition.nextStateIndexes) {
      if (!queued[nextStateIndex] && !transitions[nextStateIndex]) {
        queued[nextStateIndex] = 1;
        queue.push(nextStateIndex);
      }
    }
  }

  return transitions;
}

function buildFlatBudgetTransition(
  remainingBudget: number,
  stateIndex: number,
  states: OverloadState[],
  result: OverloadBudgetOptimizationResult,
  successProbabilityByOption: number[],
  optionProbabilityByIndex: number[],
  costWeights: OverloadCostWeights,
) {
  const state = states[stateIndex]!;
  const currentValue = readBudgetStateValue(result, remainingBudget, state);
  if (!currentValue || currentValue.actionType === -1) {
    return {
      terminalOutcome: "failure" as const,
      weightedCost: 0,
      moduleCost: 0,
      lockKeyCost: 0,
      nextRemainingBudget: remainingBudget,
      nextStateIndexes: new Int32Array(0),
      nextStateCumulativeProbabilities: new Float64Array(0),
    };
  }

  if (currentValue.actionType === ACTION_DONE) {
    return {
      terminalOutcome: "success" as const,
      weightedCost: 0,
      moduleCost: 0,
      lockKeyCost: 0,
      nextRemainingBudget: remainingBudget,
      nextStateIndexes: new Int32Array(0),
      nextStateCumulativeProbabilities: new Float64Array(0),
    };
  }

  const nextModuleMask = currentValue.actionModuleMask;
  const keyMask = currentValue.actionKeyMask;
  const protectedMask = nextModuleMask | keyMask;
  const currentModuleMask = buildMask([state[6]!, state[7]!, state[8]!]);
  const { moduleCost, lockKeyCost } = buildActionCosts(
    currentModuleMask,
    nextModuleMask,
    keyMask,
    countBits(protectedMask),
  );
  const nextRemainingBudget = remainingBudget - moduleCost;
  if (nextRemainingBudget < 0) {
    return {
      terminalOutcome: "failure" as const,
      weightedCost: 0,
      moduleCost: 0,
      lockKeyCost: 0,
      nextRemainingBudget: remainingBudget,
      nextStateIndexes: new Int32Array(0),
      nextStateCumulativeProbabilities: new Float64Array(0),
    };
  }

  const sampledDistribution =
    currentValue.actionType === ACTION_GRADE
      ? buildGradeTransitionDistribution(
          state,
          nextModuleMask,
          protectedMask,
          successProbabilityByOption,
          result.stateIndexByKey,
        )
      : buildOptionTransitionDistribution(
          state,
          nextModuleMask,
          protectedMask,
          states,
          successProbabilityByOption,
          optionProbabilityByIndex,
          result.stateIndexByKey,
        );

  const keptStateValue = readBudgetStateValue(result, nextRemainingBudget, state);
  const effectiveDistribution = new Map<number, number>();
  for (const [sampledStateIndex, probability] of sampledDistribution) {
    const sampledState = states[sampledStateIndex]!;
    const sampledStateValue = readBudgetStateValue(result, nextRemainingBudget, sampledState);
    const nextStateIndex =
      sampledStateValue &&
      keptStateValue &&
      isBetterBudgetState(
        sampledStateValue.successProbability,
        sampledStateValue.expectedLockKeyCost,
        keptStateValue.successProbability,
        keptStateValue.expectedLockKeyCost,
      )
        ? sampledStateIndex
        : stateIndex;
    effectiveDistribution.set(nextStateIndex, (effectiveDistribution.get(nextStateIndex) ?? 0) + probability);
  }

  const normalizedTransitions = normalizeTransitionDistribution(effectiveDistribution);
  return {
    terminalOutcome: null,
    weightedCost: costWeights.module * moduleCost + costWeights.lockKey * lockKeyCost,
    moduleCost,
    lockKeyCost,
    nextRemainingBudget,
    nextStateIndexes: Int32Array.from(normalizedTransitions.map((transition) => transition.stateIndex)),
    nextStateCumulativeProbabilities: Float64Array.from(
      normalizedTransitions.map((transition) => transition.cumulativeProbability),
    ),
  };
}

function buildPrebuiltBudgetTransitions(
  startStateIndex: number,
  states: OverloadState[],
  result: OverloadBudgetOptimizationResult,
  successProbabilityByOption: number[],
  optionProbabilityByIndex: number[],
  costWeights: OverloadCostWeights,
) {
  const transitions = new Map<number, FlatBudgetTransition>();
  const queued = new Set<number>();
  const startTableIndex = buildBudgetTableIndex(result, result.moduleBudget, startStateIndex);
  const queue = [startTableIndex];
  queued.add(startTableIndex);

  while (queue.length > 0) {
    const tableIndex = queue.pop()!;
    if (transitions.has(tableIndex)) {
      continue;
    }

    const remainingBudget = Math.floor(tableIndex / result.stateCount);
    const stateIndex = tableIndex % result.stateCount;
    const transition = buildFlatBudgetTransition(
      remainingBudget,
      stateIndex,
      states,
      result,
      successProbabilityByOption,
      optionProbabilityByIndex,
      costWeights,
    );
    transitions.set(tableIndex, transition);

    if (transition.terminalOutcome !== null) {
      continue;
    }

    for (const nextStateIndex of transition.nextStateIndexes) {
      const nextTableIndex = buildBudgetTableIndex(result, transition.nextRemainingBudget, nextStateIndex);
      if (!queued.has(nextTableIndex) && !transitions.has(nextTableIndex)) {
        queued.add(nextTableIndex);
        queue.push(nextTableIndex);
      }
    }
  }

  return transitions;
}

function simulateFlatPolicyEpisode(
  startStateIndex: number,
  result: OverloadPolicyOptimizationResult,
  transitions: Array<FlatPolicyTransition | undefined>,
  maxSteps: number,
): SimulationEpisodeResult {
  let stateIndex = startStateIndex;
  let totalCost = 0;
  let totalModuleCost = 0;
  let totalLockKeyCost = 0;

  for (let step = 0; step < maxSteps; step++) {
    const transition = transitions[stateIndex];
    if (!transition) {
      throw new Error("Prebuilt policy transition is missing for a reachable state");
    }

    if (transition.isTerminal) {
      return {
        totalCost,
        totalModuleCost,
        totalLockKeyCost,
        terminalState: [...result.states[stateIndex]!] as OverloadState,
        outcome: "success",
      };
    }

    totalCost += transition.weightedCost;
    totalModuleCost += transition.moduleCost;
    totalLockKeyCost += transition.lockKeyCost;

    const draw = Math.random();
    let nextStateIndex = transition.nextStateIndexes[transition.nextStateIndexes.length - 1]!;
    for (let transitionIndex = 0; transitionIndex < transition.nextStateIndexes.length; transitionIndex++) {
      if (draw <= transition.nextStateCumulativeProbabilities[transitionIndex]!) {
        nextStateIndex = transition.nextStateIndexes[transitionIndex]!;
        break;
      }
    }

    stateIndex = nextStateIndex;
  }

  throw new Error("Simulation exceeded max steps before reaching a done state");
}

function simulateFlatBudgetEpisode(
  startStateIndex: number,
  states: OverloadState[],
  result: OverloadBudgetOptimizationResult,
  transitions: Map<number, FlatBudgetTransition>,
  maxSteps: number,
): SimulationEpisodeResult {
  let stateIndex = startStateIndex;
  let remainingBudget = result.moduleBudget;
  let totalCost = 0;
  let totalModuleCost = 0;
  let totalLockKeyCost = 0;

  for (let step = 0; step < maxSteps; step++) {
    const tableIndex = buildBudgetTableIndex(result, remainingBudget, stateIndex);
    const transition = transitions.get(tableIndex);
    if (!transition) {
      throw new Error("Prebuilt budget transition is missing for a reachable state-budget pair");
    }

    if (transition.terminalOutcome !== null) {
      return {
        totalCost,
        totalModuleCost,
        totalLockKeyCost,
        terminalState: [...states[stateIndex]!] as OverloadState,
        outcome: transition.terminalOutcome,
      };
    }

    totalCost += transition.weightedCost;
    totalModuleCost += transition.moduleCost;
    totalLockKeyCost += transition.lockKeyCost;

    const draw = Math.random();
    let nextStateIndex = transition.nextStateIndexes[transition.nextStateIndexes.length - 1]!;
    for (let transitionIndex = 0; transitionIndex < transition.nextStateIndexes.length; transitionIndex++) {
      if (draw <= transition.nextStateCumulativeProbabilities[transitionIndex]!) {
        nextStateIndex = transition.nextStateIndexes[transitionIndex]!;
        break;
      }
    }

    remainingBudget = transition.nextRemainingBudget;
    stateIndex = nextStateIndex;
  }

  throw new Error("Budget simulation exceeded max steps before reaching a terminal state");
}

function buildCumulativeCostDistribution(costSamples: number[], costBucketSize: number) {
  const bucketCounts = new Map<number, number>();

  for (const cost of costSamples) {
    const bucketStart = Math.floor(cost / costBucketSize) * costBucketSize;
    bucketCounts.set(bucketStart, (bucketCounts.get(bucketStart) ?? 0) + 1);
  }

  const sortedBuckets = Array.from(bucketCounts.entries()).sort(([left], [right]) => left - right);
  const distribution: MonteCarloCostBucket[] = [];

  let cumulativeCount = 0;
  for (const [bucketStart, count] of sortedBuckets) {
    cumulativeCount += count;
    distribution.push({
      upperBound: bucketStart + costBucketSize,
      cumulativeCount,
      cumulativeShare: cumulativeCount / costSamples.length,
    });
  }

  return distribution;
}

function buildTerminalStateDistribution(
  terminalStates: Array<{ state: OverloadState; outcome: "success" | "failure" }>,
) {
  const terminalCounts = new Map<string, { state: OverloadState; count: number; outcome: "success" | "failure" }>();

  for (const { state, outcome } of terminalStates) {
    const key = `${outcome}|${createStateKey(state)}`;
    const entry = terminalCounts.get(key);
    if (entry) {
      entry.count += 1;
      continue;
    }

    terminalCounts.set(key, { state: [...state] as OverloadState, count: 1, outcome });
  }

  const sortedTerminalStates = Array.from(terminalCounts.values())
    .sort((left, right) => right.count - left.count)
    .map(({ state, count, outcome }) => ({
      state,
      count,
      share: count / terminalStates.length,
      outcome,
    }));

  return {
    terminalStateDistribution: sortedTerminalStates,
    terminalStateCount: terminalCounts.size,
  };
}

export function runMonteCarloPolicySimulation(
  startState: OverloadState,
  result: OverloadPolicyOptimizationResult,
  targetGrades: OverloadOptionTarget[],
  options: MonteCarloSimulationOptions = {},
): MonteCarloSimulationSummary {
  const trialCount = options.trialCount ?? DEFAULT_TRIAL_COUNT;
  const maxSteps = options.maxSteps ?? DEFAULT_MAX_STEPS;
  const costBucketSize = options.costBucketSize ?? DEFAULT_COST_BUCKET_SIZE;
  const costWeights = options.costWeights ?? defaultCostWeights;

  const { successProbabilityByOption, optionProbabilityByIndex } = buildSimulationData(targetGrades);
  const stateIndexByKey = buildStateIndexByKey(result.states);
  const startStateIndex = stateIndexByKey[encodeStateKey(startState)];
  const transitions = buildPrebuiltPolicyTransitions(
    startStateIndex,
    result,
    stateIndexByKey,
    successProbabilityByOption,
    optionProbabilityByIndex,
    costWeights,
  );
  const episodes = Array.from({ length: trialCount }, () =>
    simulateFlatPolicyEpisode(startStateIndex, result, transitions, maxSteps),
  );

  const costSamples = episodes.map((episode) => episode.totalCost);
  const moduleCostSamples = episodes.map((episode) => episode.totalModuleCost);
  const lockKeyCostSamples = episodes.map((episode) => episode.totalLockKeyCost);
  const convertedLockKeyCostSamples = lockKeyCostSamples.map((value) => value * costWeights.lockKey);
  const terminalStates = episodes.map((episode) => ({ state: episode.terminalState, outcome: episode.outcome }));
  const sampleMean = costSamples.reduce((sum, value) => sum + value, 0) / costSamples.length;
  const sampleMeanModuleCost = moduleCostSamples.reduce((sum, value) => sum + value, 0) / moduleCostSamples.length;
  const sampleMeanLockKeyCost = lockKeyCostSamples.reduce((sum, value) => sum + value, 0) / lockKeyCostSamples.length;
  const sampleMeanConvertedLockKeyCost =
    convertedLockKeyCostSamples.reduce((sum, value) => sum + value, 0) / convertedLockKeyCostSamples.length;
  const sampleVariance =
    costSamples.reduce((sum, value) => sum + (value - sampleMean) ** 2, 0) / Math.max(1, costSamples.length - 1);
  const standardError = Math.sqrt(sampleVariance / costSamples.length);
  const successCount = episodes.length;
  const failureCount = 0;
  const { terminalStateDistribution, terminalStateCount } = buildTerminalStateDistribution(terminalStates);
  const startStateValue = getStateValue(result.stateValues, startState);

  return {
    trialCount,
    estimatedCost: startStateValue.cost,
    estimatedModuleCost: startStateValue.expectedCosts.module,
    estimatedLockKeyCost: startStateValue.expectedCosts.lockKey,
    estimatedConvertedLockKeyCost: startStateValue.expectedCosts.lockKey * costWeights.lockKey,
    estimatedSuccessProbability: 1,
    sampleMean,
    sampleMeanModuleCost,
    sampleMeanLockKeyCost,
    sampleMeanConvertedLockKeyCost,
    sampleSuccessRate: 1,
    successCount,
    failureCount,
    standardError,
    successRateStandardError: 0,
    cumulativeCostDistribution: buildCumulativeCostDistribution(costSamples, costBucketSize),
    cumulativeModuleCostDistribution: buildCumulativeCostDistribution(moduleCostSamples, costBucketSize),
    cumulativeLockKeyCostDistribution: buildCumulativeCostDistribution(lockKeyCostSamples, costBucketSize),
    cumulativeConvertedLockKeyCostDistribution: buildCumulativeCostDistribution(
      convertedLockKeyCostSamples,
      costBucketSize,
    ),
    terminalStateDistribution,
    terminalStateCount,
  };
}

export function runMonteCarloBudgetSimulation(
  startState: OverloadState,
  result: OverloadBudgetOptimizationResult,
  targetGrades: OverloadOptionTarget[],
  options: MonteCarloSimulationOptions = {},
): MonteCarloSimulationSummary {
  const trialCount = options.trialCount ?? DEFAULT_TRIAL_COUNT;
  const maxSteps = options.maxSteps ?? DEFAULT_MAX_STEPS;
  const costBucketSize = options.costBucketSize ?? DEFAULT_COST_BUCKET_SIZE;
  const costWeights = options.costWeights ?? defaultCostWeights;

  const states = Array.from(iterateOverloadStates());
  const { successProbabilityByOption, optionProbabilityByIndex } = buildSimulationData(targetGrades);
  const startStateIndex = result.stateIndexByKey[encodeStateKey(startState)];
  const transitions = buildPrebuiltBudgetTransitions(
    startStateIndex,
    states,
    result,
    successProbabilityByOption,
    optionProbabilityByIndex,
    costWeights,
  );
  const episodes = Array.from({ length: trialCount }, () =>
    simulateFlatBudgetEpisode(startStateIndex, states, result, transitions, maxSteps),
  );

  const costSamples = episodes.map((episode) => episode.totalCost);
  const moduleCostSamples = episodes.map((episode) => episode.totalModuleCost);
  const lockKeyCostSamples = episodes.map((episode) => episode.totalLockKeyCost);
  const convertedLockKeyCostSamples = lockKeyCostSamples.map((value) => value * costWeights.lockKey);
  const terminalStates = episodes.map((episode) => ({ state: episode.terminalState, outcome: episode.outcome }));
  const sampleMean = costSamples.reduce((sum, value) => sum + value, 0) / costSamples.length;
  const sampleMeanModuleCost = moduleCostSamples.reduce((sum, value) => sum + value, 0) / moduleCostSamples.length;
  const sampleMeanLockKeyCost = lockKeyCostSamples.reduce((sum, value) => sum + value, 0) / lockKeyCostSamples.length;
  const sampleMeanConvertedLockKeyCost =
    convertedLockKeyCostSamples.reduce((sum, value) => sum + value, 0) / convertedLockKeyCostSamples.length;
  const sampleVariance =
    costSamples.reduce((sum, value) => sum + (value - sampleMean) ** 2, 0) / Math.max(1, costSamples.length - 1);
  const standardError = Math.sqrt(sampleVariance / costSamples.length);
  const successCount = episodes.filter((episode) => episode.outcome === "success").length;
  const failureCount = episodes.length - successCount;
  const sampleSuccessRate = successCount / episodes.length;
  const successRateVariance = sampleSuccessRate * (1 - sampleSuccessRate);
  const { terminalStateDistribution, terminalStateCount } = buildTerminalStateDistribution(terminalStates);
  const startStateValue = readBudgetStateValue(result, result.moduleBudget, startState);

  return {
    trialCount,
    estimatedCost: result.moduleBudget,
    estimatedModuleCost: result.moduleBudget,
    estimatedLockKeyCost: startStateValue?.expectedLockKeyCost ?? 0,
    estimatedConvertedLockKeyCost: (startStateValue?.expectedLockKeyCost ?? 0) * costWeights.lockKey,
    estimatedSuccessProbability: startStateValue?.successProbability ?? 0,
    sampleMean,
    sampleMeanModuleCost,
    sampleMeanLockKeyCost,
    sampleMeanConvertedLockKeyCost,
    sampleSuccessRate,
    successCount,
    failureCount,
    standardError,
    successRateStandardError: Math.sqrt(successRateVariance / episodes.length),
    cumulativeCostDistribution: buildCumulativeCostDistribution(costSamples, costBucketSize),
    cumulativeModuleCostDistribution: buildCumulativeCostDistribution(moduleCostSamples, costBucketSize),
    cumulativeLockKeyCostDistribution: buildCumulativeCostDistribution(lockKeyCostSamples, costBucketSize),
    cumulativeConvertedLockKeyCostDistribution: buildCumulativeCostDistribution(
      convertedLockKeyCostSamples,
      costBucketSize,
    ),
    terminalStateDistribution,
    terminalStateCount,
  };
}
