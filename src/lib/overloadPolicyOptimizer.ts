import {
  defaultCostWeights,
  lockCosts,
  lockKeyCosts,
  overloadGradeProbabilities,
  overloadOptions,
  OVERLOAD_GRADE_COUNT,
  OVERLOAD_OPTION_COUNT,
  rerollCosts,
  slotOptionProbabilities,
  type OverloadCostWeights,
  type OverloadOptionIds,
  type OverloadOptionTarget,
} from "./overloadOptions";

// 비트마스크 관련 상수.
// 각 비트는 3개 슬롯 중 어떤 슬롯을 보호하거나 유지할지를 나타낸다.
const MASKS = [0, 1, 2, 3, 4, 5, 6] as const;
const SINGLE_MASK_INDEX = [-1, 0, 1, -1, 2, -1, -1] as const;
const PAIR_MASK_INDEX = [-1, -1, -1, 0, -1, 1, 2] as const;

// 상태를 flat 배열로 다루기 위한 인코딩 관련 상수.
const OPTION_RADIX = OVERLOAD_OPTION_COUNT + 1;
const STATE_KEY_SIZE = OPTION_RADIX * OPTION_RADIX * OPTION_RADIX * 2 * 2 * 2 * 2 * 2 * 2;
const SLOT_VALUE_KEY_SIZE = (OVERLOAD_OPTION_COUNT + 1) * 2;
const TWO_SLOT_KEY_SIZE = SLOT_VALUE_KEY_SIZE * SLOT_VALUE_KEY_SIZE;

// 내부적으로 행동 종류도 숫자 코드로 다뤄서 분기 비용을 줄인다.
const ACTION_DONE = 0;
const ACTION_OPTION = 1;
const ACTION_GRADE = 2;
const MASK_COUNT = MASKS.length;
const KEY_COUNT_PER_STATE = 3;
// 등급 재설정은 3개의 성공/실패 비트만 있으면 되므로 결과 경우의 수는 최대 8개다.
const GRADE_MASK_COUNT = 8;
const MAX_GRADE_TRANSITIONS_PER_MASK = GRADE_MASK_COUNT;
// 3개 슬롯에서 보호 가능한 슬롯이 최대 2개이므로 가능한 옵션 재설정 행동 수는 이 상한 안에 들어간다.
const MAX_OPTION_ACTION_CANDIDATES = 19;
const ACTION_COST_TABLE_SIZE = MASK_COUNT * MASK_COUNT * MASK_COUNT;
const SINGLE_WEIGHT_INDEX_BY_SLOT = [1, 2, 4] as const;
const PAIR_WEIGHT_INDEX_BY_SLOT = [3, 5, 6] as const;

export type SlotLockState = [boolean, boolean, boolean];
export type OverloadAction =
  | {
      type: "option" | "grade";
      moduleLock: SlotLockState;
      keyLock: SlotLockState;
    }
  | { type: "done" };
export type OverloadState = [number, number, number, number, number, number, number, number, number];

type AggregateSums = {
  singleProbSums: Float64Array[];
  singleDistSums: Float64Array[];
  pairProbSums: Float64Array[];
  pairDistSums: Float64Array[];
  allProbSum: number;
  allDistSum: number;
};
type ExpectationAggregatePair = {
  module: AggregateSums[];
  lockKey: AggregateSums[];
};

export type OverloadExpectedCosts = {
  module: number;
  lockKey: number;
};

export type OverloadForcedLockAlternative = {
  protectedMask: SlotLockState;
  action: Exclude<OverloadAction, { type: "done" }>;
  cost: number;
  expectedCosts: OverloadExpectedCosts;
  deltaFromOptimal: number;
  isCurrentOptimal: boolean;
};

export type OverloadStateTensor<T> = T[][][][][][][][][];
export type OverloadStateValue = { cost: number; expectedCosts: OverloadExpectedCosts; action: OverloadAction };

export type OverloadPolicyOptimizationResult = {
  stateValues: OverloadStateTensor<OverloadStateValue>;
  iterationsRun: number;
  states: OverloadState[];
};

export type OverloadOptimizationProgress = {
  phase: "policy" | "expectation" | "done";
  completedIterations: number;
  totalIterations: number;
  percent: number;
};

export type OptimizeOverloadPolicyOptions = {
  onProgress?: (progress: OverloadOptimizationProgress) => void;
  yieldEveryIterations?: number;
};

// 진행도 표시가 필요한 경우에만 이벤트 루프에 제어를 잠깐 반환한다.
async function yieldToMainThread() {
  await new Promise<void>((resolve) => {
    globalThis.setTimeout(resolve, 0);
  });
}

// --------
// 비트마스크 / 상태 인코딩 유틸리티
// --------

function countBits(mask: number) {
  return Number(Boolean(mask & 1)) + Number(Boolean(mask & 2)) + Number(Boolean(mask & 4));
}

function decodeMask(mask: number): [number, number, number] {
  return [mask & 1 ? 1 : 0, mask & 2 ? 1 : 0, mask & 4 ? 1 : 0];
}

function decodeBooleanMask(mask: number): SlotLockState {
  return [Boolean(mask & 1), Boolean(mask & 2), Boolean(mask & 4)];
}

function buildMask([b1, b2, b3]: [number, number, number]) {
  return b1 | (b2 << 1) | (b3 << 2);
}

function moduleMaskFromState([, , , , , , m1, m2, m3]: OverloadState) {
  return buildMask([m1, m2, m3]);
}

function encodeOptionTripleKey(o1: number, o2: number, o3: number) {
  return (o1 * OPTION_RADIX + o2) * OPTION_RADIX + o3;
}

function encodeStateKeyFromParts(
  o1: number,
  o2: number,
  o3: number,
  g1: number,
  g2: number,
  g3: number,
  m1: number,
  m2: number,
  m3: number,
) {
  // 9차원 상태를 조밀한 정수 키로 압축해 내부에서는 flat typed array 인덱스로 사용한다.
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

function encodeStateKey([o1, o2, o3, g1, g2, g3, m1, m2, m3]: OverloadState) {
  return encodeStateKeyFromParts(o1, o2, o3, g1, g2, g3, m1, m2, m3);
}

function buildActionFromMasks(actionType: number, moduleMask: number, keyMask: number): OverloadAction {
  if (actionType === ACTION_DONE) {
    return { type: "done" };
  }

  return {
    type: actionType === ACTION_OPTION ? "option" : "grade",
    moduleLock: decodeBooleanMask(moduleMask),
    keyLock: decodeBooleanMask(keyMask),
  };
}

function buildRequiredGradeSuccessProbabilities(targetGradeTargets: OverloadOptionTarget[]) {
  const optionIndexById = new Map<string, number>();
  const gradeTailProbabilityByThreshold = Array<number>(OVERLOAD_GRADE_COUNT + 1).fill(0);

  for (let index = 1; index < overloadOptions.length; index++) {
    const option = overloadOptions[index];
    if (option) {
      optionIndexById.set(option.id, index);
    }
  }

  for (let grade = OVERLOAD_GRADE_COUNT - 1; grade >= 0; grade--) {
    gradeTailProbabilityByThreshold[grade] =
      gradeTailProbabilityByThreshold[grade + 1]! + overloadGradeProbabilities[grade]!;
  }

  const requiredGradeByOption = Array(OVERLOAD_OPTION_COUNT + 1).fill(0);
  for (const target of targetGradeTargets) {
    const optionIndex = optionIndexById.get(target.id);
    if (optionIndex !== undefined) {
      requiredGradeByOption[optionIndex] = target.grade;
    }
  }

  return requiredGradeByOption.map((requiredGrade) => {
    const successProbability = gradeTailProbabilityByThreshold[requiredGrade]!;
    return [1 - successProbability, successProbability] as [number, number];
  });
}

function readStateValueFromExactState(
  stateValues: OverloadStateTensor<OverloadStateValue>,
  [o1, o2, o3, g1, g2, g3, m1, m2, m3]: OverloadState,
) {
  return stateValues[o1][o2][o3][g1][g2][g3][m1][m2][m3];
}

function masksMatchAction(action: OverloadAction, actionType: number, nextModuleMask: number, keyMask: number) {
  if (action.type === "done") {
    return false;
  }

  if (
    (actionType === ACTION_OPTION && action.type !== "option") ||
    (actionType === ACTION_GRADE && action.type !== "grade")
  ) {
    return false;
  }

  return (
    buildMask([Number(action.moduleLock[0]), Number(action.moduleLock[1]), Number(action.moduleLock[2])]) ===
      nextModuleMask &&
    buildMask([Number(action.keyLock[0]), Number(action.keyLock[1]), Number(action.keyLock[2])]) === keyMask
  );
}

export function readForcedLockAlternatives(
  result: OverloadPolicyOptimizationResult,
  state: OverloadState,
  targetOptionIds: OverloadOptionIds[],
  targetGradeTargets: OverloadOptionTarget[],
  costWeights: OverloadCostWeights = defaultCostWeights,
): OverloadForcedLockAlternative[] {
  const currentStateValue = readStateValueFromExactState(result.stateValues, state);
  if (currentStateValue.action.type === "done") {
    return [];
  }

  const currentModuleMask = moduleMaskFromState(state);
  const [o1, o2, o3, g1, g2, g3] = state;
  const unresolvedSlotMask =
    (o1 !== 0 && g1 === 0 ? 1 : 0) | (o2 !== 0 && g2 === 0 ? 2 : 0) | (o3 !== 0 && g3 === 0 ? 4 : 0);
  const stateKey = encodeStateKey(state);
  const meetsTargetGradeProbabilities = buildRequiredGradeSuccessProbabilities(targetGradeTargets);
  const alternatives: OverloadForcedLockAlternative[] = [];

  const isProtectedMaskCompatibleWithTargets = (protectedMask: number) => {
    if (protectedMask === 0) {
      return true;
    }

    return targetOptionIds.some((targetState) => {
      for (let slot = 0; slot < 3; slot++) {
        if (((protectedMask >> slot) & 1) === 0) {
          continue;
        }

        const currentOptionId = overloadOptions[state[slot]]?.id;
        const targetOptionId = targetState[slot];
        if (targetOptionId !== undefined && currentOptionId !== targetOptionId) {
          return false;
        }
      }

      return true;
    });
  };

  const chooseBetterAlternative = (
    current: OverloadForcedLockAlternative | null,
    candidate: OverloadForcedLockAlternative,
  ) => {
    if (!current) {
      return candidate;
    }

    if (candidate.cost < current.cost - 1e-9) {
      return candidate;
    }

    if (Math.abs(candidate.cost - current.cost) <= 1e-9) {
      if (candidate.expectedCosts.lockKey < current.expectedCosts.lockKey - 1e-9) {
        return candidate;
      }

      if (
        Math.abs(candidate.expectedCosts.lockKey - current.expectedCosts.lockKey) <= 1e-9 &&
        candidate.expectedCosts.module < current.expectedCosts.module - 1e-9
      ) {
        return candidate;
      }
    }

    return current;
  };

  const buildAlternative = (
    actionType: number,
    nextModuleMask: number,
    keyMask: number,
    sameWeight: number,
    totalWeight: number,
    externalWeightedCostSum: number,
    externalModuleCostSum: number,
    externalLockKeyCostSum: number,
  ): OverloadForcedLockAlternative | null => {
    const leaveWeight = totalWeight - sameWeight;
    if (leaveWeight <= 1e-12) {
      return null;
    }

    const protectedMask = nextModuleMask | keyMask;
    const actionCosts = buildActionCosts(
      currentModuleMask,
      nextModuleMask,
      keyMask,
      countBits(protectedMask),
      costWeights,
    );
    const expectedCosts = {
      module: (actionCosts.moduleCost * totalWeight + externalModuleCostSum) / leaveWeight,
      lockKey: (actionCosts.lockKeyCost * totalWeight + externalLockKeyCostSum) / leaveWeight,
    };
    const cost = (actionCosts.weightedCost * totalWeight + externalWeightedCostSum) / leaveWeight;

    return {
      protectedMask: decodeBooleanMask(protectedMask),
      action: buildActionFromMasks(actionType, nextModuleMask, keyMask) as Exclude<OverloadAction, { type: "done" }>,
      cost,
      expectedCosts,
      deltaFromOptimal: cost - currentStateValue.cost,
      isCurrentOptimal: masksMatchAction(currentStateValue.action, actionType, nextModuleMask, keyMask),
    };
  };

  const evaluateGradeAction = (protectedMask: number, nextModuleMask: number) => {
    const keyMask = protectedMask ^ nextModuleMask;
    let sameWeight = 0;
    let totalWeight = 0;
    let externalWeightedCostSum = 0;
    let externalModuleCostSum = 0;
    let externalLockKeyCostSum = 0;

    for (let gradeMask = 0; gradeMask < GRADE_MASK_COUNT; gradeMask++) {
      const ng1 = gradeMask & 1 ? 1 : 0;
      const ng2 = gradeMask & 2 ? 1 : 0;
      const ng3 = gradeMask & 4 ? 1 : 0;
      if ((protectedMask & 1) !== 0 && ng1 !== g1) continue;
      if ((protectedMask & 2) !== 0 && ng2 !== g2) continue;
      if ((protectedMask & 4) !== 0 && ng3 !== g3) continue;

      let probability = 1;
      if ((protectedMask & 1) === 0) probability *= meetsTargetGradeProbabilities[o1]![ng1]!;
      if ((protectedMask & 2) === 0) probability *= meetsTargetGradeProbabilities[o2]![ng2]!;
      if ((protectedMask & 4) === 0) probability *= meetsTargetGradeProbabilities[o3]![ng3]!;
      if (probability <= 0) continue;

      totalWeight += probability;
      const nextState = [
        o1,
        o2,
        o3,
        ng1,
        ng2,
        ng3,
        Number(Boolean(nextModuleMask & 1)),
        Number(Boolean(nextModuleMask & 2)),
        Number(Boolean(nextModuleMask & 4)),
      ] as OverloadState;
      if (encodeStateKey(nextState) === stateKey) {
        sameWeight += probability;
        continue;
      }

      const nextStateValue = readStateValueFromExactState(result.stateValues, nextState);
      externalWeightedCostSum += probability * nextStateValue.cost;
      externalModuleCostSum += probability * nextStateValue.expectedCosts.module;
      externalLockKeyCostSum += probability * nextStateValue.expectedCosts.lockKey;
    }

    return buildAlternative(
      ACTION_GRADE,
      nextModuleMask,
      keyMask,
      sameWeight,
      totalWeight,
      externalWeightedCostSum,
      externalModuleCostSum,
      externalLockKeyCostSum,
    );
  };

  const evaluateOptionAction = (protectedMask: number, nextModuleMask: number) => {
    const keyMask = protectedMask ^ nextModuleMask;
    let totalWeight = 0;
    const weightedOutcomes: Array<{
      weight: number;
      cost: number;
      moduleCost: number;
      lockKeyCost: number;
    }> = [];

    for (const candidateState of result.states) {
      if (
        candidateState[6] !== Number(Boolean(nextModuleMask & 1)) ||
        candidateState[7] !== Number(Boolean(nextModuleMask & 2)) ||
        candidateState[8] !== Number(Boolean(nextModuleMask & 4))
      ) {
        continue;
      }

      let weight = 1;
      for (let slot = 0; slot < 3; slot++) {
        const currentOption = state[slot]!;
        const currentGrade = state[slot + 3]!;
        const nextOption = candidateState[slot]!;
        const nextGrade = candidateState[slot + 3]!;

        if (((protectedMask >> slot) & 1) !== 0) {
          if (currentOption !== nextOption || currentGrade !== nextGrade) {
            weight = 0;
            break;
          }
          continue;
        }

        if (nextOption === 0) {
          if (nextGrade !== 0) {
            weight = 0;
            break;
          }

          weight *= 1 - slotOptionProbabilities[slot]!;
          continue;
        }

        weight *=
          slotOptionProbabilities[slot]! *
          overloadOptions[nextOption]!.probability *
          meetsTargetGradeProbabilities[nextOption]![nextGrade]!;
      }

      if (weight <= 0) {
        continue;
      }

      totalWeight += weight;
      if (encodeStateKey(candidateState) === stateKey) {
        continue;
      }

      const nextStateValue = readStateValueFromExactState(result.stateValues, candidateState);
      weightedOutcomes.push({
        weight,
        cost: nextStateValue.cost,
        moduleCost: nextStateValue.expectedCosts.module,
        lockKeyCost: nextStateValue.expectedCosts.lockKey,
      });
    }

    if (weightedOutcomes.length === 0) {
      return null;
    }

    weightedOutcomes.sort((left, right) => left.cost - right.cost);

    const protectedMaskValue = nextModuleMask | keyMask;
    const actionCosts = buildActionCosts(
      currentModuleMask,
      nextModuleMask,
      keyMask,
      countBits(protectedMaskValue),
      costWeights,
    );
    let acceptedWeight = 0;
    let acceptedWeightedCostSum = 0;
    let acceptedModuleCostSum = 0;
    let acceptedLockKeyCostSum = 0;

    for (let outcomeIndex = 0; outcomeIndex < weightedOutcomes.length; outcomeIndex++) {
      const outcome = weightedOutcomes[outcomeIndex]!;
      acceptedWeight += outcome.weight;
      acceptedWeightedCostSum += outcome.weight * outcome.cost;
      acceptedModuleCostSum += outcome.weight * outcome.moduleCost;
      acceptedLockKeyCostSum += outcome.weight * outcome.lockKeyCost;

      const cost = (actionCosts.weightedCost * totalWeight + acceptedWeightedCostSum) / acceptedWeight;
      const lowerBound = outcome.cost;
      const upperBound = weightedOutcomes[outcomeIndex + 1]?.cost ?? Number.POSITIVE_INFINITY;
      if (cost <= lowerBound + 1e-9 || cost > upperBound + 1e-9) {
        continue;
      }

      const expectedCosts = {
        module: (actionCosts.moduleCost * totalWeight + acceptedModuleCostSum) / acceptedWeight,
        lockKey: (actionCosts.lockKeyCost * totalWeight + acceptedLockKeyCostSum) / acceptedWeight,
      };

      return {
        protectedMask: decodeBooleanMask(protectedMaskValue),
        action: buildActionFromMasks(ACTION_OPTION, nextModuleMask, keyMask) as Exclude<
          OverloadAction,
          { type: "done" }
        >,
        cost,
        expectedCosts,
        deltaFromOptimal: cost - currentStateValue.cost,
        isCurrentOptimal: masksMatchAction(currentStateValue.action, ACTION_OPTION, nextModuleMask, keyMask),
      };
    }

    return null;
  };

  for (const protectedMask of MASKS) {
    if (!canUseMask([o1, o2, o3], protectedMask)) {
      continue;
    }

    if (!isProtectedMaskCompatibleWithTargets(protectedMask)) {
      continue;
    }

    if ((protectedMask & unresolvedSlotMask) !== 0) {
      continue;
    }

    let bestAlternative: OverloadForcedLockAlternative | null = null;
    for (const nextModuleMask of MASKS) {
      if (!canUseMask([o1, o2, o3], nextModuleMask)) {
        continue;
      }

      if ((nextModuleMask & protectedMask) !== nextModuleMask) {
        continue;
      }

      const gradeAlternative = evaluateGradeAction(protectedMask, nextModuleMask);
      if (gradeAlternative) {
        bestAlternative = chooseBetterAlternative(bestAlternative, gradeAlternative);
      }

      const optionAlternative = evaluateOptionAction(protectedMask, nextModuleMask);
      if (optionAlternative) {
        bestAlternative = chooseBetterAlternative(bestAlternative, optionAlternative);
      }
    }

    if (bestAlternative) {
      alternatives.push(bestAlternative);
    }
  }

  alternatives.sort((left, right) => left.cost - right.cost);
  return alternatives;
}

// 외부 API는 기존 9차원 tensor 형태를 유지하므로 마지막 반환용 구조도 그대로 제공한다.
function createOverloadStateTensor<T>(factory: () => T): OverloadStateTensor<T> {
  return Array.from({ length: OVERLOAD_OPTION_COUNT + 1 }, () =>
    Array.from({ length: OVERLOAD_OPTION_COUNT + 1 }, () =>
      Array.from({ length: OVERLOAD_OPTION_COUNT + 1 }, () =>
        Array.from({ length: 2 }, () =>
          Array.from({ length: 2 }, () =>
            Array.from({ length: 2 }, () =>
              Array.from({ length: 2 }, () => Array.from({ length: 2 }, () => Array.from({ length: 2 }, factory))),
            ),
          ),
        ),
      ),
    ),
  );
}

// 가능한 모든 유효 상태를 순회한다.
// 중복 옵션, 존재하지 않는 슬롯, 모듈 잠금 3개 같은 불가능한 상태는 여기서 제거한다.
function* iterateOverloadStates() {
  for (let o1 = 0; o1 <= OVERLOAD_OPTION_COUNT; o1++) {
    for (let o2 = 0; o2 <= OVERLOAD_OPTION_COUNT; o2++) {
      if (o1 === o2 && o1 !== 0) continue;
      for (let o3 = 0; o3 <= OVERLOAD_OPTION_COUNT; o3++) {
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

// 주어진 슬롯 조합에서 해당 마스크를 실제로 사용할 수 있는지 확인한다.
function canUseMask([o1, o2, o3]: [number, number, number], mask: number) {
  if (countBits(mask) > 2) return false;
  if (mask & 1 && o1 === 0) return false;
  if (mask & 2 && o2 === 0) return false;
  if (mask & 4 && o3 === 0) return false;
  return true;
}

// 목표 상태는 "옵션 일치 + 필요한 등급 충족"까지 만족해야 달성으로 본다.
function matchesTargetPattern(state: OverloadState, [t1, t2, t3]: [number, number, number]) {
  const [o1, o2, o3, g1, g2, g3] = state;

  if (t1 !== 0 && o1 !== t1) return false;
  if (t2 !== 0 && o2 !== t2) return false;
  if (t3 !== 0 && o3 !== t3) return false;

  if (t1 !== 0 && g1 !== 1) return false;
  if (t2 !== 0 && g2 !== 1) return false;
  if (t3 !== 0 && g3 !== 1) return false;

  return true;
}

// --------
// 목표 조건에서 파생되는 확률 / 인덱스 데이터 전처리
// --------

function buildDerivedOverloadData(targetOptionIds: OverloadOptionIds[], targetGradeTargets: OverloadOptionTarget[]) {
  const optionIndexById = new Map<string, number>();
  const optionProbabilityByIndex = overloadOptions.map((option) => option?.probability ?? 0);
  const gradeTailProbabilityByThreshold = Array<number>(OVERLOAD_GRADE_COUNT + 1).fill(0);

  for (let index = 1; index < overloadOptions.length; index++) {
    const option = overloadOptions[index];
    if (option) {
      optionIndexById.set(option.id, index);
    }
  }

  for (let grade = OVERLOAD_GRADE_COUNT - 1; grade >= 0; grade--) {
    gradeTailProbabilityByThreshold[grade] =
      gradeTailProbabilityByThreshold[grade + 1]! + overloadGradeProbabilities[grade]!;
  }

  const targetStates = targetOptionIds.map(
    (optionIds) =>
      optionIds.map((id) => {
        if (!id) {
          return 0;
        }

        const optionIndex = optionIndexById.get(id);
        if (optionIndex === undefined) {
          throw new Error(`알 수 없는 목표 옵션 ID입니다: ${id}`);
        }

        return optionIndex;
      }) as [number, number, number],
  );

  const requiredGradeByOption = Array(OVERLOAD_OPTION_COUNT + 1).fill(0);
  for (const target of targetGradeTargets) {
    const optionIndex = optionIndexById.get(target.id);
    if (optionIndex !== undefined) {
      requiredGradeByOption[optionIndex] = target.grade;
    }
  }

  const meetsTargetGradeProbabilities = requiredGradeByOption.map((requiredGrade) => {
    const successProbability = gradeTailProbabilityByThreshold[requiredGrade]!;
    return [1 - successProbability, successProbability] as [number, number];
  });

  return {
    meetsTargetGradeProbabilities,
    optionProbabilityByIndex,
    targetStates,
  };
}

// 옵션 재설정 기대값 계산에서 쓰는 누적 합 버퍼.
// 단일 슬롯 호환, 2슬롯 호환, 전체 호환 경우를 각각 따로 쌓아 빠르게 조회한다.
function createAggregateSums(): AggregateSums {
  return {
    singleProbSums: Array.from({ length: 3 }, () => new Float64Array(SLOT_VALUE_KEY_SIZE)),
    singleDistSums: Array.from({ length: 3 }, () => new Float64Array(SLOT_VALUE_KEY_SIZE)),
    pairProbSums: Array.from({ length: 3 }, () => new Float64Array(TWO_SLOT_KEY_SIZE)),
    pairDistSums: Array.from({ length: 3 }, () => new Float64Array(TWO_SLOT_KEY_SIZE)),
    allProbSum: 0,
    allDistSum: 0,
  };
}

// 기대값 집계 버퍼는 반복마다 다시 할당하지 않고 0으로만 초기화해서 재사용한다.
function resetAggregateSums(aggregates: AggregateSums) {
  aggregates.allProbSum = 0;
  aggregates.allDistSum = 0;
  for (let slot = 0; slot < 3; slot++) {
    aggregates.singleProbSums[slot]!.fill(0);
    aggregates.singleDistSums[slot]!.fill(0);
    aggregates.pairProbSums[slot]!.fill(0);
    aggregates.pairDistSums[slot]!.fill(0);
  }
}

// 행동 비용 테이블을 1차원 배열로 저장하기 위한 인덱스 계산.
function buildActionCostIndex(currentModuleMask: number, nextModuleMask: number, keyMask: number) {
  return (currentModuleMask * MASK_COUNT + nextModuleMask) * MASK_COUNT + keyMask;
}

// 즉시 소모되는 모듈 / 락키 / 가중 비용을 계산한다.
// keptModuleCount를 따로 계산하는 이유는 기존에 잠겨 있던 모듈 보호 혜택을 비용에 반영하기 위해서다.
function buildActionCosts(
  currentModuleMask: number,
  nextModuleMask: number,
  keyMask: number,
  protectedCount: number,
  costWeights: OverloadCostWeights,
) {
  const keptModuleCount = countBits(currentModuleMask & nextModuleMask);
  const nextModuleCount = countBits(nextModuleMask);
  const keyCount = countBits(keyMask);
  const moduleCost = rerollCosts[protectedCount]! + lockCosts[nextModuleCount]! - lockCosts[keptModuleCount]!;
  const lockKeyCost = lockKeyCosts[keyCount]!;
  return {
    moduleCost,
    lockKeyCost,
    weightedCost: costWeights.module * moduleCost + costWeights.lockKey * lockKeyCost,
  };
}

export async function optimizeOverloadPolicy(
  targetOptionIds: OverloadOptionIds[],
  targetGradeTargets: OverloadOptionTarget[],
  iterations: number,
  costWeights: OverloadCostWeights = defaultCostWeights,
  options: OptimizeOverloadPolicyOptions = {},
): Promise<OverloadPolicyOptimizationResult> {
  // 전체 계산은 크게 두 단계로 진행된다.
  // 1) policy improvement: 각 상태에서 어떤 행동이 최적인지 찾는다.
  // 2) policy evaluation: 선택된 행동 정책 아래의 기대 모듈/락키 소모량을 계산한다.
  const reportProgress = options.onProgress;
  const shouldYieldToMainThread = reportProgress !== undefined;
  const yieldEveryIterations = Math.max(1, options.yieldEveryIterations ?? 12);
  const INFINITE_COST = 1e5;
  const states = Array.from(iterateOverloadStates());
  const stateCount = states.length;
  const { meetsTargetGradeProbabilities, optionProbabilityByIndex, targetStates } = buildDerivedOverloadData(
    targetOptionIds,
    targetGradeTargets,
  );

  const costs = new Float64Array(stateCount).fill(INFINITE_COST);
  const expectedModuleCosts = new Float64Array(stateCount);
  const expectedLockKeyCosts = new Float64Array(stateCount);
  const actionTypeByState = new Int8Array(stateCount).fill(ACTION_OPTION);
  const actionModuleMaskByState = new Int8Array(stateCount);
  const actionKeyMaskByState = new Int8Array(stateCount);
  const stateIndexByKey = new Int32Array(STATE_KEY_SIZE).fill(-1);
  const slotValueKey = (option: number, grade: number) => option * 2 + grade;

  for (let stateIndex = 0; stateIndex < stateCount; stateIndex++) {
    stateIndexByKey[encodeStateKey(states[stateIndex]!)] = stateIndex;
  }

  // 상태별로 반복 참조되는 정보들을 미리 평탄화해 둔다.
  // 여기서 계산한 값들은 이후 핫루프에서 중복 계산을 피하기 위한 캐시 역할을 한다.
  const stateModuleMasks = new Int8Array(stateCount);
  const optionTripleKeysByState = new Int32Array(stateCount);
  const singleCompatibilityKeys = new Int32Array(stateCount * KEY_COUNT_PER_STATE);
  const pairCompatibilityKeys = new Int32Array(stateCount * KEY_COUNT_PER_STATE);
  // 등급 재설정은 가능한 module lock mask와 3비트 등급 결과마다 다음 상태를 미리 계산해 둔다.
  const nextStateIndexByModuleMaskAndGradeMask = new Int32Array(stateCount * MASK_COUNT * GRADE_MASK_COUNT).fill(-1);
  for (let stateIndex = 0; stateIndex < stateCount; stateIndex++) {
    const [o1, o2, o3, g1, g2, g3] = states[stateIndex]!;
    const key1 = slotValueKey(o1, g1);
    const key2 = slotValueKey(o2, g2);
    const key3 = slotValueKey(o3, g3);
    const keyOffset = stateIndex * KEY_COUNT_PER_STATE;
    const nextStateOffset = stateIndex * MASK_COUNT * GRADE_MASK_COUNT;

    stateModuleMasks[stateIndex] = moduleMaskFromState(states[stateIndex]!);
    optionTripleKeysByState[stateIndex] = encodeOptionTripleKey(o1, o2, o3);
    singleCompatibilityKeys[keyOffset] = key1;
    singleCompatibilityKeys[keyOffset + 1] = key2;
    singleCompatibilityKeys[keyOffset + 2] = key3;
    pairCompatibilityKeys[keyOffset] = key1 * SLOT_VALUE_KEY_SIZE + key2;
    pairCompatibilityKeys[keyOffset + 1] = key1 * SLOT_VALUE_KEY_SIZE + key3;
    pairCompatibilityKeys[keyOffset + 2] = key2 * SLOT_VALUE_KEY_SIZE + key3;

    for (const nextModuleMask of MASKS) {
      if (!canUseMask([o1, o2, o3], nextModuleMask)) {
        continue;
      }

      const [nm1, nm2, nm3] = decodeMask(nextModuleMask);
      const maskOffset = nextStateOffset + nextModuleMask * GRADE_MASK_COUNT;
      for (let gradeMask = 0; gradeMask < GRADE_MASK_COUNT; gradeMask++) {
        nextStateIndexByModuleMaskAndGradeMask[maskOffset + gradeMask] =
          stateIndexByKey[
            encodeStateKeyFromParts(
              o1,
              o2,
              o3,
              gradeMask & 1 ? 1 : 0,
              gradeMask & 2 ? 1 : 0,
              gradeMask & 4 ? 1 : 0,
              nm1,
              nm2,
              nm3,
            )
          ];
      }
    }
  }

  const optionTargetMatch = new Uint8Array(OPTION_RADIX * OPTION_RADIX * OPTION_RADIX);
  // 옵션 조합만 보고도 "목표 후보가 될 가능성"이 전혀 없는 상태는 빠르게 걸러낸다.
  for (let o1 = 0; o1 <= OVERLOAD_OPTION_COUNT; o1++) {
    for (let o2 = 0; o2 <= OVERLOAD_OPTION_COUNT; o2++) {
      for (let o3 = 0; o3 <= OVERLOAD_OPTION_COUNT; o3++) {
        if ((o1 === o2 && o1 !== 0) || (o1 === o3 && o1 !== 0) || (o2 === o3 && o2 !== 0)) {
          continue;
        }

        optionTargetMatch[encodeOptionTripleKey(o1, o2, o3)] = Number(
          targetStates.some(
            ([t1, t2, t3]) => (t1 === 0 || t1 === o1) && (t2 === 0 || t2 === o2) && (t3 === 0 || t3 === o3),
          ),
        );
      }
    }
  }

  for (let stateIndex = 0; stateIndex < stateCount; stateIndex++) {
    const state = states[stateIndex]!;
    if (!targetStates.some((targetState) => matchesTargetPattern(state, targetState))) {
      continue;
    }

    // 이미 목표를 만족한 상태는 종결 상태이므로 비용 0으로 시작한다.
    costs[stateIndex] = 0;
    actionTypeByState[stateIndex] = ACTION_DONE;
  }

  const optionTransitionWeightsByState = new Float64Array(stateCount * MASK_COUNT);
  // 등급 재설정 전이는 핫루프에서 작은 객체를 다시 만들지 않도록 병렬 배열로 평탄화한다.
  const gradeTransitionCountsByStateAndMask = new Uint8Array(stateCount * MASK_COUNT);
  const gradeTransitionGradeMasksByStateAndMask = new Uint8Array(
    stateCount * MASK_COUNT * MAX_GRADE_TRANSITIONS_PER_MASK,
  );
  const gradeTransitionProbabilitiesByStateAndMask = new Float64Array(
    stateCount * MASK_COUNT * MAX_GRADE_TRANSITIONS_PER_MASK,
  );
  for (let stateIndex = 0; stateIndex < stateCount; stateIndex++) {
    const [o1, o2, o3, g1, g2, g3] = states[stateIndex]!;
    // optionTransitionWeightsByState는 "어떤 슬롯을 유지했을 때" 곱해지는 확률 항을 미리 쪼개 저장한다.
    // 이후 누적 합 버퍼에서 단일 슬롯 / 2슬롯 호환 값을 빠르게 합치기 위한 준비 단계다.
    const term1 = o1
      ? slotOptionProbabilities[0] * overloadOptions[o1]!.probability * meetsTargetGradeProbabilities[o1][g1]
      : 1 - slotOptionProbabilities[0];
    const term2 = o2
      ? slotOptionProbabilities[1] * overloadOptions[o2]!.probability * meetsTargetGradeProbabilities[o2][g2]
      : 1 - slotOptionProbabilities[1];
    const term3 = o3
      ? slotOptionProbabilities[2] * overloadOptions[o3]!.probability * meetsTargetGradeProbabilities[o3][g3]
      : 1 - slotOptionProbabilities[2];
    const weightOffset = stateIndex * MASK_COUNT;
    optionTransitionWeightsByState[weightOffset] = term1 * term2 * term3;
    optionTransitionWeightsByState[weightOffset + 1] = term2 * term3;
    optionTransitionWeightsByState[weightOffset + 2] = term1 * term3;
    optionTransitionWeightsByState[weightOffset + 3] = term3;
    optionTransitionWeightsByState[weightOffset + 4] = term1 * term2;
    optionTransitionWeightsByState[weightOffset + 5] = term2;
    optionTransitionWeightsByState[weightOffset + 6] = term1;

    for (const protectedMask of MASKS) {
      if (!canUseMask([o1, o2, o3], protectedMask)) {
        continue;
      }

      const [p1, p2, p3] = decodeMask(protectedMask);
      const transitionOffset = (stateIndex * MASK_COUNT + protectedMask) * MAX_GRADE_TRANSITIONS_PER_MASK;
      let transitionCount = 0;

      for (let gradeMask = 0; gradeMask < GRADE_MASK_COUNT; gradeMask++) {
        const ng1 = gradeMask & 1 ? 1 : 0;
        const ng2 = gradeMask & 2 ? 1 : 0;
        const ng3 = gradeMask & 4 ? 1 : 0;
        if (o1 === 0 && ng1 !== 0) continue;
        if (o2 === 0 && ng2 !== 0) continue;
        if (o3 === 0 && ng3 !== 0) continue;
        if (p1 && ng1 !== g1) continue;
        if (p2 && ng2 !== g2) continue;
        if (p3 && ng3 !== g3) continue;

        let prob = 1;
        if (!p1 && o1 !== 0) prob *= meetsTargetGradeProbabilities[o1][ng1];
        if (!p2 && o2 !== 0) prob *= meetsTargetGradeProbabilities[o2][ng2];
        if (!p3 && o3 !== 0) prob *= meetsTargetGradeProbabilities[o3][ng3];
        if (prob <= 0) continue;

        gradeTransitionGradeMasksByStateAndMask[transitionOffset + transitionCount] = gradeMask;
        gradeTransitionProbabilitiesByStateAndMask[transitionOffset + transitionCount] = prob;
        transitionCount += 1;
      }

      gradeTransitionCountsByStateAndMask[stateIndex * MASK_COUNT + protectedMask] = transitionCount;
    }
  }

  const weightedActionCostByMaskTriplet = new Float64Array(ACTION_COST_TABLE_SIZE);
  const moduleActionCostByMaskTriplet = new Float64Array(ACTION_COST_TABLE_SIZE);
  const lockKeyActionCostByMaskTriplet = new Float64Array(ACTION_COST_TABLE_SIZE);
  // 행동 비용은 현재 잠금 상태, 다음 module lock, key lock 조합으로만 결정되므로 한 번 미리 계산해 둘 수 있다.
  for (const currentModuleMask of MASKS) {
    for (const nextModuleMask of MASKS) {
      for (const keyMask of MASKS) {
        const actionCostIndex = buildActionCostIndex(currentModuleMask, nextModuleMask, keyMask);
        const actionCosts = buildActionCosts(
          currentModuleMask,
          nextModuleMask,
          keyMask,
          countBits(nextModuleMask | keyMask),
          costWeights,
        );
        weightedActionCostByMaskTriplet[actionCostIndex] = actionCosts.weightedCost;
        moduleActionCostByMaskTriplet[actionCostIndex] = actionCosts.moduleCost;
        lockKeyActionCostByMaskTriplet[actionCostIndex] = actionCosts.lockKeyCost;
      }
    }
  }

  const getAggregatedOptionTransitionSums = (
    protectedMask: (typeof MASKS)[number],
    stateIndex: number,
    aggregates: AggregateSums,
  ) => {
    // protectedMask에 따라 필요한 호환성 범위가 달라진다.
    // 0이면 전체 합, 1개 보호면 단일 슬롯 기준 합, 2개 보호면 슬롯 쌍 기준 합을 꺼낸다.
    const keyOffset = stateIndex * KEY_COUNT_PER_STATE;
    if (protectedMask === 0) {
      return { probSum: aggregates.allProbSum, distSum: aggregates.allDistSum };
    }

    const singleSlot = SINGLE_MASK_INDEX[protectedMask];
    if (singleSlot !== -1) {
      const key = singleCompatibilityKeys[keyOffset + singleSlot]!;
      return {
        probSum: aggregates.singleProbSums[singleSlot]![key],
        distSum: aggregates.singleDistSums[singleSlot]![key],
      };
    }

    const pairSlot = PAIR_MASK_INDEX[protectedMask];
    if (pairSlot === -1) {
      return { probSum: 0, distSum: 0 };
    }

    const key = pairCompatibilityKeys[keyOffset + pairSlot]!;
    return {
      probSum: aggregates.pairProbSums[pairSlot]![key],
      distSum: aggregates.pairDistSums[pairSlot]![key],
    };
  };

  const getOptionTransitionProbabilityMass = (() => {
    let squaredOptionProbabilitySum = 0;
    let cubedOptionProbabilitySum = 0;

    for (let index = 0; index <= OVERLOAD_OPTION_COUNT; index++) {
      squaredOptionProbabilitySum += optionProbabilityByIndex[index]! ** 2;
      cubedOptionProbabilitySum += optionProbabilityByIndex[index]! ** 3;
    }

    return (o1: number, o2: number, o3: number, protectedMask: number) => {
      // 옵션 재설정은 "보호한 슬롯과 충돌하지 않는 유효 옵션 조합"만 남아야 하므로
      // 보호 슬롯 수에 따라 확률 질량을 닫힌 형태로 계산한다.
      const protectedCount = countBits(protectedMask);
      if (protectedCount > 2) {
        return 0;
      }

      const options = [o1, o2, o3] as const;

      if (protectedCount === 0) {
        return (
          1 -
          (slotOptionProbabilities[0] * slotOptionProbabilities[1] +
            slotOptionProbabilities[0] * slotOptionProbabilities[2] +
            slotOptionProbabilities[1] * slotOptionProbabilities[2]) *
            squaredOptionProbabilitySum +
          2 *
            slotOptionProbabilities[0] *
            slotOptionProbabilities[1] *
            slotOptionProbabilities[2] *
            cubedOptionProbabilitySum
        );
      }

      if (protectedCount === 1) {
        const protectedSlot = SINGLE_MASK_INDEX[protectedMask];
        if (protectedSlot === -1) return 0;

        const protectedOptionProbability = optionProbabilityByIndex[options[protectedSlot]]!;
        const unlockedSlots = [0, 1, 2].filter((slot) => slot !== protectedSlot) as [number, number];
        const [slotA, slotB] = unlockedSlots;
        return (
          1 -
          (slotOptionProbabilities[slotA] + slotOptionProbabilities[slotB]) * protectedOptionProbability +
          slotOptionProbabilities[slotA] *
            slotOptionProbabilities[slotB] *
            (2 * protectedOptionProbability * protectedOptionProbability - squaredOptionProbabilitySum)
        );
      }

      const protectedOptionProbabilitySum =
        optionProbabilityByIndex[o1]! + optionProbabilityByIndex[o2]! + optionProbabilityByIndex[o3]!;
      const unlockedSlot = [0, 1, 2].find((slot) => ((protectedMask >> slot) & 1) === 0);
      if (unlockedSlot === undefined) {
        return 0;
      }

      return (
        1 -
        slotOptionProbabilities[unlockedSlot] *
          (protectedOptionProbabilitySum - optionProbabilityByIndex[options[unlockedSlot]]!)
      );
    };
  })();

  const optionCandidateCountsByState = new Uint8Array(stateCount);
  const optionCandidateNextModuleMasksByState = new Int8Array(stateCount * MAX_OPTION_ACTION_CANDIDATES);
  const optionCandidateProtectedMasksByState = new Int8Array(stateCount * MAX_OPTION_ACTION_CANDIDATES);
  const optionCandidateKeyMasksByState = new Int8Array(stateCount * MAX_OPTION_ACTION_CANDIDATES);
  const optionCandidateProbabilityMassesByState = new Float64Array(stateCount * MAX_OPTION_ACTION_CANDIDATES);
  // 옵션 재설정 행동도 평탄화해서 정책 개선 루프가 조밀한 숫자 테이블만 순회하게 만든다.
  for (let stateIndex = 0; stateIndex < stateCount; stateIndex++) {
    const [o1, o2, o3] = states[stateIndex]!;
    const candidateOffset = stateIndex * MAX_OPTION_ACTION_CANDIDATES;
    let candidateCount = 0;

    for (const nextModuleMask of MASKS) {
      if (!canUseMask([o1, o2, o3], nextModuleMask)) continue;

      for (const protectedMask of MASKS) {
        if (!canUseMask([o1, o2, o3], protectedMask)) continue;
        if ((nextModuleMask & protectedMask) !== nextModuleMask) continue;

        const probabilityMass = getOptionTransitionProbabilityMass(o1, o2, o3, protectedMask);
        if (probabilityMass <= 0) continue;

        const candidateIndex = candidateOffset + candidateCount;
        optionCandidateNextModuleMasksByState[candidateIndex] = nextModuleMask;
        optionCandidateProtectedMasksByState[candidateIndex] = protectedMask;
        optionCandidateKeyMasksByState[candidateIndex] = protectedMask ^ nextModuleMask;
        optionCandidateProbabilityMassesByState[candidateIndex] = probabilityMass;
        candidateCount += 1;
      }
    }

    optionCandidateCountsByState[stateIndex] = candidateCount;
  }

  const orderedStateIndexesScratch = Array.from({ length: stateCount }, (_, index) => index);
  const orderedStateIndexesByModuleMask = MASKS.map(() => new Int32Array(stateCount));
  const orderedStateCountsByModuleMask = new Int32Array(MASK_COUNT);
  const optionSnapshotAggregatesByModuleMask = MASKS.map(() => createAggregateSums());
  const optionSnapshotCandidatePointers = new Int32Array(MASK_COUNT);
  const bestOptionCostScratch = new Float64Array(stateCount);
  const bestOptionModuleMaskScratch = new Int8Array(stateCount);
  const bestOptionKeyMaskScratch = new Int8Array(stateCount);

  const buildOptionRerollSnapshot = () => {
    // 옵션 재설정은 현재 반복에서 즉시 갱신된 값을 섞지 않고,
    // 이전 value function의 스냅샷을 기준으로 한 번에 평가한다.
    const snapshotCosts = Float64Array.from(costs);
    orderedStateIndexesScratch.sort((left, right) => snapshotCosts[left]! - snapshotCosts[right]!);
    orderedStateCountsByModuleMask.fill(0);
    optionSnapshotCandidatePointers.fill(0);
    bestOptionCostScratch.fill(INFINITE_COST);
    bestOptionModuleMaskScratch.fill(-1);
    bestOptionKeyMaskScratch.fill(-1);
    for (const moduleMask of MASKS) {
      resetAggregateSums(optionSnapshotAggregatesByModuleMask[moduleMask]!);
    }
    for (const stateIndex of orderedStateIndexesScratch) {
      const moduleMask = stateModuleMasks[stateIndex]!;
      orderedStateIndexesByModuleMask[moduleMask]![orderedStateCountsByModuleMask[moduleMask]!] = stateIndex;
      orderedStateCountsByModuleMask[moduleMask] += 1;
    }

    const addCandidate = (moduleMask: number, candidateIndex: number) => {
      // 비용이 더 낮은 상태부터 차례대로 aggregate에 누적해 두면,
      // 현재 상태보다 좋은 후보들만으로 기대 비용을 빠르게 계산할 수 있다.
      const aggregates = optionSnapshotAggregatesByModuleMask[moduleMask]!;
      const weightOffset = candidateIndex * MASK_COUNT;
      const keyOffset = candidateIndex * KEY_COUNT_PER_STATE;
      const candidateCost = snapshotCosts[candidateIndex]!;

      aggregates.allProbSum += optionTransitionWeightsByState[weightOffset]!;
      aggregates.allDistSum += optionTransitionWeightsByState[weightOffset]! * candidateCost;

      for (let slot = 0; slot < 3; slot++) {
        const singleKey = singleCompatibilityKeys[keyOffset + slot]!;
        const weightIndex = SINGLE_WEIGHT_INDEX_BY_SLOT[slot]!;
        const weight = optionTransitionWeightsByState[weightOffset + weightIndex]!;
        aggregates.singleProbSums[slot]![singleKey] += weight;
        aggregates.singleDistSums[slot]![singleKey] += weight * candidateCost;
      }

      for (let pairIndex = 0; pairIndex < 3; pairIndex++) {
        const pairKey = pairCompatibilityKeys[keyOffset + pairIndex]!;
        const weightIndex = PAIR_WEIGHT_INDEX_BY_SLOT[pairIndex]!;
        const weight = optionTransitionWeightsByState[weightOffset + weightIndex]!;
        aggregates.pairProbSums[pairIndex]![pairKey] += weight;
        aggregates.pairDistSums[pairIndex]![pairKey] += weight * candidateCost;
      }
    };

    for (const stateIndex of orderedStateIndexesScratch) {
      const currentCost = snapshotCosts[stateIndex]!;

      for (const moduleMask of MASKS) {
        const orderedCandidates = orderedStateIndexesByModuleMask[moduleMask]!;
        let candidatePointer = optionSnapshotCandidatePointers[moduleMask]!;
        while (
          candidatePointer < orderedStateCountsByModuleMask[moduleMask]! &&
          snapshotCosts[orderedCandidates[candidatePointer]!] < currentCost - 1e-4
        ) {
          addCandidate(moduleMask, orderedCandidates[candidatePointer]!);
          candidatePointer += 1;
        }
        optionSnapshotCandidatePointers[moduleMask] = candidatePointer;
      }

      const currentModuleMask = stateModuleMasks[stateIndex]!;
      const candidateOffset = stateIndex * MAX_OPTION_ACTION_CANDIDATES;
      const candidateCount = optionCandidateCountsByState[stateIndex]!;

      for (let candidateIndex = 0; candidateIndex < candidateCount; candidateIndex++) {
        const optionCandidateIndex = candidateOffset + candidateIndex;
        const nextModuleMask = optionCandidateNextModuleMasksByState[optionCandidateIndex]!;
        const protectedMask = optionCandidateProtectedMasksByState[optionCandidateIndex]!;
        const keyMask = optionCandidateKeyMasksByState[optionCandidateIndex]!;
        const probabilityMass = optionCandidateProbabilityMassesByState[optionCandidateIndex]!;
        const { probSum, distSum } = getAggregatedOptionTransitionSums(
          protectedMask as (typeof MASKS)[number],
          stateIndex,
          optionSnapshotAggregatesByModuleMask[nextModuleMask]!,
        );

        const nextCost =
          distSum / probabilityMass +
          (1 - probSum / probabilityMass) * currentCost +
          weightedActionCostByMaskTriplet[buildActionCostIndex(currentModuleMask, nextModuleMask, keyMask)]!;

        if (nextCost < bestOptionCostScratch[stateIndex]! - 1e-4) {
          bestOptionCostScratch[stateIndex] = nextCost;
          bestOptionModuleMaskScratch[stateIndex] = nextModuleMask;
          bestOptionKeyMaskScratch[stateIndex] = keyMask;
        }
      }
    }

    return {
      bestOptionCost: bestOptionCostScratch,
      bestOptionModuleMask: bestOptionModuleMaskScratch,
      bestOptionKeyMask: bestOptionKeyMaskScratch,
    };
  };

  reportProgress?.({
    phase: "policy",
    completedIterations: 0,
    totalIterations: iterations,
    percent: 0,
  });

  let iterationsRun = 0;
  for (let iteration = 0; iteration < iterations; iteration++) {
    iterationsRun = iteration + 1;
    const { bestOptionCost, bestOptionModuleMask, bestOptionKeyMask } = buildOptionRerollSnapshot();

    let totalImprovement = 0;
    // 정책 개선 단계.
    // 각 비종결 상태에서 등급 재설정과 옵션 재설정 중 더 싼 행동을 골라 value function을 낮춘다.
    for (let stateIndex = 0; stateIndex < stateCount; stateIndex++) {
      const currentModuleMask = stateModuleMasks[stateIndex]!;
      const currentCost = costs[stateIndex]!;
      if (actionTypeByState[stateIndex] === ACTION_DONE) continue;

      if (optionTargetMatch[optionTripleKeysByState[stateIndex]!]) {
        // 목표 후보가 될 수 있는 옵션 조합에서만 등급 재설정을 검토할 가치가 있다.
        let currentBestCost = currentCost;
        const candidateOffset = stateIndex * MAX_OPTION_ACTION_CANDIDATES;
        const candidateCount = optionCandidateCountsByState[stateIndex]!;

        for (let candidateIndex = 0; candidateIndex < candidateCount; candidateIndex++) {
          const optionCandidateIndex = candidateOffset + candidateIndex;
          const nextModuleMask = optionCandidateNextModuleMasksByState[optionCandidateIndex]!;
          const protectedMask = optionCandidateProtectedMasksByState[optionCandidateIndex]!;
          const keyMask = optionCandidateKeyMasksByState[optionCandidateIndex]!;
          let nextCost =
            weightedActionCostByMaskTriplet[buildActionCostIndex(currentModuleMask, nextModuleMask, keyMask)]!;
          const transitionOffset = (stateIndex * MASK_COUNT + protectedMask) * MAX_GRADE_TRANSITIONS_PER_MASK;
          const transitionCount = gradeTransitionCountsByStateAndMask[stateIndex * MASK_COUNT + protectedMask]!;
          const nextStateOffset = stateIndex * MASK_COUNT * GRADE_MASK_COUNT + nextModuleMask * GRADE_MASK_COUNT;

          for (let transitionIndex = 0; transitionIndex < transitionCount; transitionIndex++) {
            const gradeMask = gradeTransitionGradeMasksByStateAndMask[transitionOffset + transitionIndex]!;
            const prob = gradeTransitionProbabilitiesByStateAndMask[transitionOffset + transitionIndex]!;
            const nextStateIndex = nextStateIndexByModuleMaskAndGradeMask[nextStateOffset + gradeMask]!;
            nextCost += prob * Math.min(costs[nextStateIndex]!, currentBestCost);
          }

          const improvement = currentBestCost - nextCost;
          if (improvement > 1e-4) {
            totalImprovement += improvement;
            currentBestCost = nextCost;
            costs[stateIndex] = nextCost;
            actionTypeByState[stateIndex] = ACTION_GRADE;
            actionModuleMaskByState[stateIndex] = nextModuleMask;
            actionKeyMaskByState[stateIndex] = keyMask;
          }
        }
      }

      const bestNextOptionCost = bestOptionCost[stateIndex]!;
      const bestModuleMask = bestOptionModuleMask[stateIndex]!;
      const bestKeyMask = bestOptionKeyMask[stateIndex]!;
      // 옵션 재설정 스냅샷 결과와 현재 값을 비교해 실제 개선이 있을 때만 정책을 갱신한다.
      const improvement = costs[stateIndex]! - bestNextOptionCost;
      if (bestModuleMask !== -1 && bestKeyMask !== -1 && improvement > 1e-4) {
        totalImprovement += improvement;
        costs[stateIndex] = bestNextOptionCost;
        actionTypeByState[stateIndex] = ACTION_OPTION;
        actionModuleMaskByState[stateIndex] = bestModuleMask;
        actionKeyMaskByState[stateIndex] = bestKeyMask;
      }
    }

    if (totalImprovement === 0) {
      reportProgress?.({
        phase: "policy",
        completedIterations: iterationsRun,
        totalIterations: iterations,
        percent: (iterationsRun / iterations) * 80,
      });
      break;
    }

    if (reportProgress && (iteration === iterations - 1 || iteration % yieldEveryIterations === 0)) {
      reportProgress?.({
        phase: "policy",
        completedIterations: iterationsRun,
        totalIterations: iterations,
        percent: (iterationsRun / iterations) * 80,
      });
      if (shouldYieldToMainThread) {
        await yieldToMainThread();
      }
    }
  }

  const expectationModuleAggregatesByModuleMask = MASKS.map(() => createAggregateSums());
  const expectationLockKeyAggregatesByModuleMask = MASKS.map(() => createAggregateSums());

  const buildExpectationAggregates = (): ExpectationAggregatePair => {
    // 행동 정책이 고정되면 기대값 계산도 같은 호환성 집계 기법을 재사용한다.
    // 차이는 "최소 비용" 대신 "기대 모듈 소모량 / 기대 락키 소모량"을 누적한다는 점이다.
    for (const moduleMask of MASKS) {
      resetAggregateSums(expectationModuleAggregatesByModuleMask[moduleMask]!);
      resetAggregateSums(expectationLockKeyAggregatesByModuleMask[moduleMask]!);
    }

    for (let stateIndex = 0; stateIndex < stateCount; stateIndex++) {
      const moduleMask = stateModuleMasks[stateIndex]!;
      const moduleAggregates = expectationModuleAggregatesByModuleMask[moduleMask]!;
      const lockKeyAggregates = expectationLockKeyAggregatesByModuleMask[moduleMask]!;
      const weightOffset = stateIndex * MASK_COUNT;
      const keyOffset = stateIndex * KEY_COUNT_PER_STATE;
      const moduleValue = expectedModuleCosts[stateIndex]!;
      const lockKeyValue = expectedLockKeyCosts[stateIndex]!;

      moduleAggregates.allProbSum += optionTransitionWeightsByState[weightOffset]!;
      moduleAggregates.allDistSum += optionTransitionWeightsByState[weightOffset]! * moduleValue;
      lockKeyAggregates.allProbSum += optionTransitionWeightsByState[weightOffset]!;
      lockKeyAggregates.allDistSum += optionTransitionWeightsByState[weightOffset]! * lockKeyValue;

      for (let slot = 0; slot < 3; slot++) {
        const singleKey = singleCompatibilityKeys[keyOffset + slot]!;
        const weightIndex = SINGLE_WEIGHT_INDEX_BY_SLOT[slot]!;
        const weight = optionTransitionWeightsByState[weightOffset + weightIndex]!;
        moduleAggregates.singleProbSums[slot]![singleKey] += weight;
        moduleAggregates.singleDistSums[slot]![singleKey] += weight * moduleValue;
        lockKeyAggregates.singleProbSums[slot]![singleKey] += weight;
        lockKeyAggregates.singleDistSums[slot]![singleKey] += weight * lockKeyValue;
      }

      for (let pairIndex = 0; pairIndex < 3; pairIndex++) {
        const pairKey = pairCompatibilityKeys[keyOffset + pairIndex]!;
        const weightIndex = PAIR_WEIGHT_INDEX_BY_SLOT[pairIndex]!;
        const weight = optionTransitionWeightsByState[weightOffset + weightIndex]!;
        moduleAggregates.pairProbSums[pairIndex]![pairKey] += weight;
        moduleAggregates.pairDistSums[pairIndex]![pairKey] += weight * moduleValue;
        lockKeyAggregates.pairProbSums[pairIndex]![pairKey] += weight;
        lockKeyAggregates.pairDistSums[pairIndex]![pairKey] += weight * lockKeyValue;
      }
    }

    return {
      module: expectationModuleAggregatesByModuleMask,
      lockKey: expectationLockKeyAggregatesByModuleMask,
    };
  };

  reportProgress?.({
    phase: "expectation",
    completedIterations: 0,
    totalIterations: iterations,
    percent: 80,
  });

  for (let evaluationIteration = 0; evaluationIteration < iterations; evaluationIteration++) {
    const expectationAggregates = buildExpectationAggregates();
    let maxDelta = 0;

    // 정책 평가 단계.
    // 이미 고정된 행동 정책을 따라갔을 때 각 상태의 기대 모듈/락키 소모량을 수렴시킨다.
    for (let stateIndex = 0; stateIndex < stateCount; stateIndex++) {
      const state = states[stateIndex]!;
      const actionType = actionTypeByState[stateIndex]!;
      if (actionType === ACTION_DONE) {
        expectedModuleCosts[stateIndex] = 0;
        expectedLockKeyCosts[stateIndex] = 0;
        continue;
      }

      const currentModuleMask = stateModuleMasks[stateIndex]!;
      const nextModuleMask = actionModuleMaskByState[stateIndex]!;
      const keyMask = actionKeyMaskByState[stateIndex]!;
      const protectedMask = (nextModuleMask | keyMask) as (typeof MASKS)[number];
      const actionCostIndex = buildActionCostIndex(currentModuleMask, nextModuleMask, keyMask);

      let nextExpectedModuleCost = moduleActionCostByMaskTriplet[actionCostIndex]!;
      let nextExpectedLockKeyCost = lockKeyActionCostByMaskTriplet[actionCostIndex]!;

      if (actionType === ACTION_GRADE) {
        // 등급 재설정은 미리 계산한 다음 상태 분포를 그대로 따라가면 된다.
        const transitionOffset = (stateIndex * MASK_COUNT + protectedMask) * MAX_GRADE_TRANSITIONS_PER_MASK;
        const transitionCount = gradeTransitionCountsByStateAndMask[stateIndex * MASK_COUNT + protectedMask]!;
        const nextStateOffset = stateIndex * MASK_COUNT * GRADE_MASK_COUNT + nextModuleMask * GRADE_MASK_COUNT;
        for (let transitionIndex = 0; transitionIndex < transitionCount; transitionIndex++) {
          const gradeMask = gradeTransitionGradeMasksByStateAndMask[transitionOffset + transitionIndex]!;
          const prob = gradeTransitionProbabilitiesByStateAndMask[transitionOffset + transitionIndex]!;
          const nextStateIndex = nextStateIndexByModuleMaskAndGradeMask[nextStateOffset + gradeMask]!;
          nextExpectedModuleCost += prob * expectedModuleCosts[nextStateIndex]!;
          nextExpectedLockKeyCost += prob * expectedLockKeyCosts[nextStateIndex]!;
        }
      } else {
        const [o1, o2, o3] = state;
        // 옵션 재설정은 호환되는 후보 상태들의 누적 기대값을 확률 질량으로 나눠 사용한다.
        const probabilityMass = getOptionTransitionProbabilityMass(o1, o2, o3, protectedMask);
        if (probabilityMass > 0) {
          const moduleSums = getAggregatedOptionTransitionSums(
            protectedMask,
            stateIndex,
            expectationAggregates.module[nextModuleMask]!,
          );
          const lockKeySums = getAggregatedOptionTransitionSums(
            protectedMask,
            stateIndex,
            expectationAggregates.lockKey[nextModuleMask]!,
          );
          nextExpectedModuleCost += moduleSums.distSum / probabilityMass;
          nextExpectedLockKeyCost += lockKeySums.distSum / probabilityMass;
        }
      }

      maxDelta = Math.max(
        maxDelta,
        Math.abs(nextExpectedModuleCost - expectedModuleCosts[stateIndex]!),
        Math.abs(nextExpectedLockKeyCost - expectedLockKeyCosts[stateIndex]!),
      );
      expectedModuleCosts[stateIndex] = nextExpectedModuleCost;
      expectedLockKeyCosts[stateIndex] = nextExpectedLockKeyCost;
    }

    if (maxDelta < 1e-9) {
      reportProgress?.({
        phase: "expectation",
        completedIterations: evaluationIteration + 1,
        totalIterations: iterations,
        percent: 80 + ((evaluationIteration + 1) / iterations) * 20,
      });
      break;
    }

    if (
      reportProgress &&
      (evaluationIteration === iterations - 1 || evaluationIteration % yieldEveryIterations === 0)
    ) {
      reportProgress?.({
        phase: "expectation",
        completedIterations: evaluationIteration + 1,
        totalIterations: iterations,
        percent: 80 + ((evaluationIteration + 1) / iterations) * 20,
      });
      if (shouldYieldToMainThread) {
        await yieldToMainThread();
      }
    }
  }

  const stateValues = createOverloadStateTensor<OverloadStateValue>(() => ({
    cost: INFINITE_COST,
    expectedCosts: { module: 0, lockKey: 0 },
    action: { type: "done" },
  }));

  // 공개 API는 기존의 중첩 tensor 형태를 유지하므로 마지막에 한 번만 다시 복원한다.
  // 내부 계산은 전부 flat 배열로 끝냈기 때문에 이 단계는 결과 포장 단계에 가깝다.
  for (let stateIndex = 0; stateIndex < stateCount; stateIndex++) {
    const state = states[stateIndex]!;
    const [o1, o2, o3, g1, g2, g3, m1, m2, m3] = state;
    stateValues[o1][o2][o3][g1][g2][g3][m1][m2][m3] = {
      cost: costs[stateIndex]!,
      expectedCosts: {
        module: expectedModuleCosts[stateIndex]!,
        lockKey: expectedLockKeyCosts[stateIndex]!,
      },
      action: buildActionFromMasks(
        actionTypeByState[stateIndex]!,
        actionModuleMaskByState[stateIndex]!,
        actionKeyMaskByState[stateIndex]!,
      ),
    };
  }

  reportProgress?.({
    phase: "done",
    completedIterations: iterationsRun,
    totalIterations: iterations,
    percent: 100,
  });

  return {
    stateValues,
    iterationsRun,
    states,
  };
}
