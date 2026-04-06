import {
  lockCosts,
  lockKeyCosts,
  overloadGradeProbabilities,
  overloadOptions,
  OVERLOAD_GRADE_COUNT,
  OVERLOAD_OPTION_COUNT,
  rerollCosts,
  slotOptionProbabilities,
  type OverloadOptionIds,
  type OverloadOptionTarget,
} from "./overloadOptions";
import { type OverloadAction, type OverloadState } from "./overloadPolicyOptimizer";

const MASKS = [0, 1, 2, 3, 4, 5, 6] as const;
const SINGLE_MASK_INDEX = [-1, 0, 1, -1, 2, -1, -1] as const;
const PAIR_MASK_INDEX = [-1, -1, -1, 0, -1, 1, 2] as const;
const OPTION_RADIX = OVERLOAD_OPTION_COUNT + 1;
const STATE_KEY_SIZE = OPTION_RADIX * OPTION_RADIX * OPTION_RADIX * 2 * 2 * 2 * 2 * 2 * 2;
const SLOT_VALUE_KEY_SIZE = (OVERLOAD_OPTION_COUNT + 1) * 2;
const TWO_SLOT_KEY_SIZE = SLOT_VALUE_KEY_SIZE * SLOT_VALUE_KEY_SIZE;
const MASK_COUNT = MASKS.length;
const KEY_COUNT_PER_STATE = 3;
const GRADE_MASK_COUNT = 8;
const MAX_GRADE_TRANSITIONS_PER_MASK = GRADE_MASK_COUNT;
const MAX_ACTION_CANDIDATES = 19;
const ACTION_COST_TABLE_SIZE = MASK_COUNT * MASK_COUNT * MASK_COUNT;
const ACTION_DONE = 0;
const ACTION_OPTION = 1;
const ACTION_GRADE = 2;
const SINGLE_WEIGHT_INDEX_BY_SLOT = [1, 2, 4] as const;
const PAIR_WEIGHT_INDEX_BY_SLOT = [3, 5, 6] as const;
const COMPARISON_EPSILON = 1e-12;

type ValueAggregates = {
  singleSums: Float64Array[];
  pairSums: Float64Array[];
  allSum: number;
};

export type OverloadBudgetCurvePoint = {
  moduleBudget: number;
  successProbability: number;
  expectedLockKeyCost: number;
};

export type OverloadBudgetRecommendation = {
  successProbability: number;
  expectedLockKeyCost: number;
  action: OverloadAction | null;
};

export type OverloadBudgetOptimizationSummary = {
  moduleBudget: number;
  current: OverloadBudgetRecommendation;
  curve: OverloadBudgetCurvePoint[];
};

export type OverloadBudgetOptimizationResult = {
  moduleBudget: number;
  stateCount: number;
  stateIndexByKey: Int32Array;
  successProbabilityTable: Float64Array;
  expectedLockKeyCostTable: Float64Array;
  actionTypeTable: Int8Array;
  actionModuleMaskTable: Int8Array;
  actionKeyMaskTable: Int8Array;
};

function countBits(mask: number) {
  return Number(Boolean(mask & 1)) + Number(Boolean(mask & 2)) + Number(Boolean(mask & 4));
}

function decodeMask(mask: number): [number, number, number] {
  return [mask & 1 ? 1 : 0, mask & 2 ? 1 : 0, mask & 4 ? 1 : 0];
}

function decodeBooleanMask(mask: number): [boolean, boolean, boolean] {
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

function canUseMask([o1, o2, o3]: [number, number, number], mask: number) {
  if (countBits(mask) > 2) return false;
  if (mask & 1 && o1 === 0) return false;
  if (mask & 2 && o2 === 0) return false;
  if (mask & 4 && o3 === 0) return false;
  return true;
}

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

function createValueAggregates(): ValueAggregates {
  return {
    singleSums: Array.from({ length: 3 }, () => new Float64Array(SLOT_VALUE_KEY_SIZE)),
    pairSums: Array.from({ length: 3 }, () => new Float64Array(TWO_SLOT_KEY_SIZE)),
    allSum: 0,
  };
}

function buildActionCostIndex(currentModuleMask: number, nextModuleMask: number, keyMask: number) {
  return (currentModuleMask * MASK_COUNT + nextModuleMask) * MASK_COUNT + keyMask;
}

function buildActionCosts(currentModuleMask: number, nextModuleMask: number, keyMask: number) {
  const keptModuleCount = countBits(currentModuleMask & nextModuleMask);
  const nextModuleCount = countBits(nextModuleMask);
  const keyCount = countBits(keyMask);
  return {
    moduleCost:
      rerollCosts[countBits(nextModuleMask | keyMask)]! + lockCosts[nextModuleCount]! - lockCosts[keptModuleCount]!,
    lockKeyCost: lockKeyCosts[keyCount]!,
  };
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

function readBudgetTableIndex(result: OverloadBudgetOptimizationResult, moduleBudget: number, stateIndex: number) {
  return moduleBudget * result.stateCount + stateIndex;
}

function buildDerivedBudgetData(targetOptionIds: OverloadOptionIds[], targetGradeTargets: OverloadOptionTarget[]) {
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
    targetStates,
  };
}

function isBetterCandidate(
  probability: number,
  expectedLockKeyCost: number,
  bestProbability: number,
  bestExpectedLockKeyCost: number,
) {
  if (probability > bestProbability + COMPARISON_EPSILON) {
    return true;
  }

  if (Math.abs(probability - bestProbability) <= COMPARISON_EPSILON) {
    return expectedLockKeyCost + COMPARISON_EPSILON < bestExpectedLockKeyCost;
  }

  return false;
}

export function readOverloadBudgetOptimizationSummary(
  result: OverloadBudgetOptimizationResult,
  startState: OverloadState,
): OverloadBudgetOptimizationSummary | null {
  const startStateIndex = result.stateIndexByKey[encodeStateKey(startState)];
  if (startStateIndex === -1) {
    return null;
  }

  const curve = Array.from({ length: result.moduleBudget + 1 }, (_, moduleBudget) => {
    const tableIndex = readBudgetTableIndex(result, moduleBudget, startStateIndex);
    return {
      moduleBudget,
      successProbability: result.successProbabilityTable[tableIndex]!,
      expectedLockKeyCost: result.expectedLockKeyCostTable[tableIndex]!,
    };
  });

  const currentTableIndex = readBudgetTableIndex(result, result.moduleBudget, startStateIndex);
  const currentActionType = result.actionTypeTable[currentTableIndex]!;
  return {
    moduleBudget: result.moduleBudget,
    current: {
      successProbability: result.successProbabilityTable[currentTableIndex]!,
      expectedLockKeyCost: result.expectedLockKeyCostTable[currentTableIndex]!,
      action:
        currentActionType === -1
          ? null
          : buildActionFromMasks(
              currentActionType,
              result.actionModuleMaskTable[currentTableIndex]!,
              result.actionKeyMaskTable[currentTableIndex]!,
            ),
    },
    curve,
  };
}

export function optimizeOverloadBudgetSuccess(
  targetOptionIds: OverloadOptionIds[],
  targetGradeTargets: OverloadOptionTarget[],
  maxModuleBudget: number,
): OverloadBudgetOptimizationResult {
  const normalizedBudget = Math.max(0, Math.floor(maxModuleBudget));
  const states = Array.from(iterateOverloadStates());
  const stateCount = states.length;
  const { meetsTargetGradeProbabilities, targetStates } = buildDerivedBudgetData(targetOptionIds, targetGradeTargets);
  const stateIndexByKey = new Int32Array(STATE_KEY_SIZE).fill(-1);
  const slotValueKey = (option: number, grade: number) => option * 2 + grade;

  for (let stateIndex = 0; stateIndex < stateCount; stateIndex++) {
    stateIndexByKey[encodeStateKey(states[stateIndex]!)] = stateIndex;
  }

  const stateModuleMasks = new Int8Array(stateCount);
  const singleCompatibilityKeys = new Int32Array(stateCount * KEY_COUNT_PER_STATE);
  const pairCompatibilityKeys = new Int32Array(stateCount * KEY_COUNT_PER_STATE);
  const nextStateIndexByModuleMaskAndGradeMask = new Int32Array(stateCount * MASK_COUNT * GRADE_MASK_COUNT).fill(-1);
  const targetStateByIndex = new Uint8Array(stateCount);

  for (let stateIndex = 0; stateIndex < stateCount; stateIndex++) {
    const state = states[stateIndex]!;
    const [o1, o2, o3, g1, g2, g3] = state;
    const key1 = slotValueKey(o1, g1);
    const key2 = slotValueKey(o2, g2);
    const key3 = slotValueKey(o3, g3);
    const keyOffset = stateIndex * KEY_COUNT_PER_STATE;
    const nextStateOffset = stateIndex * MASK_COUNT * GRADE_MASK_COUNT;

    stateModuleMasks[stateIndex] = moduleMaskFromState(state);
    singleCompatibilityKeys[keyOffset] = key1;
    singleCompatibilityKeys[keyOffset + 1] = key2;
    singleCompatibilityKeys[keyOffset + 2] = key3;
    pairCompatibilityKeys[keyOffset] = key1 * SLOT_VALUE_KEY_SIZE + key2;
    pairCompatibilityKeys[keyOffset + 1] = key1 * SLOT_VALUE_KEY_SIZE + key3;
    pairCompatibilityKeys[keyOffset + 2] = key2 * SLOT_VALUE_KEY_SIZE + key3;
    targetStateByIndex[stateIndex] = Number(
      targetStates.some((targetState) => matchesTargetPattern(state, targetState)),
    );

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

  const optionTransitionWeightsByState = new Float64Array(stateCount * MASK_COUNT);
  const gradeTransitionCountsByStateAndMask = new Uint8Array(stateCount * MASK_COUNT);
  const gradeTransitionGradeMasksByStateAndMask = new Uint8Array(
    stateCount * MASK_COUNT * MAX_GRADE_TRANSITIONS_PER_MASK,
  );
  const gradeTransitionProbabilitiesByStateAndMask = new Float64Array(
    stateCount * MASK_COUNT * MAX_GRADE_TRANSITIONS_PER_MASK,
  );

  for (let stateIndex = 0; stateIndex < stateCount; stateIndex++) {
    const [o1, o2, o3, g1, g2, g3] = states[stateIndex]!;
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
        if (p1 && ng1 !== g1) continue;
        if (p2 && ng2 !== g2) continue;
        if (p3 && ng3 !== g3) continue;

        let probability = 1;
        if (!p1) probability *= meetsTargetGradeProbabilities[o1][ng1];
        if (!p2) probability *= meetsTargetGradeProbabilities[o2][ng2];
        if (!p3) probability *= meetsTargetGradeProbabilities[o3][ng3];
        if (probability <= 0) continue;

        gradeTransitionGradeMasksByStateAndMask[transitionOffset + transitionCount] = gradeMask;
        gradeTransitionProbabilitiesByStateAndMask[transitionOffset + transitionCount] = probability;
        transitionCount += 1;
      }

      gradeTransitionCountsByStateAndMask[stateIndex * MASK_COUNT + protectedMask] = transitionCount;
    }
  }

  const actionModuleCostByMaskTriplet = new Int8Array(ACTION_COST_TABLE_SIZE);
  const actionLockKeyCostByMaskTriplet = new Int8Array(ACTION_COST_TABLE_SIZE);
  for (const currentModuleMask of MASKS) {
    for (const nextModuleMask of MASKS) {
      for (const keyMask of MASKS) {
        const actionCostIndex = buildActionCostIndex(currentModuleMask, nextModuleMask, keyMask);
        const actionCosts = buildActionCosts(currentModuleMask, nextModuleMask, keyMask);
        actionModuleCostByMaskTriplet[actionCostIndex] = actionCosts.moduleCost;
        actionLockKeyCostByMaskTriplet[actionCostIndex] = actionCosts.lockKeyCost;
      }
    }
  }

  const getAggregatedValueSum = (
    protectedMask: (typeof MASKS)[number],
    stateIndex: number,
    aggregates: ValueAggregates,
  ) => {
    const keyOffset = stateIndex * KEY_COUNT_PER_STATE;
    if (protectedMask === 0) {
      return aggregates.allSum;
    }

    const singleSlot = SINGLE_MASK_INDEX[protectedMask];
    if (singleSlot !== -1) {
      return aggregates.singleSums[singleSlot]![singleCompatibilityKeys[keyOffset + singleSlot]!];
    }

    const pairSlot = PAIR_MASK_INDEX[protectedMask];
    if (pairSlot === -1) {
      return 0;
    }

    return aggregates.pairSums[pairSlot]![pairCompatibilityKeys[keyOffset + pairSlot]!];
  };

  const getOptionTransitionProbabilityMass = (() => {
    let squaredOptionProbabilitySum = 0;
    let cubedOptionProbabilitySum = 0;

    for (let index = 0; index <= OVERLOAD_OPTION_COUNT; index++) {
      squaredOptionProbabilitySum += (overloadOptions[index]?.probability ?? 0) ** 2;
      cubedOptionProbabilitySum += (overloadOptions[index]?.probability ?? 0) ** 3;
    }

    return (o1: number, o2: number, o3: number, protectedMask: number) => {
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

        const protectedOptionProbability = overloadOptions[options[protectedSlot]]?.probability ?? 0;
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
        (overloadOptions[o1]?.probability ?? 0) +
        (overloadOptions[o2]?.probability ?? 0) +
        (overloadOptions[o3]?.probability ?? 0);
      const unlockedSlot = [0, 1, 2].find((slot) => ((protectedMask >> slot) & 1) === 0);
      if (unlockedSlot === undefined) {
        return 0;
      }

      return (
        1 -
        slotOptionProbabilities[unlockedSlot] *
          (protectedOptionProbabilitySum - (overloadOptions[options[unlockedSlot]]?.probability ?? 0))
      );
    };
  })();

  const optionCandidateCountsByState = new Uint8Array(stateCount);
  const optionCandidateNextModuleMasksByState = new Int8Array(stateCount * MAX_ACTION_CANDIDATES);
  const optionCandidateProtectedMasksByState = new Int8Array(stateCount * MAX_ACTION_CANDIDATES);
  const optionCandidateKeyMasksByState = new Int8Array(stateCount * MAX_ACTION_CANDIDATES);
  const optionCandidateProbabilityMassesByState = new Float64Array(stateCount * MAX_ACTION_CANDIDATES);
  const gradeCandidateCountsByState = new Uint8Array(stateCount);
  const gradeCandidateNextModuleMasksByState = new Int8Array(stateCount * MAX_ACTION_CANDIDATES);
  const gradeCandidateProtectedMasksByState = new Int8Array(stateCount * MAX_ACTION_CANDIDATES);
  const gradeCandidateKeyMasksByState = new Int8Array(stateCount * MAX_ACTION_CANDIDATES);

  for (let stateIndex = 0; stateIndex < stateCount; stateIndex++) {
    const [o1, o2, o3] = states[stateIndex]!;
    const candidateOffset = stateIndex * MAX_ACTION_CANDIDATES;
    let optionCandidateCount = 0;
    let gradeCandidateCount = 0;

    for (const nextModuleMask of MASKS) {
      if (!canUseMask([o1, o2, o3], nextModuleMask)) continue;

      for (const protectedMask of MASKS) {
        if (!canUseMask([o1, o2, o3], protectedMask)) continue;
        if ((nextModuleMask & protectedMask) !== nextModuleMask) continue;

        const keyMask = protectedMask ^ nextModuleMask;

        const optionProbabilityMass = getOptionTransitionProbabilityMass(o1, o2, o3, protectedMask);
        if (optionProbabilityMass > 0) {
          const optionIndex = candidateOffset + optionCandidateCount;
          optionCandidateNextModuleMasksByState[optionIndex] = nextModuleMask;
          optionCandidateProtectedMasksByState[optionIndex] = protectedMask;
          optionCandidateKeyMasksByState[optionIndex] = keyMask;
          optionCandidateProbabilityMassesByState[optionIndex] = optionProbabilityMass;
          optionCandidateCount += 1;
        }

        if (optionTargetMatch[encodeOptionTripleKey(o1, o2, o3)]) {
          const gradeIndex = candidateOffset + gradeCandidateCount;
          gradeCandidateNextModuleMasksByState[gradeIndex] = nextModuleMask;
          gradeCandidateProtectedMasksByState[gradeIndex] = protectedMask;
          gradeCandidateKeyMasksByState[gradeIndex] = keyMask;
          gradeCandidateCount += 1;
        }
      }
    }

    optionCandidateCountsByState[stateIndex] = optionCandidateCount;
    gradeCandidateCountsByState[stateIndex] = gradeCandidateCount;
  }

  const probabilityTable = new Float64Array((normalizedBudget + 1) * stateCount);
  const expectedLockKeyTable = new Float64Array((normalizedBudget + 1) * stateCount);
  const actionTypeTable = new Int8Array((normalizedBudget + 1) * stateCount).fill(-1);
  const actionModuleMaskTable = new Int8Array((normalizedBudget + 1) * stateCount).fill(-1);
  const actionKeyMaskTable = new Int8Array((normalizedBudget + 1) * stateCount).fill(-1);

  for (let budget = 0; budget <= normalizedBudget; budget++) {
    const probabilityAggregatesByBudget = new Map<number, ValueAggregates[]>();
    const lockKeyAggregatesByBudget = new Map<number, ValueAggregates[]>();

    const buildAggregatesForBudget = (remainingBudget: number, table: Float64Array) => {
      const byModuleMask = MASKS.map(() => createValueAggregates());
      const baseOffset = remainingBudget * stateCount;

      for (let stateIndex = 0; stateIndex < stateCount; stateIndex++) {
        const value = table[baseOffset + stateIndex]!;
        if (value === 0) {
          continue;
        }

        const aggregates = byModuleMask[stateModuleMasks[stateIndex]!]!;
        const weightOffset = stateIndex * MASK_COUNT;
        const keyOffset = stateIndex * KEY_COUNT_PER_STATE;

        aggregates.allSum += optionTransitionWeightsByState[weightOffset]! * value;

        for (let slot = 0; slot < 3; slot++) {
          const singleKey = singleCompatibilityKeys[keyOffset + slot]!;
          const weight = optionTransitionWeightsByState[weightOffset + SINGLE_WEIGHT_INDEX_BY_SLOT[slot]!]!;
          aggregates.singleSums[slot]![singleKey] += weight * value;
        }

        for (let pairIndex = 0; pairIndex < 3; pairIndex++) {
          const pairKey = pairCompatibilityKeys[keyOffset + pairIndex]!;
          const weight = optionTransitionWeightsByState[weightOffset + PAIR_WEIGHT_INDEX_BY_SLOT[pairIndex]!]!;
          aggregates.pairSums[pairIndex]![pairKey] += weight * value;
        }
      }

      return byModuleMask;
    };

    const getAggregates = (cache: Map<number, ValueAggregates[]>, remainingBudget: number, table: Float64Array) => {
      const cached = cache.get(remainingBudget);
      if (cached) {
        return cached;
      }

      const nextAggregates = buildAggregatesForBudget(remainingBudget, table);
      cache.set(remainingBudget, nextAggregates);
      return nextAggregates;
    };

    for (let stateIndex = 0; stateIndex < stateCount; stateIndex++) {
      const tableIndex = budget * stateCount + stateIndex;

      if (targetStateByIndex[stateIndex]) {
        probabilityTable[tableIndex] = 1;
        expectedLockKeyTable[tableIndex] = 0;
        actionTypeTable[tableIndex] = ACTION_DONE;
        actionModuleMaskTable[tableIndex] = 0;
        actionKeyMaskTable[tableIndex] = 0;
        continue;
      }

      const currentModuleMask = stateModuleMasks[stateIndex]!;
      let bestProbability = 0;
      let bestExpectedLockKeyCost = Number.POSITIVE_INFINITY;
      let bestActionType = -1;
      let bestModuleMask = -1;
      let bestKeyMask = -1;
      const candidateOffset = stateIndex * MAX_ACTION_CANDIDATES;

      for (let candidateIndex = 0; candidateIndex < optionCandidateCountsByState[stateIndex]!; candidateIndex++) {
        const actionIndex = candidateOffset + candidateIndex;
        const nextModuleMask = optionCandidateNextModuleMasksByState[actionIndex]!;
        const protectedMask = optionCandidateProtectedMasksByState[actionIndex]! as (typeof MASKS)[number];
        const keyMask = optionCandidateKeyMasksByState[actionIndex]!;
        const actionCostIndex = buildActionCostIndex(currentModuleMask, nextModuleMask, keyMask);
        const moduleCost = actionModuleCostByMaskTriplet[actionCostIndex]!;
        if (moduleCost > budget) {
          continue;
        }

        const remainingBudget = budget - moduleCost;
        const probabilityMass = optionCandidateProbabilityMassesByState[actionIndex]!;
        const probabilityAggregates = getAggregates(probabilityAggregatesByBudget, remainingBudget, probabilityTable);
        const lockKeyAggregates = getAggregates(lockKeyAggregatesByBudget, remainingBudget, expectedLockKeyTable);
        const nextProbability =
          getAggregatedValueSum(protectedMask, stateIndex, probabilityAggregates[nextModuleMask]!) / probabilityMass;

        if (nextProbability <= COMPARISON_EPSILON) {
          continue;
        }

        const nextExpectedLockKeyCost =
          actionLockKeyCostByMaskTriplet[actionCostIndex]! +
          getAggregatedValueSum(protectedMask, stateIndex, lockKeyAggregates[nextModuleMask]!) / probabilityMass;

        if (isBetterCandidate(nextProbability, nextExpectedLockKeyCost, bestProbability, bestExpectedLockKeyCost)) {
          bestProbability = nextProbability;
          bestExpectedLockKeyCost = nextExpectedLockKeyCost;
          bestActionType = ACTION_OPTION;
          bestModuleMask = nextModuleMask;
          bestKeyMask = keyMask;
        }
      }

      for (let candidateIndex = 0; candidateIndex < gradeCandidateCountsByState[stateIndex]!; candidateIndex++) {
        const actionIndex = candidateOffset + candidateIndex;
        const nextModuleMask = gradeCandidateNextModuleMasksByState[actionIndex]!;
        const protectedMask = gradeCandidateProtectedMasksByState[actionIndex]!;
        const keyMask = gradeCandidateKeyMasksByState[actionIndex]!;
        const actionCostIndex = buildActionCostIndex(currentModuleMask, nextModuleMask, keyMask);
        const moduleCost = actionModuleCostByMaskTriplet[actionCostIndex]!;
        if (moduleCost > budget) {
          continue;
        }

        const remainingBudget = budget - moduleCost;
        const gradeTransitionCount = gradeTransitionCountsByStateAndMask[stateIndex * MASK_COUNT + protectedMask]!;
        if (gradeTransitionCount === 0) {
          continue;
        }

        const transitionOffset = (stateIndex * MASK_COUNT + protectedMask) * MAX_GRADE_TRANSITIONS_PER_MASK;
        const nextTableOffset = remainingBudget * stateCount;
        let nextProbability = 0;
        let nextExpectedLockKeyCost = actionLockKeyCostByMaskTriplet[actionCostIndex]!;

        for (let transitionIndex = 0; transitionIndex < gradeTransitionCount; transitionIndex++) {
          const probability = gradeTransitionProbabilitiesByStateAndMask[transitionOffset + transitionIndex]!;
          const gradeMask = gradeTransitionGradeMasksByStateAndMask[transitionOffset + transitionIndex]!;
          const nextStateIndex =
            nextStateIndexByModuleMaskAndGradeMask[
              stateIndex * MASK_COUNT * GRADE_MASK_COUNT + nextModuleMask * GRADE_MASK_COUNT + gradeMask
            ]!;
          nextProbability += probability * probabilityTable[nextTableOffset + nextStateIndex]!;
          nextExpectedLockKeyCost += probability * expectedLockKeyTable[nextTableOffset + nextStateIndex]!;
        }

        if (nextProbability <= COMPARISON_EPSILON) {
          continue;
        }

        if (isBetterCandidate(nextProbability, nextExpectedLockKeyCost, bestProbability, bestExpectedLockKeyCost)) {
          bestProbability = nextProbability;
          bestExpectedLockKeyCost = nextExpectedLockKeyCost;
          bestActionType = ACTION_GRADE;
          bestModuleMask = nextModuleMask;
          bestKeyMask = keyMask;
        }
      }

      probabilityTable[tableIndex] = bestProbability;
      expectedLockKeyTable[tableIndex] = bestActionType === -1 ? 0 : bestExpectedLockKeyCost;
      actionTypeTable[tableIndex] = bestActionType;
      actionModuleMaskTable[tableIndex] = bestModuleMask;
      actionKeyMaskTable[tableIndex] = bestKeyMask;
    }
  }

  return {
    moduleBudget: normalizedBudget,
    stateCount,
    stateIndexByKey,
    successProbabilityTable: probabilityTable,
    expectedLockKeyCostTable: expectedLockKeyTable,
    actionTypeTable,
    actionModuleMaskTable,
    actionKeyMaskTable,
  };
}
