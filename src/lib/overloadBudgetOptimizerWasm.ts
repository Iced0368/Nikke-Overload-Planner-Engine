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
import { optimizeOverloadBudgetSuccess, type OverloadBudgetOptimizationResult } from "./overloadBudgetOptimizer";

const MASKS = [0, 1, 2, 3, 4, 5, 6] as const;
const SINGLE_MASK_INDEX = [-1, 0, 1, -1, 2, -1, -1] as const;
const OPTION_RADIX = OVERLOAD_OPTION_COUNT + 1;
const STATE_KEY_SIZE = OPTION_RADIX * OPTION_RADIX * OPTION_RADIX * 2 * 2 * 2 * 2 * 2 * 2;
const SLOT_VALUE_KEY_SIZE = (OVERLOAD_OPTION_COUNT + 1) * 2;
const MASK_COUNT = MASKS.length;
const KEY_COUNT_PER_STATE = 3;
const GRADE_MASK_COUNT = 8;
const MAX_GRADE_TRANSITIONS_PER_MASK = GRADE_MASK_COUNT;
const MAX_ACTION_CANDIDATES = 19;
const ACTION_COST_TABLE_SIZE = MASK_COUNT * MASK_COUNT * MASK_COUNT;

type OverloadState = [number, number, number, number, number, number, number, number, number];

type PreparedBudgetOptimizerInput = {
  moduleBudget: number;
  stateCount: number;
  stateIndexByKey: Int32Array;
  stateModuleMasks: Int8Array;
  singleCompatibilityKeys: Int32Array;
  pairCompatibilityKeys: Int32Array;
  nextStateIndexByModuleMaskAndGradeMask: Int32Array;
  targetStateByIndex: Uint8Array;
  optionTransitionWeightsByState: Float64Array;
  gradeTransitionCountsByStateAndMask: Uint8Array;
  gradeTransitionGradeMasksByStateAndMask: Uint8Array;
  gradeTransitionProbabilitiesByStateAndMask: Float64Array;
  actionModuleCostByMaskTriplet: Int8Array;
  actionLockKeyCostByMaskTriplet: Int8Array;
  optionCandidateCountsByState: Uint8Array;
  optionCandidateNextModuleMasksByState: Int8Array;
  optionCandidateProtectedMasksByState: Int8Array;
  optionCandidateKeyMasksByState: Int8Array;
  optionCandidateProbabilityMassesByState: Float64Array;
  gradeCandidateCountsByState: Uint8Array;
  gradeCandidateNextModuleMasksByState: Int8Array;
  gradeCandidateProtectedMasksByState: Int8Array;
  gradeCandidateKeyMasksByState: Int8Array;
  successProbabilityTable: Float64Array;
  expectedLockKeyCostTable: Float64Array;
  actionTypeTable: Int8Array;
  actionModuleMaskTable: Int8Array;
  actionKeyMaskTable: Int8Array;
};

type EmscriptenBudgetOptimizerModule = {
  HEAP8: Int8Array;
  HEAPU8: Uint8Array;
  HEAP32: Int32Array;
  HEAPF64: Float64Array;
  _malloc(size: number): number;
  _free(ptr: number): void;
  _run_budget_optimizer_core(
    stateCount: number,
    moduleBudget: number,
    stateModuleMasksPtr: number,
    singleCompatibilityKeysPtr: number,
    pairCompatibilityKeysPtr: number,
    nextStateIndexByModuleMaskAndGradeMaskPtr: number,
    targetStateByIndexPtr: number,
    optionTransitionWeightsByStatePtr: number,
    gradeTransitionCountsByStateAndMaskPtr: number,
    gradeTransitionGradeMasksByStateAndMaskPtr: number,
    gradeTransitionProbabilitiesByStateAndMaskPtr: number,
    actionModuleCostByMaskTripletPtr: number,
    actionLockKeyCostByMaskTripletPtr: number,
    optionCandidateCountsByStatePtr: number,
    optionCandidateNextModuleMasksByStatePtr: number,
    optionCandidateProtectedMasksByStatePtr: number,
    optionCandidateKeyMasksByStatePtr: number,
    optionCandidateProbabilityMassesByStatePtr: number,
    gradeCandidateCountsByStatePtr: number,
    gradeCandidateNextModuleMasksByStatePtr: number,
    gradeCandidateProtectedMasksByStatePtr: number,
    gradeCandidateKeyMasksByStatePtr: number,
    successProbabilityTablePtr: number,
    expectedLockKeyCostTablePtr: number,
    actionTypeTablePtr: number,
    actionModuleMaskTablePtr: number,
    actionKeyMaskTablePtr: number,
  ): void;
};

type BudgetOptimizerModuleFactory = (options?: {
  locateFile?: (path: string, scriptDirectory: string) => string;
}) => Promise<EmscriptenBudgetOptimizerModule>;

let budgetOptimizerModulePromise: Promise<EmscriptenBudgetOptimizerModule> | null = null;

function countBits(mask: number) {
  return Number(Boolean(mask & 1)) + Number(Boolean(mask & 2)) + Number(Boolean(mask & 4));
}

function decodeMask(mask: number): [number, number, number] {
  return [mask & 1 ? 1 : 0, mask & 2 ? 1 : 0, mask & 4 ? 1 : 0];
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

function prepareBudgetOptimizerKernelInput(
  targetOptionIds: OverloadOptionIds[],
  targetGradeTargets: OverloadOptionTarget[],
  maxModuleBudget: number,
) {
  const moduleBudget = Math.max(0, Math.floor(maxModuleBudget));
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

  const tableSize = (moduleBudget + 1) * stateCount;
  return {
    moduleBudget,
    stateCount,
    stateIndexByKey,
    stateModuleMasks,
    singleCompatibilityKeys,
    pairCompatibilityKeys,
    nextStateIndexByModuleMaskAndGradeMask,
    targetStateByIndex,
    optionTransitionWeightsByState,
    gradeTransitionCountsByStateAndMask,
    gradeTransitionGradeMasksByStateAndMask,
    gradeTransitionProbabilitiesByStateAndMask,
    actionModuleCostByMaskTriplet,
    actionLockKeyCostByMaskTriplet,
    optionCandidateCountsByState,
    optionCandidateNextModuleMasksByState,
    optionCandidateProtectedMasksByState,
    optionCandidateKeyMasksByState,
    optionCandidateProbabilityMassesByState,
    gradeCandidateCountsByState,
    gradeCandidateNextModuleMasksByState,
    gradeCandidateProtectedMasksByState,
    gradeCandidateKeyMasksByState,
    successProbabilityTable: new Float64Array(tableSize),
    expectedLockKeyCostTable: new Float64Array(tableSize),
    actionTypeTable: new Int8Array(tableSize).fill(-1),
    actionModuleMaskTable: new Int8Array(tableSize).fill(-1),
    actionKeyMaskTable: new Int8Array(tableSize).fill(-1),
  } satisfies PreparedBudgetOptimizerInput;
}

function allocateInt8Array(module: EmscriptenBudgetOptimizerModule, values: Int8Array) {
  const pointer = module._malloc(values.byteLength);
  module.HEAP8.set(values, pointer);
  return pointer;
}

function allocateUint8Array(module: EmscriptenBudgetOptimizerModule, values: Uint8Array) {
  const pointer = module._malloc(values.byteLength);
  module.HEAPU8.set(values, pointer);
  return pointer;
}

function allocateInt32Array(module: EmscriptenBudgetOptimizerModule, values: Int32Array) {
  const pointer = module._malloc(values.byteLength);
  module.HEAP32.set(values, pointer >> 2);
  return pointer;
}

function allocateFloat64Array(module: EmscriptenBudgetOptimizerModule, values: Float64Array) {
  const pointer = module._malloc(values.byteLength);
  module.HEAPF64.set(values, pointer >> 3);
  return pointer;
}

async function loadBudgetOptimizerWasmModule() {
  if (!budgetOptimizerModulePromise) {
    budgetOptimizerModulePromise = (async () => {
      const moduleUrl = new URL(`${import.meta.env.BASE_URL}wasm/budgetOptimizerCore.mjs`, self.location.origin);
      const wasmUrl = new URL(`${import.meta.env.BASE_URL}wasm/budgetOptimizerCore.wasm`, self.location.origin);
      const imported = (await import(/* @vite-ignore */ moduleUrl.href)) as { default: BudgetOptimizerModuleFactory };
      return imported.default({
        locateFile: (path, scriptDirectory) => {
          if (path.endsWith(".wasm")) {
            return wasmUrl.href;
          }

          return new URL(path, scriptDirectory || moduleUrl.href).href;
        },
      });
    })().catch((error: unknown) => {
      budgetOptimizerModulePromise = null;
      throw error;
    });
  }

  return budgetOptimizerModulePromise;
}

function runBudgetOptimizerWithWasmCore(
  module: EmscriptenBudgetOptimizerModule,
  prepared: PreparedBudgetOptimizerInput,
) {
  const stateModuleMasksPtr = allocateInt8Array(module, prepared.stateModuleMasks);
  const singleCompatibilityKeysPtr = allocateInt32Array(module, prepared.singleCompatibilityKeys);
  const pairCompatibilityKeysPtr = allocateInt32Array(module, prepared.pairCompatibilityKeys);
  const nextStateIndexByModuleMaskAndGradeMaskPtr = allocateInt32Array(
    module,
    prepared.nextStateIndexByModuleMaskAndGradeMask,
  );
  const targetStateByIndexPtr = allocateUint8Array(module, prepared.targetStateByIndex);
  const optionTransitionWeightsByStatePtr = allocateFloat64Array(module, prepared.optionTransitionWeightsByState);
  const gradeTransitionCountsByStateAndMaskPtr = allocateUint8Array(
    module,
    prepared.gradeTransitionCountsByStateAndMask,
  );
  const gradeTransitionGradeMasksByStateAndMaskPtr = allocateUint8Array(
    module,
    prepared.gradeTransitionGradeMasksByStateAndMask,
  );
  const gradeTransitionProbabilitiesByStateAndMaskPtr = allocateFloat64Array(
    module,
    prepared.gradeTransitionProbabilitiesByStateAndMask,
  );
  const actionModuleCostByMaskTripletPtr = allocateInt8Array(module, prepared.actionModuleCostByMaskTriplet);
  const actionLockKeyCostByMaskTripletPtr = allocateInt8Array(module, prepared.actionLockKeyCostByMaskTriplet);
  const optionCandidateCountsByStatePtr = allocateUint8Array(module, prepared.optionCandidateCountsByState);
  const optionCandidateNextModuleMasksByStatePtr = allocateInt8Array(
    module,
    prepared.optionCandidateNextModuleMasksByState,
  );
  const optionCandidateProtectedMasksByStatePtr = allocateInt8Array(
    module,
    prepared.optionCandidateProtectedMasksByState,
  );
  const optionCandidateKeyMasksByStatePtr = allocateInt8Array(module, prepared.optionCandidateKeyMasksByState);
  const optionCandidateProbabilityMassesByStatePtr = allocateFloat64Array(
    module,
    prepared.optionCandidateProbabilityMassesByState,
  );
  const gradeCandidateCountsByStatePtr = allocateUint8Array(module, prepared.gradeCandidateCountsByState);
  const gradeCandidateNextModuleMasksByStatePtr = allocateInt8Array(
    module,
    prepared.gradeCandidateNextModuleMasksByState,
  );
  const gradeCandidateProtectedMasksByStatePtr = allocateInt8Array(
    module,
    prepared.gradeCandidateProtectedMasksByState,
  );
  const gradeCandidateKeyMasksByStatePtr = allocateInt8Array(module, prepared.gradeCandidateKeyMasksByState);
  const successProbabilityTablePtr = allocateFloat64Array(module, prepared.successProbabilityTable);
  const expectedLockKeyCostTablePtr = allocateFloat64Array(module, prepared.expectedLockKeyCostTable);
  const actionTypeTablePtr = allocateInt8Array(module, prepared.actionTypeTable);
  const actionModuleMaskTablePtr = allocateInt8Array(module, prepared.actionModuleMaskTable);
  const actionKeyMaskTablePtr = allocateInt8Array(module, prepared.actionKeyMaskTable);

  try {
    module._run_budget_optimizer_core(
      prepared.stateCount,
      prepared.moduleBudget,
      stateModuleMasksPtr,
      singleCompatibilityKeysPtr,
      pairCompatibilityKeysPtr,
      nextStateIndexByModuleMaskAndGradeMaskPtr,
      targetStateByIndexPtr,
      optionTransitionWeightsByStatePtr,
      gradeTransitionCountsByStateAndMaskPtr,
      gradeTransitionGradeMasksByStateAndMaskPtr,
      gradeTransitionProbabilitiesByStateAndMaskPtr,
      actionModuleCostByMaskTripletPtr,
      actionLockKeyCostByMaskTripletPtr,
      optionCandidateCountsByStatePtr,
      optionCandidateNextModuleMasksByStatePtr,
      optionCandidateProtectedMasksByStatePtr,
      optionCandidateKeyMasksByStatePtr,
      optionCandidateProbabilityMassesByStatePtr,
      gradeCandidateCountsByStatePtr,
      gradeCandidateNextModuleMasksByStatePtr,
      gradeCandidateProtectedMasksByStatePtr,
      gradeCandidateKeyMasksByStatePtr,
      successProbabilityTablePtr,
      expectedLockKeyCostTablePtr,
      actionTypeTablePtr,
      actionModuleMaskTablePtr,
      actionKeyMaskTablePtr,
    );

    prepared.successProbabilityTable.set(
      module.HEAPF64.subarray(
        successProbabilityTablePtr >> 3,
        (successProbabilityTablePtr >> 3) + prepared.successProbabilityTable.length,
      ),
    );
    prepared.expectedLockKeyCostTable.set(
      module.HEAPF64.subarray(
        expectedLockKeyCostTablePtr >> 3,
        (expectedLockKeyCostTablePtr >> 3) + prepared.expectedLockKeyCostTable.length,
      ),
    );
    prepared.actionTypeTable.set(
      module.HEAP8.subarray(actionTypeTablePtr, actionTypeTablePtr + prepared.actionTypeTable.length),
    );
    prepared.actionModuleMaskTable.set(
      module.HEAP8.subarray(actionModuleMaskTablePtr, actionModuleMaskTablePtr + prepared.actionModuleMaskTable.length),
    );
    prepared.actionKeyMaskTable.set(
      module.HEAP8.subarray(actionKeyMaskTablePtr, actionKeyMaskTablePtr + prepared.actionKeyMaskTable.length),
    );
  } finally {
    module._free(stateModuleMasksPtr);
    module._free(singleCompatibilityKeysPtr);
    module._free(pairCompatibilityKeysPtr);
    module._free(nextStateIndexByModuleMaskAndGradeMaskPtr);
    module._free(targetStateByIndexPtr);
    module._free(optionTransitionWeightsByStatePtr);
    module._free(gradeTransitionCountsByStateAndMaskPtr);
    module._free(gradeTransitionGradeMasksByStateAndMaskPtr);
    module._free(gradeTransitionProbabilitiesByStateAndMaskPtr);
    module._free(actionModuleCostByMaskTripletPtr);
    module._free(actionLockKeyCostByMaskTripletPtr);
    module._free(optionCandidateCountsByStatePtr);
    module._free(optionCandidateNextModuleMasksByStatePtr);
    module._free(optionCandidateProtectedMasksByStatePtr);
    module._free(optionCandidateKeyMasksByStatePtr);
    module._free(optionCandidateProbabilityMassesByStatePtr);
    module._free(gradeCandidateCountsByStatePtr);
    module._free(gradeCandidateNextModuleMasksByStatePtr);
    module._free(gradeCandidateProtectedMasksByStatePtr);
    module._free(gradeCandidateKeyMasksByStatePtr);
    module._free(successProbabilityTablePtr);
    module._free(expectedLockKeyCostTablePtr);
    module._free(actionTypeTablePtr);
    module._free(actionModuleMaskTablePtr);
    module._free(actionKeyMaskTablePtr);
  }
}

export async function optimizeOverloadBudgetSuccessWithRuntimeWasm(
  targetOptionIds: OverloadOptionIds[],
  targetGradeTargets: OverloadOptionTarget[],
  maxModuleBudget: number,
): Promise<OverloadBudgetOptimizationResult> {
  try {
    const prepared = prepareBudgetOptimizerKernelInput(targetOptionIds, targetGradeTargets, maxModuleBudget);
    const module = await loadBudgetOptimizerWasmModule();
    runBudgetOptimizerWithWasmCore(module, prepared);

    return {
      moduleBudget: prepared.moduleBudget,
      stateCount: prepared.stateCount,
      stateIndexByKey: prepared.stateIndexByKey,
      successProbabilityTable: prepared.successProbabilityTable,
      expectedLockKeyCostTable: prepared.expectedLockKeyCostTable,
      actionTypeTable: prepared.actionTypeTable,
      actionModuleMaskTable: prepared.actionModuleMaskTable,
      actionKeyMaskTable: prepared.actionKeyMaskTable,
    };
  } catch (error) {
    console.warn("WASM budget optimizer load failed, falling back to TypeScript budget optimizer.", error);
    return optimizeOverloadBudgetSuccess(targetOptionIds, targetGradeTargets, maxModuleBudget);
  }
}
