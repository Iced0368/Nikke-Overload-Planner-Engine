import { performance } from "node:perf_hooks";
import process from "node:process";
import { optimizeOverloadPolicy, type OverloadState } from "../src/lib/overloadPolicyOptimizer";
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
} from "../src/lib/overloadOptions";

const MASKS = [0, 1, 2, 3, 4, 5, 6] as const;
const SINGLE_MASK_INDEX = [-1, 0, 1, -1, 2, -1, -1] as const;
const OPTION_RADIX = OVERLOAD_OPTION_COUNT + 1;
const STATE_KEY_SIZE = OPTION_RADIX * OPTION_RADIX * OPTION_RADIX * 2 * 2 * 2 * 2 * 2 * 2;
const SLOT_VALUE_KEY_SIZE = (OVERLOAD_OPTION_COUNT + 1) * 2;
const ACTION_DONE = 0;
const ACTION_OPTION = 1;
const MASK_COUNT = MASKS.length;
const KEY_COUNT_PER_STATE = 3;
const GRADE_MASK_COUNT = 8;
const MAX_GRADE_TRANSITIONS_PER_MASK = GRADE_MASK_COUNT;
const MAX_OPTION_ACTION_CANDIDATES = 19;
const ACTION_COST_TABLE_SIZE = MASK_COUNT * MASK_COUNT * MASK_COUNT;
const INFINITE_COST = 1e5;
const BENCH_RUNS = 3;

const targetOptionIds = [
  ["elementdmg", "atk", "ammunition"],
  ["elementdmg", "ammunition", "atk"],
  ["atk", "elementdmg", "ammunition"],
  ["atk", "ammunition", "elementdmg"],
  ["ammunition", "elementdmg", "atk"],
  ["ammunition", "atk", "elementdmg"],
] as const satisfies readonly OverloadOptionIds[];

const targetGrades: OverloadOptionTarget[] = [
  { id: "elementdmg", grade: 9 },
  { id: "atk", grade: 9 },
  { id: "ammunition", grade: 9 },
];

const startState: OverloadState = [1, 2, 5, 1, 1, 1, 0, 0, 0];
const ITERATIONS = 1000;

type PreparedOptimizerInput = {
  states: OverloadState[];
  stateIndexByKey: Int32Array;
  stateModuleMasks: Int8Array;
  optionTripleKeysByState: Int32Array;
  singleCompatibilityKeys: Int32Array;
  pairCompatibilityKeys: Int32Array;
  nextStateIndexByModuleMaskAndGradeMask: Int32Array;
  optionTargetMatch: Uint8Array;
  optionTransitionWeightsByState: Float64Array;
  gradeTransitionCountsByStateAndMask: Uint8Array;
  gradeTransitionGradeMasksByStateAndMask: Uint8Array;
  gradeTransitionProbabilitiesByStateAndMask: Float64Array;
  weightedActionCostByMaskTriplet: Float64Array;
  moduleActionCostByMaskTriplet: Float64Array;
  lockKeyActionCostByMaskTriplet: Float64Array;
  optionCandidateCountsByState: Uint8Array;
  optionCandidateNextModuleMasksByState: Int8Array;
  optionCandidateProtectedMasksByState: Int8Array;
  optionCandidateKeyMasksByState: Int8Array;
  optionCandidateProbabilityMassesByState: Float64Array;
  costs: Float64Array;
  expectedModuleCosts: Float64Array;
  expectedLockKeyCosts: Float64Array;
  actionTypeByState: Int8Array;
  actionModuleMaskByState: Int8Array;
  actionKeyMaskByState: Int8Array;
};

type OptimizerKernelSummary = {
  iterationsRun: number;
  startCost: number;
  startExpectedModuleCost: number;
  startExpectedLockKeyCost: number;
};

type EmscriptenOptimizerModule = {
  HEAP8: Int8Array;
  HEAPU8: Uint8Array;
  HEAP32: Int32Array;
  HEAPF64: Float64Array;
  _malloc(size: number): number;
  _free(ptr: number): void;
  _run_optimizer_core(
    stateCount: number,
    iterations: number,
    progressEveryIterations: number,
    infiniteCost: number,
    stateModuleMasksPtr: number,
    optionTripleKeysByStatePtr: number,
    singleCompatibilityKeysPtr: number,
    pairCompatibilityKeysPtr: number,
    nextStateIndexByModuleMaskAndGradeMaskPtr: number,
    optionTargetMatchPtr: number,
    optionTransitionWeightsByStatePtr: number,
    gradeTransitionCountsByStateAndMaskPtr: number,
    gradeTransitionGradeMasksByStateAndMaskPtr: number,
    gradeTransitionProbabilitiesByStateAndMaskPtr: number,
    weightedActionCostByMaskTripletPtr: number,
    moduleActionCostByMaskTripletPtr: number,
    lockKeyActionCostByMaskTripletPtr: number,
    optionCandidateCountsByStatePtr: number,
    optionCandidateNextModuleMasksByStatePtr: number,
    optionCandidateProtectedMasksByStatePtr: number,
    optionCandidateKeyMasksByStatePtr: number,
    optionCandidateProbabilityMassesByStatePtr: number,
    costsPtr: number,
    expectedModuleCostsPtr: number,
    expectedLockKeyCostsPtr: number,
    actionTypeByStatePtr: number,
    actionModuleMaskByStatePtr: number,
    actionKeyMaskByStatePtr: number,
    iterationsRunOutPtr: number,
  ): void;
};

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

function buildActionCostIndex(currentModuleMask: number, nextModuleMask: number, keyMask: number) {
  return (currentModuleMask * MASK_COUNT + nextModuleMask) * MASK_COUNT + keyMask;
}

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

function prepareOptimizerKernelInput(
  targetOptionIdsInput: OverloadOptionIds[],
  targetGradeTargets: OverloadOptionTarget[],
  costWeights: OverloadCostWeights,
) {
  const states = Array.from(iterateOverloadStates());
  const stateCount = states.length;
  const { meetsTargetGradeProbabilities, optionProbabilityByIndex, targetStates } = buildDerivedOverloadData(
    targetOptionIdsInput,
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

  const stateModuleMasks = new Int8Array(stateCount);
  const optionTripleKeysByState = new Int32Array(stateCount);
  const singleCompatibilityKeys = new Int32Array(stateCount * KEY_COUNT_PER_STATE);
  const pairCompatibilityKeys = new Int32Array(stateCount * KEY_COUNT_PER_STATE);
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

    costs[stateIndex] = 0;
    actionTypeByState[stateIndex] = ACTION_DONE;
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

        let prob = 1;
        if (!p1) prob *= meetsTargetGradeProbabilities[o1][ng1];
        if (!p2) prob *= meetsTargetGradeProbabilities[o2][ng2];
        if (!p3) prob *= meetsTargetGradeProbabilities[o3][ng3];
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

  let squaredOptionProbabilitySum = 0;
  let cubedOptionProbabilitySum = 0;
  for (let index = 0; index <= OVERLOAD_OPTION_COUNT; index++) {
    squaredOptionProbabilitySum += optionProbabilityByIndex[index]! ** 2;
    cubedOptionProbabilitySum += optionProbabilityByIndex[index]! ** 3;
  }

  const getOptionTransitionProbabilityMass = (o1: number, o2: number, o3: number, protectedMask: number) => {
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

  const optionCandidateCountsByState = new Uint8Array(stateCount);
  const optionCandidateNextModuleMasksByState = new Int8Array(stateCount * MAX_OPTION_ACTION_CANDIDATES);
  const optionCandidateProtectedMasksByState = new Int8Array(stateCount * MAX_OPTION_ACTION_CANDIDATES);
  const optionCandidateKeyMasksByState = new Int8Array(stateCount * MAX_OPTION_ACTION_CANDIDATES);
  const optionCandidateProbabilityMassesByState = new Float64Array(stateCount * MAX_OPTION_ACTION_CANDIDATES);
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

  return {
    states,
    stateIndexByKey,
    stateModuleMasks,
    optionTripleKeysByState,
    singleCompatibilityKeys,
    pairCompatibilityKeys,
    nextStateIndexByModuleMaskAndGradeMask,
    optionTargetMatch,
    optionTransitionWeightsByState,
    gradeTransitionCountsByStateAndMask,
    gradeTransitionGradeMasksByStateAndMask,
    gradeTransitionProbabilitiesByStateAndMask,
    weightedActionCostByMaskTriplet,
    moduleActionCostByMaskTriplet,
    lockKeyActionCostByMaskTriplet,
    optionCandidateCountsByState,
    optionCandidateNextModuleMasksByState,
    optionCandidateProtectedMasksByState,
    optionCandidateKeyMasksByState,
    optionCandidateProbabilityMassesByState,
    costs,
    expectedModuleCosts,
    expectedLockKeyCosts,
    actionTypeByState,
    actionModuleMaskByState,
    actionKeyMaskByState,
  };
}

function allocateInt8Array(module: EmscriptenOptimizerModule, values: Int8Array) {
  const pointer = module._malloc(values.byteLength);
  module.HEAP8.set(values, pointer);
  return pointer;
}

function allocateUint8Array(module: EmscriptenOptimizerModule, values: Uint8Array) {
  const pointer = module._malloc(values.byteLength);
  module.HEAPU8.set(values, pointer);
  return pointer;
}

function allocateInt32Array(module: EmscriptenOptimizerModule, values: Int32Array) {
  const pointer = module._malloc(values.byteLength);
  module.HEAP32.set(values, pointer >> 2);
  return pointer;
}

function allocateFloat64Array(module: EmscriptenOptimizerModule, values: Float64Array) {
  const pointer = module._malloc(values.byteLength);
  module.HEAPF64.set(values, pointer >> 3);
  return pointer;
}

async function loadWasmModule() {
  const moduleUrl = new URL("../public/wasm/optimizerCore.mjs", import.meta.url);
  const imported = (await import(moduleUrl.href)) as { default: () => Promise<EmscriptenOptimizerModule> };
  return imported.default();
}

function runOptimizerWithWasmCore(
  module: EmscriptenOptimizerModule,
  prepared: PreparedOptimizerInput,
  iterations: number,
  simulationStartState: OverloadState,
): OptimizerKernelSummary {
  const stateModuleMasksPtr = allocateInt8Array(module, prepared.stateModuleMasks);
  const optionTripleKeysByStatePtr = allocateInt32Array(module, prepared.optionTripleKeysByState);
  const singleCompatibilityKeysPtr = allocateInt32Array(module, prepared.singleCompatibilityKeys);
  const pairCompatibilityKeysPtr = allocateInt32Array(module, prepared.pairCompatibilityKeys);
  const nextStateIndexByModuleMaskAndGradeMaskPtr = allocateInt32Array(
    module,
    prepared.nextStateIndexByModuleMaskAndGradeMask,
  );
  const optionTargetMatchPtr = allocateUint8Array(module, prepared.optionTargetMatch);
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
  const weightedActionCostByMaskTripletPtr = allocateFloat64Array(module, prepared.weightedActionCostByMaskTriplet);
  const moduleActionCostByMaskTripletPtr = allocateFloat64Array(module, prepared.moduleActionCostByMaskTriplet);
  const lockKeyActionCostByMaskTripletPtr = allocateFloat64Array(module, prepared.lockKeyActionCostByMaskTriplet);
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
  const costsPtr = allocateFloat64Array(module, prepared.costs);
  const expectedModuleCostsPtr = allocateFloat64Array(module, prepared.expectedModuleCosts);
  const expectedLockKeyCostsPtr = allocateFloat64Array(module, prepared.expectedLockKeyCosts);
  const actionTypeByStatePtr = allocateInt8Array(module, prepared.actionTypeByState);
  const actionModuleMaskByStatePtr = allocateInt8Array(module, prepared.actionModuleMaskByState);
  const actionKeyMaskByStatePtr = allocateInt8Array(module, prepared.actionKeyMaskByState);
  const iterationsRunOutPtr = module._malloc(Int32Array.BYTES_PER_ELEMENT);

  try {
    module._run_optimizer_core(
      prepared.states.length,
      iterations,
      0,
      INFINITE_COST,
      stateModuleMasksPtr,
      optionTripleKeysByStatePtr,
      singleCompatibilityKeysPtr,
      pairCompatibilityKeysPtr,
      nextStateIndexByModuleMaskAndGradeMaskPtr,
      optionTargetMatchPtr,
      optionTransitionWeightsByStatePtr,
      gradeTransitionCountsByStateAndMaskPtr,
      gradeTransitionGradeMasksByStateAndMaskPtr,
      gradeTransitionProbabilitiesByStateAndMaskPtr,
      weightedActionCostByMaskTripletPtr,
      moduleActionCostByMaskTripletPtr,
      lockKeyActionCostByMaskTripletPtr,
      optionCandidateCountsByStatePtr,
      optionCandidateNextModuleMasksByStatePtr,
      optionCandidateProtectedMasksByStatePtr,
      optionCandidateKeyMasksByStatePtr,
      optionCandidateProbabilityMassesByStatePtr,
      costsPtr,
      expectedModuleCostsPtr,
      expectedLockKeyCostsPtr,
      actionTypeByStatePtr,
      actionModuleMaskByStatePtr,
      actionKeyMaskByStatePtr,
      iterationsRunOutPtr,
    );

    const startStateIndex = prepared.stateIndexByKey[encodeStateKey(simulationStartState)];
    const costsOffset = costsPtr >> 3;
    const expectedModuleCostsOffset = expectedModuleCostsPtr >> 3;
    const expectedLockKeyCostsOffset = expectedLockKeyCostsPtr >> 3;
    const iterationsRunOffset = iterationsRunOutPtr >> 2;
    return {
      iterationsRun: module.HEAP32[iterationsRunOffset]!,
      startCost: module.HEAPF64[costsOffset + startStateIndex]!,
      startExpectedModuleCost: module.HEAPF64[expectedModuleCostsOffset + startStateIndex]!,
      startExpectedLockKeyCost: module.HEAPF64[expectedLockKeyCostsOffset + startStateIndex]!,
    };
  } finally {
    module._free(stateModuleMasksPtr);
    module._free(optionTripleKeysByStatePtr);
    module._free(singleCompatibilityKeysPtr);
    module._free(pairCompatibilityKeysPtr);
    module._free(nextStateIndexByModuleMaskAndGradeMaskPtr);
    module._free(optionTargetMatchPtr);
    module._free(optionTransitionWeightsByStatePtr);
    module._free(gradeTransitionCountsByStateAndMaskPtr);
    module._free(gradeTransitionGradeMasksByStateAndMaskPtr);
    module._free(gradeTransitionProbabilitiesByStateAndMaskPtr);
    module._free(weightedActionCostByMaskTripletPtr);
    module._free(moduleActionCostByMaskTripletPtr);
    module._free(lockKeyActionCostByMaskTripletPtr);
    module._free(optionCandidateCountsByStatePtr);
    module._free(optionCandidateNextModuleMasksByStatePtr);
    module._free(optionCandidateProtectedMasksByStatePtr);
    module._free(optionCandidateKeyMasksByStatePtr);
    module._free(optionCandidateProbabilityMassesByStatePtr);
    module._free(costsPtr);
    module._free(expectedModuleCostsPtr);
    module._free(expectedLockKeyCostsPtr);
    module._free(actionTypeByStatePtr);
    module._free(actionModuleMaskByStatePtr);
    module._free(actionKeyMaskByStatePtr);
    module._free(iterationsRunOutPtr);
  }
}

async function benchmarkAsync<T>(runs: number, execute: () => Promise<T>) {
  const timings: number[] = [];
  let lastResult = await execute();
  for (let runIndex = 0; runIndex < runs; runIndex++) {
    const startedAt = performance.now();
    lastResult = await execute();
    timings.push(performance.now() - startedAt);
  }
  return {
    result: lastResult,
    timings,
    averageMs: timings.reduce((sum, timing) => sum + timing, 0) / timings.length,
    bestMs: Math.min(...timings),
  };
}

function formatTimings(label: string, timings: number[]) {
  return `${label}: ${timings.map((timing) => `${timing.toFixed(2)}ms`).join(", ")}`;
}

console.log(`Node ${process.version}`);
console.log("Loading optimizer WASM module...");
const wasmModule = await loadWasmModule();

console.log("Running optimizer benchmark...\n");

const baselineBenchmark = await benchmarkAsync(BENCH_RUNS, async () =>
  optimizeOverloadPolicy(
    targetOptionIds.map((ids) => [...ids] as OverloadOptionIds),
    targetGrades,
    ITERATIONS,
  ),
);

const wasmBenchmark = await benchmarkAsync(BENCH_RUNS, async () => {
  const prepared = prepareOptimizerKernelInput(
    targetOptionIds.map((ids) => [...ids] as OverloadOptionIds),
    targetGrades,
    defaultCostWeights,
  );
  return runOptimizerWithWasmCore(wasmModule, prepared, ITERATIONS, startState);
});

const baselineStateValue =
  baselineBenchmark.result.stateValues[startState[0]][startState[1]][startState[2]][startState[3]][startState[4]][
    startState[5]
  ][startState[6]][startState[7]][startState[8]];

console.log(formatTimings("Current optimizeOverloadPolicy", baselineBenchmark.timings));
console.log(formatTimings("WASM core prototype", wasmBenchmark.timings));
console.log("");
console.log(
  `Average runtime: current ${baselineBenchmark.averageMs.toFixed(2)}ms, wasm-prototype ${wasmBenchmark.averageMs.toFixed(2)}ms`,
);
console.log(
  `Best runtime: current ${baselineBenchmark.bestMs.toFixed(2)}ms, wasm-prototype ${wasmBenchmark.bestMs.toFixed(2)}ms`,
);
console.log(`Speedup(best): x${(baselineBenchmark.bestMs / wasmBenchmark.bestMs).toFixed(2)}`);
console.log("");
console.log(
  `Start-state weighted cost: current ${baselineStateValue.cost.toFixed(4)}, wasm-prototype ${wasmBenchmark.result.startCost.toFixed(4)}`,
);
console.log(
  `Start-state module/lockKey expectation: current ${baselineStateValue.expectedCosts.module.toFixed(4)} / ${baselineStateValue.expectedCosts.lockKey.toFixed(4)}, wasm-prototype ${wasmBenchmark.result.startExpectedModuleCost.toFixed(4)} / ${wasmBenchmark.result.startExpectedLockKeyCost.toFixed(4)}`,
);
console.log(
  `Iterations run: current ${baselineBenchmark.result.iterationsRun}, wasm-prototype ${wasmBenchmark.result.iterationsRun}`,
);
console.log("");
console.log(
  "Note: the WASM prototype keeps preprocessing in TypeScript and moves only the iterative optimizer core to C++/WASM.",
);
