import { performance } from "node:perf_hooks";
import process from "node:process";
import { runMonteCarloPolicySimulation } from "../src/lib/overloadMonteCarlo";
import {
  optimizeOverloadPolicy,
  type OverloadPolicyOptimizationResult,
  type OverloadState,
} from "../src/lib/overloadPolicyOptimizer";
import {
  defaultCostWeights,
  lockCosts,
  lockKeyCosts,
  overloadGradeProbabilities,
  overloadOptions,
  rerollCosts,
  slotOptionProbabilities,
  type OverloadCostWeights,
  type OverloadOptionIds,
  type OverloadOptionTarget,
} from "../src/lib/overloadOptions";

const OPTION_RADIX = overloadOptions.length;
const STATE_KEY_SIZE = OPTION_RADIX * OPTION_RADIX * OPTION_RADIX * 2 * 2 * 2 * 2 * 2 * 2;
const TRIAL_COUNT = 200_000;
const MAX_STEPS = 10_000;
const BENCH_RUNS = 3;
const SEED = 0x1234abcd;

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

const simulationStartState: OverloadState = [1, 2, 5, 1, 1, 1, 0, 0, 0];

type FlatPolicyBenchmarkData = {
  startStateIndex: number;
  terminalFlags: Int32Array;
  weightedCosts: Float64Array;
  moduleCosts: Float64Array;
  lockKeyCosts: Float64Array;
  transitionOffsets: Int32Array;
  transitionNextStates: Int32Array;
  transitionCumulativeProbabilities: Float64Array;
};

type TrialAggregate = {
  meanWeightedCost: number;
  meanModuleCost: number;
  meanLockKeyCost: number;
  meanSteps: number;
  successRate: number;
};

type EmscriptenBenchmarkModule = {
  HEAP32: Int32Array;
  HEAPF64: Float64Array;
  _malloc(size: number): number;
  _free(ptr: number): void;
  _run_policy_trials(
    trialCount: number,
    maxSteps: number,
    startStateIndex: number,
    seed: number,
    terminalFlagsPtr: number,
    weightedCostsPtr: number,
    moduleCostsPtr: number,
    lockKeyCostsPtr: number,
    transitionOffsetsPtr: number,
    transitionNextStatesPtr: number,
    transitionCumProbPtr: number,
    outputPtr: number,
  ): void;
};

function countBits(mask: number) {
  return Number(Boolean(mask & 1)) + Number(Boolean(mask & 2)) + Number(Boolean(mask & 4));
}

function buildMask([b1, b2, b3]: [number, number, number]) {
  return b1 | (b2 << 1) | (b3 << 2);
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

function getStateValue(
  stateValues: OverloadPolicyOptimizationResult["stateValues"],
  [o1, o2, o3, g1, g2, g3, m1, m2, m3]: OverloadState,
) {
  return stateValues[o1][o2][o3][g1][g2][g3][m1][m2][m3];
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

function buildSimulationData(targets: OverloadOptionTarget[]) {
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
  for (const target of targets) {
    const optionIndex = optionIndexById.get(target.id);
    if (optionIndex !== undefined) {
      requiredGradeByOption[optionIndex] = target.grade;
    }
  }

  return {
    optionProbabilityByIndex,
    successProbabilityByOption: requiredGradeByOption.map(
      (requiredGrade) => gradeTailProbabilityByThreshold[requiredGrade] ?? 0,
    ),
  };
}

function buildWeightedActionCost(
  currentModuleMask: number,
  nextModuleMask: number,
  keyMask: number,
  protectedCount: number,
  costWeights: OverloadCostWeights,
) {
  const keptModuleCount = countBits(currentModuleMask & nextModuleMask);
  const nextModuleCount = countBits(nextModuleMask);
  const moduleCost = rerollCosts[protectedCount]! + lockCosts[nextModuleCount]! - lockCosts[keptModuleCount]!;
  const lockKeyCost = lockKeyCosts[countBits(keyMask)]!;
  return {
    moduleCost,
    lockKeyCost,
    weightedCost: costWeights.module * moduleCost + costWeights.lockKey * lockKeyCost,
  };
}

function addProbability(target: Map<number, number>, stateIndex: number, probability: number) {
  target.set(stateIndex, (target.get(stateIndex) ?? 0) + probability);
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
    addProbability(distribution, nextStateIndex, probability);
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
    addProbability(distribution, nextStateIndex, probability);
  }

  return distribution;
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

function buildFlatPolicyBenchmarkData(
  startState: OverloadState,
  result: OverloadPolicyOptimizationResult,
  targets: OverloadOptionTarget[],
  costWeights: OverloadCostWeights,
): FlatPolicyBenchmarkData {
  const { optionProbabilityByIndex, successProbabilityByOption } = buildSimulationData(targets);
  const stateIndexByKey = new Int32Array(STATE_KEY_SIZE).fill(-1);

  for (let stateIndex = 0; stateIndex < result.states.length; stateIndex++) {
    stateIndexByKey[encodeStateKey(result.states[stateIndex]!)] = stateIndex;
  }

  const startStateIndex = stateIndexByKey[encodeStateKey(startState)];
  const terminalFlags = new Int32Array(result.states.length);
  const weightedCosts = new Float64Array(result.states.length);
  const moduleCosts = new Float64Array(result.states.length);
  const lockKeyCostsArray = new Float64Array(result.states.length);
  const transitionOffsets = new Int32Array(result.states.length + 1);
  const transitionNextStates: number[] = [];
  const transitionCumulativeProbabilities: number[] = [];

  for (let stateIndex = 0; stateIndex < result.states.length; stateIndex++) {
    const state = result.states[stateIndex]!;
    const currentValue = getStateValue(result.stateValues, state);
    transitionOffsets[stateIndex] = transitionNextStates.length;

    if (currentValue.action.type === "done") {
      terminalFlags[stateIndex] = 1;
      continue;
    }

    const nextModuleMask = actionModuleMask(currentValue.action);
    const keyMask = actionKeyMask(currentValue.action);
    const protectedMask = nextModuleMask | keyMask;
    const currentModuleMask = buildMask([state[6]!, state[7]!, state[8]!]);
    const costs = buildWeightedActionCost(
      currentModuleMask,
      nextModuleMask,
      keyMask,
      countBits(protectedMask),
      costWeights,
    );
    weightedCosts[stateIndex] = costs.weightedCost;
    moduleCosts[stateIndex] = costs.moduleCost;
    lockKeyCostsArray[stateIndex] = costs.lockKeyCost;

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
      addProbability(effectiveDistribution, nextStateIndex, probability);
    }

    for (const transition of normalizeTransitionDistribution(effectiveDistribution)) {
      transitionNextStates.push(transition.stateIndex);
      transitionCumulativeProbabilities.push(transition.cumulativeProbability);
    }
  }

  transitionOffsets[result.states.length] = transitionNextStates.length;

  return {
    startStateIndex,
    terminalFlags,
    weightedCosts,
    moduleCosts,
    lockKeyCosts: lockKeyCostsArray,
    transitionOffsets,
    transitionNextStates: Int32Array.from(transitionNextStates),
    transitionCumulativeProbabilities: Float64Array.from(transitionCumulativeProbabilities),
  };
}

function createXorShift32(seed: number) {
  let state = seed >>> 0;
  if (state === 0) {
    state = 0x9e3779b9;
  }

  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state >>>= 0;
    state ^= state << 5;
    state >>>= 0;
    return state / 0x100000000;
  };
}

function runFlatPolicyTrialsJs(
  data: FlatPolicyBenchmarkData,
  trialCount: number,
  maxSteps: number,
  seed: number,
): TrialAggregate {
  const nextRandom = createXorShift32(seed);
  let weightedCostSum = 0;
  let moduleCostSum = 0;
  let lockKeyCostSum = 0;
  let totalSteps = 0;
  let successCount = 0;

  for (let trial = 0; trial < trialCount; trial++) {
    let stateIndex = data.startStateIndex;

    for (let step = 0; step < maxSteps; step++) {
      if (data.terminalFlags[stateIndex] === 1) {
        successCount += 1;
        totalSteps += step;
        break;
      }

      weightedCostSum += data.weightedCosts[stateIndex]!;
      moduleCostSum += data.moduleCosts[stateIndex]!;
      lockKeyCostSum += data.lockKeyCosts[stateIndex]!;

      const transitionStart = data.transitionOffsets[stateIndex]!;
      const transitionEnd = data.transitionOffsets[stateIndex + 1]!;
      const draw = nextRandom();

      let nextStateIndex = data.transitionNextStates[transitionEnd - 1]!;
      for (let transitionIndex = transitionStart; transitionIndex < transitionEnd; transitionIndex++) {
        if (draw <= data.transitionCumulativeProbabilities[transitionIndex]!) {
          nextStateIndex = data.transitionNextStates[transitionIndex]!;
          break;
        }
      }

      stateIndex = nextStateIndex;
    }
  }

  return {
    meanWeightedCost: weightedCostSum / trialCount,
    meanModuleCost: moduleCostSum / trialCount,
    meanLockKeyCost: lockKeyCostSum / trialCount,
    meanSteps: totalSteps / trialCount,
    successRate: successCount / trialCount,
  };
}

function allocateInt32Array(module: EmscriptenBenchmarkModule, values: Int32Array) {
  const pointer = module._malloc(values.byteLength);
  module.HEAP32.set(values, pointer >> 2);
  return pointer;
}

function allocateFloat64Array(module: EmscriptenBenchmarkModule, values: Float64Array) {
  const pointer = module._malloc(values.byteLength);
  module.HEAPF64.set(values, pointer >> 3);
  return pointer;
}

async function loadWasmModule() {
  const moduleUrl = new URL("../dist/wasm-bench/monteCarloPolicy.mjs", import.meta.url);
  const imported = (await import(moduleUrl.href)) as { default: () => Promise<EmscriptenBenchmarkModule> };
  return imported.default();
}

function runFlatPolicyTrialsWasm(
  module: EmscriptenBenchmarkModule,
  data: FlatPolicyBenchmarkData,
  trialCount: number,
  maxSteps: number,
  seed: number,
): TrialAggregate {
  const terminalFlagsPtr = allocateInt32Array(module, data.terminalFlags);
  const weightedCostsPtr = allocateFloat64Array(module, data.weightedCosts);
  const moduleCostsPtr = allocateFloat64Array(module, data.moduleCosts);
  const lockKeyCostsPtr = allocateFloat64Array(module, data.lockKeyCosts);
  const transitionOffsetsPtr = allocateInt32Array(module, data.transitionOffsets);
  const transitionNextStatesPtr = allocateInt32Array(module, data.transitionNextStates);
  const transitionCumProbPtr = allocateFloat64Array(module, data.transitionCumulativeProbabilities);
  const outputPtr = module._malloc(Float64Array.BYTES_PER_ELEMENT * 5);

  try {
    module._run_policy_trials(
      trialCount,
      maxSteps,
      data.startStateIndex,
      seed >>> 0,
      terminalFlagsPtr,
      weightedCostsPtr,
      moduleCostsPtr,
      lockKeyCostsPtr,
      transitionOffsetsPtr,
      transitionNextStatesPtr,
      transitionCumProbPtr,
      outputPtr,
    );

    const outputOffset = outputPtr >> 3;
    const weightedCostSum = module.HEAPF64[outputOffset]!;
    const moduleCostSum = module.HEAPF64[outputOffset + 1]!;
    const lockKeyCostSum = module.HEAPF64[outputOffset + 2]!;
    const totalSteps = module.HEAPF64[outputOffset + 3]!;
    const successCount = module.HEAPF64[outputOffset + 4]!;

    return {
      meanWeightedCost: weightedCostSum / trialCount,
      meanModuleCost: moduleCostSum / trialCount,
      meanLockKeyCost: lockKeyCostSum / trialCount,
      meanSteps: totalSteps / trialCount,
      successRate: successCount / trialCount,
    };
  } finally {
    module._free(terminalFlagsPtr);
    module._free(weightedCostsPtr);
    module._free(moduleCostsPtr);
    module._free(lockKeyCostsPtr);
    module._free(transitionOffsetsPtr);
    module._free(transitionNextStatesPtr);
    module._free(transitionCumProbPtr);
    module._free(outputPtr);
  }
}

function benchmark<T>(runs: number, execute: () => T) {
  const timings: number[] = [];
  let lastResult = execute();

  for (let runIndex = 0; runIndex < runs; runIndex++) {
    const startedAt = performance.now();
    lastResult = execute();
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

function formatSteps(value: number) {
  return Number.isFinite(value) ? value.toFixed(4) : "n/a";
}

function printAggregate(label: string, aggregate: TrialAggregate) {
  console.log(
    `${label} mean(weighted/module/lockKey/steps/success) = ${aggregate.meanWeightedCost.toFixed(4)} / ${aggregate.meanModuleCost.toFixed(4)} / ${aggregate.meanLockKeyCost.toFixed(4)} / ${formatSteps(aggregate.meanSteps)} / ${(aggregate.successRate * 100).toFixed(2)}%`,
  );
}

console.log(`Node ${process.version}`);
console.log(`Optimizing classic policy for ${TRIAL_COUNT.toLocaleString()}-trial benchmark input...`);

const optimizeStartedAt = performance.now();
const optimizationResult = await optimizeOverloadPolicy(
  targetOptionIds.map((ids) => [...ids] as OverloadOptionIds),
  targetGrades,
  1000,
);
console.log(`Policy optimization finished in ${(performance.now() - optimizeStartedAt).toFixed(2)}ms`);

console.log("Building flattened transition table...");
const flatData = buildFlatPolicyBenchmarkData(
  simulationStartState,
  optimizationResult,
  targetGrades,
  defaultCostWeights,
);
console.log(
  `Flattened ${optimizationResult.states.length.toLocaleString()} states into ${flatData.transitionNextStates.length.toLocaleString()} effective transitions.`,
);

console.log("Loading WASM module...");
const wasmModule = await loadWasmModule();

console.log("Running benchmark...\n");

const currentBenchmark = benchmark(BENCH_RUNS, () =>
  runMonteCarloPolicySimulation(simulationStartState, optimizationResult, targetGrades, {
    trialCount: TRIAL_COUNT,
    maxSteps: MAX_STEPS,
    costWeights: defaultCostWeights,
  }),
);

const flatJsBenchmark = benchmark(BENCH_RUNS, () => runFlatPolicyTrialsJs(flatData, TRIAL_COUNT, MAX_STEPS, SEED));
const flatWasmBenchmark = benchmark(BENCH_RUNS, () =>
  runFlatPolicyTrialsWasm(wasmModule, flatData, TRIAL_COUNT, MAX_STEPS, SEED),
);

console.log(formatTimings("Current Monte Carlo", currentBenchmark.timings));
console.log(formatTimings("Flat JS core", flatJsBenchmark.timings));
console.log(formatTimings("Flat WASM core", flatWasmBenchmark.timings));
console.log("");

console.log(
  `Average runtime: current ${currentBenchmark.averageMs.toFixed(2)}ms, flat-js ${flatJsBenchmark.averageMs.toFixed(2)}ms, flat-wasm ${flatWasmBenchmark.averageMs.toFixed(2)}ms`,
);
console.log(
  `Best runtime: current ${currentBenchmark.bestMs.toFixed(2)}ms, flat-js ${flatJsBenchmark.bestMs.toFixed(2)}ms, flat-wasm ${flatWasmBenchmark.bestMs.toFixed(2)}ms`,
);
console.log(
  `Speedup vs current(best): flat-js x${(currentBenchmark.bestMs / flatJsBenchmark.bestMs).toFixed(2)}, flat-wasm x${(currentBenchmark.bestMs / flatWasmBenchmark.bestMs).toFixed(2)}`,
);
console.log(`Speedup vs flat-js(best): flat-wasm x${(flatJsBenchmark.bestMs / flatWasmBenchmark.bestMs).toFixed(2)}`);
console.log("");

printAggregate("Current Monte Carlo", {
  meanWeightedCost: currentBenchmark.result.sampleMean,
  meanModuleCost: currentBenchmark.result.sampleMeanModuleCost,
  meanLockKeyCost: currentBenchmark.result.sampleMeanLockKeyCost,
  meanSteps: Number.NaN,
  successRate: currentBenchmark.result.sampleSuccessRate,
});
printAggregate("Flat JS core", flatJsBenchmark.result);
printAggregate("Flat WASM core", flatWasmBenchmark.result);

console.log(
  `Mean weighted-cost delta: current-flat-js ${Math.abs(currentBenchmark.result.sampleMean - flatJsBenchmark.result.meanWeightedCost).toFixed(4)}, flat-js-flat-wasm ${Math.abs(flatJsBenchmark.result.meanWeightedCost - flatWasmBenchmark.result.meanWeightedCost).toFixed(4)}`,
);
