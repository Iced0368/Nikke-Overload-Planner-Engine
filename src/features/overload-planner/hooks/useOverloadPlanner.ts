import { startTransition, useEffect, useMemo, useRef, useState } from "react";
import {
  readOverloadBudgetActionAlternatives,
  readOverloadBudgetOptimizationSummary,
  type OverloadBudgetActionAlternative,
  type OverloadBudgetOptimizationResult,
  type OverloadBudgetOptimizationSummary,
} from "../../../lib/overloadBudgetOptimizer.ts";
import {
  OVERLOAD_GRADE_COUNT,
  defaultCostWeights,
  overloadOptions,
  type OverloadCostWeights,
  type OverloadOptionIds,
  type OverloadOptionTarget,
} from "../../../lib/overloadOptions";
import {
  type OverloadOptimizationProgress,
  type OverloadPolicyOptimizationResult,
  readForcedLockAlternatives,
  type OverloadForcedLockAlternative,
} from "../../../lib/overloadPolicyOptimizer.ts";
import { type MonteCarloSimulationSummary } from "../../../lib/overloadMonteCarlo.ts";
import {
  buildBinaryStartState,
  buildSimulationStartState,
  createTargetStateKey,
  defaultStartModuleLocks,
  defaultStartState,
  defaultTargetStates,
  getTargetStatePermutations,
  isValidStartModuleLockState,
  isValidOptionTriple,
  isValidTargetOptionTriple,
  normalizeStartState,
  readStateValue,
  type StartModuleLockState,
  type StartStateDraft,
  type TargetStateDraft,
} from "../model/model";
import { type PlannerMode } from "../components/PlannerModeSection";
import PlannerWorker from "../workers/planner.worker?worker";
import { type PlannerWorkerResponse } from "../workers/plannerWorkerMessages";

const PLANNER_STORAGE_KEY = "overload-planner:state";
const LEGACY_TARGET_GRADE_STORAGE_KEY = "overload-planner:target-grades";
const DEFAULT_MODULE_BUDGET = 50;
const DEFAULT_PLANNER_MODE: PlannerMode = "classic";
const PLANNER_STORAGE_VERSION = 4;

type StoredPlannerState = {
  version: typeof PLANNER_STORAGE_VERSION;
  targetStates: TargetStateDraft[];
  targetGrades: Record<string, number>;
  costWeights: OverloadCostWeights;
  moduleBudget: number;
  plannerMode: PlannerMode;
};

type PlannerStateSnapshot = {
  targetStates: TargetStateDraft[];
  targetGrades: Map<string, number>;
  costWeights: OverloadCostWeights;
  moduleBudget: number;
  plannerMode: PlannerMode;
};

function sanitizePlannerMode(value: unknown): PlannerMode {
  return value === "budget" ? "budget" : "classic";
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isValidOptionIndex(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 0 && value < overloadOptions.length;
}

function isValidGrade(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 0 && value < OVERLOAD_GRADE_COUNT;
}

function sanitizeTargetStates(value: unknown): TargetStateDraft[] {
  if (!Array.isArray(value)) {
    return defaultTargetStates;
  }

  const nextStates = value.flatMap((entry) => {
    if (!Array.isArray(entry) || entry.length !== 3 || entry.some((slot) => !isValidOptionIndex(slot))) {
      return [];
    }

    const nextState = [...entry] as TargetStateDraft;
    return isValidTargetOptionTriple(nextState) ? [nextState] : [];
  });

  return nextStates.length > 0 ? nextStates : defaultTargetStates;
}

function sanitizeTargetGrades(value: unknown) {
  if (!isPlainObject(value)) {
    return new Map<string, number>();
  }

  const entries = Object.entries(value).flatMap(([optionId, grade]) => {
    if (typeof optionId !== "string" || !isValidGrade(grade)) {
      return [];
    }

    return [[optionId, grade] as const];
  });

  return new Map(entries);
}

function sanitizeCostWeights(value: unknown): OverloadCostWeights {
  if (!isPlainObject(value)) {
    return defaultCostWeights;
  }

  const module =
    typeof value.module === "number" && Number.isFinite(value.module) && value.module > 0
      ? value.module
      : defaultCostWeights.module;
  const lockKey =
    typeof value.lockKey === "number" && Number.isFinite(value.lockKey) && value.lockKey > 0
      ? value.lockKey
      : defaultCostWeights.lockKey;

  return {
    module,
    lockKey,
  };
}

function sanitizeModuleBudget(value: unknown) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return DEFAULT_MODULE_BUDGET;
  }

  return Math.min(200, Math.max(0, Math.round(value)));
}

function collectTargetOptionIdsInUse(targetStates: TargetStateDraft[]) {
  const ids = new Set<string>();

  for (const targetState of targetStates) {
    for (const optionIndex of targetState) {
      const optionId = overloadOptions[optionIndex]?.id;
      if (optionId) {
        ids.add(optionId);
      }
    }
  }

  return Array.from(ids);
}

function buildTargetGradesForOptions(targetOptionIdsInUse: string[], storedTargetGrades: Map<string, number>) {
  return targetOptionIdsInUse.map((id) => ({
    id,
    grade: storedTargetGrades.get(id) ?? 0,
  }));
}

function readLegacyTargetGradeMap() {
  if (typeof window === "undefined") {
    return new Map<string, number>();
  }

  try {
    const rawValue = window.localStorage.getItem(LEGACY_TARGET_GRADE_STORAGE_KEY);
    if (!rawValue) {
      return new Map<string, number>();
    }

    return sanitizeTargetGrades(JSON.parse(rawValue));
  } catch {
    return new Map<string, number>();
  }
}

function readStoredPlannerState(): PlannerStateSnapshot {
  const fallbackTargetGrades = readLegacyTargetGradeMap();

  if (typeof window === "undefined") {
    return {
      targetStates: defaultTargetStates,
      targetGrades: fallbackTargetGrades,
      costWeights: defaultCostWeights,
      moduleBudget: DEFAULT_MODULE_BUDGET,
      plannerMode: DEFAULT_PLANNER_MODE,
    };
  }

  try {
    const rawValue = window.localStorage.getItem(PLANNER_STORAGE_KEY);
    if (!rawValue) {
      return {
        targetStates: defaultTargetStates,
        targetGrades: fallbackTargetGrades,
        costWeights: defaultCostWeights,
        moduleBudget: DEFAULT_MODULE_BUDGET,
        plannerMode: DEFAULT_PLANNER_MODE,
      };
    }

    const parsed = JSON.parse(rawValue);
    if (!isPlainObject(parsed)) {
      return {
        targetStates: defaultTargetStates,
        targetGrades: fallbackTargetGrades,
        costWeights: defaultCostWeights,
        moduleBudget: DEFAULT_MODULE_BUDGET,
        plannerMode: DEFAULT_PLANNER_MODE,
      };
    }

    if (parsed.version === 2 || parsed.version === 3) {
      return {
        targetStates: sanitizeTargetStates(parsed.targetStates),
        targetGrades: sanitizeTargetGrades(parsed.targetGrades),
        costWeights: sanitizeCostWeights(parsed.costWeights),
        moduleBudget: DEFAULT_MODULE_BUDGET,
        plannerMode: DEFAULT_PLANNER_MODE,
      };
    }

    if (parsed.version !== PLANNER_STORAGE_VERSION) {
      return {
        targetStates: defaultTargetStates,
        targetGrades: fallbackTargetGrades,
        costWeights: defaultCostWeights,
        moduleBudget: DEFAULT_MODULE_BUDGET,
        plannerMode: DEFAULT_PLANNER_MODE,
      };
    }

    return {
      targetStates: sanitizeTargetStates(parsed.targetStates),
      targetGrades: sanitizeTargetGrades(parsed.targetGrades),
      costWeights: sanitizeCostWeights(parsed.costWeights),
      moduleBudget: sanitizeModuleBudget(parsed.moduleBudget),
      plannerMode: sanitizePlannerMode(parsed.plannerMode),
    };
  } catch {
    return {
      targetStates: defaultTargetStates,
      targetGrades: fallbackTargetGrades,
      costWeights: defaultCostWeights,
      moduleBudget: DEFAULT_MODULE_BUDGET,
      plannerMode: DEFAULT_PLANNER_MODE,
    };
  }
}

function writeStoredPlannerState(snapshot: PlannerStateSnapshot) {
  if (typeof window === "undefined") {
    return;
  }

  const payload: StoredPlannerState = {
    version: PLANNER_STORAGE_VERSION,
    targetStates: snapshot.targetStates,
    targetGrades: Object.fromEntries(snapshot.targetGrades.entries()),
    costWeights: snapshot.costWeights,
    moduleBudget: snapshot.moduleBudget,
    plannerMode: snapshot.plannerMode,
  };

  window.localStorage.setItem(PLANNER_STORAGE_KEY, JSON.stringify(payload));
  window.localStorage.removeItem(LEGACY_TARGET_GRADE_STORAGE_KEY);
}

export function useOverloadPlanner() {
  const initialStoredPlannerState = useMemo(() => readStoredPlannerState(), []);
  const workerRef = useRef<Worker | null>(null);
  const optimizeRequestIdRef = useRef(0);
  const simulationRequestIdRef = useRef(0);
  const budgetRequestIdRef = useRef(0);
  const optimizeRequestSignatureRef = useRef<string | null>(null);
  const simulationRequestSignatureRef = useRef<string | null>(null);
  const budgetRequestSignatureRef = useRef<string | null>(null);
  const storedTargetGradeByIdRef = useRef<Map<string, number>>(new Map(initialStoredPlannerState.targetGrades));
  const [startState, setStartState] = useState<StartStateDraft>(defaultStartState);
  const [startModuleLocks, setStartModuleLocks] = useState<StartModuleLockState>(defaultStartModuleLocks);
  const [targetStates, setTargetStates] = useState<TargetStateDraft[]>(initialStoredPlannerState.targetStates);
  const [targetGrades, setTargetGrades] = useState<OverloadOptionTarget[]>(() =>
    buildTargetGradesForOptions(
      collectTargetOptionIdsInUse(initialStoredPlannerState.targetStates),
      storedTargetGradeByIdRef.current,
    ),
  );
  const [costWeights, setCostWeights] = useState<OverloadCostWeights>(initialStoredPlannerState.costWeights);
  const [moduleBudget, setModuleBudget] = useState(initialStoredPlannerState.moduleBudget);
  const [plannerMode, setPlannerMode] = useState<PlannerMode>(initialStoredPlannerState.plannerMode);
  const [iterations] = useState(3000);
  const [result, setResult] = useState<OverloadPolicyOptimizationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [optimizationProgress, setOptimizationProgress] = useState<OverloadOptimizationProgress | null>(null);
  const [lastOptimizedPolicySignature, setLastOptimizedPolicySignature] = useState<string | null>(null);
  const [simulationResult, setSimulationResult] = useState<MonteCarloSimulationSummary | null>(null);
  const [simulationError, setSimulationError] = useState<string | null>(null);
  const [isSimulationRunning, setIsSimulationRunning] = useState(false);
  const [lastSimulationSignature, setLastSimulationSignature] = useState<string | null>(null);
  const [budgetOptimizationResult, setBudgetOptimizationResult] = useState<OverloadBudgetOptimizationResult | null>(
    null,
  );
  const [budgetOptimizationError, setBudgetOptimizationError] = useState<string | null>(null);
  const [isBudgetOptimizationRunning, setIsBudgetOptimizationRunning] = useState(false);
  const [lastBudgetOptimizationSignature, setLastBudgetOptimizationSignature] = useState<string | null>(null);

  const normalizedStartState = useMemo(() => normalizeStartState(startState), [startState]);

  const targetOptionIdsInUse = useMemo(() => collectTargetOptionIdsInUse(targetStates), [targetStates]);

  useEffect(() => {
    setTargetGrades((current) => {
      const currentGradeById = new Map(current.map((target) => [target.id, target.grade]));

      return buildTargetGradesForOptions(
        targetOptionIdsInUse,
        new Map([...storedTargetGradeByIdRef.current, ...currentGradeById]),
      );
    });
  }, [targetOptionIdsInUse]);

  useEffect(() => {
    for (const target of targetGrades) {
      storedTargetGradeByIdRef.current.set(target.id, target.grade);
    }

    writeStoredPlannerState({
      targetStates,
      targetGrades: storedTargetGradeByIdRef.current,
      costWeights,
      moduleBudget,
      plannerMode,
    });
  }, [costWeights, moduleBudget, plannerMode, targetGrades, targetStates]);

  const targetGradeById = useMemo(
    () => new Map(targetGrades.map((target) => [target.id, target.grade])),
    [targetGrades],
  );

  const binaryStartState = useMemo(
    () => buildBinaryStartState(normalizedStartState, targetGradeById),
    [normalizedStartState, targetGradeById],
  );

  const policySignature = useMemo(
    () =>
      JSON.stringify({
        targetStates,
        targetGrades,
        iterations,
        costWeights,
      }),
    [costWeights, iterations, targetGrades, targetStates],
  );

  const budgetSignature = useMemo(
    () =>
      JSON.stringify({
        targetStates,
        targetGrades,
        moduleBudget,
      }),
    [moduleBudget, targetGrades, targetStates],
  );

  const needsOptimization = result === null || lastOptimizedPolicySignature !== policySignature;
  const hasStaleResult = result !== null && needsOptimization;
  const hasStaleBudgetOptimization =
    budgetOptimizationResult !== null && lastBudgetOptimizationSignature !== budgetSignature;

  useEffect(() => {
    const worker = new PlannerWorker();
    workerRef.current = worker;

    worker.onmessage = (event: MessageEvent<PlannerWorkerResponse>) => {
      const message = event.data;

      if (message.kind === "optimize-progress") {
        if (message.requestId !== optimizeRequestIdRef.current) {
          return;
        }

        startTransition(() => {
          setOptimizationProgress(message.progress);
        });
        return;
      }

      if (message.kind === "optimize-success") {
        if (message.requestId !== optimizeRequestIdRef.current) {
          return;
        }

        startTransition(() => {
          setResult(message.result);
          setLastOptimizedPolicySignature(optimizeRequestSignatureRef.current);
          setIsRunning(false);
          setOptimizationProgress(null);
        });
        return;
      }

      if (message.kind === "optimize-error") {
        if (message.requestId !== optimizeRequestIdRef.current) {
          return;
        }

        startTransition(() => {
          setError(message.message);
          setIsRunning(false);
          setOptimizationProgress(null);
        });
        return;
      }

      if (message.kind === "simulate-success") {
        if (message.requestId !== simulationRequestIdRef.current) {
          return;
        }

        startTransition(() => {
          setSimulationResult(message.result);
          setLastSimulationSignature(simulationRequestSignatureRef.current);
          setIsSimulationRunning(false);
        });
        return;
      }

      if (message.kind === "budget-optimize-success") {
        if (message.requestId !== budgetRequestIdRef.current) {
          return;
        }

        startTransition(() => {
          setBudgetOptimizationResult(message.result);
          setLastBudgetOptimizationSignature(budgetRequestSignatureRef.current);
          setIsBudgetOptimizationRunning(false);
        });
        return;
      }

      if (message.kind === "budget-optimize-error") {
        if (message.requestId !== budgetRequestIdRef.current) {
          return;
        }

        startTransition(() => {
          setBudgetOptimizationError(message.message);
          setIsBudgetOptimizationRunning(false);
        });
        return;
      }

      if (message.requestId !== simulationRequestIdRef.current) {
        return;
      }

      startTransition(() => {
        setSimulationError(message.message);
        setIsSimulationRunning(false);
      });
    };

    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  const simulationSignature = useMemo(() => {
    if (plannerMode === "classic") {
      if (!result || hasStaleResult || !lastOptimizedPolicySignature) {
        return null;
      }

      return JSON.stringify({
        mode: plannerMode,
        optimizedPolicySignature: lastOptimizedPolicySignature,
        binaryStartState,
        startModuleLocks,
      });
    }

    if (!budgetOptimizationResult || hasStaleBudgetOptimization || !lastBudgetOptimizationSignature) {
      return null;
    }

    return JSON.stringify({
      mode: plannerMode,
      optimizedBudgetSignature: lastBudgetOptimizationSignature,
      binaryStartState,
      startModuleLocks,
    });
  }, [
    binaryStartState,
    budgetOptimizationResult,
    hasStaleBudgetOptimization,
    hasStaleResult,
    lastBudgetOptimizationSignature,
    lastOptimizedPolicySignature,
    plannerMode,
    result,
    startModuleLocks,
  ]);

  const detailedSimulationResult = useMemo(() => {
    if (!simulationSignature || simulationSignature !== lastSimulationSignature) {
      return null;
    }

    return simulationResult;
  }, [lastSimulationSignature, simulationResult, simulationSignature]);

  const currentStateValue = useMemo(
    () => (result && !hasStaleResult ? readStateValue(result, binaryStartState, startModuleLocks) : null),
    [binaryStartState, hasStaleResult, result, startModuleLocks],
  );

  const forcedLockAlternatives = useMemo<OverloadForcedLockAlternative[]>(() => {
    if (!result || hasStaleResult || !currentStateValue || plannerMode !== "classic") {
      return [];
    }

    return readForcedLockAlternatives(
      result,
      buildSimulationStartState(binaryStartState, startModuleLocks),
      targetStates.map(
        (targetState) => targetState.map((optionIndex) => overloadOptions[optionIndex]?.id) as OverloadOptionIds,
      ),
      targetGrades,
      costWeights,
    );
  }, [
    binaryStartState,
    costWeights,
    currentStateValue,
    hasStaleResult,
    plannerMode,
    result,
    startModuleLocks,
    targetStates,
    targetGrades,
  ]);

  const displayedBudgetOptimizationResult = useMemo<OverloadBudgetOptimizationSummary | null>(() => {
    if (!budgetOptimizationResult || hasStaleBudgetOptimization) {
      return null;
    }

    return readOverloadBudgetOptimizationSummary(
      budgetOptimizationResult,
      buildSimulationStartState(binaryStartState, startModuleLocks),
    );
  }, [binaryStartState, budgetOptimizationResult, hasStaleBudgetOptimization, startModuleLocks]);

  const budgetActionAlternatives = useMemo<OverloadBudgetActionAlternative[]>(() => {
    if (!budgetOptimizationResult || hasStaleBudgetOptimization || plannerMode !== "budget") {
      return [];
    }

    return readOverloadBudgetActionAlternatives(
      budgetOptimizationResult,
      buildSimulationStartState(binaryStartState, startModuleLocks),
      targetGrades,
    );
  }, [
    binaryStartState,
    budgetOptimizationResult,
    hasStaleBudgetOptimization,
    plannerMode,
    startModuleLocks,
    targetGrades,
  ]);

  const updateStartStateSlot = (slot: 0 | 1 | 2, value: number) => {
    setStartState((current) => {
      const next = [...current] as StartStateDraft;
      next[slot] = value;
      if (slot === 1 && value === 0) {
        next[4] = 0;
      }
      if (slot === 2 && value === 0) {
        next[5] = 0;
      }
      return normalizeStartState(next);
    });

    if (value === 0) {
      setStartModuleLocks((current) => {
        const next = [...current] as StartModuleLockState;
        next[slot] = false;
        return next;
      });
    }
  };

  const updateStartStateGrade = (slot: 0 | 1 | 2, value: number) => {
    setStartState((current) => {
      const next = [...current] as StartStateDraft;
      next[slot + 3] = value;
      return normalizeStartState(next);
    });
  };

  const updateStartModuleLock = (slot: 0 | 1 | 2, value: boolean) => {
    setStartModuleLocks((current) => {
      const next = [...current] as StartModuleLockState;
      next[slot] = value;
      return next;
    });
  };

  const lockKeysPerModule = Number((costWeights.module / costWeights.lockKey).toFixed(1));

  const updateLockKeysPerModule = (value: number) => {
    const normalizedKeysPerModule = Number.isFinite(value)
      ? Math.min(999, Math.max(1, Math.round(value)))
      : Number((costWeights.module / defaultCostWeights.lockKey).toFixed(1));

    setCostWeights((current) => ({
      ...current,
      lockKey: Number((current.module / normalizedKeysPerModule).toFixed(6)),
    }));
  };

  const updateModuleBudget = (value: number) => {
    const normalizedBudget = Number.isFinite(value)
      ? Math.min(200, Math.max(0, Math.round(value)))
      : DEFAULT_MODULE_BUDGET;
    setModuleBudget(normalizedBudget);
  };

  const updatePlannerMode = (mode: PlannerMode) => {
    setPlannerMode(mode);
  };

  const updateTargetStateSlot = (targetIndex: number, slot: 0 | 1 | 2, value: number) => {
    setTargetStates((current) =>
      current.map((targetState, index) => {
        if (index !== targetIndex) {
          return targetState;
        }

        const next = [...targetState] as TargetStateDraft;
        next[slot] = value;
        return next;
      }),
    );
  };

  const updateTargetGrade = (optionId: string, grade: number) => {
    setTargetGrades((current) => current.map((target) => (target.id === optionId ? { ...target, grade } : target)));
  };

  const addTargetState = () => {
    setTargetStates((current) => [...current, [0, 0, 0]]);
  };

  const clearTargetStates = () => {
    setTargetStates(defaultTargetStates);
  };

  const removeTargetState = (targetIndex: number) => {
    setTargetStates((current) => current.filter((_, index) => index !== targetIndex));
  };

  const addTargetStatePermutations = (targetIndex: number) => {
    setTargetStates((current) => {
      const baseState = current[targetIndex];
      if (!baseState) {
        return current;
      }

      const existingKeys = new Set(current.map(createTargetStateKey));
      const nextStates = [...current];

      for (const permutation of getTargetStatePermutations(baseState)) {
        const key = createTargetStateKey(permutation);
        if (existingKeys.has(key)) {
          continue;
        }

        existingKeys.add(key);
        nextStates.push(permutation);
      }

      return nextStates;
    });
  };

  const runDetailedSimulation = () => {
    if (!simulationSignature || !workerRef.current) {
      return;
    }

    setSimulationError(null);
    setIsSimulationRunning(true);
    simulationRequestIdRef.current += 1;
    simulationRequestSignatureRef.current = simulationSignature;

    if (plannerMode === "classic") {
      if (!result || hasStaleResult) {
        setIsSimulationRunning(false);
        return;
      }

      workerRef.current.postMessage({
        kind: "simulate",
        requestId: simulationRequestIdRef.current,
        startState: buildSimulationStartState(binaryStartState, startModuleLocks),
        result,
        targetGrades,
        costWeights,
      });
      return;
    }

    if (!budgetOptimizationResult || hasStaleBudgetOptimization) {
      setIsSimulationRunning(false);
      return;
    }

    workerRef.current.postMessage({
      kind: "budget-simulate",
      requestId: simulationRequestIdRef.current,
      startState: buildSimulationStartState(binaryStartState, startModuleLocks),
      result: budgetOptimizationResult,
      targetGrades,
      costWeights,
    });
  };

  const runOptimizer = () => {
    setError(null);

    if (!isValidOptionTriple([normalizedStartState[0], normalizedStartState[1], normalizedStartState[2]])) {
      setError("시작 상태의 옵션 조합이 유효하지 않습니다. 중복 옵션은 허용되지 않습니다.");
      return;
    }

    if (targetStates.length === 0) {
      setError("목표 상태를 하나 이상 추가해야 합니다.");
      return;
    }

    for (const [index, targetState] of targetStates.entries()) {
      if (!isValidTargetOptionTriple(targetState)) {
        setError(`${index + 1}번째 목표 상태의 옵션 조합이 유효하지 않습니다.`);
        return;
      }
    }

    if (!workerRef.current) {
      setError("계산 워커를 초기화하지 못했습니다.");
      return;
    }

    setIsRunning(true);
    setOptimizationProgress({
      phase: "policy",
      completedIterations: 0,
      totalIterations: iterations,
      percent: 0,
    });
    optimizeRequestIdRef.current += 1;
    optimizeRequestSignatureRef.current = policySignature;
    workerRef.current.postMessage({
      kind: "optimize",
      requestId: optimizeRequestIdRef.current,
      targetOptionIds: targetStates.map(
        (targetState) => targetState.map((optionIndex) => overloadOptions[optionIndex]?.id) as OverloadOptionIds,
      ),
      targetGrades,
      iterations,
      costWeights,
    });
  };

  const runBudgetOptimization = () => {
    setBudgetOptimizationError(null);

    if (!isValidOptionTriple([normalizedStartState[0], normalizedStartState[1], normalizedStartState[2]])) {
      setBudgetOptimizationError("시작 상태의 옵션 조합이 유효하지 않습니다. 중복 옵션은 허용되지 않습니다.");
      return;
    }

    if (
      !isValidStartModuleLockState(
        [normalizedStartState[0], normalizedStartState[1], normalizedStartState[2]],
        startModuleLocks,
      )
    ) {
      setBudgetOptimizationError("현재 시작 모듈 잠금 조합이 유효하지 않습니다.");
      return;
    }

    if (targetStates.length === 0) {
      setBudgetOptimizationError("목표 상태를 하나 이상 추가해야 합니다.");
      return;
    }

    for (const [index, targetState] of targetStates.entries()) {
      if (!isValidTargetOptionTriple(targetState)) {
        setBudgetOptimizationError(`${index + 1}번째 목표 상태의 옵션 조합이 유효하지 않습니다.`);
        return;
      }
    }

    if (!workerRef.current) {
      setBudgetOptimizationError("계산 워커를 초기화하지 못했습니다.");
      return;
    }

    setIsBudgetOptimizationRunning(true);
    budgetRequestIdRef.current += 1;
    budgetRequestSignatureRef.current = budgetSignature;
    workerRef.current.postMessage({
      kind: "budget-optimize",
      requestId: budgetRequestIdRef.current,
      targetOptionIds: targetStates.map(
        (targetState) => targetState.map((optionIndex) => overloadOptions[optionIndex]?.id) as OverloadOptionIds,
      ),
      targetGrades,
      moduleBudget,
    });
  };

  return {
    startState: normalizedStartState,
    plannerMode,
    startModuleLocks,
    costWeights,
    lockKeysPerModule,
    moduleBudget,
    targetStates,
    targetGrades,
    iterations,
    result,
    error,
    isRunning,
    optimizationProgress,
    needsOptimization,
    hasStaleResult,
    binaryStartState,
    currentStateValue,
    forcedLockAlternatives,
    budgetActionAlternatives,
    budgetOptimizationResult: displayedBudgetOptimizationResult,
    budgetOptimizationError,
    isBudgetOptimizationRunning,
    hasStaleBudgetOptimization,
    detailedSimulationResult,
    simulationError,
    isSimulationRunning,
    updateStartStateSlot,
    updateStartStateGrade,
    updateStartModuleLock,
    updateLockKeysPerModule,
    updateModuleBudget,
    updatePlannerMode,
    updateTargetStateSlot,
    updateTargetGrade,
    addTargetState,
    clearTargetStates,
    addTargetStatePermutations,
    removeTargetState,
    runOptimizer,
    runBudgetOptimization,
    runDetailedSimulation,
  };
}
