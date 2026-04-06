/// <reference lib="webworker" />

import { optimizeOverloadBudgetSuccessWithRuntimeWasm } from "../../../lib/overloadBudgetOptimizerWasm.ts";
import { runMonteCarloBudgetSimulation, runMonteCarloPolicySimulation } from "../../../lib/overloadMonteCarlo.ts";
import { optimizeOverloadPolicyWithRuntimeWasm } from "../../../lib/overloadPolicyOptimizerWasm.ts";
import { type PlannerWorkerRequest, type PlannerWorkerResponse } from "./plannerWorkerMessages";

const workerScope: DedicatedWorkerGlobalScope = self as DedicatedWorkerGlobalScope;

function postMessage(message: PlannerWorkerResponse) {
  workerScope.postMessage(message);
}

function getErrorMessage(caughtError: unknown) {
  return caughtError instanceof Error ? caughtError.message : "알 수 없는 오류가 발생했습니다.";
}

workerScope.onmessage = async (event: MessageEvent<PlannerWorkerRequest>) => {
  const message = event.data;

  if (message.kind === "optimize") {
    try {
      const result = await optimizeOverloadPolicyWithRuntimeWasm(
        message.targetOptionIds,
        message.targetGrades,
        message.iterations,
        message.costWeights,
        {
          onProgress: (progress) => {
            postMessage({
              kind: "optimize-progress",
              requestId: message.requestId,
              progress,
            });
          },
        },
      );

      postMessage({
        kind: "optimize-success",
        requestId: message.requestId,
        result,
      });
    } catch (caughtError) {
      postMessage({
        kind: "optimize-error",
        requestId: message.requestId,
        message: getErrorMessage(caughtError),
      });
    }
    return;
  }

  if (message.kind === "budget-optimize") {
    try {
      const result = await optimizeOverloadBudgetSuccessWithRuntimeWasm(
        message.targetOptionIds,
        message.targetGrades,
        message.moduleBudget,
      );

      postMessage({
        kind: "budget-optimize-success",
        requestId: message.requestId,
        result,
      });
    } catch (caughtError) {
      postMessage({
        kind: "budget-optimize-error",
        requestId: message.requestId,
        message: getErrorMessage(caughtError),
      });
    }
    return;
  }

  if (message.kind === "budget-simulate") {
    try {
      const result = runMonteCarloBudgetSimulation(message.startState, message.result, message.targetGrades, {
        costWeights: message.costWeights,
      });

      postMessage({
        kind: "simulate-success",
        requestId: message.requestId,
        result,
      });
    } catch (caughtError) {
      postMessage({
        kind: "simulate-error",
        requestId: message.requestId,
        message: getErrorMessage(caughtError),
      });
    }
    return;
  }

  try {
    const result = runMonteCarloPolicySimulation(message.startState, message.result, message.targetGrades, {
      costWeights: message.costWeights,
    });

    postMessage({
      kind: "simulate-success",
      requestId: message.requestId,
      result,
    });
  } catch (caughtError) {
    postMessage({
      kind: "simulate-error",
      requestId: message.requestId,
      message: getErrorMessage(caughtError),
    });
  }
};

export {};
