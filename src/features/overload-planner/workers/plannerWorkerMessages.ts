import { type MonteCarloSimulationSummary } from "../../../lib/overloadMonteCarlo.ts";
import { type OverloadBudgetOptimizationResult } from "../../../lib/overloadBudgetOptimizer.ts";
import {
  type OverloadCostWeights,
  type OverloadOptionIds,
  type OverloadOptionTarget,
} from "../../../lib/overloadOptions";
import {
  type OverloadOptimizationProgress,
  type OverloadPolicyOptimizationResult,
  type OverloadState,
} from "../../../lib/overloadPolicyOptimizer.ts";

export type OptimizeWorkerRequest = {
  kind: "optimize";
  requestId: number;
  targetOptionIds: OverloadOptionIds[];
  targetGrades: OverloadOptionTarget[];
  iterations: number;
  costWeights: OverloadCostWeights;
};

export type SimulateWorkerRequest = {
  kind: "simulate";
  requestId: number;
  startState: OverloadState;
  result: OverloadPolicyOptimizationResult;
  targetGrades: OverloadOptionTarget[];
  costWeights: OverloadCostWeights;
};

export type BudgetSimulateWorkerRequest = {
  kind: "budget-simulate";
  requestId: number;
  startState: OverloadState;
  result: OverloadBudgetOptimizationResult;
  targetGrades: OverloadOptionTarget[];
  costWeights: OverloadCostWeights;
};

export type BudgetOptimizeWorkerRequest = {
  kind: "budget-optimize";
  requestId: number;
  targetOptionIds: OverloadOptionIds[];
  targetGrades: OverloadOptionTarget[];
  moduleBudget: number;
};

export type PlannerWorkerRequest =
  | OptimizeWorkerRequest
  | SimulateWorkerRequest
  | BudgetOptimizeWorkerRequest
  | BudgetSimulateWorkerRequest;

export type OptimizeProgressWorkerResponse = {
  kind: "optimize-progress";
  requestId: number;
  progress: OverloadOptimizationProgress;
};

export type OptimizeSuccessWorkerResponse = {
  kind: "optimize-success";
  requestId: number;
  result: OverloadPolicyOptimizationResult;
};

export type OptimizeErrorWorkerResponse = {
  kind: "optimize-error";
  requestId: number;
  message: string;
};

export type SimulateSuccessWorkerResponse = {
  kind: "simulate-success";
  requestId: number;
  result: MonteCarloSimulationSummary;
};

export type SimulateErrorWorkerResponse = {
  kind: "simulate-error";
  requestId: number;
  message: string;
};

export type BudgetOptimizeSuccessWorkerResponse = {
  kind: "budget-optimize-success";
  requestId: number;
  result: OverloadBudgetOptimizationResult;
};

export type BudgetOptimizeErrorWorkerResponse = {
  kind: "budget-optimize-error";
  requestId: number;
  message: string;
};

export type PlannerWorkerResponse =
  | OptimizeProgressWorkerResponse
  | OptimizeSuccessWorkerResponse
  | OptimizeErrorWorkerResponse
  | SimulateSuccessWorkerResponse
  | SimulateErrorWorkerResponse
  | BudgetOptimizeSuccessWorkerResponse
  | BudgetOptimizeErrorWorkerResponse;
