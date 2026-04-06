import "./App.css";
import "./features/overload-planner/styles/overloadPlanner.css";
import {
  BudgetOptimizationSection,
  CostWeightsSection,
  PlannerHeader,
  PlannerModeSection,
  PlannerResultPanel,
  StartStateSection,
  TargetGradesSection,
  TargetStatesSection,
  useOverloadPlanner,
} from "./features/overload-planner/index.ts";

function App() {
  const {
    plannerMode,
    startState,
    startModuleLocks,
    lockKeysPerModule,
    targetStates,
    targetGrades,
    result,
    error,
    isRunning,
    optimizationProgress,
    needsOptimization,
    hasStaleResult,
    binaryStartState,
    currentStateValue,
    forcedLockAlternatives,
    detailedSimulationResult,
    simulationError,
    isSimulationRunning,
    moduleBudget,
    budgetOptimizationResult,
    budgetOptimizationError,
    isBudgetOptimizationRunning,
    hasStaleBudgetOptimization,
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
  } = useOverloadPlanner();

  const isOptimizeButtonDisabled = isRunning || !needsOptimization;
  const optimizeButtonLabel = isRunning ? "계산 중..." : needsOptimization ? "최적화 실행" : "최적화 완료됨";
  const optimizeButtonStateClass = isRunning ? "is-running" : needsOptimization ? "is-ready" : "is-complete";
  const needsBudgetOptimization = hasStaleBudgetOptimization || !budgetOptimizationResult;
  const isBudgetButtonDisabled = isBudgetOptimizationRunning || isRunning || !needsBudgetOptimization;
  const isClassicMode = plannerMode === "classic";
  const activeRunHandler = isClassicMode ? runOptimizer : runBudgetOptimization;
  const activeRunDisabled = isClassicMode ? isOptimizeButtonDisabled : isBudgetButtonDisabled;
  const activeRunLabel = isClassicMode
    ? optimizeButtonLabel
    : isBudgetOptimizationRunning
      ? "계산 중..."
      : needsBudgetOptimization
        ? "예산 기반 실행"
        : "예산 계산 완료됨";
  const activeRunStateClass = isClassicMode
    ? optimizeButtonStateClass
    : isBudgetOptimizationRunning
      ? "is-running"
      : needsBudgetOptimization
        ? "is-ready"
        : "is-complete";
  const activeError = isClassicMode ? error : null;

  return (
    <main className="app-shell">
      <PlannerHeader />

      <section className="workspace-grid">
        <div className="panel form-panel">
          <div className="panel-header">
            <h2>입력 설정</h2>
            <button
              className={`primary-button ${activeRunStateClass}`}
              onClick={activeRunHandler}
              disabled={activeRunDisabled}
            >
              <span className="button-content">
                <span className="button-icon" aria-hidden="true">
                  <svg viewBox="0 0 20 20" focusable="false">
                    {isClassicMode ? (
                      isRunning ? (
                        <path d="M10 3.25a6.75 6.75 0 1 1-4.773 1.977" />
                      ) : needsOptimization ? (
                        <>
                          <path d="M4.25 3.75v12.5" />
                          <path d="M7.25 5.25 15.75 10l-8.5 4.75Z" />
                        </>
                      ) : (
                        <path d="M4.5 10.25 8 13.75l7.5-7.5" />
                      )
                    ) : isBudgetOptimizationRunning ? (
                      <path d="M10 3.25a6.75 6.75 0 1 1-4.773 1.977" />
                    ) : hasStaleBudgetOptimization || !budgetOptimizationResult ? (
                      <>
                        <path d="M4.25 3.75v12.5" />
                        <path d="M7.25 5.25 15.75 10l-8.5 4.75Z" />
                      </>
                    ) : (
                      <path d="M4.5 10.25 8 13.75l7.5-7.5" />
                    )}
                  </svg>
                </span>
                <span>{activeRunLabel}</span>
              </span>
            </button>
          </div>

          <PlannerModeSection mode={plannerMode} onModeChange={updatePlannerMode} />

          <StartStateSection
            startState={startState}
            binaryStartState={binaryStartState}
            startModuleLocks={startModuleLocks}
            onSlotChange={updateStartStateSlot}
            onGradeChange={updateStartStateGrade}
            onStartModuleLockChange={updateStartModuleLock}
          />

          {isClassicMode ? (
            <CostWeightsSection
              lockKeysPerModule={lockKeysPerModule}
              onLockKeysPerModuleChange={updateLockKeysPerModule}
            />
          ) : (
            <BudgetOptimizationSection moduleBudget={moduleBudget} onModuleBudgetChange={updateModuleBudget} />
          )}

          <TargetStatesSection
            targetStates={targetStates}
            onAdd={addTargetState}
            onClear={clearTargetStates}
            onAddPermutations={addTargetStatePermutations}
            onRemove={removeTargetState}
            onSlotChange={updateTargetStateSlot}
          />

          {activeError ? <div className="error-banner">{activeError}</div> : null}
        </div>

        <div className="side-column">
          <PlannerResultPanel
            mode={plannerMode}
            result={result}
            isStale={hasStaleResult}
            isOptimizing={isRunning}
            optimizationProgress={optimizationProgress}
            currentStateValue={currentStateValue}
            forcedLockAlternatives={forcedLockAlternatives}
            moduleBudget={moduleBudget}
            budgetOptimizationResult={budgetOptimizationResult}
            budgetOptimizationError={budgetOptimizationError}
            isBudgetOptimizing={isBudgetOptimizationRunning}
            isBudgetStale={hasStaleBudgetOptimization}
            detailedSimulationResult={detailedSimulationResult}
            simulationError={simulationError}
            isSimulationRunning={isSimulationRunning}
            onRunDetailedSimulation={runDetailedSimulation}
            targetGrades={targetGrades}
          />
          <div className="panel side-grade-panel">
            <TargetGradesSection targetGrades={targetGrades} onTargetGradeChange={updateTargetGrade} />
          </div>
        </div>
      </section>
    </main>
  );
}

export default App;
