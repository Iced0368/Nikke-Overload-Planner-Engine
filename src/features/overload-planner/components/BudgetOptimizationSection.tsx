const MODULE_BUDGET_PRESETS = [20, 50, 100, 150] as const;

type BudgetOptimizationSectionProps = {
  moduleBudget: number;
  onModuleBudgetChange: (value: number) => void;
};

export function BudgetOptimizationSection({ moduleBudget, onModuleBudgetChange }: BudgetOptimizationSectionProps) {
  return (
    <div className="section-block">
      <div className="section-title-row">
        <h3>예산 내 달성 확률</h3>
        <span className="section-caption">입력한 모듈 개수 안에서 목표 달성 확률이 가장 높은 정책을 계산합니다.</span>
      </div>

      <div className="cost-weight-card budget-config-card">
        <div className="cost-weight-grid budget-config-grid">
          <label className="cost-weight-slider-block">
            <span className="cost-weight-label-row">
              <span className="ingame-field-label">모듈 예산</span>
              <span className="start-lock-tooltip-shell cost-weight-tooltip-shell">
                <button type="button" className="start-lock-info-button" aria-label="예산 최적화 설명 보기">
                  i
                </button>
                <span className="start-lock-tooltip" role="tooltip">
                  락키는 직접 제한하지 않고, 같은 달성 확률이면 기대 락키 소모가 더 적은 정책을 우선합니다.
                </span>
              </span>
            </span>
            <input
              className="weight-slider"
              type="range"
              min={0}
              max={200}
              step={1}
              value={moduleBudget}
              onChange={(event) => onModuleBudgetChange(Number(event.target.value))}
            />
          </label>

          <label className="cost-weight-number-block">
            <span className="ingame-field-label">직접 입력</span>
            <input
              type="number"
              min={0}
              max={200}
              step={1}
              value={moduleBudget}
              onChange={(event) => onModuleBudgetChange(Number(event.target.value))}
            />
          </label>
        </div>

        <div className="cost-weight-presets" role="group" aria-label="모듈 예산 빠른 선택">
          {MODULE_BUDGET_PRESETS.map((preset) => (
            <button
              key={preset}
              type="button"
              className={moduleBudget === preset ? "toggle-chip active" : "toggle-chip"}
              onClick={() => onModuleBudgetChange(preset)}
            >
              {preset}개
            </button>
          ))}
        </div>

        <div className="budget-config-footer">
          <div className="metric-card budget-config-metric">
            <span className="metric-label">현재 입력 예산</span>
            <strong>{moduleBudget.toLocaleString()} 모듈</strong>
          </div>
          <p className="section-caption budget-config-caption">상단 실행 버튼으로 현재 모드 계산을 시작합니다.</p>
        </div>
      </div>
    </div>
  );
}
