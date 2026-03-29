const LOCK_KEYS_PER_MODULE_PRESETS = [10, 20, 50, 100] as const;

type CostWeightsSectionProps = {
  lockKeysPerModule: number;
  onLockKeysPerModuleChange: (value: number) => void;
};

function formatModuleEquivalent(value: number) {
  return `${value.toFixed(2)} 모듈`;
}

export function CostWeightsSection({ lockKeysPerModule, onLockKeysPerModuleChange }: CostWeightsSectionProps) {
  return (
    <div className="section-block">
      <div className="section-title-row">
        <h3>비용 가치</h3>
        <span className="section-caption">모듈 1개를 락키 몇 개로 볼지 정해 내부 가중치로 변환합니다.</span>
      </div>

      <div className="cost-weight-card">
        <div className="cost-weight-grid">
          <label className="cost-weight-slider-block">
            <span className="cost-weight-label-row">
              <span className="ingame-field-label">모듈 1개 = 락키 몇 개</span>
              <span className="start-lock-tooltip-shell cost-weight-tooltip-shell">
                <button type="button" className="start-lock-info-button" aria-label="비용 가치 설명 보기">
                  i
                </button>
                <span className="start-lock-tooltip" role="tooltip">
                  값이 작을수록 락키를 더 아끼는 방향으로 계산하고, 값이 클수록 락키를 더 적극적으로 사용하는 방향으로
                  계산합니다.
                </span>
              </span>
            </span>
            <input
              className="weight-slider"
              type="range"
              min={1}
              max={200}
              step={1}
              value={lockKeysPerModule}
              onChange={(event) => onLockKeysPerModuleChange(Number(event.target.value))}
            />
          </label>

          <label className="cost-weight-number-block">
            <span className="ingame-field-label">직접 입력</span>
            <input
              type="number"
              min={1}
              max={999}
              step={1}
              value={lockKeysPerModule}
              onChange={(event) => onLockKeysPerModuleChange(Number(event.target.value))}
            />
          </label>
        </div>

        <div className="cost-weight-presets" role="group" aria-label="모듈 대비 락키 가치 빠른 선택">
          {LOCK_KEYS_PER_MODULE_PRESETS.map((preset) => (
            <button
              key={preset}
              type="button"
              className={lockKeysPerModule === preset ? "toggle-chip active" : "toggle-chip"}
              onClick={() => onLockKeysPerModuleChange(preset)}
            >
              {preset}개
            </button>
          ))}
        </div>

        <div className="cost-weight-summary">
          <div className="metric-card cost-weight-metric">
            <span className="metric-label">락키 20개 환산</span>
            <strong>{formatModuleEquivalent(20 / lockKeysPerModule)}</strong>
          </div>
          <div className="metric-card cost-weight-metric">
            <span className="metric-label">락키 30개 환산</span>
            <strong>{formatModuleEquivalent(30 / lockKeysPerModule)}</strong>
          </div>
        </div>
      </div>
    </div>
  );
}
