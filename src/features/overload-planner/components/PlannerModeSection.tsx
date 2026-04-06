export type PlannerMode = "classic" | "budget";

type PlannerModeSectionProps = {
  mode: PlannerMode;
  onModeChange: (mode: PlannerMode) => void;
};

const MODE_OPTIONS: Array<{ mode: PlannerMode; label: string; description: string }> = [
  {
    mode: "classic",
    label: "평균 소모 최적화",
    description: "기대 소모량을 최소화하는 정책을 계산합니다.",
  },
  {
    mode: "budget",
    label: "예산 내 확률 최적화",
    description: "모듈 예산 안에서 목표 달성 확률이 가장 높은 정책을 계산합니다.",
  },
];

export function PlannerModeSection({ mode, onModeChange }: PlannerModeSectionProps) {
  const activeOption = MODE_OPTIONS.find((option) => option.mode === mode) ?? MODE_OPTIONS[0]!;

  return (
    <div className="section-block">
      <div className="section-title-row">
        <h3>최적화 모드</h3>
        <span className="section-caption">계산 기준을 바꾸면 입력 패널과 결과 패널이 함께 전환됩니다.</span>
      </div>

      <div className="mode-selector-group" role="radiogroup" aria-label="최적화 모드 선택">
        {MODE_OPTIONS.map((option) => (
          <button
            key={option.mode}
            type="button"
            role="radio"
            aria-checked={mode === option.mode}
            className={mode === option.mode ? "mode-selector-pill is-active" : "mode-selector-pill"}
            onClick={() => onModeChange(option.mode)}
          >
            <span>{option.label}</span>
          </button>
        ))}
      </div>

      <p className="mode-selector-caption">{activeOption.description}</p>
    </div>
  );
}
