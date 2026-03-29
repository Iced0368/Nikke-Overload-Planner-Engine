export function PlannerHeader() {
  return (
    <section className="hero-panel">
      <div className="hero-copy">
        <p className="eyebrow">NOPE • Nikke Overload Planner Engine</p>
        <h1>오버로드 최적화 플래너</h1>
        <p className="hero-note">현재 상태 기준 기대 소모량과 권장 리롤 행동을 빠르게 확인합니다.</p>
      </div>
      <div className="hero-meta-panel" aria-label="버전 및 마지막 업데이트 정보">
        <div className="hero-meta">
          <span>버전 {__APP_VERSION__}</span>
          <span>마지막 업데이트 {__LAST_UPDATED_AT__}</span>
        </div>
      </div>
    </section>
  );
}
