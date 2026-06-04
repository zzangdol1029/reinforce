"""공통 환경 설정. 학습/평가가 동일한 시나리오를 쓰도록 한 곳에서 관리."""

ENV_KWARGS = dict(
    n_instances=4,
    arrival_rate=9.0,       # 평균 총 처리율(1.5+2.17+2.83+3.5=10)에 근접 -> 혼잡 발생
    mean_demand=1.0,
    sla_threshold=1.5,
    episode_len=500,
    lambda_balance=0.1,
    sla_penalty=2.0,
)
