"""
통합 admission control 설정 (게이트웨이 1 → 역할별 WAS N대 → 공유 DB 1개)
=====================================================================
- 각 라우트(역할)는 자기 WAS 자원으로 처리 (서로 독립)
- 모든 라우트의 요청은 하나의 공유 DB를 사용 → 여기서 결합(coupling) 발생
- 공유 DB 용량 C_db(t)는 비노출(시변): 락 경합/백업/열화로 떨어짐
"""

# 라우트별(역할별) 설정 -- 실제 시스템 구성에 맞게 숫자만 바꾸면 됨
ROUTES = [
    # name      s(서비스시간) was_cap  db_cost  sla    priority  base_rps peak_rps
    dict(name="search",  s=0.05, was_cap=300, db_cost=1.0, sla=0.15, priority=1.0, base_rps=140, peak_rps=380),
    dict(name="payment", s=0.15, was_cap=110, db_cost=3.0, sla=0.40, priority=3.0, base_rps=55,  peak_rps=170),
    dict(name="report",  s=0.50, was_cap=40,  db_cost=5.0, sla=2.00, priority=0.4, base_rps=18,  peak_rps=60),
]

# 공유 DB (단위: DB 처리 work/s). cost 가중 수요 합이 cap을 넘으면 전 라우트 지연 폭증
DB = dict(cap_healthy=520.0, cap_deg_min=160.0, cap_deg_max=260.0, degrade_prob=0.05, recover_prob=0.20)

S_DB_BASE = 0.03     # DB 기본 처리시간(정상시 라우트 지연에 더해짐)
EPISODE_STEPS = 300
MIN_L = 2
MAX_L = 200
W_REJECT = 0.5       # 거절 벌점
W_SLA = 10.0         # SLA 위반 벌점 (>> w_reject : 지연붕괴가 거절보다 훨씬 나쁨)

ENV_KWARGS = dict(routes=ROUTES, db=DB, s_db_base=S_DB_BASE, episode_steps=EPISODE_STEPS,
                  min_L=MIN_L, max_L=MAX_L, w_reject=W_REJECT, w_sla=W_SLA)
