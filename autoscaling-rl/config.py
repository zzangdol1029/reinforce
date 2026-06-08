"""실험 공통 설정값.

하이퍼파라미터 권장값과 그 근거는 각 항목 주석 참고.
v1(공식 기반)과 v2(이산사건 기반) 환경 설정을 모두 포함한다.
"""

# ---------------------------------------------------------------- 환경 공통
SLA_LATENCY = 0.2          # v1 평균지연 SLA (초)
EPISODE_STEPS = 288        # 1 에피소드 step 수

# ------------------------------------------------- v1 Stage 1: 인스턴스 (공식 기반)
CONTAINER = dict(
    c_min=1, c_max=20, c_init=4,
    c_base=4,              # 기본 확보 인스턴스 — 이하로는 비용 0, 초과분만 과금
    w_under=0.2,           # 기본값 미만으로 줄일 때 페널티 (안전 여유 훼손 방지)
    mu=20.0,               # 인스턴스 1개 처리율 (req/s)
    cold_start_steps=2,    # scale-out 후 활성화까지 step 수
    lam_base=80.0, lam_max=400.0,
    w_sla=1.0, w_cost=0.35, w_thrash=0.02,
)

# ------------------------------------------------- v1 Stage 2: thread (공식 기반)
THREADPOOL = dict(
    c_min=4, c_max=64, c_init=16, c_step=4,
    c_base=16,             # 기본 worker thread 수
    w_under=0.2,
    mu_thread=25.0,
    c_knee=32, alpha_cs=2.0,   # 비단조 용량: knee 초과 시 오버헤드
    lam_base=250.0, lam_max=900.0,
    w_sla=1.0, w_cost=0.30, w_thrash=0.02,
)

# ------------------------------------------------- v2: 이산사건 (p95 SLA 기반)
EVENT_CONTAINER = dict(
    c_min=1, c_max=20, c_init=4,
    c_base=4,              # 기본 확보 인스턴스 — 이하로는 비용 0, 초과분만 과금
    w_under=0.2,           # 기본값 미만으로 줄일 때 페널티 (안전 여유 훼손 방지)
    workers_per_instance=2,    # 인스턴스당 동시 처리 슬롯
    cold_start=20.0,           # 초 (= 윈도우 2개)
    window=10.0,               # 결정 주기 (초)
    lam_base=30.0, lam_max=160.0,
    service_mean=0.1,          # 평균 처리시간 100ms
    sigma_range=(0.6, 1.0),    # heavy-tail 강도 — 에피소드마다 랜덤 (과적합 방지)
    sla=0.4,                   # p95 SLA 400ms
    w_sla=1.0, w_cost=0.35, w_thrash=0.02,
)

EVENT_THREADPOOL = dict(
    c_min=4, c_max=64, c_init=16, c_step=4,
    c_base=16,             # 기본 worker thread 수
    w_under=0.2,
    c_knee=32, alpha_cs=2.0,
    window=5.0,
    lam_base=15.0, lam_max=100.0,
    service_mean=0.3,          # 평균 처리시간 300ms -> thread당 약 3.3 req/s
    sigma_range=(0.6, 1.0),
    sla=1.0,                   # p95 SLA 1.0s
    w_sla=1.0, w_cost=0.30, w_thrash=0.02,
)

# ---------------------------------------------------------------- DQN 권장값
# 근거:
#  - hidden 128 : v2는 state 8차원 + tail 분포 학습 필요 → 64보다 여유
#  - lr 5e-4    : 1e-3은 v1에서 후반 진동 관찰 → 절반으로 안정화
#  - batch 128  : 보상 분산(버스트 난이도 차)이 커서 큰 배치가 유리
#  - buffer 100k: 드문 '위기 상태(backlog 폭증)' 경험을 오래 보존
#  - eps decay 0.5 : 전체의 절반까지 탐험 유지
DQN = dict(
    hidden=128,
    lr=5e-4,
    gamma=0.99,
    buffer_size=100_000,
    batch_size=128,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_frac=0.5,
    target_update=1_000,   # hard sync 주기 (step)
    learn_start=2_000,
)

# ---------------------------------------------------------------- PPO 권장값
# 근거:
#  - rollout 4096 : 에피소드(288 step) 14개 분량 — 다양한 트래픽 난이도가
#                   한 배치에 섞여 gradient 분산 감소 (v1 2048 대비 안정)
#  - epochs 10 / minibatch 256 : 표준 설정, ratio 폭주 없이 충분한 재사용
#  - ent_coef 0.01 : 조기 결정적 수렴(탐험 소실) 방지
#  - clip 0.2     : PPO 논문 기본값, 행동 3개의 좁은 정책엔 충분
PPO = dict(
    hidden=128,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    rollout_steps=4_096,
    epochs=10,
    minibatch=256,
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
)

# ---------------------------------------------------------------- 학습/평가
# 과적합 방지의 핵심: 시드(=트래픽 패턴) 완전 분리
#   train      : seed*10_000 + ep        (에피소드마다 새 패턴)
#   validation : VAL_SEED + i            (학습 중 best 체크포인트 선택용)
#   test       : EVAL_SEED + i           (최종 보고용 — 학습/검증에 미사용)
TRAIN_EPISODES = 300       # v1 기준. v2는 시뮬레이션이 무거워 200 권장
EVAL_EPISODES = 20         # 최종 평가 에피소드 수
EVAL_SEED = 7_777
VAL_SEED = 5_555_555
VAL_EPISODES = 3           # 검증 1회당 에피소드 수
VAL_INTERVAL = 10          # 몇 에피소드마다 검증할지
