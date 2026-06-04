# portal-pool-tuning 전체 구조 설명

강화학습으로 포털 WAS의 워커 스레드(W)와 DB 커넥션풀(D)을 동시에 자동 최적화하는 프로젝트.

---

## 1. 프로젝트 목적

포털 WAS는 Undertow 워커 스레드 수와 HikariCP DB 커넥션풀 크기를 정적으로 설정해 운영한다.
부하와 요청 패턴이 변해도 두 값이 고정되어 있어 아래와 같은 문제가 반복된다.

| 포털 | 현재 문제 |
|------|-----------|
| 연구 | 워커 2048개 대부분 과다(메모리 낭비), DB-heavy 구간에서 DB풀 150 부족 |
| 사용자 | 16코어 CPU 병목, 워커·DB풀 조절만으로는 한계 존재 |
| 관리 | 워커 512개 완전 과다, 실제 필요량은 ~37개 수준 |

RL 에이전트는 실시간 운영 메트릭을 관측하고 두 값을 매 제어 주기마다 동시에 조절하는 방법을 학습한다.

---

## 2. 파일 구성

```
portal-pool-tuning/
├── config.py               포털별 실사양 데이터 (하드웨어·부하·비용 계수)
├── envs/
│   └── portal_env.py       Gymnasium 환경 — 3중 병목 시뮬레이터 (핵심)
├── baselines.py            비교용 규칙 기반 정책 5종 + 평가 함수
├── train.py                DQN/PPO/SAC 학습 실행 + 학습 곡선 저장
├── optimize.py             Optuna 베이지안 HP 자동 탐색
├── evaluate.py             RL vs 베이스라인 성능 비교·시각화
├── requirements.txt        패키지 의존성
├── ARCHITECTURE.md         이 파일 — 전체 구조 설명
└── DGX_SERVER_GUIDE.md     DGX 서버 실행 가이드
```

---

## 3. 파일 의존 관계

```
config.py
    │  포털 사양 딕셔너리 제공
    ▼
envs/portal_env.py          ← 시뮬레이터 핵심
    │
    ├──▶ baselines.py        규칙 기반 정책 5종
    │         │
    │         └──▶ evaluate.py
    │
    ├──▶ train.py            SB3 학습
    │         │
    │         ▼
    │    models/<portal>_<algo>.zip
    │         │
    │         └──▶ evaluate.py
    │
    ├──▶ optimize.py         Optuna HP 탐색
    │         │
    │         ▼
    │    results/optuna_*.db / *.png / *.txt
    │
    └──▶ evaluate.py         최종 비교·시각화
              │
              ▼
         results/comparison_*.png / timeseries_*.png
```

---

## 4. 강화학습 문제 정의 (MDP)

### State — 10차원 관측 벡터

| 인덱스 | 값 | 의미 |
|--------|-----|------|
| [0] | lam / peak_rps | 현재 부하 비율 (0=야간, 1=피크) |
| [1] | Δlam / peak_rps | 부하 증가 추세 (양수=증가 중) |
| [2] | cpu_util | CPU 이용률 0~1 |
| [3] | worker_util | **워커풀 이용률** (1이면 워커 병목) |
| [4] | db_util | **DB풀 이용률** (1이면 DB 병목) |
| [5] | tanh(lat/sla) | 현재 지연 수준 |
| [6] | Δtanh(lat/sla) | 지연 추세 (양수=악화 중) |
| [7] | W / max_w | 현재 워커 비율 |
| [8] | D / max_d | 현재 DB풀 비율 |
| [9] | drop_rate | 직전 드롭율 |

> **부분 관측(POMDP)**: 요청 체류시간(mean_s)과 DB 비중(db_frac)은 관측 불가.
> 에이전트는 워커/DB 이용률([3][4])로 어느 자원이 병목인지 간접 추론해야 한다.

### Action — 이산 25가지 / 연속 2차원

```
이산 (DQN/PPO): Discrete(25) = ΔW 5단계 × ΔD 5단계
  단계: -8% / -2% / 0% / +2% / +8%  (max 기준 비율)
  예) max_workers=2560, 단계 +8% → +205개

연속 (SAC): Box(-1, 1, shape=(2,))
  action[0] → ΔW = action[0] × 0.08 × max_w
  action[1] → ΔD = action[1] × 0.08 × max_d
```

### Reward — 5개 항목의 합

```
R = -sla_weight  × max(0, 지연 - sla_threshold)   SLA 위반 페널티
  - drop_penalty × drop_rate                       요청 드롭 페널티
  - mem_cost     × W                               워커 유지 비용
  - db_cost      × D                               DB풀 유지 비용
  - thrash       × (|ΔW|/max_w + |ΔD|/max_d)      자원 변동 페널티
```

| 항목 | 역할 |
|------|------|
| SLA 페널티 | 응답 지연이 목표 초과 시 강하게 억제 |
| 드롭 페널티 | 처리 못한 요청 발생 시 강력 억제 |
| 자원 비용 | 불필요한 과다 할당 억제 (메모리·DB 절약 유도) |
| 변동 페널티 | 매 스텝 큰 폭 변경 억제 (운영 안정성) |

---

## 5. 핵심 모델: 3중 병목 (portal_env.py)

```
요청 한 건의 생애주기:
  [워커 스레드 점유] ─────────────────── 전체 체류시간(mean_s)
                     └── [DB 커넥션 점유] DB 처리시간(mean_s × db_frac)

처리능력 = min(cpu_cap, worker_cap, db_cap)
  cpu_cap    = cores / cpu_처리시간
  worker_cap = W / 유효_체류시간
  db_cap     = D / 유효_DB처리시간          ← 공유 DB 경합 반영
```

**공유 DB 과부하 효과**:
```
db_t_eff = db_t × (1 + db_contention × D / db_shared_max)
```
D를 늘릴수록 공유 DB가 느려져 효과가 상쇄된다. 이것이 D를 무작정 최대로 올리면 안 되는 이유다.

**응답 지연 계산 (M/M/1 대기열 근사)**:
```
rho = lam / capacity        (이용률)

rho < 1: latency = mean_s_eff / (1 - rho)   ← 정상, 부하 증가 시 지연 증가
rho ≥ 1: latency = sla × 3                 ← 과부하, 드롭 발생
```

---

## 6. 부하 시계열 생성 (`_build_traces`)

에피소드마다 3개의 독립 시계열이 생성된다.

```
lam (RPS)   : sin 일간 패턴 (base→peak) + 노이즈 4%   ← 에이전트 관측 가능
mean_s      : 독립 sin + 랜덤 위상 + 노이즈            ← 관측 불가 (숨겨진 상태)
db_frac     : 독립 sin + 랜덤 위상 + 노이즈            ← 관측 불가 (숨겨진 상태)
```

랜덤 위상으로 에피소드마다 다른 패턴이 생성되어 에이전트의 일반화를 강제한다.
mean_s와 db_frac의 변동이 '어느 자원이 병목인가'를 결정하므로,
에이전트는 이용률([3][4])을 통해 간접 추론해야 한다.

---

## 7. 비교 기준 정책 (baselines.py)

```
성능 (이론적 순서):
StaticLow < StaticCurrent < HillClimb < [RL 목표] < Oracle
  (최소)      (현재 운영)    (반응형)                (미래 알고 선행)
```

| 정책 | 방식 | 핵심 한계 |
|------|------|-----------|
| StaticCurrent | 현재 운영값 고정 | 부하 변화 완전 무시 |
| StaticMax | 최대값 고정 | SLA는 안전하지만 자원 낭비 극심 |
| StaticLow | 최솟값 고정 | 피크·DB-heavy 구간 병목 |
| HillClimb | 이용률 보고 즉시 반응 | 한 스텝 항상 뒤처짐, 추세 예측 불가 |
| Oracle | 미래 H스텝 알고 선행 조절 | 실운영 불가능, 상한 참조용 |

> RL이 HillClimb를 이기고 Oracle에 가까울수록 학습 성공.

---

## 8. 학습 파이프라인 (train.py)

```
train_one(algo, portal, timesteps)
    │
    ├─ 환경 생성: PortalPoolEnv → Monitor 래핑 (에피소드 통계 기록)
    ├─ 모델 생성: DQN / PPO / SAC (SB3)
    ├─ 콜백 등록
    │    ├─ ProgressBarCallback: tqdm 진행 바 (스텝 진행률·남은 시간)
    │    └─ EpisodeLogCallback:  보상 콘솔 출력 + 학습 곡선 그래프 저장
    └─ model.learn(total_timesteps)
         └─ models/<portal>_<algo>.zip 저장
```

### 알고리즘 비교

| 알고리즘 | 행동공간 | 특징 |
|----------|----------|------|
| DQN | Discrete(25) | 경험 리플레이 + 타겟 네트워크, off-policy |
| PPO | Discrete(25) | 클리핑으로 안정적 업데이트, on-policy |
| SAC | Box(2) 연속 | 최대 엔트로피, 연속 행동, off-policy |

### 학습 곡선 분석 지표

| 표시 | 의미 | 해석 |
|------|------|------|
| ★ 금색 | 이동평균 최고점 | 이 시점 모델이 최선 |
| ▼ 초록 | 수렴 시작 | 이후 학습해도 개선 없음 |
| 빨간 음영 | 과적합 구간 | 최고점 대비 5% 이상 하락 |

---

## 9. HP 탐색 파이프라인 (optimize.py)

```
run_optimize(portal, algo, n_trials, timesteps)
    │
    ├─ optuna.create_study(TPESampler, MedianPruner)
    │
    └─ trial 반복:
         sample_*_params()   HP 샘플링 (TPE 베이지안)
         make_model()         모델 생성
         model.learn()        학습 (timesteps 스텝)
         evaluate_policy()    평가 (다른 시드로 일반화 측정)
         return -mean_reward  목적함수 (최소화 = 보상 최대화)
```

### TPE 탐색 원리

```
1~20회 : 랜덤 탐색 (warm-up, 탐색 공간 전체 파악)
21회~  : TPE 베이지안 탐색
          ├─ 좋은 결과의 HP 분포 l(x) 모델링
          ├─ 나쁜 결과의 HP 분포 g(x) 모델링
          └─ l(x)/g(x) 높은 HP를 다음 trial에 제안
             → 좋은 영역을 집중 탐색
```

### 탐색 대상 HP

| 알고리즘 | 주요 HP |
|----------|---------|
| DQN | learning_rate, batch_size, buffer_size, exploration_fraction, target_update_interval |
| PPO | learning_rate, n_steps, batch_size, n_epochs, ent_coef, gae_lambda, clip_range |
| SAC | learning_rate, batch_size, buffer_size, tau, ent_coef |

---

## 10. 평가 파이프라인 (evaluate.py)

```
main(portal, n_episodes)
    │
    ├─ 베이스라인 5종 평가 (evaluate_policy)
    │    └─ 에피소드별: 다른 시드로 n_episodes 실행 → 평균
    │
    ├─ RL 3종 평가 (evaluate_rl)
    │    └─ model.predict(deterministic=True) ← 탐색 없이 greedy
    │
    ├─ _save_csv()        comparison_<portal>.csv
    ├─ _plot_bars()       comparison_<portal>.png  (2×2 막대 그래프)
    └─ _plot_timeseries() timeseries_<portal>.png  (부하 vs W·D)
```

### 생성 결과물

| 파일 | 내용 |
|------|------|
| `comparison_*.csv` | 정책별 보상·SLA위반율·지연·자원 수치표 |
| `comparison_*.png` | 4개 지표 막대 비교 (RL=빨강, 베이스라인=회색) |
| `timeseries_*.png` | RPS 변화 vs 에이전트의 W·D 조절 추이 |
| `learning_curve_*.png` | 에피소드별 보상 + 수렴·과적합 표시 |
| `optuna_*_history.png` | HP 탐색 진행 + HP 중요도 |

---

## 11. 결과 해석 기준

| 지표 | 좋은 방향 | 판단 기준 |
|------|-----------|-----------|
| 보상 | 높을수록 | StaticCurrent 초과 → RL 성공 |
| SLA 위반율 | 낮을수록 | HillClimb 미만 → 반응형 이상 |
| 평균 워커 수 | 낮을수록 | current_workers 대비 절감 |
| 평균 DB 커넥션 | 낮을수록 | current_db 대비 절감 |
| Oracle 대비 보상 | 100%에 가까울수록 | 80% 이상이면 우수 |

---

## 12. 전체 실행 순서

```
Step 1 [선택]  python optimize.py --portal research --algo DQN --n-trials 50
               → results/optuna_research_DQN_best.txt 확인 후 최적 HP 적용

Step 2 [필수]  python train.py --portal research --algo all --timesteps 250000
               → models/research_DQN.zip, PPO.zip, SAC.zip 생성
               → results/learning_curve_research_*.png 생성

Step 3 [필수]  python evaluate.py --portal research --episodes 20
               → results/comparison_research.png
               → results/timeseries_research.png
```
