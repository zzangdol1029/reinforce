# 강화학습 기반 Worker Thread Pool 동적 튜닝

**인스턴스가 고정된 운영 환경**(온프레미스, 수평 확장 불가)에서 트래픽 변화에 따라
Undertow worker thread pool 크기를 동적으로 조절하는 정책을 강화학습으로 학습합니다.

> 딥러닝 라이브러리는 교재의 **DeZero**를 사용합니다.  
> DQN·PPO 알고리즘은 SB3 없이 `agents/` 에 직접 구현되어 있습니다.  
> CPU만으로 충분합니다: v1 학습 1~2분, v2 학습 3~5분 수준.

---

## 환경 구성

주력 환경은 thread 2종이며, 인스턴스 환경 2종은 확장 실험용입니다.

| 환경 이름 | 방식 | 조절 대상 | SLA 지표 | State 차원 |
|-----------|------|-----------|----------|-----------|
| `threadpool` **(주력)** | v1 큐잉 공식 | thread 4~64 (±4) | 평균 지연 | 6 |
| `ev-threadpool` **(주력)** | v2 이산사건 | thread 4~64 (±4) | **p95 지연** | 8 |
| `container` (확장용) | v1 큐잉 공식 | 인스턴스 1~20 | 평균 지연 | 6 |
| `ev-container` (확장용) | v2 이산사건 | 인스턴스 1~20 | p95 지연 | 8 |

**v2**는 요청 하나하나를 FCFS 큐에서 처리해 응답시간을 정확히 계산합니다
(`envs/event_engine.py`). 실무 SLA와 같은 **p95 tail latency** 기준 평가가 가능하고,
lognormal heavy-tail 처리시간으로 '가끔 오는 무거운 요청'을 재현합니다.

---

## MDP 정의

### threadpool / container (v1)

| 요소 | 정의 |
|------|------|
| **State (6,)** | [λ 정규화, λ 변화량, 이용률 ρ, 지연/SLA, c 정규화, backlog 정규화] |
| **Action (3)** | 0 = thread −step / 1 = 유지 / 2 = thread +step |
| **Reward** | −w_sla·SLA위반정도 − w_cost·(기본값 초과 비용) − w_under·(기본값 미만) − w_thrash·(행동) |
| **Episode** | 288 step, 트래픽 패턴 에피소드마다 무작위 |

### ev-threadpool / ev-container (v2)

| 요소 | 정의 |
|------|------|
| **State (8,)** | [λ 정규화, λ 변화량, 이용률, p95/SLA, 평균지연/SLA, c 정규화, 기동중 비율, backlog] |
| **Action (3)** | 0 = 축소 / 1 = 유지 / 2 = 확대 |
| **Reward** | −w_sla·(p95 SLA 초과 비율) − w_cost·(자원 비용) − w_thrash·(행동) |
| **Episode** | 288 step |

### 비단조 용량 곡선 (threadpool 환경의 핵심)

```
capacity(c) = (c × μ_thread) / (1 + α × max(0, c − c_knee)² / c_knee²)

c ≤ c_knee (=32): thread 증가 → 용량 증가  (IO-bound 병렬화 이득)
c > c_knee      : 컨텍스트 스위칭/CPU 경합 → thread당 효율 하락
                  → 무작정 늘리면 오히려 용량 감소
```

최적 thread 수가 트래픽에 따라 움직이며, 에이전트는 이를 추적해야 합니다.

### 비용 모델

기본값(`c_base=16`)까지는 비용 0, **초과 증설분에만 비용**을 부과합니다.
기본값 미만으로 줄이면 `w_under` 페널티가 부과됩니다.

> 에이전트의 목표: 평소엔 기본 16개 유지 → 요청 증가 시 증설 → 끝나면 기본값 복귀

---

## 알고리즘

SB3를 사용하지 않고 DeZero 기반으로 직접 구현되어 있습니다.

| 알고리즘 | 파일 | 주요 기법 |
|----------|------|-----------|
| **DQN** | `agents/dqn.py` | Double DQN, Experience Replay, ε-greedy (선형 감쇠) |
| **PPO** | `agents/ppo.py` | Actor-Critic, GAE(λ), Clipped Surrogate, Entropy bonus |

> **SAC는 구현되어 있지 않습니다.** `--algo` 선택지는 `dqn`, `ppo`만 유효합니다.

---

## 과적합 방지 설계

### 1. 시드 3분할

```
학습    : seed × 10000 + ep    (에피소드마다 새 트래픽 패턴)
검증    : VAL_SEED   = 5555555  (학습 중 best 체크포인트 선택용)
평가    : EVAL_SEED  = 7777     (최종 보고용 — 학습/검증/튜닝에 절대 미사용)
```

### 2. 검증 기반 best 체크포인트

`VAL_INTERVAL(=10)` 에피소드마다 검증 트래픽으로 greedy 평가하고,
최고 성능 시점의 가중치를 `*_best.npz`로 저장합니다.
학습 후반의 정책 붕괴가 최종 모델을 오염시키지 않으며,
`evaluate.py`는 자동으로 `_best` 가중치를 우선 사용합니다.

### 3. 도메인 랜덤화

트래픽 위상·피크·버스트 강도와 heavy-tail sigma가 에피소드마다 변경되어
특정 패턴 암기가 불가능합니다.

---

## 설치 및 환경 설정

이 프로젝트는 **DeZero 라이브러리**를 사용합니다. `week10-dezero` conda 환경이 필요합니다.

```bash
# 최초 1회: week10-dezero 환경 생성 (week10/ 폴더에서)
conda activate week10-dezero

# 추가 패키지 설치
pip install -r requirements.txt
```

`requirements.txt` 내용:
```
numpy>=1.20.0
matplotlib>=3.5.0
gymnasium>=0.29.0
dezero>=0.0.13
```

> NumPy 1.24+ 호환은 `agents/common.py`에서 자동 패치(`np.int`, `np.float`, `np.bool` 재정의)합니다.

---

## 실행 방법

### 전체 파이프라인 한 번에

```bash
python run_all.py                   # thread 환경 전체 (v1 + v2, 약 10분)
python run_all.py --quick           # v1 thread만 (약 3분)
python run_all.py --v2-only         # v2 thread만
python run_all.py --with-container  # 인스턴스 환경까지 (확장 실험)
```

### 단계별 실행

```bash
# 1단계: 학습
python train.py --env threadpool    --algo dqn --episodes 500
python train.py --env threadpool    --algo ppo --episodes 600
python train.py --env ev-threadpool --algo dqn --episodes 200
python train.py --env ev-threadpool --algo ppo --episodes 300

# 중단 후 이어서 학습
python train.py --env threadpool --algo dqn --resume

# N초 후 자동 저장 종료 (분할 학습)
python train.py --env threadpool --algo dqn --max-seconds 300

# 2단계: 평가 + 그래프
python evaluate.py --env threadpool
python evaluate.py --env ev-threadpool

# 3단계: 설명용 그림
python make_figures.py
```

---

## 산출물 (results/ 폴더)

모든 결과는 `results/` 단일 폴더에 파일명으로 구분됩니다.
같은 환경·알고리즘 조합을 재실행하면 덮어씁니다.

### 학습 산출물 (train.py)

```
results/
├── <env>_<algo>_w.npz       마지막 가중치 (DQN: 1개, PPO: .actor/.critic 2개)
├── <env>_<algo>_best.npz    검증 최고 가중치 ← evaluate.py가 이 파일 사용
├── <env>_<algo>_curve.npy   에피소드별 학습 보상
├── <env>_<algo>_val.npy     검증 보상 [(episode, val_reward), ...]
└── <env>_<algo>_state.pkl   분할 학습 재개용 상태 (--resume)
```

### 평가 산출물 (evaluate.py)

```
results/
├── <env>_learning_curves.png  학습 곡선 + 검증 곡선 (DQN/PPO 비교)
├── <env>_behavior.png         트래픽/자원량/지연 시계열 (정책별 거동 비교)
├── <env>_comparison.png       보상/SLA위반율/비용 막대 그래프 (3종)
└── <env>_summary.csv          수치 요약표 (PPT 기입용)
```

### 설명용 그림 (make_figures.py)

```
results/
├── traffic_sample.png    트래픽 생성기 샘플 (일일 사이클 + 버스트)
└── capacity_curve.png    thread 비단조 용량 곡선 (c_knee=32 표시)
```

---

## 산출물 → PPT 매핑

| 실행 | 산출물 | PPT 사용처 |
|------|--------|-----------|
| `evaluate.py` | `<env>_summary.csv` | 결과 표 (수치 기입) |
| `evaluate.py` | `<env>_learning_curves.png` | 학습 결과 슬라이드 |
| `evaluate.py` | `<env>_behavior.png` | 거동 비교 슬라이드 |
| `evaluate.py` | `<env>_comparison.png` | 종합 비교 슬라이드 |
| `make_figures.py` | `traffic_sample.png` | 트래픽 모델 설명 슬라이드 |
| `make_figures.py` | `capacity_curve.png` | 비단조 용량 설명 슬라이드 |
| `node build_deck.js` | `autoscaling_rl_발표.pptx` | 그림 자동 삽입 PPT 재생성 |

---

## 시각화 (발표 데모용)

```bash
# 실시간 시뮬레이션 — matplotlib 창에서 step별로 그려짐
python simulate_live.py --env threadpool --algo dqn   # 학습 후
python simulate_live.py --env threadpool --algo rule  # 학습 전에도 가능
# --interval 0.02 (재생 속도 조절) / --save (창 없이 png 저장)

# 정책 내장 인터랙티브 HTML (브라우저에서 RL/룰/수동 전환 데모)
python export_policy.py --algo dqn    # → viz_simulation_dqn.html 생성

# 애니메이션 GIF (PPT 삽입용)
python make_gif.py --env threadpool --algo dqn
```

---

## 하이퍼파라미터 탐색

```bash
python hp_search.py --env ev-threadpool --algo dqn --episodes 80
```

검증 보상 기준 순위가 출력됩니다. 평가 시드(`EVAL_SEED=7777`)는 사용하지 않습니다.
권장값과 근거는 `config.py` 주석 참고.

---

## 프로젝트 구조

```
autoscaling-rl/
├── config.py              환경·하이퍼파라미터 (권장값 근거 주석 포함)
│
├── envs/
│   ├── traffic.py         일일 패턴 + 버스트 트래픽 생성기
│   ├── threadpool_env.py  v1: thread 환경 (큐잉 공식, State 6차원)
│   ├── autoscale_env.py   v1: 인스턴스 환경 (큐잉 공식, State 6차원)
│   ├── event_engine.py    v2: 이산사건 FCFS 큐 엔진 (요청 단위 시뮬레이션)
│   └── event_env.py       v2: thread/인스턴스 환경 (p95 SLA, State 8차원)
│
├── agents/
│   ├── common.py          DeZero 호환 패치 + MLP + gradient clipping
│   ├── dqn.py             Double DQN (직접 구현)
│   └── ppo.py             PPO + GAE (직접 구현)
│
├── baselines.py           ThresholdAutoscaler(HPA식), StaticPolicy
├── train.py               학습 (검증 best 체크포인트, 분할 학습 지원)
├── evaluate.py            평가 + 그래프 3종 + summary.csv
├── run_all.py             전체 파이프라인 (학습→평가→그림) 한 번에
│
├── hp_search.py           하이퍼파라미터 grid search
├── make_figures.py        설명용 그림 (트래픽/용량 곡선)
├── simulate_live.py       실시간 시뮬레이션 뷰어 (발표 라이브 데모)
├── export_policy.py       정책 내장 인터랙티브 HTML 데모 생성
├── make_gif.py            정책 거동 애니메이션 GIF (PPT용)
├── build_deck.js          발표 PPT 자동 생성 (npm i pptxgenjs 필요)
├── viz_template.html      인터랙티브 데모 HTML 템플릿
│
└── results/               모든 산출물 (학습 후 생성)
    ├── <env>_<algo>_best.npz   검증 최고 가중치
    ├── <env>_<algo>_curve.npy  학습 곡선
    ├── <env>_learning_curves.png
    ├── <env>_behavior.png
    ├── <env>_comparison.png
    ├── <env>_summary.csv
    ├── traffic_sample.png
    └── capacity_curve.png
```

---

## 베이스라인

| 정책 | 방식 | 특징 |
|------|------|------|
| **ThresholdAutoscaler** | 이용률 > 0.7 → 확대, < 0.3 → 축소 (HPA 방식) | cooldown 3 step으로 진동 방지 |
| **StaticPolicy** | c_init 고정 (아무것도 안 함) | 동적 제어의 가치 비교용 |

---

## 학습 로그 읽는 법

```
[DQN/threadpool] ep  10/500  R(10)= -82.34  val= -76.12  best= -76.12  eps=0.912  (23s)
                              ↑               ↑            ↑
                         최근 10ep 평균    검증 보상    검증 최고 보상 (best 저장 기준)
```

`val=` 이 `best=` 를 갱신할 때 `*_best.npz` 파일이 저장됩니다.
