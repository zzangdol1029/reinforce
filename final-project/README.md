# 강화학습 기반 MSA 게이트웨이 부하분산

Spring Boot 기반 MSA 환경의 API 게이트웨이가 요청을 백엔드 서비스
인스턴스로 라우팅(부하분산)할 때, **강화학습으로 동적 라우팅 정책을
학습**하여 평균 응답시간과 SLA 위반을 줄이는 프로젝트입니다.
정적/휴리스틱 알고리즘(Round Robin, Least Connection 등)을 베이스라인으로
두고 DQN · PPO · SAC 세 가지 알고리즘과 비교합니다.

> 학습은 실제 Spring 환경이 아니라, MSA 트래픽을 모사하는 **시뮬레이터
> (Gymnasium 환경)** 안에서 수행합니다. 강화학습은 수십만 step의 시행착오가
> 필요해 실제 시스템을 직접 학습 환경으로 쓰기 어렵기 때문입니다.

## 문제를 MDP로 정의

| 요소 | 정의 |
|------|------|
| **State** | 인스턴스 N개 각각의 [정규화 큐길이, 정규화 잔여작업, 최근 EWMA 응답시간, 정규화 처리율] + 전역 부하 → `(N*4+1,)` 벡터 |
| **Action** | (이산) `Discrete(N)` — 요청 보낼 인스턴스 선택. DQN/PPO용<br>(연속) `Box(-1,1,(N,))` — 인스턴스별 가중치, argmax로 라우팅. SAC용 |
| **Reward** | `-(라우팅된 요청 예상 완료시간) - λ·(인스턴스 부하 표준편차) - (SLA 위반 시 페널티)` |
| **Episode** | 요청 500건 라우팅. 한 step = 한 요청 도착/결정 |

핵심 설계: 인스턴스마다 처리율(`service_rate`)이 **이질적**(느린 노드~빠른 노드)이고
요청 작업량도 변동(지수분포)이라, 단순 Round Robin은 느린 노드에 과부하를 만들어
불리합니다. 이 지점에서 RL이 학습으로 우위를 보이는지가 관전 포인트입니다.

## 프로젝트 구조

```
final-project/
├── envs/load_balancer_env.py   # Gymnasium 시뮬레이터 환경
├── baselines.py                # RR, WeightedRR, LeastConn, LeastWork, Random + 평가함수
├── config.py                   # 공통 환경 파라미터(학습/평가 동일 시나리오)
├── train.py                    # DQN/PPO/SAC 학습 (Stable-Baselines3)
├── evaluate.py                 # RL vs 베이스라인 비교 표(CSV)·그래프(PNG)
├── requirements.txt
└── results/                    # comparison.csv, comparison.png (실행 후 생성)
```

## 실행법

```bash
pip install -r requirements.txt

# 1) 학습 (각 알고리즘 모델은 models/ 에 저장)
python train.py --algo all --timesteps 150000
#   개별: python train.py --algo DQN --timesteps 150000

# 2) 평가 + 비교 그래프/표 생성
python evaluate.py --episodes 30
```

생성물: `results/comparison.csv`(지표 표), `results/comparison.png`(막대그래프 4종 —
평균 응답시간 / p95 / SLA 위반율 / 부하 불균형).

## 평가 지표
- **mean_latency / p95_latency** — 평균·꼬리 응답시간 (낮을수록 좋음)
- **throughput** — 처리 완료 요청 수 (높을수록 좋음)
- **sla_violation_rate** — SLA(기본 1.5s) 초과 비율 (낮을수록 좋음)
- **load_imbalance** — 인스턴스 간 예상 대기시간 표준편차 (낮을수록 균형)

## 발표(PPT) 구성 매핑

1. **적용 문제** — MSA 게이트웨이 부하분산의 한계(정적 알고리즘은 이질적/변동
   트래픽에 취약). 동적 학습 라우팅의 필요성.
2. **모델: state / action 정의** — 위 MDP 표. 시뮬레이터 가정(Poisson 도착,
   이질적 처리율, 지수분포 작업량).
3. **방법/알고리즘** — 가치기반(DQN), 정책기반(PPO), 엔트로피 정규화 Actor-Critic
   (SAC). 이산 vs 연속 행동공간을 같은 환경에서 다룬 방식.
4. **프로그램 소스 설명** — `load_balancer_env.py`의 step/reward, `train.py`의 SB3
   하이퍼파라미터, `evaluate.py`의 비교 절차.
5. **결과 제시/분석** — `comparison.png`로 RL이 베이스라인 대비 응답시간·SLA
   위반을 얼마나 줄였는지. 알고리즘 간(DQN/PPO/SAC) 차이 분석.
6. **기여도 / 향후 연구** — 트래픽 변동에 적응하는 학습 라우팅의 가능성. 향후:
   실제 Spring Cloud Gateway 커스텀 LoadBalancer(또는 GlobalFilter)에 학습된 정책을
   gRPC/REST로 연동, 오토스케일링/서킷브레이커와 결합, 실트래픽 로그 기반 학습.
```
```
