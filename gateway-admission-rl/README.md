# 강화학습 기반 게이트웨이 통합 동시성 제어 (Joint Admission Control over a Shared DB)

MSA 게이트웨이가 **역할별 백엔드(WAS)로 라우팅**하면서, 각 라우트의 **허용 동시
처리 수 L_i 를 동시에(통합) 조절**한다. WAS 자원은 라우트마다 독립이지만 **모든
라우트가 하나의 공유 DB를 사용**하므로, 한 라우트가 욕심내면 DB가 포화되어 **다른
라우트까지 느려진다.** 이 결합(coupling) 때문에 라우트별 독립 제어로는 풀 수 없고,
**우선순위를 고려해 공유 DB 예산을 배분하는 통합 제어**가 필요하다. 이를 강화학습
(DQN·PPO·SAC)으로 학습하고 독립/정적/AIMD 베이스라인과 비교한다.

## 실제 시스템 구조 → 모델 매핑

```
                          ┌─ WAS-search (자기 CPU) ─┐
게이트웨이 ─ L_search,    ├─ WAS-payment(자기 CPU) ─┤─→  공유 DB  (C_db, 진짜 병목)
  L_payment, L_report ─→ └─ WAS-report (자기 CPU) ─┘     락 경합·열화로 시변·비노출
   (통합 컨트롤러)
```

- 게이트웨이 자체 자원(CPU·힙·스레드풀)은 충분하다고 가정(병목 아님).
- 각 WAS는 **자기 자원으로만 처리**(라우트별 독립). 사양·역할·SLA가 라우트마다 다름.
- **공유 DB가 유일한 결합점이자 진짜 병목.** 용량 C_db(t)는 직접 관측 불가(시변).

## 2단 병목 물리 (시뮬레이터)

```
was_tp_i = min(arrival_i, L_i / s_i, was_cap_i)     # ① 라우트별 WAS 처리량 후보(독립)
D        = Σ was_tp_i · db_cost_i                    # 공유 DB 총수요(cost 가중)
db_util  = D / C_db(t)                               # C_db 는 비노출·시변

db_util ≤ 1 :  served_i = was_tp_i                   # DB 여유 → 그대로 처리
db_util > 1 :  served_i = was_tp_i / db_util         # ② DB 포화 → 비례 throttle
               지연_i ×= db_util²                     #    + 큐잉 폭증(전 라우트 공통!)
```

- 라우트가 받아들이는 양이 늘면 D가 커지고, **D가 C_db를 넘으면 db_util>1 → 모든
  라우트의 지연이 db_util² 로 폭증**(공유 DB 큐잉). 여기서 라우트들이 서로 간섭한다.
- 따라서 **최적은 Σ served_i·db_cost_i ≈ C_db** 를 유지하되, **DB가 부족하면 저우선
  라우트를 조여 고우선(결제)을 보호**하는 배분이다.

## 문제를 MDP로 정의 (통합)

| 요소 | 정의 |
|------|------|
| **State** (6·N 차원, 관측 가능) | 라우트별 [정규화 L_i, 지연_i/SLA_i, 지연추세_i, 처리율 util_i, 거절율_i, 정규화 도착_i] — **공유 DB 용량 C_db(t)는 비노출** |
| **Action** | 라우트별 ΔL 벡터 {−16,−4,−1,0,+1,+4,+16}. DQN=`Discrete(7^N)`(flat) / PPO=`MultiDiscrete([7]×N)` / SAC=`Box(N)` |
| **Reward** | `Σ_i 우선순위_i·(SLA내 처리량_i) − w_reject·Σ거절 − w_sla·Σ max(0, 지연_i−SLA_i)` |
| **Episode** | 300 제어 스텝. 한 스텝 = 모든 라우트의 동시성 한계를 1회 재조정 |

기본 라우트 구성(`config.py`, 실제 구성에 맞게 숫자만 교체):

| 라우트 | 서비스시간 s | WAS 용량 | DB 비용 | SLA | 우선순위 | 특성 |
|---|---|---|---|---|---|---|
| search  | 0.05s | 300 | 1.0 | 0.15s | 1.0 | 가벼움·고빈도·빠른 응답 |
| payment | 0.15s | 110 | 3.0 | 0.40s | 3.0 | DB 무거움·**최우선** |
| report  | 0.50s | 40  | 5.0 | 2.00s | 0.4 | CPU 집약·느려도 됨·저우선 |

공유 DB: 정상 520 work/s ↔ **열화 시 160~260**(락 경합·백업·장애). 열화 때 적정
배분이 급변 → 정적·독립 제어 실패.

## 검증된 베이스라인 격차 (15 에피소드 평균, numpy 검증 완료)

| 정책 | 평균 보상 | SLA 위반 | 처리량 | DB util | 우선순위 goodput | 설명 |
|------|----------:|---------:|-------:|--------:|----------------:|------|
| Static-High | 41542 | 37.0% | 229 | **1.67** | 243 | 공유 DB 과부하 → 전 라우트 지연 |
| Static-Mid | 38510 | 19.5% | 188 | 1.04 | 242 | 보수적, 용량 낭비(거절↑) |
| Greedy-Indep | 16663 | 57.0% | 229 | 1.78 | 172 | DB 무시 → 집단 과부하 |
| Unlimited | −108088 | 99.5% | 229 | 1.79 | 68 | 붕괴(지연 폭발) |
| Indep-AIMD | 36807 | 32.5% | 210 | 1.36 | 230 | 라우트별 반응형 → 전환 지연·진동 |
| **Oracle-Joint (참조 상한)** | **80540** | **6.5%** | 222 | **1.05** | **364** | DB 용량 알고 우선순위 배분 — 최적 |

핵심 관찰(발표 논리):

1. **독립·정적 제어는 못 이긴다** — 공격적이면 공유 DB 과부하(util 1.67~1.79)로 전
   라우트 지연 폭증, 보수적이면 용량 낭비. 무제한은 붕괴.
2. **라우트별 AIMD도 부족** — 각자 자기 지연만 보고 반응하므로 공유 DB 결합을 모르고
   전환 구간에서 뒤처지고 진동한다.
3. **DB 용량을 아는 Oracle-Joint가 천장** — util≈1.05로 과부하를 피하면서 저우선
   (report)을 조여 **최우선(payment)을 보호**(우선순위 goodput 364, 최고). 독립 제어
   대비 보상 **+38998**. 이것이 RL의 목표: **관측만으로 숨은 DB 용량을 추론해 라우트별
   L을 선제·우선순위 배분**.

라우트별 효과(8 에피소드): Oracle은 search/payment/report SLA 위반을 7.2/6.4/4.2%로
고르게 낮춤. Static-High는 search 62.3%로 붕괴(공유 DB 과부하로 고빈도 라우트가 가장
먼저 무너짐).

## 알고리즘 비교 포인트 (다차원 행동)

행동이 **라우트별 ΔL 벡터**라서 알고리즘별 적합도가 갈린다 — 발표의 좋은 분석거리:

- **DQN** : `Discrete(7^N)` 로 평탄화 → **행동 수가 7^N 으로 조합 폭증**(N=3이면 343).
  단일 제어엔 되지만 라우트가 늘면 한계 — 가치기반의 다차원 행동 약점 노출.
- **PPO** : `MultiDiscrete([7]×N)` 를 자연스럽게 처리(라우트별 독립 분포).
- **SAC** : `Box(N)` 연속 벡터 + 엔트로피 → 부드러운 배분, 표본효율 좋음.

## 프로젝트 구조

```
gateway-admission-rl/
├── envs/admission_env.py  # 다중 라우트 + 공유 DB 시뮬레이터(2단 병목, 부분관측)
├── baselines.py           # Static(고/중), Greedy-Indep, Unlimited, Indep-AIMD, Oracle-Joint
├── config.py              # 라우트별·공유 DB 파라미터(실제 구성에 맞게 교체)
├── train.py               # DQN(flat)/PPO(MultiDiscrete)/SAC(Box) 학습
├── evaluate.py            # 비교 표·막대 + 숨은 C_db·라우트별 L·우선순위 보호 시계열
├── requirements.txt
├── models/ · results/     # 실행 후 생성
```

## 실행법

```bash
pip install -r requirements.txt
python train.py --algo all --timesteps 300000
python evaluate.py --episodes 20
```

산출물: `results/comparison.csv`, `results/comparison.png`(보상/SLA/처리량/DB util),
`results/timeseries.png`(숨은 C_db vs db_util, 라우트별 L, 라우트별 지연/SLA — 에이전트가
**용량은 못 보는데도 DB util을 1 근처로 유지하며 우선순위로 L을 배분**하는 거동).

## 실무 연동 메모

State는 게이트웨이가 이미 측정하는 지표(라우트별 응답시간·요청률·in-flight·429율,
Micrometer/Actuator)에서 얻는다. RL 출력(라우트별 목표 동시성 L)을 게이트웨이의
라우트별 동시성 limiter(또는 RequestRateLimiter)에 런타임 반영하거나, 안전하게는
권고값으로 표시 후 적용한다. 공유 DB가 결합점인 멀티-백엔드 환경에서, 본 연구는
독립 제어를 우선순위 인지 통합 제어로 대체·개선하는 것이다.

## 전제 / 한계 / 향후

- **전제**: 게이트웨이 자원은 비병목. 공유 DB가 유일한 결합·병목.
- **한계**: 시뮬레이터 물리(M/M/1형 근사) — sim-to-real 갭 존재.
- **향후**: 실제 Spring Cloud Gateway 필터(GlobalFilter) 연동, 라우트 수 N 확장,
  WAS별 헬스 기반 라우팅까지 통합, 실트래픽 보정.
