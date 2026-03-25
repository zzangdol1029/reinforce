# Markov Decision Process (03-Markov Decision Process.pdf)  
## 페이지별 상세 정리 · 풀이 과정

> **출처**: Prof. Tae-Hyoung Park, Dept. of Intelligent Systems & Robotics, CBNU  
> 본 문서는 강의 슬라이드 17페이지 구성에 맞춰 **개념 정리**, **수식**, **예시 풀이**를 포함합니다.

---

## 목차

1. [페이지 1 — 표지](#페이지-1--표지)
2. [페이지 2 — Agent–Environment Interface](#페이지-2--agentenvironment-interface)
3. [페이지 3 — 예: Recycling Robot](#페이지-3--예-recycling-robot)
4. [페이지 4 — 예: 3×4 Grid World](#페이지-4--예-34-grid-world)
5. [페이지 5 — Episodic vs Continuing Task](#페이지-5--episodic-vs-continuing-task)
6. [페이지 6 — Return (수익)과 할인율 γ](#페이지-6--return-수익과-할인율-γ)
7. [페이지 7 — Markov Property](#페이지-7--markov-property)
8. [페이지 8–9 — MDP 정의 · Recycling Robot MDP](#페이지-89--mdp-정의--recycling-robot-mdp)
9. [페이지 10 — 1×2 Grid World (전이표)](#페이지-10--12-grid-world-전이표)
10. [페이지 11 — (Q1) 1×5 Grid World 전이도](#페이지-11--q1-15-grid-world-전이도-풀이-가이드)
11. [페이지 12 — Value Functions](#페이지-12--value-functions)
12. [페이지 13 — Optimal Value & Policy](#페이지-13--optimal-value--policy)
13. [페이지 14 — Grid World 문제 (v_π, v*, π*)](#페이지-14--grid-world-문제-v_π-v-π)
14. [페이지 15 — 기댓값 Expectation 예시](#페이지-15--기댓값-expectation-예시-풀이)
15. [페이지 16 — 1×2 Grid World와 Bellman 방정식](#페이지-16--12-grid-world와-bellman-방정식-풀이)
16. [페이지 17 — (Q2) 1×2 Grid World 4가지 정책](#페이지-17--q2-12-grid-world-4가지-정책-풀이)

---

## 페이지 1 — 표지

**내용**: 강의 제목 *Markov Decision Process (MDP)*, 교수·학과 정보.

**정리**: 이후 슬라이드에서 **순차적 의사결정**을 **상태–행동–보상–전이**로 모델링하고, **가치 함수**와 **최적 정책**을 다룹니다.

---

## 페이지 2 — Agent–Environment Interface

### 슬라이드 요지

- **Agent**: 학습·의사결정 주체 (controller)
- **Environment**: 에이전트와 상호작용하는 대상 (plant / system)
- **State** \(S_t \in \mathcal{S}\): 가능한 상태 집합
- **Action** \(A_t \in \mathcal{A}(S_t)\): 상태 \(S_t\)에서 선택 가능한 행동 집합
- **Reward** \(R_{t+1} \in \mathcal{R} \subset \mathbb{R}\): 스칼라 보상
- **Policy** \(\pi_t(a|s)\): 상태 \(s\)에서 행동 \(a\)를 택할 **확률**  
  - deterministic / stochastic 모두 가능
- **목표**: 누적 보상을 최대화하는 정책 \(\pi\)를 찾는 것

### 풀이·이해 포인트

1. 한 스텝의 흐름: \(S_t \xrightarrow{A_t} R_{t+1}, S_{t+1}\) (슬라이드 표기에 따라 보상 인덱스는 \(t+1\)일 수 있음)
2. 정책은 **상태에 대한 행동 분포**이므로, 같은 상태에서도 확률적으로 다른 행동을 고를 수 있음

---

## 페이지 3 — 예: Recycling Robot

### 슬라이드 요지

재활용 로봇은 **배터리 잔량(상태)**에 따라 **충전·캔 수집 등(행동)**을 선택하고 **보상**을 받는 전형적인 MDP 예시입니다.

### 풀이·이해 포인트 (일반적 모델)

- **상태 예**: `High`, `Low` (배터리)
- **행동 예**: `search`, `wait`, `recharge` 등
- **보상**: 캔 수집 성공 등
- **전이**: 배터리가 줄거나 충전되는 확률적 규칙

> 슬라이드에 그림만 있는 경우, **상태 집합 \(\mathcal{S}\)**, **행동 집합 \(\mathcal{A}(s)\)**, **보상 \(r(s,a,s')\)**, **전이 \(p(s'|s,a)\)** 를 표로 적어보면 “MDP를 안다”는 것과 동일한 연습이 됩니다.

---

## 페이지 4 — 예: 3×4 Grid World

### 슬라이드 요지

격자 세계에서 **\(\mathcal{S}\)**, **\(\mathcal{A}\)**, **\(r\)** 를 채우는 연습용 예시입니다.

### 풀이 과정 (표준 설정 예시)

격자가 3행×4열이고, 셀을 상태로 둡니다.

1. **상태 \(\mathcal{S}\)**  
   - 각 칸이 하나의 상태 (단, 장애물·목표는 별도 처리)  
   - 예: 12개 칸이면 \(|\mathcal{S}| = 12\) (또는 시작/목표/벽 정의에 따라 감소)

2. **행동 \(\mathcal{A}\)**  
   - 보통 `{상, 하, 좌, 우}`  
   - 벽에 부딪히면 **제자리** 또는 **이동 불가** 중 하나로 정의 (문제에서 통일)

3. **보상 \(r\)**  
   - 목표 칸 도착: \(+1\)  
   - 함정: \(-1\)  
   - 그 외: \(0\)  
   (슬라이드의 빈칸은 위와 같이 **문제 정의**를 스스로 쓰는 연습)

### 체크리스트

- [ ] Episodic인가? (목표 도착 시 에피소드 종료?)
- [ ] 전이는 deterministic인가 stochastic인가?

---

## 페이지 5 — Episodic vs Continuing Task

### 정의

| 구분 | Episodic (일회성) | Continuing (지속적) |
|------|-------------------|---------------------|
| 종료 | **Terminal state** 존재 | Terminal 없음 |
| 예 | 게임, 바둑, Grid World (목표 도달) | 재고 관리, 로봇 작업, Recycling Robot 등 |

### 풀이·이해 포인트

- **Episodic**: 보통 \(t = 1, 2, \ldots, T\) 까지의 궤적(trajectory)으로 생각
- **Continuing**: \(T \to \infty\) 이므로 **할인율 \(\gamma < 1\)** 로 무한합을 수렴시키는 경우가 많음 (다음 페이지와 연결)

---

## 페이지 6 — Return (수익)과 할인율 γ

### 정의

- **목표**: 장기적으로 받는 보상의 합(또는 할인합)을 최대화
- **Return** \(G_t\): 시점 \(t\) 이후의 보상 합

**Episodic** (종료 시각 \(T\)):

\[
G_t = R_{t+1} + R_{t+2} + \cdots + R_T
\]

**Continuing** (\(T = \infty\)):

\[
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
\]

- **\(\gamma\)**: discount rate, \(0 \le \gamma \le 1\)
- **주의**: Continuing에서 \(\gamma = 1\)이면 보상이 계속 쌓이면 \(G_t \to \infty\) 가능 → 분석·수렴을 위해 \(\gamma < 1\)을 쓰는 경우가 많음

### 소규모 풀이 예시

\(R_{t+1}=1, R_{t+2}=2, R_{t+3}=0, \gamma=0.9\), 이후 보상 0이라 가정하면:

\[
G_t = 1 + 0.9 \times 2 + 0.9^2 \times 0 = 1 + 1.8 = 2.8
\]

---

## 페이지 7 — Markov Property

### 정의

상태 \(S_t\)가 **Markov**라는 것은:

\[
\Pr(S_{t+1} \mid S_0, S_1, \ldots, S_t) = \Pr(S_{t+1} \mid S_t)
\]

**직관**: “미래는 과거와 독립이고, **현재 상태**만 주어지면 충분하다.”

### RL에서의 형태

\[
\Pr(R_{t+1}=r,\, S_{t+1}=s' \mid S_0,A_0,\ldots,S_t,A_t)
= p(s', r \mid s, a)
\]

즉 **한 스텝 역학**은 \((s, a)\) 만으로 결정됩니다.

### 풀이·이해 포인트

- 상태 설계가 나쁘면 Markov가 깨짐 → **상태에 과거 정보를 넣어** Markov에 가깝게 만드는 것이 설계 과제

---

## 페이지 8–9 — MDP 정의 · Recycling Robot MDP

### MDP 정의

**MDP**는 보통 \((\mathcal{S}, \mathcal{A}, \mathcal{P}, r, \gamma)\) 로 씁니다.

- \(\mathcal{S}\): 상태 집합  
- \(\mathcal{A}\): 행동 집합 (상태별로 \(\mathcal{A}(s)\) 가능)  
- **전이** \(\mathcal{P}\) 또는 \(p(s' \mid s, a)\): \(s,a\) 다음 \(s'\) 확률  
- **보상** \(r(s,a)\) 또는 \(r(s,a,s')\): 기댓값 보상  
- \(\gamma\): 할인율  

**Finite MDP**: \(\mathcal{S}, \mathcal{A}\) 가 유한

### One-step dynamics

- \((s', r)\) 쌍의 확률: \(p(s', r \mid s, a)\)
- **기댓값 보상**:  
  \[
  r(s,a) = \mathbb{E}[R_{t+1} \mid S_t=s, A_t=a] = \sum_{s',r} p(s',r\mid s,a)\, r
  \]

### Recycling Robot (복습)

슬라이드의 **전이 그래프**를 다음으로 옮기는 연습:

- 각 **화살표**: \((s, a) \to s'\) 와 확률, 보상(또는 기댓값 보상)

---

## 페이지 10 — 1×2 Grid World (전이표)

### 설정

- **상태**: \(\mathcal{S} = \{L_1, L_2\}\)
- **행동**: \(\mathcal{A}(L_1) = \mathcal{A}(L_2) = \{\text{right}, \text{left}\}\)

### 슬라이드 전이·보상 (파라미터 \(\alpha, \beta\))

| \(s\) | \(s'\) | \(a\)   | \(p(s'|s,a)\) | \(r(s,a,s')\) |
|-------|--------|---------|---------------|----------------|
| \(L_1\) | \(L_1\) | left  | \(1-\alpha\)  | \(-1\)         |
| \(L_1\) | \(L_2\) | right | \(\alpha\)    | \(+1\)         |
| \(L_2\) | \(L_1\) | left  | \(\beta\)     | \(0\)          |
| \(L_2\) | \(L_2\) | right | \(1-\beta\)   | \(-1\)         |

### 풀이 포인트

- **확률 합**: 각 \((s,a)\) 에 대해 \(\sum_{s'} p(s'|s,a) = 1\) 인지 확인  
  예: \(L_1\), left → \(L_1\) 만 \(1-\alpha\) 이면, 나머지 \(L_2\)로 가는 확률이 슬라이드에 없다면 **문제 정의**를 “left는 \(L_1\)에 머무름만”으로 해석 (일반적 1×2 예시와 동일)

---

## 페이지 11 — (Q1) 1×5 Grid World 전이도 (풀이 가이드)

### 문제 요지

- **상태**: \(\mathcal{S} = \{L_1, L_2, \ldots, L_5\}\)
- **보상 규칙 (예시)**  
  - 사과: \(+1\)  
  - 폭탄: \(-2\)  
  - 빈칸: \(0\)  
  - 벽: \(0\) (이동 불가 또는 제자리)

### 풀이 과정 (전이도 그리는 법)

1. **노드 5개**: \(L_1, \ldots, L_5\)
2. **행동** 정의 (예: left / right 만):
   - \(L_1\)에서 left → **벽** → \(L_1\) 유지, 보상 0  
   - \(L_i\)에서 right → \(L_{i+1}\) (단, \(i=5\)면 벽)
3. 각 **화살표**에 \((p, r)\) 또는 deterministic이면 **보상만** 표기
4. **사과·폭탄**이 있는 칸은 그 **상태에 머무를 때** 또는 **진입 시** 보상을 주도록 문제와 일치시킴

### 예시 (한 칸씩만 오른쪽 이동 가능한 단순 선형 세계)

- 행동 `right`: \(L_i \to L_{i+1}\) (i<5), \(L_5\)에서는 제자리  
- 행동 `left`: \(L_i \to L_{i-1}\) (i>1), \(L_1\)에서는 제자리  
- \(L_3\)에 사과면 **\(L_3\)에 도착하는 전이**에 \(+1\) 부여

> 실제 그림은 **본인이 그린 전이도**가 정답이며, 위는 **채점 기준**: 상태·행동·보상·전이의 **일관성**.

---

## 페이지 12 — Value Functions

### 상태 가치 (State-value)

\[
v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]
\]

**의미**: 정책 \(\pi\)를 따를 때, 상태 \(s\)에서 출발한 **기댓값 Return**.

### 행동 가치 (Action-value)

\[
q_\pi(s, a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]
\]

**의미**: \(s\)에서 **먼저 \(a\)를 한 뒤** \(\pi\)를 따를 때의 기댓값 Return.

### 풀이·연결

- **Control**: \(q_\pi(s,a)\) 가 크면 그 행동이 유리 → greedy 개선의 기반

---

## 페이지 13 — Optimal Value & Policy

### 정의

\[
v_*(s) = \max_\pi v_\pi(s), \qquad
q_*(s, a) = \max_\pi q_\pi(s, a)
\]

**최적 정책** \(\pi_*\) 는 모든 상태에서 \(v_*\)를 달성하는 정책 (일반적으로 존재).

**Deterministic 최적 행동** (슬라이드 \(\mu^*\)):

\[
\mu^*(s) \in \arg\max_a q_*(s, a)
\]

---

## 페이지 14 — Grid World 문제 (v_π, v*, π*)

### 슬라이드 요지

어떤 Grid World에 대해 **\(\gamma = 0.9\)** 일 때:

- 정책 \(\pi\)에 대한 \(v_\pi(s)\)
- \(v_*(s)\)
- \(\pi_*(a|s)\)

를 구하거나 그림으로 나타내는 문제입니다.

### 풀이 과정 (일반 알고리즘)

1. **MDP 명시**: \(\mathcal{S}, \mathcal{A}, p(s'|s,a), r, \gamma\)
2. **Policy Evaluation**: Bellman expectation을 반복  
   \[
   v_\pi(s) \leftarrow \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)\,[r + \gamma v_\pi(s')]
   \]
3. **Optimal**: Value Iteration 또는 Policy Iteration  
   \[
   v_{k+1}(s) = \max_a \sum_{s',r} p(s',r|s,a)\,[r + \gamma v_k(s')]
   \]

> 슬라이드의 **구체적 숫자 그리드**가 있으면, 위 반복을 표로 계산하면 됩니다.

---

## 페이지 15 — 기댓값 Expectation 예시 (풀이)

### 슬라이드 설정 (요약)

- 주사위 \(X \in \{1,\ldots,6\}\), \(p(x) = 1/6\)
- 동전 \(Y \in \{\text{뒤}=0, \text{앞}=1\}\), 조건부 \(p(y|x)\) 예시: \(p(\text{앞}|x=4) = 4/5\) 등 (슬라이드 수치 따름)
- 보상 \(r(x, y)\) 예: \(r(4, \text{앞}) = 4\)

### 결합확률

\[
p(x, y) = p(x)\, p(y|x)
\]

예: \(p(4, \text{앞}) = \frac{1}{6} \cdot \frac{4}{5} = \frac{4}{30}\) (슬라이드와 동일한 가정일 때)

### 기댓값

\[
\mathbb{E}[r(X,Y)] = \sum_x \sum_y p(x,y)\, r(x,y)
= \sum_x p(x) \sum_y p(y|x)\, r(x,y)
\]

### MDP와의 연결

Bellman 방정식의 **\(\sum_{s'} p(s'|s,a)\, v(s')\)** 항은 **조건부 기댓값**입니다.  
즉 “다음 상태에 대한 가중 평균”이 **기댓값 연산**입니다.

---

## 페이지 16 — 1×2 Grid World와 Bellman 방정식 (풀이)

### Deterministic policy \(\pi_1\) (예시 표)

| \(a \setminus s\) | \(L_1\) | \(L_2\) |
|-------------------|---------|---------|
| left              | 0       | 0       |
| right             | 1       | 1       |

즉 **항상 right** 선택: \(\mu_1(L_1)=\text{right}, \mu_1(L_2)=\text{right}\).

### Bellman Expectation (일반형)

\[
v_\pi(s) = \sum_a \pi(a|s) \left[ r(s,a) + \gamma \sum_{s'} p(s'|s,a)\, v_\pi(s') \right]
\]

여기서 \(r(s,a) = \sum_{s'} p(s'|s,a)\, r(s,a,s')\) (기댓값 보상).

### \(\alpha = \beta = 0.5\), \(\gamma = 0.9\), 위 정책 “항상 right” 인 경우

**기댓값 보상** (페이지 10 표):

- \(L_1\), right: \(L_2\)로 \(\alpha=0.5\), 보상 \(+1\); \(L_1\)으로? 표에는 right→\(L_2\) 만 \(\alpha\) 이므로, 나머지 확률 \(1-\alpha\)는 **정의에 따라** (슬라이드 그림이 \(L_1\to L_2\) 만 있으면) \(L_1\)에 남는 전이를 추가해야 합이 1이 됨.  
  **표준 1×2 예시**(강의 자료와 동일하게)는 다음을 만족:
  - \(L_1\), right → \(L_2\) : \(p=\alpha, r=+1\)
  - \(L_1\), right → \(L_1\) : \(p=1-\alpha, r=-1\) (슬라이드에 **right의 실패**가 없다면, 강의록의 그림과 표를 함께 확인)

슬라이드 **텍스트 표**는 네 줄만 제시하므로, **실습 코드 `week2/grid_world_1x2.py`** 와 같이:

- \(L_1\)+left → \(L_1\) (\(1-\alpha\), -1)  
- \(L_1\)+right → \(L_2\) (\(\alpha\), +1)  
- \(L_2\)+left → \(L_1\) (\(\beta\), 0)  
- \(L_2\)+right → \(L_2\) (\(1-\beta\), -1)  

으로 두면, **항상 right** 일 때:

\[
v(L_1) = r(L_1,\text{right}) + \gamma\,[\alpha\, v(L_2) + (1-\alpha)\, v(L_1)]
\]

\[
v(L_2) = r(L_2,\text{right}) + \gamma\,[(1-\beta)\, v(L_2)]
\]

여기서 \(r(L_1,\text{right}) = \alpha\cdot 1 + (1-\alpha)\cdot(-1) = 2\alpha - 1\),  
\(r(L_2,\text{right}) = (1-\beta)\cdot(-1) = -(1-\beta)\) (해당 \(s'\)로만 간다면).

**\(\alpha=\beta=0.5\)** 이면:

- \(r(L_1,\text{right}) = 0\)
- \(r(L_2,\text{right}) = -0.5\)

\(L_2\) 방정식:  
\(v_2 = -0.5 + 0.9 \cdot 0.5 \cdot v_2 = -0.5 + 0.45 v_2\)  
\(\Rightarrow 0.55 v_2 = -0.5 \Rightarrow v_2 = -\frac{10}{11} \approx -0.909\)

\(L_1\) 방정식:  
\(v_1 = 0 + 0.9(0.5 v_2 + 0.5 v_1)\)  
\(\Rightarrow v_1 = 0.45 v_2 + 0.45 v_1 \Rightarrow 0.55 v_1 = 0.45 v_2\)  
\(\Rightarrow v_1 = \frac{9}{11} v_2 \approx -0.744\)

> 슬라이드의 “?????” 는 위와 같이 **연립방정식**으로 채우면 됩니다.

### Stochastic policy (0.5 / 0.5)

각 상태에서 left/right를 각 0.5로 고르면, Bellman 식에서 **\(\sum_a \pi(a|s)[\cdots]\)** 를 **두 행동에 대해 평균**하면 됩니다.

---

## 페이지 17 — (Q2) 1×2 Grid World 4가지 정책 (풀이)

### Deterministic policy 네 가지 (슬라이드)

| 정책 | \(L_1\) | \(L_2\) |
|------|---------|---------|
| \(\mu_1\) | right | right |
| \(\mu_2\) | right | left  | ← **Optimal** |
| \(\mu_3\) | left  | right |
| \(\mu_4\) | left  | left  |

### 풀이 방법

1. 각 \(\mu_i\)에 대해 **선형 연립방정식** \(v = r_\pi + \gamma P_\pi v\) 풀기  
   또는 **반복 정책 평가**
2. \(v^{\mu_i}(L_1), v^{\mu_i}(L_2)\) 를 비교
3. **\(\mu_2\)** 가 두 상태 모두에서 가장 유리한지(또는 최적 정책인지) 확인

### 왜 \(\mu_2\) 가 최적인가 (직관)

- \(L_1\)에서는 **right**로 가야 \(+1\) 기회(\(\alpha\))가 있음  
- \(L_2\)에서는 **left**로 \(L_1\) 쪽으로 돌아가 **다시 +1 기회**를 노리는 것이, **right로 \(L_2\)에 갇혀 -1만 반복**하는 것보다 유리할 수 있음  

정확한 우열은 **\(\alpha, \beta, \gamma\)** 에 따라 달라지므로, 숫자를 넣어 \(v\)를 계산하는 것이 정석입니다.

### 코드로 검증

프로젝트의 `week2/grid_world_1x2.py`, `week2/bellman.py`에서 동일한 MDP로 **policy evaluation / value iteration**을 실행하면 슬라이드 결론과 비교할 수 있습니다.

---

## 부록 — 자주 쓰는 식 모음

**Bellman Expectation (벡터 형태)**  
\(v_\pi = r_\pi + \gamma P_\pi v_\pi\)  
\(\Rightarrow v_\pi = (I - \gamma P_\pi)^{-1} r_\pi\) (차원이 작을 때 해석해)

**Bellman Optimality**  
\[
v_*(s) = \max_a \left[ r(s,a) + \gamma \sum_{s'} p(s'|s,a)\, v_*(s') \right]
\]

---

## 참고

- 본 정리는 **슬라이드에 인쇄된 텍스트·표**를 기준으로 작성하였으며, **그림 전용 슬라이드**(Recycling Robot 그림, 3×4 격자 그림 등)는 일반적인 RL 교재 표기로 보완하였습니다.
- 수치 실습: `c:\reinforce\week2\` 의 `grid_world_1x2.py`, `bellman.py` 와 함께 보면 학습 효과가 큽니다.

---

*문서 끝*
