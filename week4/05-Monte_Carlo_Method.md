# Monte Carlo Method

> Prof. Tae-Hyoung Park  
> Dept. of Intelligent Systems & Robotics, CBNU

---

## 목차

1. [기초 (Basics)](#1-기초-basics)
   - 1.1 몬테카를로 방법이란?
   - 1.2 원주율 π 구하기
   - 1.3 주사위 눈의 합
   - 1.4 분포 모델 vs 샘플 모델
   - 1.5 몬테카를로 업데이트 공식
2. [몬테카를로 예측 (Monte Carlo Prediction)](#2-몬테카를로-예측-monte-carlo-prediction)
   - 2.1 상태 가치 함수
   - 2.2 계산 효율성
   - 2.3 정책 평가 알고리즘
3. [정책 평가 실습 — GridWorld](#3-정책-평가-실습--gridworld)
   - 3.1 GridWorld 환경
   - 3.2 RandomAgent
   - 3.3 실습 mc_eval.py
4. [몬테카를로 제어 (Monte Carlo Control)](#4-몬테카를로-제어-monte-carlo-control)
   - 4.1 정책 개선 정리
   - 4.2 행동 가치 함수 (Q-function)
   - 4.3 McAgent 클래스
   - 4.4 update()
5. [핵심 기법](#5-핵심-기법)
   - 5.1 ε-Greedy 정책
   - 5.2 지수 이동 평균 (Exponential Moving Average)
6. [실습 mc_control.py](#6-실습-mc_controlpy)
7. [퀴즈 (Quiz)](#7-퀴즈-quiz)
8. [요약 — DP vs MC](#8-요약--dp-vs-mc)

---

## 1. 기초 (Basics)

### 1.1 몬테카를로 방법이란?

| 항목 | 내용 |
|------|------|
| **정의** | 반복적인 무작위 샘플링(Repeated Random Sampling)에 의존하는 계산 알고리즘 |
| **어원** | 모나코 공국의 도박 도시 **Monte Carlo Casino** |
| **핵심 아이디어** | 수식으로 정확히 계산하기 어려운 값을 무작위 시뮬레이션을 수없이 반복한 결과의 평균으로 **근사** |

### 1.2 원주율 π 구하기

공식 없이 무작위 샘플링만으로 π를 추정할 수 있다.

```
1 × 1 정사각형 안에 반지름 1인 부채꼴(1/4 원)을 그린다.
무작위 점을 n개 찍는다.
원점에서 거리 ≤ 1 인 점(부채꼴 안)의 개수를 r로 센다.
```

#### 왜 (부채꼴 넓이)/(정사각형 넓이) ≈ (안에 들어간 점 개수)/(전체 점 개수) 인가?

**왼쪽 (기하학):**  
- 정사각형 한 변이 1이면 넓이는 1×1 = **1**.  
- 반지름 1인 원의 넓이는 π×1² = **π**. 그중 **1/4만** 쓰면(첫 사분면 부채꼴) 넓이는 **π/4**.  
- 따라서 **이론적인 면적 비** = (부채꼴 넓이) ÷ (정사각형 넓이) = **(π/4) ÷ 1 = π/4**.

**오른쪽 (몬테카를로):**  
- 정사각형 안에 점을 **균일하게** 무작위로 찍으면, **어느 작은 영역에 들어갈 확률**은 그 영역의 **넓이 비율**과 같다.  
- 그래서 “부채꼴 안에 들어간 점의 비율” **r/n**은, 점을 아주 많이 찍을수록 **진짜 면적 비 π/4**에 가까워진다.  
- 정리하면: **π/4 ≈ r/n** 이므로 **π ≈ 4×(r/n)** = **4r/n**.

(기호만으로 쓰면: 이론 비 = π/4, 샘플로 얻는 비 ≈ r/n, 둘을 연결한 것이 **≈** 이다.)

#### 수식 (미리보기에서 보기 좋은 형태)

$$\frac{\text{부채꼴 넓이}}{\text{정사각형 넓이}} = \frac{\pi/4}{1} = \frac{\pi}{4}, \qquad
\frac{\text{부채꼴 안 점 비율}}{\text{전체 점}} = \frac{r}{n} \approx \frac{\pi}{4}$$

$$\therefore \quad \pi \approx \frac{4r}{n}$$

- 점을 많이 찍을수록 π에 **수렴**한다.
- 정확한 공식 없이 **경험(샘플)만으로 근사**하는 MC의 핵심 원리를 보여준다.

#### `.md` 파일에서 `$$` · `\frac` 은?

- **메모장·일반 텍스트**로 열면: `$$`, `\frac{...}{...}` 같은 **기호가 그대로** 보인다. “특수문자 없이 예쁜 분수”로 바뀌지는 **않는다**.
- **Cursor / VS Code 미리보기**, **GitHub**, **Obsidian**, **Typora** 등은 보통 **LaTeX 수식을 렌더링**해서 분수·그리스 문자로 보여 준다.
- 수식 없이 읽고 싶다면 위 **“왼쪽(기하학) / 오른쪽(몬테카를로)”** 한글 단락만 보면 된다.

### 1.3 주사위 눈의 합

두 개의 주사위를 던질 때 나오는 눈의 합 분포:

| 합 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|----|---|---|---|---|---|---|---|---|----|----|----|
| 경우의 수 | 1 | 2 | 3 | 4 | 5 | **6** | 5 | 4 | 3 | 2 | 1 |
| 확률 | 1/36 | 2/36 | 3/36 | 4/36 | 5/36 | **6/36** | 5/36 | 4/36 | 3/36 | 2/36 | 1/36 |

$$E[\text{합}] = \sum_{k=2}^{12} k \cdot P(k) = 7.0$$

### 1.4 분포 모델 vs 샘플 모델

| 구분 | 분포 모델 (Distribution Model) | 샘플 모델 (Sample Model) |
|------|-------------------------------|--------------------------|
| 방법 | 모든 경우의 수를 수식으로 계산 | 컴퓨터로 실제 주사위를 반복 시뮬레이션 |
| 결과 | 정확한 확률 분포 | 시행 횟수가 늘수록 참값에 수렴 |
| 필요 정보 | 환경의 완전한 확률 모델 필요 | 환경 모델 불필요 (모델-프리) |
| MC 방식 | ✗ | ✓ |

### 1.5 몬테카를로 업데이트 공식

> 📄 이 절의 **심화 설명(메모리 절약 · 실시간 학습 이유)** → [05-1-증분방식의_장점.md](./05-1-증분방식의_장점.md)

**일반 평균:**

$$V_n = \frac{s_1 + s_2 + \cdots + s_n}{n}$$

**증분 방식 (Incremental Update):**

$$\boxed{V_n = V_{n-1} + \frac{1}{n}(s_n - V_{n-1})}$$

- 매번 처음부터 다시 계산할 필요 없이, 새 샘플이 올 때마다 **이전 평균을 조금씩 수정**한다.
- `오차(error) = s_n - V_{n-1}`: 새 샘플과 현재 추정값의 차이만큼 업데이트
- 1,000번 시뮬레이션 시 기댓값 **≈ 6.96** (참값 7.0에 근사)

---

## 2. 몬테카를로 예측 (Monte Carlo Prediction)

> 📄 PDF **6페이지** (상태 가치 함수 정의 · 슬라이드 에피소드 예시 · DP 비교) 상세 → [05-2-Monte_Carlo_Prediction_p6.md](./05-2-Monte_Carlo_Prediction_p6.md)

### 2.1 상태 가치 함수

$$\boxed{v_\pi(s) = \mathbb{E}[G \mid s]}$$

- **정책 π에 따라 행동**하는 경우, 상태 s에서 출발하여 얻을 수 있는 **기대 수익(return)**
- 수익 G는 할인된 미래 보상의 합: $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots$

**MC 방법으로 추정:**

$$V_\pi(s) \approx \frac{G^{(1)} + G^{(2)} + \cdots + G^{(n)}}{n}$$

- $G^{(n)}$: n번째 에피소드에서 상태 s를 방문했을 때 얻은 실제 수익
- 에피소드를 많이 경험할수록 참값에 수렴

**DP vs MC 비교:**

| | Dynamic Programming | Monte Carlo |
|--|---------------------|-------------|
| 방식 | Computing (계산) | Learning (학습) |
| 환경 모델 | 필요 | 불필요 |
| 특징 | 벨만 방정식으로 직접 계산 | 실제 에피소드 경험으로 추정 |

### 2.2 계산 효율성

> 📄 PDF **7페이지** (비효율 vs 역방향 점화식·$G=R+\gamma G_{\text{next}}$) 상세 → [05-3-Monte_Carlo_Prediction_p7.md](./05-3-Monte_Carlo_Prediction_p7.md)

에피소드 A → B → C → (종료) 가 있을 때:

**비효율적 방법** — 각 상태에서 독립적으로 G를 처음부터 계산:

```
G_C = R_C
G_B = R_B + γ·R_C + γ²·... (중복 계산 발생)
G_A = R_A + γ·R_B + γ²·R_C + ... (더 많은 중복 계산)
```

**효율적 방법** — **역방향(reversed)** 계산:

$$G_C = R_C$$
$$G_B = R_B + \gamma \cdot G_C \quad \leftarrow \text{이미 계산한 } G_C \text{ 재사용}$$
$$G_A = R_A + \gamma \cdot G_B \quad \leftarrow \text{이미 계산한 } G_B \text{ 재사용}$$

```python
G = 0.0
for (state, action, reward) in reversed(memory):   # 역방향 순회
    G = gamma * G + reward                          # 점화식으로 누적
    V[state] += (G - V[state]) / cnts[state]        # 증분 업데이트
```

### 2.3 정책 평가 알고리즘

> 📄 PDF **8페이지** (Initialize · 에피소드 생성 · Returns(s) · First-Visit MC) 상세 → [05-4-Monte_Carlo_Policy_Evaluation_p8.md](./05-4-Monte_Carlo_Policy_Evaluation_p8.md)

$$\boxed{V_n(s) = V_{n-1}(s) + \frac{1}{n}\bigl(G^{(n)} - V_{n-1}(s)\bigr)}$$

---

## 3. 정책 평가 실습 — GridWorld

### 3.1 GridWorld 환경

3×4 격자 환경:

```
[ ][ ][ ][ G +1.0 ]
[ ][ # ][ ][ T -1.0 ]
[ ][ ][ ][ ]
```

| 기호 | 의미 |
|------|------|
| `G` | 목표(Goal), 보상 +1.0 |
| `T` | 함정(Trap), 보상 −1.0 |
| `#` | 벽(Wall), 이동 불가 |

**주요 메서드:**

```python
env.reset()           # 에이전트를 시작 위치로 초기화, 시작 state 반환
env.step(action)      # action 수행 → (next_state, reward, done) 반환
env.render_v(V, pi)   # V(s)와 정책 시각화 (matplotlib 히트맵)
```

행동 공간: `{0: UP, 1: DOWN, 2: LEFT, 3: RIGHT}`

### 3.2 RandomAgent

```python
class RandomAgent:
    def __init__(self):
        self.gamma = 0.9
        self.action_size = 4
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)  # 균등 무작위 정책
        self.V   = defaultdict(lambda: 0.0)            # 가치 함수
        self.cnts = defaultdict(lambda: 0)             # 방문 횟수 N(s)
        self.memory = []                               # 에피소드 기록

    def get_action(self, state):
        """π(s)에 따라 행동 샘플링"""

    def add(self, state, action, reward):
        """한 스텝 (s, a, r)을 memory에 저장"""

    def eval(self):
        """에피소드 종료 후 역방향으로 V(s) 증분 업데이트"""
        G = 0.0
        for (state, action, reward) in reversed(self.memory):
            G = self.gamma * G + reward
            self.cnts[state] += 1
            self.V[state] += (G - self.V[state]) / self.cnts[state]
            #             ↑ 증분 공식: Vₙ = Vₙ₋₁ + 1/n · (G − Vₙ₋₁)
```

- `No computation model → Learning`: 환경의 전이 확률을 전혀 사용하지 않음
- 에피소드가 끝날 때(`done`)만 `eval()` 호출

### 3.3 실습 mc_eval.py

```python
if __name__ == "__main__":
    env   = GridWorld()
    agent = RandomAgent()

    episodes = 1000
    for episode in range(episodes):
        state = env.reset()
        agent.reset()

        while True:
            action                    = agent.get_action(state)
            next_state, reward, done  = env.step(action)
            agent.add(state, action, reward)

            if done:
                agent.eval()   # MC 가치 함수 갱신
                break

            state = next_state

    # V(s) 격자 히트맵 시각화 (슬라이드 스타일)
    env.render_v(agent.V)
```

**결과:** 랜덤 정책으로도 1,000 에피소드 후 DP로 계산한 참값에 근접한 V(s) 획득

---

## 4. 몬테카를로 제어 (Monte Carlo Control)

### 4.1 정책 개선 정리 (Policy Improvement Theorem)

**목표:** 가치 평가(Evaluation)와 정책 개선(Improvement)을 반복하여 **최적 정책 π\*** 도달

$$q \rightarrow q_\pi, \quad \pi \rightarrow \text{greedy}(q)$$

탐욕 정책(Greedy Policy):

$$\mu(s) = \arg\max_a q(s, a)$$

**정책 개선 정리** — greedy 정책 $\pi_{k+1}$은 이전 정책 $\pi_k$보다 항상 같거나 낫다:

$$q_{\pi_k}(s, \mu_{k+1}(s)) = q_{\pi_k}(s, \arg\max_a q_{\pi_k}(s,a))
= \max_a q_{\pi_k}(s,a) \geq q_{\pi_k}(s,a) \geq v_{\pi_k}(s)$$

### 4.2 행동 가치 함수 (Q-function)

**V-함수 대신 Q-함수를 쓰는 이유:**

- V(s)만으로는 "어떤 행동이 좋은가"를 바로 알 수 없음 (환경 모델 필요)
- Q(s, a)는 (상태, 행동) 쌍의 가치를 직접 저장 → **모델 없이 greedy 정책 도출 가능**

**Q-함수 정의:**

$$Q_n(s,a) = \frac{G^{(1)} + G^{(2)} + \cdots + G^{(n)}}{n}$$

**증분 업데이트:**

$$\boxed{Q_n(s,a) = Q_{n-1}(s,a) + \frac{1}{n}\bigl\{G^{(n)} - Q_{n-1}(s,a)\bigr\}}$$

- $G^{(n)}$: n번째 에피소드에서 (s, a) 방문 후 얻은 실제 수익

### 4.3 McAgent 클래스

```python
class McAgent:
    def __init__(self):
        self.gamma      = 0.9
        self.epsilon    = 0.1   # ε-greedy 탐색률
        self.alpha      = 0.1   # 고정 step-size (지수 이동 평균용)
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi    = defaultdict(lambda: random_actions)  # 초기: 균등 정책
        self.Q     = defaultdict(lambda: 0.0)             # Q(s, a)
        self.cnts  = defaultdict(lambda: 0)               # 방문 횟수
        self.memory = []
```

### 4.4 update()

```python
def update(self):
    G = 0.0
    for (state, action, reward) in reversed(self.memory):
        G = self.gamma * G + reward

        key = (state, action)
        self.cnts[key] += 1

        # Q 함수 업데이트 (고정 α 방식)
        self.Q[key] += (G - self.Q[key]) * self.alpha

        # ε-greedy로 정책 개선
        self.pi[state] = greedy_probs(self.Q, state, self.epsilon, self.action_size)
```

**정책 표현:**

$$\pi(a \mid s) = \begin{cases} 1 & a = \mu(s) = \arg\max_a Q(s,a) \\ 0 & \text{otherwise} \end{cases} \quad \text{(순수 greedy)}$$

ε-greedy를 적용하면 0이 아닌 작은 확률 ε/|A|가 부여됨.

---

## 5. 핵심 기법

### 5.1 ε-Greedy 정책

**문제:** 탐욕적(greedy) 정책만 사용하면 아직 방문하지 않은 더 좋은 경로를 **영영 발견하지 못할** 수 있다.  
→ **탐험(Exploration) vs 활용(Exploitation) 딜레마**

**해결:** 작은 확률 ε로 **무작위 행동**을 섞는다.

$$\pi(a \mid s) = \begin{cases} 1 - \varepsilon + \dfrac{\varepsilon}{|A|} & a = \arg\max_{a'} Q(s,a') \\ \dfrac{\varepsilon}{|A|} & \text{otherwise} \end{cases}$$

**예시 (|A| = 4, max_action = 1):**

| 방식 | a=0 | **a=1 (최적)** | a=2 | a=3 |
|------|-----|--------------|-----|-----|
| Pure greedy | 0.0 | **1.0** | 0.0 | 0.0 |
| ε-greedy (ε=0.4) | 0.1 | **0.7** | 0.1 | 0.1 |

```python
def greedy_probs(Q, state, epsilon, action_size=4):
    qs         = [Q[(state, a)] for a in range(action_size)]
    max_action = int(np.argmax(qs))
    base_prob  = epsilon / action_size
    probs      = {a: base_prob for a in range(action_size)}
    probs[max_action] += (1.0 - epsilon)
    return probs
```

### 5.2 지수 이동 평균 (Exponential Moving Average)

**문제:** 학습이 진행될수록 정책이 계속 바뀐다.  
초기에 무작위 정책으로 수집한 데이터와 나중에 좋은 정책으로 수집한 데이터를 **동등하게** 취급하는 것은 비효율적.

**기존 증분 방식 (1/n 가중치):**

$$Q_n(s,a) = Q_{n-1}(s,a) + \frac{1}{n}\bigl\{G^{(n)} - Q_{n-1}(s,a)\bigr\}$$

$$\text{가중치}: \quad G^{(1)}, G^{(2)}, \ldots, G^{(n)} \; \text{모두 동일하게} \; \frac{1}{n}$$

**고정 α 방식 (지수 이동 평균):**

$$\boxed{Q_n(s,a) = Q_{n-1}(s,a) + \alpha\bigl\{G^{(n)} - Q_{n-1}(s,a)\bigr\}}$$

$$\text{가중치}: \quad G^{(1)}: \alpha(1-\alpha)^{n-1}, \quad G^{(2)}: \alpha(1-\alpha)^{n-2}, \quad \ldots, \quad G^{(n)}: \alpha$$

| 데이터 | 가중치 크기 |
|--------|------------|
| $G^{(1)}$ (오래된 데이터) | $\alpha^{n-1}$ → 매우 작음 (거의 잊혀짐) |
| $G^{(n)}$ (최신 데이터) | $\alpha$ → 가장 큼 |

- **최신 데이터의 가중치가 exponential하게 더 크다**
- α = 0.1 사용 시, 오래된 샘플의 영향은 급격히 감소

---

## 6. 실습 mc_control.py

### 전체 흐름

```python
if __name__ == "__main__":
    env   = GridWorld()
    agent = McAgent()

    episodes = 10000
    for episode in range(episodes):
        state = env.reset()
        agent.reset()

        while True:
            action                    = agent.get_action(state)
            next_state, reward, done  = env.step(action)
            agent.add(state, action, reward)

            if done:
                agent.update()   # Q 갱신 + 정책 개선
                break

            state = next_state

    # Q 마름모 시각화 + greedy policy 화살표 (2패널)
    from common.gridworld_render import Renderer
    renderer = Renderer(env.reward_map, env.goal_state, env.wall_state)
    renderer.render_q_and_policy(agent.Q)
```

### 시각화 결과 (두 패널)

| 왼쪽 패널: Q 함수 시각화 | 오른쪽 패널: Greedy Policy |
|--------------------------|---------------------------|
| 각 셀을 4등분 삼각형으로 분할 | 최적 행동을 화살표(↑↓←→)로 표시 |
| 삼각형 색상: 녹색(높은 Q) ↔ 적색(낮은 Q) | 벽: 회색 |
| 각 삼각형에 Q(s,a) 수치 표시 | R 1.0 (GOAL) / R -1.0 라벨 |

### RandomAgent vs McAgent 비교

| 항목 | RandomAgent (mc_eval) | McAgent (mc_control) |
|------|----------------------|----------------------|
| 목적 | V(s) 평가 (Policy Evaluation) | 최적 정책 탐색 (Policy Control) |
| 정책 | 고정 균등 무작위 | ε-greedy로 계속 개선 |
| 가치 함수 | V(s) | Q(s, a) |
| 업데이트 | 1/n 증분 | 고정 α (지수 이동 평균) |
| epsilon | 없음 | 0.1 |
| alpha | 없음 | 0.1 |
| 에피소드 수 | 1,000 | 10,000 |

---

## 7. 퀴즈 (Quiz)

> **[Q]** Monte Carlo Method를 적용하여 **5×5 Grid World**에 대한 Q함수를 시각화하고 policy를 구하라.

**환경 설정:**

```
5×5 격자
+1 : 사과 (보상)
-1 : 폭탄 (패널티, 2개)
```

**실험 과제:**

1. 파라미터 **ε (epsilon)** 을 변경하며 결과 비교
   - ε ↑ → 탐험 증가, 수렴 느릴 수 있으나 더 다양한 경로 발견
   - ε ↓ → 활용 증가, 조기 수렴하나 지역 최적에 빠질 위험

2. 파라미터 **α (alpha)** 를 변경하며 결과 비교
   - α ↑ → 최신 데이터 반영 빠름, 불안정할 수 있음
   - α ↓ → 안정적이나 수렴 느림

---

## 8. 요약 — DP vs MC

| 비교 항목 | Dynamic Programming (DP) | Monte Carlo (MC) |
|-----------|--------------------------|-----------------|
| **핵심 방식** | Computing (수식 계산) | Learning (경험으로 학습) |
| **계산 모델 필요** | ✅ 필요 | ❌ 불필요 |
| **차원의 저주** | ⚠️ 발생 가능 | ✅ 회피 가능 |
| **적용 가능 에피소드** | 일회성 ✅ / 지속성 ✅ | 일회성 ✅ / 지속성 ❌ |
| **마르코프 성질 필요** | ✅ 필요 | ❌ 불필요 |
| **벨만 방정식** | 사용 | 미사용 |
| **대표 알고리즘** | Policy/Value Iteration | MC Prediction / MC Control |

### 직관적 비교

| | DP | MC |
|-|----|----|
| 비유 | 지도와 지형 정보를 모두 갖추고 **머리로 최적 경로를 계산** | 아무 정보 없이 **직접 수백 번 부딪혀보며 체득** |
| 필요한 것 | 환경 전이 확률 $p(s',r \mid s,a)$ | 에피소드 경험만으로 충분 |
| 한계 | 상태 공간이 클 때 계산 폭발 | 반드시 **종료하는(Episodic) 환경**이어야 함 |

### Policy Evaluation / Control 비교

```
DP:   π → V_π (Bellman Expectation Eq.)  →  π' (greedy)  →  π* 
MC:   π → V_π (에피소드 평균)             →  π' (ε-greedy) →  π*
```

두 방법 모두 **Generalized Policy Iteration (GPI)** 구조를 따른다.

---

> **참고 파일:** `mc_eval.py` (정책 평가 실습), `mc_control.py` (정책 제어 실습)
