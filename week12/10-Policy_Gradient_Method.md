# Policy Gradient Method

> Prof. Tae-Hyoung Park  
> Dept. of Intelligent Systems & Robotics, CBNU  
> 원본 PDF: [`10-Policy Gradient Method.pdf`](./10-Policy%20Gradient%20Method.pdf)

---

## 목차

1. [기초 (Basics)](#1-기초-basics)
   - [1.1 Value-Based vs Policy Gradient](#11-value-based-vs-policy-gradient)
   - [1.2 Policy Function $\pi_\theta(a|s)$](#12-policy-function-pi_thetaas)
   - [1.3 Trajectory와 Objective Function $J(\theta)$](#13-trajectory와-objective-function-jtheta)
2. [Policy Gradient의 수학적 유도](#2-policy-gradient의-수학적-유도)
   - [2.1 Gradient Ascent](#21-gradient-ascent)
   - [2.2 로그 미분 트릭 (Log-Derivative Trick)](#22-로그-미분-트릭-log-derivative-trick)
   - [2.3 Monte Carlo 샘플링 적용](#23-monte-carlo-샘플링-적용)
3. [Simple Policy Gradient 구현](#3-simple-policy-gradient-구현)
   - [3.1 Policy Network 구조](#31-policy-network-구조)
   - [3.2 Agent 클래스](#32-agent-클래스)
   - [3.3 실습 Simple_pg2.py](#33-실습-simple_pg2py)
4. [REINFORCE](#4-reinforce)
   - [4.1 Simple PG의 문제점 — Noise](#41-simple-pg의-문제점--noise)
   - [4.2 $G_t$ 의 도입](#42-g_t-의-도입)
   - [4.3 실습 Reinforce2.py](#43-실습-reinforce2py)
5. [Baseline](#5-baseline)
   - [5.1 분산(Variance) 감소의 원리 — 시험 성적 비유](#51-분산variance-감소의-원리--시험-성적-비유)
   - [5.2 Baseline 적용 수식](#52-baseline-적용-수식)
   - [5.3 State Value Function을 Baseline으로](#53-state-value-function을-baseline으로)
6. [Actor-Critic](#6-actor-critic)
   - [6.1 Actor와 Critic의 역할](#61-actor와-critic의-역할)
   - [6.2 TD 에러를 이용한 업데이트](#62-td-에러를-이용한-업데이트)
   - [6.3 PolicyNet & ValueNet 구현](#63-policynet--valuenet-구현)
   - [6.4 실습 Actor_critic2.py](#64-실습-actor_critic2py)
7. [Mountain Car 환경](#7-mountain-car-환경)
8. [퀴즈 (Quiz)](#8-퀴즈-quiz)
9. [요약 — $\Phi_t$의 진화와 Value vs Policy 비교](#9-요약--phi_t의-진화와-value-vs-policy-비교)

---

## 1. 기초 (Basics)

### 1.1 Value-Based vs Policy Gradient

**강화학습의 목적:** 최적의 **policy** 를 구하는 것.

| 구분 | Value-Based Method (가치 기반) | Policy Gradient Method (정책 경사) |
|------|------------------------------|---------------------------------|
| **핵심 아이디어** | Value function (state value, action value) 을 학습/평가 후, 이를 통해 policy 개선 | Policy function 을 **직접** 파라미터화하여 학습/개선 |
| **학습 대상** | $V(s)$, $Q(s,a)$ | $\pi_\theta(a\|s)$ (policy network) |
| **정책 도출** | 학습된 Q값에 **간접적**으로 derive (예: $\arg\max_a Q$) | 신경망이 **직접** 행동 확률 출력 |
| **탐험 방식** | ε-greedy (인위적) | softmax (자연스러움) |
| **대표 알고리즘** | SARSA, Q-Learning, DQN | Policy Gradient, REINFORCE, PPO, A3C |
| **적합한 행동공간** | 이산(discrete)에 강함 | 이산 + **연속(continuous)** 모두 가능 |

```
       [ State s ]                        [ State s ]
            │                                  │
      Q network                         Policy network
            │                                  │
       Q(s,a)                           π(a|s) = [0.3, 0.7]
            │                                  │
       argmax                              sampling
            ↓                                  ↓
        action a                         action ~ π(·|s)
```

> 💡 Value-Based는 "**각 행동의 가치를 평가한 뒤 가장 좋은 것을 고르자**"는 접근.  
> Policy Gradient는 "**처음부터 행동할 확률 그 자체를 학습하자**"는 접근.

### 1.2 Policy Function $\pi_\theta(a|s)$

| 기호 | 의미 |
|------|------|
| $\pi(a \mid s)$ | policy function — state $s$ 에서 action $a$ 를 선택할 **확률** |
| $\pi_\theta(a \mid s)$ | **신경망(policy network)** 으로 구현한 policy function |
| $\theta$ | 신경망의 **weight 벡터** (= 학습 대상) |

```
        θ
         │
    ┌────┴────┐
    │ Policy  │
    │ Network │
    └────┬────┘
         ▼
   π(a|s) = [0.3, 0.7]
```

> 🎯 정책을 신경망으로 파라미터화 → "**정책을 학습한다 = $\theta$ 를 학습한다**"

### 1.3 Trajectory와 Objective Function $J(\theta)$

**Trajectory (궤적):**

$$\tau = (S_0, A_0, R_0,\; S_1, A_1, R_1,\; \cdots,\; S_T, A_T, R_T,\; S_{T+1})$$

**Return (수익):**

$$G(\tau) = R_0 + \gamma R_1 + \gamma^2 R_2 + \cdots + \gamma^T R_T$$

**Objective Function (목적 함수):**

$$\boxed{J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\bigl[\,G(\tau)\,\bigr]}$$

| 기호 | 의미 |
|------|------|
| $J(\theta)$ | policy network $\theta$ 에 대한 **기대 수익** → **최대화** |
| $\tau \sim \pi_\theta$ | 시계열 trajectory $\tau$ 가 policy 신경망 $\pi_\theta$ 로부터 **생성**됨 |
| $\mathbb{E}_{\tau \sim \pi_\theta}[\cdot]$ | 모든 가능한 trajectory에 대한 **기댓값** |

---

## 2. Policy Gradient의 수학적 유도

### 2.1 Gradient Ascent

$J(\theta)$ 를 **최대화**하는 policy network $\theta$ 를 구하는 문제 → **Gradient ascent method**

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\bigl[\,G(\tau)\,\bigr]$$

$$\boxed{\theta \leftarrow \theta + \alpha\, \nabla_\theta J(\theta)}$$

> 기울기의 **(+) 방향**으로 이동 → **local maxima** 도달

| 기호 | 의미 |
|------|------|
| $\alpha$ | learning rate (학습률) |
| $\nabla_\theta J(\theta)$ | $J$ 가 $\theta$ 에 대해 가장 빠르게 증가하는 방향 |

### 2.2 로그 미분 트릭 (Log-Derivative Trick)

$$\frac{d}{dx}\log f(x) = \frac{f'(x)}{f(x)} \quad \Rightarrow \quad \nabla_\theta f(\theta) = f(\theta)\, \nabla_\theta \log f(\theta)$$

이 트릭을 적용하면:

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} G(\tau)\, \nabla_\theta \log \pi_\theta(A_t \mid S_t)\right]}$$

| 기호 | 의미 |
|------|------|
| $\pi_\theta(A_t \mid S_t)$ | state $S_t$ 에서 action $A_t$ 를 선택할 **확률** |
| $\nabla_\theta \pi_\theta(A_t \mid S_t)$ | 그 확률의 **변화량(기울기)** |
| $\nabla_\theta \log \pi_\theta(A_t \mid S_t)$ | log 확률의 기울기 (policy gradient 핵심) |

> 💡 trajectory $\tau$ 가 좋은 결과($G(\tau) > 0$) → 그 안의 $(S_t, A_t)$ 선택 확률을 **올리자**  
> 나쁜 결과($G(\tau) < 0$) → 선택 확률을 **내리자**

### 2.3 Monte Carlo 샘플링 적용

$\nabla_\theta J(\theta)$ 를 구하는 알고리즘 — **Monte Carlo method** 적용:

- sampling 을 여러 번하여 **평균**을 구함
- Agent 를 policy $\pi_\theta$ 에 따라 행동하게 하여 **$n$ 개의 trajectory** $\tau$ 를 얻음

**$n$ 개 sample 평균:**

$$\nabla_\theta J(\theta) \approx \frac{1}{n}\sum_{i=1}^{n} \sum_{t=0}^{T} G(\tau^{(i)})\, \nabla_\theta \log \pi_\theta(A_t^{(i)} \mid S_t^{(i)})$$

**1 sample case ($n=1$):**

$$\nabla_\theta J(\theta) \approx \sum_{t=0}^{T} G(\tau)\, \nabla_\theta \log \pi_\theta(A_t \mid S_t)$$

| 기호 | 의미 |
|------|------|
| $\tau^{(i)}$ | $i$ 번째 episode의 trajectory |
| $A_t^{(i)},\, S_t^{(i)}$ | $i$ 번째 episode의 시간 $t$ 에서의 action, state |

---

## 3. Simple Policy Gradient 구현

### 3.1 Policy Network 구조

**Policy Network ($\pi_\theta$):** 2-layer NN, classification model (CartPole 예시)

```
Input:  state (4 x batch_size)
           │
           ▼
┌──────────────────────────┐
│  Linear (4 → hidden)      │
│       ReLU                │
│  Linear (hidden → 2)      │
│      softmax              │
└──────────────────────────┘
           ▼
Output: action probability  [π(a₀|s), π(a₁|s)]   (합 = 1)
```

| 항목 | 값 (CartPole) |
|------|---------------|
| 입력 | state = 4 × batch_size |
| 출력 | action_size = 2 (action probability, **softmax**) |
| 역할 | $\pi_\theta(a \mid s)$ 계산 |

```python
import dezero.functions as F
import dezero.layers as L
from dezero import Model

class Policy(Model):
    def __init__(self, action_size=2):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x))   # π_θ(a|s)
        return x
```

### 3.2 Agent 클래스

**학습 흐름:**

```
Policy network 초기화
        │
        ▼
Policy network 출력 π_θ(A_t|S_t) → action 선택
        │
        ▼
G(τ) 계산
        │
        ▼
loss = −∇_θ J(θ) = − Σ_t G(τ) log π_θ(A_t|S_t)
        │
        ▼
Policy network update
```

$$\theta^* = \arg\max_\theta J(\theta) = \arg\min_\theta \bigl(-J(\theta)\bigr)$$

```python
class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0002
        self.action_size = 2

        self.memory = []                       # (reward, prob) 누적
        self.pi = Policy(self.action_size)
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.pi)

    def get_action(self, state):
        state = state[np.newaxis, :]
        probs = self.pi(state)
        probs = probs[0]
        action = np.random.choice(len(probs), p=probs.data)
        return action, probs[action]

    def add(self, reward, prob):
        self.memory.append((reward, prob))

    def update(self):
        self.pi.cleargrads()

        G, loss = 0, 0
        for reward, prob in reversed(self.memory):
            G = reward + self.gamma * G                   # G(τ)

        for reward, prob in self.memory:
            loss += -F.log(prob) * G                      # −Σ G(τ) log π_θ

        loss.backward()
        self.optimizer.update()
        self.memory = []
```

| 코드 | 대응 수식 |
|------|----------|
| `probs = self.pi(state)` | $\pi_\theta(\cdot \mid S_t)$ |
| `np.random.choice(p=probs)` | $A_t \sim \pi_\theta(\cdot \mid S_t)$ |
| `G = reward + self.gamma * G` | $G(\tau) = R_0 + \gamma R_1 + \cdots$ |
| `loss += -F.log(prob) * G` | $\text{loss} = -\sum_t G(\tau) \log \pi_\theta(A_t\|S_t)$ |

### 3.3 실습 Simple_pg2.py

**실습 #1** — CartPole에서 Simple Policy Gradient 학습

```python
if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    agent = Agent()
    reward_history = []

    for episode in range(3000):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, prob = agent.get_action(state)
            next_state, reward, done, info = env.step(action)

            agent.add(reward, prob)
            state = next_state
            total_reward += reward

        agent.update()                     # 에피소드 종료 후 한 번에 업데이트
        reward_history.append(total_reward)
```

**실험 결과 (슬라이드):**

| Episode별 Total Reward | 100회 이동평균 |
|------------------------|----------------|
| 학습 진행에 따라 reward 증가 | 흔들림이 큰 학습 곡선 (high variance) |

> ⚠️ 학습은 되지만 **분산이 매우 크고 불안정** → REINFORCE → Baseline → Actor-Critic 으로 발전

---

## 4. REINFORCE

### 4.1 Simple PG의 문제점 — Noise

**REINFORCE** — Ronald J. Williams, 1992  
**어원:** **RE**ward **I**ncrement = **N**onnegative **F**actor × **O**ffset **R**einforcement × **C**haracter **E**ligibility

Simple PG (①):

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} \underbrace{G(\tau)}_{\text{전체 기간 } t=0\sim T \text{ 수익}} \nabla_\theta \log \pi_\theta(A_t \mid S_t)\right] \quad \cdots \;①$$

$$G(\tau) = R_0 + \gamma R_1 + \gamma^2 R_2 + \cdots + \gamma^T R_T$$

**문제점:**

- 시간 $t$ 에서의 action $A_t$ 에 **항상 일정한 가중치 $G(\tau)$** 적용 → **noise**
- action $A_t$ **이후** 발생한 수익을 가중치로 부여하는 것이 **합리적**

### 4.2 $G_t$ 의 도입

REINFORCE (②):

$$\boxed{G_t = R_t + \gamma R_{t+1} + \cdots + \gamma^{T-t} R_T}$$

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} \underbrace{G_t}_{t \text{ 이후 수익}} \nabla_\theta \log \pi_\theta(A_t \mid S_t)\right] \quad \cdots \;②$$

| | Simple PG (①) | REINFORCE (②) |
|--|---------------|---------------|
| 가중치 | $G(\tau)$ (전체 수익) | $G_t$ ($t$ **이후** 수익) |
| 샘플 ↑ | 정확한 $\nabla_\theta J$ 에 수렴 | 동일하게 수렴 |
| **분산** | **큼** (관련 없는 noise 포함) | **작음** |
| 학습 | 불안정 | ✅ **안정적이고 빠른 학습** |

### 4.3 실습 Reinforce2.py

**실습 #2** — 핵심 차이는 `update()` 메서드:

```python
def update(self):
    self.pi.cleargrads()

    G, loss = 0, 0
    for reward, prob in reversed(self.memory):
        G = reward + self.gamma * G                # G_t (역방향 누적)
        loss += -F.log(prob) * G                   # 각 t에 G_t 가중치

    loss.backward()
    self.optimizer.update()
    self.memory = []
```

**Simple_pg vs REINFORCE 학습 곡선 (슬라이드):**

```
Simple_pg     ──→  늦게 수렴, 흔들림 큼
REINFORCE     ──→  ✓ 안정적이고 빠른 학습
```

---

## 5. Baseline

### 5.1 분산(Variance) 감소의 원리 — 시험 성적 비유

**3명의 시험 성적 예시:**

| | ① 실제값 | ② 실제값 − 예측값 |
|--|---------|-----------------|
| 학생 A | 90 | +5 |
| 학생 B | 70 | −5 |
| 학생 C | 50 | −5 |
| **분산** | **466.667** | **32.667** |

> 🎯 **실제값 − 예측값** 으로 변환하면 분산이 **466.667 → 32.667** 로 급감  
> → 데이터의 분산을 줄이기 위해 **실제값과 예측값의 차이** 활용  
> → 예측값의 정확도가 높을수록 분산이 작아짐

### 5.2 Baseline 적용 수식

REINFORCE:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} G_t\, \nabla_\theta \log \pi_\theta(A_t \mid S_t)\right]$$

REINFORCE with baseline:

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} \bigl(G_t - b(S_t)\bigr)\, \nabla_\theta \log \pi_\theta(A_t \mid S_t)\right]}$$

**Proof:** $b(S_t)$ 가 $A_t$ 와 **독립**이면

$$\mathbb{E}\bigl[b(S_t)\, \nabla_\theta \log \pi_\theta(A_t \mid S_t)\bigr] = 0 \quad (t = 0 \sim T \text{ 모두 성립})$$

→ baseline을 빼도 **기울기의 평균(=$\nabla_\theta J$)은 유지**, **분산만 감소**

### 5.3 State Value Function을 Baseline으로

| baseline 후보 | 의미 |
|--------------|------|
| $b(S_t)$ | state $S_t$ 에서 지금까지 얻은 보상의 **평균** (시험성적 평균치) |
| ✅ $V_{\pi_\theta}(S_t)$ | **state value function** — 분산 감소 + 학습 효율 증가 |

> 🎓 "$V$를 어떻게 알지?" → **신경망(Critic)으로 학습** → Actor-Critic

---

## 6. Actor-Critic

### 6.1 Actor와 Critic의 역할

**Policy-Based & Value-Based (Hybrid method)**

| 구성 | 역할 | 비유 |
|------|------|------|
| **Actor (행위자)** | Agent의 action을 결정하는 **policy** 학습 | 무대 위 **배우** |
| **Critic (비평자)** | 주어진 state에서 **value function** 학습 | 객석 **평론가** |

| 패러다임 | 구성 |
|---------|------|
| Policy-based | actor only |
| Value-based | critic only |
| **Actor-Critic** | **actor + critic** |

```
        State s
           │
   ┌───────┴────────┐
   ▼                ▼
 Actor            Critic
π_θ(a|s)          V_w(s)
(policy net)    (value net)
```

### 6.2 TD 에러를 이용한 업데이트

| 기호 | 의미 |
|------|------|
| $\theta$ | **policy network (actor)** weight |
| $w$ | **value network (critic)** weight |
| $V_w(S_t)$ | Critic이 예측한 state value |

$$\begin{aligned}
\nabla_\theta J(\theta) &= \mathbb{E}\!\left[\sum_t \bigl(G_t - b(S_t)\bigr) \nabla_\theta \log \pi_\theta(A_t \mid S_t)\right] && \text{REINFORCE w/ baseline} \\
&= \mathbb{E}\!\left[\sum_t \bigl(G_t - V_w(S_t)\bigr) \nabla_\theta \log \pi_\theta(A_t \mid S_t)\right] && \text{Actor-Critic (MC)} \\
&= \mathbb{E}\!\left[\sum_t \underbrace{\bigl(R_t + \gamma V_w(S_{t+1}) - V_w(S_t)\bigr)}_{\text{TD error } \delta_t} \nabla_\theta \log \pi_\theta(A_t \mid S_t)\right] && \text{Actor-Critic (TD)}
\end{aligned}$$

| 방식 | 특징 |
|------|------|
| Monte Carlo | 에피소드 종료 후 $G_t$ 계산 |
| **TD** | 매 스텝 $R_t + \gamma V_w(S_{t+1})$ 로 학습 가능 |

### 6.3 PolicyNet & ValueNet 구현

```python
class PolicyNet(Model):                    # Actor (π_θ)
    def __init__(self, action_size=2):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x))
        return x

class ValueNet(Model):                     # Critic (V_w)
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(1)              # 스칼라 V(s)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x
```

**Loss (슬라이드):**

$$\text{loss}_v = \bigl(R_t + \gamma V_w(S_{t+1}) - V_w(S_t)\bigr)^2$$

$$\text{loss}_{\pi} = -\bigl(R_t + \gamma V_w(S_{t+1}) - V_w(S_t)\bigr)\, \log \pi_\theta(A_t \mid S_t)$$

```python
class Agent:
    def update(self, state, action_prob, reward, next_state, done):
        next_state = next_state[np.newaxis, :]
        target = reward + self.gamma * self.v(next_state) * (1 - done)
        target.unchain()

        v = self.v(state[np.newaxis, :])
        loss_v = F.mean_squared_error(v, target)

        delta = target - v                                 # TD error δ_t
        delta.unchain()
        loss_pi = -F.log(action_prob) * delta

        self.v.cleargrads();  self.pi.cleargrads()
        loss_v.backward();    loss_pi.backward()
        self.optimizer_v.update();  self.optimizer_pi.update()
```

| 코드 | 대응 |
|------|------|
| `target = R + γ·V_w(s')` | TD target $R_t + \gamma V_w(S_{t+1})$ |
| `loss_v = MSE(v, target)` | Critic loss |
| `delta = target - v` | TD error $\delta_t$ |
| `loss_pi = -log(π)·delta` | Actor loss |

### 6.4 실습 Actor_critic2.py

**실습 #3** — REINFORCE vs Actor-Critic

```python
if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    agent = Agent()
    reward_history = []

    for episode in range(3000):
        state = env.reset()
        done, total_reward = False, 0

        while not done:
            action, prob = agent.get_action(state)
            next_state, reward, done, info = env.step(action)

            agent.update(state, prob, reward, next_state, done)  # ★ 매 스텝 학습
            state = next_state
            total_reward += reward

        reward_history.append(total_reward)
```

| 알고리즘 | 학습 시점 | 특성 |
|---------|----------|------|
| REINFORCE | 에피소드 종료 후 | 보통 |
| **Actor-Critic** | **매 스텝** | ✅ 빠르고 안정적 |

```
REINFORCE      ──→  에피소드 끝난 뒤 update
Actor-Critic   ──→  매 step update (슬라이드 비교 그래프)
```

---

## 7. Mountain Car 환경

| 항목 | 내용 |
|------|------|
| **State (observation)** | `[position, velocity]` |
| **Action** | `0`: 왼쪽 가속 / `1`: 가속 안 함 / `2`: 오른쪽 가속 |
| **Reward** | 매 timestep **-1** (목표: 깃발에 **빨리** 도달) |
| **Starting state** | Position: `[-0.6, -0.4]` 균등 무작위 |
| **Episode 종료** | 카트 위치 ≥ **0.5** (깃발 도달) 또는 길이 ≥ **200** (truncation) |

🔗 [Gymnasium Mountain Car 소스](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/mountain_car.py)

> ⚠️ 보상이 매 timestep −1 이므로 **total reward는 0에 가까울수록(덜 음수) 좋음**. −200 = 200스텝 내 실패.

---

## 8. 퀴즈 (Quiz)

### (Q1) Mountain Car — DQN & Actor-Critic

**과제:** 다음 방법을 **Mountain Car**에 적용하고, **Hyper-parameter** 를 변경하여 **최대 total reward** 를 갖는 policy 를 결정하라.

1. **DQN**
2. **Actor-Critic**

**제출물 (PPT):**

1. 프로그램 소스
2. 최적 hyperparameter
3. Episode별 total reward graph
4. 최대 total reward 값 및 해당 policy 적용 시의 **동영상**

---

## 9. 요약 — $\Phi_t$의 진화와 Value vs Policy 비교

### 9.1 정책 경사법의 일반형

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} \Phi_t\, \nabla_\theta \log \pi_\theta(A_t \mid S_t)\right]}$$

차이점은 **가중치 $\Phi_t$** 에 있습니다.

| 알고리즘 | $\Phi_t$ | 설명 |
|---------|----------|------|
| **Simple Policy Gradient** | $\Phi_t = G(\tau)$ | trajectory 전체 수익 |
| **REINFORCE** | $\Phi_t = G_t$ | $t$ 이후 수익 |
| **REINFORCE with baseline** | $\Phi_t = G_t - b(S_t)$ | baseline으로 분산 감소 |
| **Actor-Critic (TD)** | $\Phi_t = R_t + \gamma V_w(S_{t+1}) - V_w(S_t)$ | TD error |
| **Advantage Actor-Critic (A2C)** | $\Phi_t = Q(S_t, A_t) - V(S_t) = A(S_t, A_t)$ | Advantage function |

### 9.2 Value-Based vs Policy-Based

| 비교 항목 | Value-Based | Policy-Based |
|----------|-------------|--------------|
| **학습 대상** | Value function $Q(s,a)$ | Policy function $\pi(a\|s)$ |
| **정책 도출** | $Q$ 로부터 **간접적** derive | **직접** policy 결정 (**효율적**) |
| **탐험** | ε-greedy | softmax |
| **대표 예제** | CartPole (DQN) | Pendulum (REINFORCE/AC) |
| **대표 알고리즘** | SARSA, Q-Learning, DQN | REINFORCE, A2C, PPO, A3C |

### 9.3 진화 흐름

```
Simple PG           ─→  Φ_t = G(τ)
    │
    ▼
REINFORCE           ─→  Φ_t = G_t
    │
    ▼
REINFORCE+Baseline  ─→  Φ_t = G_t − b(S_t)
    │
    ▼
Actor-Critic (MC)   ─→  Φ_t = G_t − V_w(S_t)
    │
    ▼
Actor-Critic (TD)   ─→  Φ_t = R_t + γV_w(S_{t+1}) − V_w(S_t)   ★
    │
    ▼
A2C                 ─→  Φ_t = A(S_t, A_t) = Q − V
```

> 🎓 Policy Gradient 의 발전사 = **"분산을 줄이면서 true gradient 평균은 유지"** 의 역사

---

> **심화 문서 (week10):**  
> - [`10-1-Policy_Gradient_Math.md`](../week10/10-1-Policy_Gradient_Math.md) — 로그 기울기 트릭 증명  
> - [`10-2-Baseline_Intuition.md`](../week10/10-2-Baseline_Intuition.md) — Baseline 직관  
> - [`10-3-Actor_Critic.md`](../week10/10-3-Actor_Critic.md) — Actor-Critic 코드 상세
