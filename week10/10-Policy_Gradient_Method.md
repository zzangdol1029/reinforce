# Policy Gradient Method

> Prof. Tae-Hyoung Park  
> Dept. of Intelligent Systems & Robotics, CBNU

---

## 목차

1. [기초 (Basics)](#1-기초-basics)
   - 1.1 Value-Based vs Policy Gradient
   - 1.2 Policy Function $\pi_\theta(a|s)$
   - 1.3 Trajectory와 Objective Function $J(\theta)$
2. [Policy Gradient의 수학적 유도](#2-policy-gradient의-수학적-유도)
   - 2.1 Gradient Ascent
   - 2.2 로그 미분 트릭 (Log-Derivative Trick)
   - 2.3 Monte Carlo 샘플링 적용
3. [Simple Policy Gradient 구현](#3-simple-policy-gradient-구현)
   - 3.1 Policy Network 구조
   - 3.2 Agent 클래스
   - 3.3 실습 simple_pg.py
4. [REINFORCE](#4-reinforce)
   - 4.1 Simple PG의 문제점 — Noise
   - 4.2 $G_t$ 의 도입
   - 4.3 실습 reinforce.py
5. [Baseline](#5-baseline)
   - 5.1 분산(Variance) 감소의 원리 — 시험 성적 비유
   - 5.2 Baseline 적용 수식
   - 5.3 State Value Function을 Baseline으로
6. [Actor-Critic](#6-actor-critic)
   - 6.1 Actor와 Critic의 역할
   - 6.2 TD 에러를 이용한 업데이트
   - 6.3 PolicyNet & ValueNet 구현
   - 6.4 실습 actor_critic.py
7. [퀴즈 (Quiz)](#7-퀴즈-quiz)
8. [요약 — Φₜ의 진화와 Value vs Policy 비교](#8-요약--φₜ의-진화와-value-vs-policy-비교)

---

> 📄 **심화 문서 링크**
> - [`10-1-Policy_Gradient_Math.md`](./10-1-Policy_Gradient_Math.md) — 로그 기울기 트릭과 $\nabla_\theta J(\theta)$ 의 단계별 증명
> - [`10-2-Baseline_Intuition.md`](./10-2-Baseline_Intuition.md) — Baseline & 분산 감소 원리 (시험 성적 비유)
> - [`10-3-Actor_Critic.md`](./10-3-Actor_Critic.md) — Actor-Critic 아키텍처와 DeZero 코드 상세 해설

---

## 1. 기초 (Basics)

### 1.1 Value-Based vs Policy Gradient

강화학습의 **궁극적인 목표**는 단 하나입니다 — **최적의 정책 $\pi^*$ 을 찾는 것**.  
하지만 그 목표에 도달하는 **길**은 두 가지로 갈립니다.

| 구분 | Value-Based Method (가치 기반) | Policy Gradient Method (정책 경사) |
|------|------------------------------|---------------------------------|
| **학습 대상** | Value function ($V(s)$ 또는 $Q(s,a)$) | Policy function $\pi(a\|s)$ 자체 |
| **정책 도출** | 학습된 Q값에 **간접적**으로 derive (예: $\arg\max_a Q$) | 신경망이 **직접** 행동 확률을 출력 |
| **출력 예시** | $Q(s, a) = $ 0.3, 0.7, ... (스칼라 값) | $\pi(a\|s) = $ [0.3, 0.7] (확률 분포) |
| **탐험 방식** | ε-greedy (인위적) | softmax (자연스러움) |
| **대표 알고리즘** | SARSA, Q-Learning, DQN, Rainbow | REINFORCE, Actor-Critic, PPO, A3C |
| **적합한 행동공간** | 이산(discrete)에 강함 | 이산 + **연속(continuous)** 모두 가능 |

> 💡 **직관:** Value-Based는 "**각 행동의 가치를 평가한 뒤 가장 좋은 것을 고르자**"는 접근.  
> Policy Gradient는 "**처음부터 행동할 확률 그 자체를 학습하자**"는 접근.  
> 후자는 연속 행동 공간(예: 로봇 팔의 각도 -1.57~+1.57)에서 특히 강력합니다.

```
       [ State s ]                        [ State s ]
            │                                  │
        Q network                         Policy network
            │                                  │
       Q(s,a₁)=0.3                       π(a₁|s) = 0.3
       Q(s,a₂)=0.7                       π(a₂|s) = 0.7
            │                                  │
       argmax                              sampling
            ↓                                  ↓
        action a₂                        action ~ π(·|s)
```

### 1.2 Policy Function $\pi_\theta(a|s)$

| 기호 | 의미 |
|------|------|
| $\pi(a \mid s)$ | 상태 $s$ 에서 행동 $a$ 를 선택할 **확률** (정책 함수) |
| $\pi_\theta(a \mid s)$ | **신경망(policy network)** 으로 구현한 정책 함수 |
| $\theta$ | 신경망의 **weight 벡터** (= 학습 대상) |

> 🎯 **핵심:** 정책을 신경망으로 파라미터화함으로써, "정책을 학습한다 = $\theta$ 를 학습한다" 가 됩니다.

### 1.3 Trajectory와 Objective Function $J(\theta)$

**Trajectory (궤적):** 한 에피소드 동안 에이전트가 겪은 (상태, 행동, 보상)의 시계열입니다.

$$\tau = (S_0, A_0, R_0,\; S_1, A_1, R_1,\; \cdots,\; S_T, A_T, R_T,\; S_{T+1})$$

**Return (수익):** trajectory 전체에서 받은 할인 보상 합.

$$G(\tau) = R_0 + \gamma R_1 + \gamma^2 R_2 + \cdots + \gamma^T R_T$$

**Objective Function (목적 함수):**

$$\boxed{J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\bigl[\,G(\tau)\,\bigr]}$$

| 기호 | 의미 |
|------|------|
| $\tau \sim \pi_\theta$ | trajectory $\tau$ 가 정책 신경망 $\pi_\theta$ 에 의해 **샘플링** 됨 |
| $\mathbb{E}_{\tau \sim \pi_\theta}[\cdot]$ | 모든 가능한 trajectory에 대한 **기댓값** |
| $J(\theta)$ | 정책 $\pi_\theta$ 를 따랐을 때 **기대되는 총 수익** |

> 💡 **직관:** $J(\theta)$ 는 "내 정책이 평균적으로 얼마나 잘하느냐"를 측정하는 **시험 점수의 평균값**입니다.  
> 우리의 목표는 이 평균 점수를 **최대화**하는 $\theta$ 를 찾는 것.

---

## 2. Policy Gradient의 수학적 유도

> 📄 이 절의 **자세한 증명 과정** (로그 미분 트릭, $p(\tau\|\theta)$의 분해 등) → [`10-1-Policy_Gradient_Math.md`](./10-1-Policy_Gradient_Math.md)

### 2.1 Gradient Ascent

$J(\theta)$ 를 **최대화**해야 하므로 (보통 deep learning은 loss를 minimize 하지만, 여기서는 reward를 maximize), **gradient ascent**를 사용합니다.

$$\boxed{\theta \leftarrow \theta + \alpha\, \nabla_\theta J(\theta)}$$

> 📈 기울기의 **(+) 방향**으로 조금씩 이동 → local maxima 에 도달.

| 기호 | 의미 |
|------|------|
| $\alpha$ | learning rate (학습률) |
| $\nabla_\theta J(\theta)$ | $J$ 가 $\theta$ 에 대해 가장 빠르게 증가하는 방향 |

### 2.2 로그 미분 트릭 (Log-Derivative Trick)

문제는 $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G(\tau)]$ 의 기울기를 직접 구하기 어렵다는 점입니다.  
**왜?** 기댓값의 분포 자체가 $\theta$ 에 의존하기 때문.

이때 사용하는 핵심 도구가 **로그 미분 트릭**입니다.

$$\nabla_\theta \log f(\theta) = \frac{\nabla_\theta f(\theta)}{f(\theta)} \quad \Rightarrow \quad \nabla_\theta f(\theta) = f(\theta)\, \nabla_\theta \log f(\theta)$$

이 트릭을 적용한 결과:

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} G(\tau)\, \nabla_\theta \log \pi_\theta(A_t \mid S_t)\right]}$$

| 기호 | 의미 |
|------|------|
| $\nabla_\theta \log \pi_\theta(A_t\|S_t)$ | 상태 $S_t$ 에서 행동 $A_t$ 를 선택할 **로그 확률의 기울기** |
| $G(\tau)$ | 그 trajectory 전체에서 얻은 수익 (가중치 역할) |

> 💡 **직관적 의미:**  
> "trajectory $\tau$ 가 좋은 결과($G(\tau) > 0$)를 냈다면 → 그 trajectory 안의 **모든 $(S_t, A_t)$ 의 선택 확률을 올리자**"  
> "나쁜 결과($G(\tau) < 0$)를 냈다면 → **선택 확률을 낮추자**"  
>  
> $\nabla_\theta \log \pi_\theta(A_t\|S_t)$ 는 "$A_t$ 의 확률을 올리려면 $\theta$ 를 어느 방향으로 움직여야 하는지"를 알려줍니다.

### 2.3 Monte Carlo 샘플링 적용

기댓값 $\mathbb{E}_{\tau \sim \pi_\theta}[\cdot]$ 를 **해석적으로** 계산하는 것은 사실상 불가능합니다.  
→ **Monte Carlo 방법** 적용: 정책 $\pi_\theta$ 로 직접 행동시켜서 **$n$ 개의 trajectory** 를 얻고 평균.

**$n$ 개 sample 평균:**

$$\nabla_\theta J(\theta) \approx \frac{1}{n}\sum_{i=1}^{n} \sum_{t=0}^{T} G(\tau^{(i)})\, \nabla_\theta \log \pi_\theta(A_t^{(i)} \mid S_t^{(i)})$$

**1 sample case ($n=1$, 가장 간단한 형태):**

$$\nabla_\theta J(\theta) \approx \sum_{t=0}^{T} G(\tau)\, \nabla_\theta \log \pi_\theta(A_t \mid S_t)$$

| 기호 | 의미 |
|------|------|
| $\tau^{(i)}$ | $i$ 번째 episode의 trajectory |
| $A_t^{(i)},\, S_t^{(i)}$ | $i$ 번째 episode의 시간 $t$ 에서의 action, state |

---

## 3. Simple Policy Gradient 구현

### 3.1 Policy Network 구조

CartPole 환경을 예로 들면:

```
Input:  state vector (dim=4)  ──┐
                                ▼
                    ┌──────────────────────┐
                    │  Linear (4 → 128)     │
                    │       ReLU            │
                    │  Linear (128 → 2)     │
                    │      softmax          │
                    └──────────────────────┘
                                ▼
Output: action probability  [π(a₀|s), π(a₁|s)]   (합 = 1)
```

| 항목 | 값 |
|------|-----|
| 입력 | state vector (CartPole: 4차원) |
| 은닉층 | Linear (128 units) + ReLU |
| 출력 | action_size = 2 (left, right) |
| 출력 활성화 | **softmax** → 확률 분포 |

> 🎯 **softmax의 의미:** 출력값들이 항상 0~1 사이이고 합이 1 이므로, 그대로 **행동 확률 분포**로 사용 가능.

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
        x = F.softmax(self.l2(x))   # action probability π_θ(a|s)
        return x
```

### 3.2 Agent 클래스

```python
class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0002
        self.action_size = 2

        self.memory = []                       # (reward, prob) 누적
        self.pi = Policy(self.action_size)     # policy network π_θ
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.pi)

    def get_action(self, state):
        state = state[np.newaxis, :]
        probs = self.pi(state)                 # π_θ(a|s) 출력
        probs = probs[0]
        action = np.random.choice(len(probs),
                                  p=probs.data) # 확률 분포에서 sampling
        return action, probs[action]

    def add(self, reward, prob):
        self.memory.append((reward, prob))     # (R_t, π_θ(A_t|S_t)) 저장

    def update(self):
        self.pi.cleargrads()

        G, loss = 0, 0
        for reward, prob in reversed(self.memory):
            G = reward + self.gamma * G                   # G(τ) 역방향 계산

        for reward, prob in self.memory:
            loss += -F.log(prob) * G   # ← loss = -Σ G(τ)·log π_θ(A_t|S_t)

        loss.backward()
        self.optimizer.update()
        self.memory = []
```

#### 코드와 수식 매핑

| 코드 라인 | 대응 수식 |
|-----------|----------|
| `probs = self.pi(state)` | $\pi_\theta(\cdot \mid S_t)$ 계산 |
| `np.random.choice(p=probs)` | $A_t \sim \pi_\theta(\cdot \mid S_t)$ 샘플링 |
| `G = reward + self.gamma * G` | $G(\tau) = R_0 + \gamma R_1 + \gamma^2 R_2 + \cdots$ |
| `loss += -F.log(prob) * G` | $\text{loss} = -\sum_t G(\tau) \log \pi_\theta(A_t\|S_t)$ |
| `loss.backward(); update()` | $\theta \leftarrow \theta + \alpha\,\nabla_\theta J(\theta)$ |

> ⚠️ **왜 loss 앞에 마이너스(-)가 있는가?**  
> Deep learning 프레임워크는 기본적으로 **minimize**(손실 최소화)를 합니다.  
> 우리는 $J(\theta)$ 를 maximize 해야 하므로:
> $$\theta^* = \arg\max_\theta J(\theta) = \arg\min_\theta \bigl(-J(\theta)\bigr)$$
> 즉, **`loss = -J(θ)`** 로 정의하고 minimize하면 됩니다.

### 3.3 실습 simple_pg.py

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

            agent.add(reward, prob)        # (R_t, π_θ(A_t|S_t)) 저장
            state = next_state
            total_reward += reward

        agent.update()                     # 에피소드 종료 후 한 번에 업데이트
        reward_history.append(total_reward)
```

**실험 결과 — 두 그래프:**

| 왼쪽: Episode별 Total Reward | 오른쪽: 100회 이동평균 |
|----------------------------|-----------------------|
| 학습이 진행되며 점점 200(max)에 도달 | 흔들림이 큰 학습 곡선 (high variance) |

> ⚠️ **관찰:** 학습은 되지만 **분산이 매우 크고 불안정**. 이를 해결하기 위해 **REINFORCE → Baseline → Actor-Critic** 으로 발전합니다.

---

## 4. REINFORCE

### 4.1 Simple PG의 문제점 — Noise

Simple PG의 수식을 다시 봅시다:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} \underbrace{G(\tau)}_{\text{전체 기간 수익}} \nabla_\theta \log \pi_\theta(A_t \mid S_t)\right] \quad \cdots \;①$$

**문제점:** 시간 $t$ 에서의 행동 $A_t$ 에 항상 **동일한 가중치 $G(\tau)$ (전체 수익)** 가 곱해집니다.

> 🤔 **이상한 점:** 시간 $t = 5$ 에서의 행동 $A_5$ 는, $t = 0 \sim 4$ 에서 받은 보상과 **인과적으로 무관**합니다.  
> 그런데도 $G(\tau)$ 에는 그 보상들이 포함되어 있어 **노이즈(noise)** 로 작용합니다.

### 4.2 $G_t$ 의 도입

**해결 아이디어:** 행동 $A_t$ **이후에** 발생한 수익만을 가중치로 부여.

$$\boxed{G_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots + \gamma^{T-t} R_T}$$

이를 적용한 식이 **REINFORCE** (Williams, 1992):

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} \underbrace{G_t}_{t \text{ 이후 수익}} \nabla_\theta \log \pi_\theta(A_t \mid S_t)\right] \quad \cdots \;②$$

> 📚 **REINFORCE의 어원:**  
> **RE**ward **I**ncrement = **N**onnegative **F**actor × **O**ffset **R**einforcement × **C**haracter **E**ligibility

### Simple PG vs REINFORCE 비교

| | Simple PG (①) | REINFORCE (②) |
|--|---------------|---------------|
| 가중치 | $G(\tau) = R_0 + \gamma R_1 + \cdots + \gamma^T R_T$ | $G_t = R_t + \gamma R_{t+1} + \cdots + \gamma^{T-t} R_T$ |
| 의미 | 전체 trajectory 수익 (모든 $t$ 에 동일) | $t$ **이후**에만 발생한 수익 |
| 수렴성 | 샘플 ↑ → 정확한 $\nabla_\theta J$ 에 수렴 | 샘플 ↑ → 동일하게 수렴 |
| **분산** | **큼** (관련 없는 noise 포함) | **작음** (인과적으로 의미있는 부분만) |
| 학습 안정성 | 불안정 | ✅ 안정적이고 빠름 |

### 4.3 실습 reinforce.py

핵심 차이는 **`update()`** 메서드 한 곳:

```python
def update(self):
    self.pi.cleargrads()

    G, loss = 0, 0
    for reward, prob in reversed(self.memory):
        G = reward + self.gamma * G                # ← 역방향 누적: G_t 계산
        loss += -F.log(prob) * G                   # ← 각 t에 G_t 가중치 적용

    loss.backward()
    self.optimizer.update()
    self.memory = []
```

> 💡 **포인트:** `reversed(memory)` 로 순회하면서 같은 루프 안에서 `G`(=$G_t$)를 계산함과 동시에 loss를 누적합니다.  
> 이렇게 하면 자연스럽게 각 시간 $t$ 에 **그 시점 이후의 수익**이 가중치로 부여됩니다.

**Simple_pg vs REINFORCE 학습 곡선:**

| 알고리즘 | 학습 곡선 특성 |
|---------|--------------|
| Simple_pg | 늦게 수렴, 흔들림 큼 |
| REINFORCE | ✅ **안정적이고 빠른 학습** |

---

## 5. Baseline

> 📄 이 절의 **자세한 직관 설명** (시험 성적 비유, 분산 감소 원리) → [`10-2-Baseline_Intuition.md`](./10-2-Baseline_Intuition.md)

### 5.1 분산(Variance) 감소의 원리 — 시험 성적 비유

**3명의 시험 성적 예시:**

| 학생 | 실제 점수 | 과거 평균 (예측값) | 차이 (실제 − 예측) |
|------|----------|------------------|------------------|
| A | 90 | 85 | **+5** |
| B | 70 | 75 | **−5** |
| C | 50 | 55 | **−5** |
| **분산** | **466.67** | — | **약 32.67** |

> 🎯 **관찰:** "**실제값 − 예측값**" 으로 변환하면 분산이 **466.67 → 32.67** 로 급감!  
> 그런데도 평균(혹은 기댓값)은 **변하지 않습니다** — 단지 변동성만 줄어듭니다.

**원리:** 데이터에서 **(같은 무게의)** 기준값을 빼면, 데이터의 평균은 그만큼 이동하지만 **분산은 그대로 유지**됩니다.  
하지만 **각 데이터마다 적절한 예측값**(=baseline)을 빼주면, 노이즈는 줄고 **신호(signal)** 만 남게 됩니다.

### 5.2 Baseline 적용 수식

REINFORCE의 가중치 $G_t$ 에서 baseline $b(S_t)$ 를 빼줍니다:

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} \bigl(G_t - b(S_t)\bigr)\, \nabla_\theta \log \pi_\theta(A_t \mid S_t)\right]}$$

> 🔑 **수학적 정당성 (proof):**  
> $b(S_t)$ 가 **$A_t$ 와 독립**이라면, baseline 항의 기댓값은 0이 됩니다:
> $$\mathbb{E}\bigl[b(S_t)\, \nabla_\theta \log \pi_\theta(A_t \mid S_t)\bigr] = 0$$
> 즉, **baseline을 빼도 기울기의 평균값(=$\nabla_\theta J$)은 그대로 유지**되고, **분산만 감소**합니다.  
> (이 증명은 모든 $t = 0 \sim T$ 에 대해 성립.)

### 5.3 State Value Function을 Baseline으로

**가장 자연스러운 baseline 후보는 무엇일까?**

> 💡 **답:** 상태 $S_t$ 에서 **지금까지 평균적으로 얻은 보상** = **state value function** $V_{\pi_\theta}(S_t)$

| baseline 후보 | 의미 |
|--------------|------|
| 상수 (예: 0) | 분산 감소 효과 미미 |
| 평균 reward | 보통 |
| ✅ $V_{\pi_\theta}(S_t)$ | "그 상태에서 기대되는 평균 수익" — 가장 좋은 예측값 |

> 🎓 **이 발상이 곧 Actor-Critic으로 이어집니다.**  
> "$V$를 어떻게 알지?" → **신경망(Critic)으로 학습하자!**

---

## 6. Actor-Critic

> 📄 이 절의 **자세한 코드 구조와 두 신경망의 학습 흐름** → [`10-3-Actor_Critic.md`](./10-3-Actor_Critic.md)

### 6.1 Actor와 Critic의 역할

**Actor-Critic = Policy-Based + Value-Based 의 hybrid method**

| 구성 요소 | 역할 | 비유 |
|----------|------|------|
| **Actor (행위자)** | Policy $\pi_\theta(a\|s)$ 를 학습하여 실제로 **행동을 결정** | 무대 위에서 **연기**하는 배우 |
| **Critic (비평자)** | Value function $V_w(s)$ 를 학습하여 **현재 상태의 가치를 평가** | 객석에서 **연기를 평가**하는 평론가 |

| 패러다임 | 구성 |
|---------|------|
| Policy-based 만 | actor only (순수 REINFORCE) |
| Value-based 만 | critic only (순수 DQN) |
| **Actor-Critic** | **actor + critic (둘 다)** |

```
        State s
           │
   ┌───────┴────────┐
   ▼                ▼
[ Actor ]      [ Critic ]
π_θ(a|s)        V_w(s)
   │                │
 action       state value
   │                │
   └───→ TD error ←─┘
              │
              ▼
   둘 다 이 TD error 로 업데이트
```

### 6.2 TD 에러를 이용한 업데이트

수식의 진화 과정을 살펴봅시다:

$$\begin{aligned}
\nabla_\theta J(\theta) &= \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} \bigl(G_t - b(S_t)\bigr) \nabla_\theta \log \pi_\theta(A_t \mid S_t)\right] && \text{(REINFORCE w/ baseline)} \\
&= \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} \bigl(G_t - V_w(S_t)\bigr) \nabla_\theta \log \pi_\theta(A_t \mid S_t)\right] && \text{(Actor-Critic, MC)} \\
&= \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} \underbrace{\bigl(R_t + \gamma V_w(S_{t+1}) - V_w(S_t)\bigr)}_{\text{TD error } \delta_t} \nabla_\theta \log \pi_\theta(A_t \mid S_t)\right] && \text{(Actor-Critic, TD)}
\end{aligned}$$

| 기호 | 의미 |
|------|------|
| $\theta$ | **policy network (actor)** 의 weight 벡터 |
| $w$ | **value network (critic)** 의 weight 벡터 |
| $V_w(S_t)$ | Critic이 예측한 상태 가치 |
| $R_t + \gamma V_w(S_{t+1})$ | **TD target** (Critic의 학습 목표) |
| $\delta_t = R_t + \gamma V_w(S_{t+1}) - V_w(S_t)$ | **TD error** (Critic 예측의 오차) |

> 🎯 **TD 방식의 장점:**  
> - REINFORCE/MC 방식: 에피소드가 **끝나야** 학습 가능 ($G_t$ 계산을 위해 미래 보상 모두 필요)  
> - **TD 방식: 매 스텝마다** 학습 가능! ($R_t$ 와 $V_w(S_{t+1})$ 만 있으면 됨)

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
        self.l2 = L.Linear(1)              # 출력은 스칼라 V(s)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x
```

```python
class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v  = 0.0005
        self.action_size = 2

        self.pi = PolicyNet()                              # Actor
        self.v  = ValueNet()                               # Critic
        self.optimizer_pi = optimizers.Adam(self.lr_pi).setup(self.pi)
        self.optimizer_v  = optimizers.Adam(self.lr_v).setup(self.v)

    def update(self, state, action_prob, reward, next_state, done):
        # ────── TD target ──────
        next_state = next_state[np.newaxis, :]
        target = reward + self.gamma * self.v(next_state) * (1 - done)
        target.unchain()                                   # gradient 차단

        # ────── Critic update (V_w) ──────
        v = self.v(state[np.newaxis, :])
        loss_v = F.mean_squared_error(v, target)           # ← (TD target − V_w)²

        # ────── Actor update (π_θ) ──────
        delta = target - v                                 # ← TD error δ_t
        delta.unchain()                                    # δ는 상수 취급
        loss_pi = -F.log(action_prob) * delta              # ← -δ · log π_θ(A|S)

        self.v.cleargrads();  self.pi.cleargrads()
        loss_v.backward();    loss_pi.backward()
        self.optimizer_v.update();  self.optimizer_pi.update()
```

#### 코드와 수식 매핑

| 코드 | 대응 수식 | 역할 |
|------|----------|------|
| `target = reward + γ·V_w(s')` | $R_t + \gamma V_w(S_{t+1})$ | TD target |
| `loss_v = F.mean_squared_error(v, target)` | $\mathcal{L}_v = \bigl(R_t + \gamma V_w(S_{t+1}) - V_w(S_t)\bigr)^2$ | Critic loss |
| `delta = target - v` | $\delta_t = R_t + \gamma V_w(S_{t+1}) - V_w(S_t)$ | TD error |
| `loss_pi = -F.log(action_prob) * delta` | $\mathcal{L}_\pi = -\delta_t \log \pi_\theta(A_t\|S_t)$ | Actor loss |

> ⚠️ **`unchain()` 의 역할:**  
> - `target.unchain()`: TD target은 Critic 학습의 **고정된 목표**여야 함 → backprop이 흐르면 안 됨.  
> - `delta.unchain()`: Actor 학습에서 $\delta_t$ 는 **상수 가중치**로만 작용해야 함 → policy gradient만 계산.

### 6.4 실습 actor_critic.py

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

            agent.update(state, prob, reward, next_state, done)  # ★ 매 스텝 학습!
            state = next_state
            total_reward += reward

        reward_history.append(total_reward)
```

> 💡 **핵심 차이:** REINFORCE는 에피소드가 끝난 후 한 번에 update, Actor-Critic은 **매 스텝마다 update**.

**REINFORCE vs Actor-Critic 학습 곡선:**

| 알고리즘 | 학습 속도 | 안정성 | 메모리 |
|---------|---------|--------|--------|
| REINFORCE | 보통 | 보통 | 에피소드 전체 저장 |
| **Actor-Critic** | ✅ **빠름** | ✅ **안정적** | 1-step 정보만 |

---

## 7. 퀴즈 (Quiz)

> **[Q1]** Actor-Critic 을 **Mountain Car** 문제에 적용하되, **Hyper-parameter** 를 변경하여 **최대의 total reward** 를 갖는 policy 를 결정하라.

**제출물 (PPT):**

1. 프로그램 소스
2. 최적 hyperparameter (예: $\gamma$, $\alpha_\pi$, $\alpha_v$, hidden size, ...)
3. Episode별 total reward graph
4. 최대 total reward 값 및 해당 policy 적용 시의 동영상

**힌트:**

- Mountain Car는 보상이 매우 sparse (대부분 −1) → baseline 효과가 클 가능성
- learning rate, discount factor $\gamma$, hidden layer size 를 grid search
- Reward shaping (예: 위치에 비례한 추가 보상) 도 고려

---

## 8. 요약 — Φₜ의 진화와 Value vs Policy 비교

### 8.1 정책 경사법의 일반형

모든 Policy Gradient 계열 알고리즘은 다음의 **하나의 통일된 형태**로 표현됩니다:

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} \Phi_t\, \nabla_\theta \log \pi_\theta(A_t \mid S_t)\right]}$$

차이점은 오로지 **가중치 $\Phi_t$ 가 무엇이냐** 에 있습니다.

| 알고리즘 | $\Phi_t$ 의 정의 | 직관 |
|---------|----------------|------|
| **Simple Policy Gradient** | $\Phi_t = G(\tau)$ | trajectory 전체 수익 (noise 큼) |
| **REINFORCE** | $\Phi_t = G_t$ | $t$ 이후 수익만 |
| **REINFORCE with baseline** | $\Phi_t = G_t - b(S_t)$ | baseline으로 분산 감소 |
| **Actor-Critic** | $\Phi_t = R_t + \gamma V_w(S_{t+1}) - V_w(S_t)$ | TD error (Critic이 baseline 학습) |

> 📌 **참고:** $\Phi_t$ 에 state value function 대신 **action value function**도 사용 가능:  
> - $\Phi_t = Q(S_t, A_t)$  
> - $\Phi_t = Q(S_t, A_t) - V(S_t) = A(S_t, A_t)$ ← **Advantage function** (A2C, A3C 등)

### 8.2 Value-Based vs Policy-Based 최종 비교

| 비교 항목 | Value-Based | Policy-Based |
|----------|-------------|--------------|
| **학습 대상** | Value function $Q(s,a)$ | Policy function $\pi(a\|s)$ |
| **정책 도출** | $Q$ 로부터 **간접적**으로 derive | **직접** 정책 결정 (효율적) |
| **탐험 방식** | ε-greedy (인위적) | softmax (자연스러움) |
| **연속 행동공간** | 어려움 (max 계산 곤란) | ✅ 쉬움 (분포에서 sampling) |
| **수렴성** | 빠르고 안정 | 느릴 수 있으나 **local optimum 보장** |
| **분산** | 보통 | 큼 → baseline/critic으로 감소 |
| **대표 예제** | DQN — Cart pole | REINFORCE/AC — Pendulum |
| **대표 알고리즘** | SARSA, Q-Learning, DQN, Rainbow | REINFORCE, A2C, A3C, PPO, TRPO, SAC |

### 8.3 한눈에 보는 진화 흐름

```
Simple PG           ─→  Φ_t = G(τ)                    [전체 수익, noise 큼]
    │
    │ "관련 없는 과거 보상 제거"
    ▼
REINFORCE           ─→  Φ_t = G_t                     [t 이후만, 분산 ↓]
    │
    │ "예측값을 빼서 분산 추가 감소"
    ▼
REINFORCE+Baseline  ─→  Φ_t = G_t − b(S_t)            [signal vs noise]
    │
    │ "baseline을 신경망으로 학습"
    ▼
Actor-Critic (MC)   ─→  Φ_t = G_t − V_w(S_t)
    │
    │ "에피소드 종료 기다리지 말고 TD로"
    ▼
Actor-Critic (TD)   ─→  Φ_t = R_t + γV_w(S_{t+1}) − V_w(S_t)   ★
```

> 🎓 **결론:** Policy Gradient 의 발전사는 **"어떻게 분산을 줄이면서도 평균(=true gradient)은 유지할 것인가"** 의 역사라고 요약할 수 있습니다.

---

> **실습 실행:** `week10/` 에서 아래 순서로 실행 가능 (`pip install -r requirements.txt`).
> ```text
> python simple_pg.py           # 선택: --episodes 3000 --plot
> python reinforce.py
> python actor_critic.py
> ```
> 구현은 **순수 NumPy** 로 동작하게 두었으며(현재 많은 환경에서 `dezero` 패키지의 NumPy 2.x 비호환 이슈를 피하기 위함), 수식 매핑은 교재 슬라이드와 동일합니다.  
> **참고 모듈:** `pg_numpy_core.py`(정책·가치망 역전파), `cartpole_env.py`(gymnasium/gym 선택)
>
> **심화 문서:**  
> - 수식 증명 → [`10-1-Policy_Gradient_Math.md`](./10-1-Policy_Gradient_Math.md)  
> - Baseline 직관 → [`10-2-Baseline_Intuition.md`](./10-2-Baseline_Intuition.md)  
> - Actor-Critic 상세 → [`10-3-Actor_Critic.md`](./10-3-Actor_Critic.md)
