# Deep Q-Network (DQN)

> **강사**: Prof. Tae-Hyoung Park  
> **소속**: Dept. of Intelligent Systems & Robotics, CBNU  
> **원본 슬라이드**: [`09-Deep Q-Network.pdf`](./09-Deep%20Q-Network.pdf)

---

## 목차

1. [강화학습 가치 함수 표현의 진화](#1-강화학습-가치-함수-표현의-진화)
2. [OpenAI Gym (Gymnasium) 환경 기초](#2-openai-gym-gymnasium-환경-기초)
3. [Experience Replay (경험 재생)](#3-experience-replay-경험-재생)
4. [Target Network (목표 신경망)](#4-target-network-목표-신경망)
5. [DQN 알고리즘 확장 (Advanced Techniques)](#5-dqn-알고리즘-확장-advanced-techniques)
6. [퀴즈 및 실습 과제](#6-퀴즈-및-실습-과제)
7. [참고 문헌](#7-참고-문헌)

---

## 1. 강화학습 가치 함수 표현의 진화

| 항목 | **Q-Learning** | **Q-Network** | **DQN** |
|------|---------------|---------------|---------|
| **방법론** | Value based | Value based | Value based |
| **Q 함수 표현 $Q(s,a)$** | **Table** (격자형 표) | **Neural Network** ($\theta$) | Neural Network ($\theta$) |
| **학습/업데이트 방식** | Bellman 최적 방정식 | Error Backpropagation (Gradient Descent) | Error Backpropagation + 안정화 기법 |
| **출력값 / 목표값** | $Q(S_t, A_t)$ ↔ TD target (테이블 갱신) | 출력값 $= Q(S_t, A_t)$<br>목표값 $= R_t + \gamma \max_a Q(S_{t+1}, a)$ | ← (동일) + **Target Network**로 목표값 고정 |
| **Action 선택 정책** | ε-greedy | ε-greedy | ε-greedy |
| **적용 환경** | Discrete state space, **작은 state space** | Continuous state space 가능, **큰 state space 가능** | ← (동일) |
| **한계점 / 개선** | State space 폭발 문제 | **학습 효율·안정성 문제** (정답 레이블 변동, 데이터 상관관계) | **Experience Replay + Target Network** 도입으로 해결 |

**Q-Learning 업데이트 식:**

$$Q'(S_t, A_t) = Q(S_t, A_t) + \alpha \left[ R_t + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t) \right]$$

**Q-Network / DQN의 신경망 학습 목표값 (정답 레이블):**

$$\text{Target} = R_t + \gamma \max_a Q(S_{t+1}, a)$$

> 💡 **핵심 포인트**: Q-Learning → Q-Network 전환의 본질은 "표(table)"를 "함수 근사기(neural network)"로 대체하여 **연속 / 거대 상태 공간**에 일반화한 것입니다. DQN은 여기에 **학습 안정성**을 더한 진화 형태입니다.

---

## 2. OpenAI Gym (Gymnasium) 환경 기초

### 2.1 라이브러리 소개

- **오픈소스 Python 라이브러리**로, 강화학습 알고리즘 개발 및 비교용 표준 환경 제공
- Public beta version: **2016.04** 공개
- 공식 문서: <https://www.gymlibrary.dev/>

### 2.2 환경별 설치 명령어

```bash
# Classic Control (CartPole, MountainCar, Acrobot, Pendulum)
pip install gym[classic_control]

# Box2D (Bipedal Walker, Car Racing, Lunar Lander)
pip install gym[box2d]

# MuJoCo (Ant, Half Cheetah, Hopper)
pip install gym[mujoco]
```

> 최신 프로젝트는 **Gymnasium** (`pip install gymnasium`) 을 사용합니다. 슬라이드·교재 예제는 `gym` API와 거의 동일합니다.

### 2.3 제공 환경 카테고리

| 카테고리 | 대표 환경 |
|---------|----------|
| **Classic Control** | Cart Pole, Mountain Car, Acrobot, Pendulum |
| **Box2D** | Bipedal Walker, Car Racing, Lunar Lander |
| **Atari** | Breakout, Pong, Space Invaders |
| **MuJoCo** | Ant, Half Cheetah, Hopper |
| **Toy Text** | Blackjack, Taxi, Cliff Walking |

### 2.4 Cart Pole 환경 상세 명세

#### State (Observation) — 4차원 관측값

| Index | 관측 항목 | 범위 |
|:-----:|----------|------|
| 0 | Cart Position (카트 위치) | **-4.8 ~ 4.8** |
| 1 | Cart Velocity (카트 속도) | **-∞ ~ ∞** |
| 2 | Pole Angle (막대 각도) | **-24° ~ 24°** |
| 3 | Pole Angular Velocity (막대 각속도) | **-∞ ~ ∞** |

#### Action — `Discrete(2)`

- **`0`**: 카트를 **왼쪽**으로 밀기 (Push cart to the left)
- **`1`**: 카트를 **오른쪽**으로 밀기 (Push cart to the right)

#### Reward 시스템

- 매 step마다 **+1 보상** 지급
- **목표**: 막대가 쓰러지지 않도록 최대한 오래 균형 유지

#### Starting State

- 4차원 상태 각각 **(-0.05, 0.05)** 범위의 균등 분포에서 무작위 초기화

#### Episode 종료 조건 (아래 중 하나라도 만족 시)

- **막대 각도**가 **±12° (±0.209 rad)** 초과
- **카트 위치**가 **±2.4** 초과 (디스플레이 가장자리 도달)
- 에피소드 길이가 **500 step 초과** (v0 기준 200 step) — `truncated`

### 2.5 핵심 API 사용법

```python
import gym
import numpy as np

# 환경 생성
env = gym.make('CartPole-v0', render_mode='human')

# 상태 초기화
state = env.reset()[0]
print('상태:', state)
# 예: [0.03454657 -0.01361909 -0.02143636  0.02152179]

action_space = env.action_space
print('행동의 차원 수:', action_space)  # Discrete(2)

# 환경 한 스텝 진행
action = 0  # 혹은 1
next_state, reward, terminated, truncated, info = env.step(action)
```

| 반환값 | 의미 |
|--------|------|
| `next_state` | 다음 상태 |
| `reward` | 보상 |
| `terminated` | **목표/실패 상태 도달 여부** (각도 초과, 위치 벗어남 등) |
| `truncated` | **MDP 범위 밖 종료 조건** 충족 여부 (시간 초과, 500 step 초과 등) |
| `info` | 추가 디버깅 정보 |

> 💡 **Tip**: `done = terminated | truncated` 로 두 종료 신호를 합쳐 처리하는 것이 관용적입니다.

### 2.6 Random Agent 예시

```python
import numpy as np
import gym

env = gym.make('CartPole-v0', render_mode='human')
state = env.reset()[0]
done = False

while not done:  # 에피소드가 끝날 때까지 반복
    env.render()                              # 진행 과정 시각화
    action = np.random.choice([0, 1])         # 행동 선택(무작위)
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated | truncated             # 둘 중 하나만 True면 종료

env.close()
```

---

## 3. Experience Replay (경험 재생)

### 3.1 개념: Supervised Learning과의 유사성

| 학습 패러다임 | 데이터 출처 | 추출 방식 |
|--------------|------------|----------|
| **Supervised Learning** | 고정된 Training Data Set | **무작위 추출 → mini-batch → 신경망 학습** |
| **Q-Network (Experience Replay)** | Agent가 환경과 상호작용하며 모은 **Replay Buffer** | **무작위 추출 → mini-batch → 신경망 학습** |

**경험 데이터 정의:**

$$E_t = (S_t, A_t, R_t, S_{t+1})$$

> ⚠️ **문제점**: 시간 순서대로 수집된 $E_t$와 $E_{t+1}$ 사이에는 **강한 상관관계(Correlation)** 가 존재 → 신경망 학습 시 편향(bias) 증가, 불안정한 수렴  
> 💡 **해결책**: 경험을 **버퍼에 저장 → 무작위로 샘플링**하여 상관관계를 약화하고 편향이 작은 학습 데이터를 생성

### 3.2 Experience Replay 흐름도

```
   ┌──────────────┐       ┌─────────────────────┐
   │    Agent     │ ◄───► │     Environment     │
   │   (Q-Net)    │       │      (Gym Env)      │
   └──────┬───────┘       └─────────────────────┘
          │
          │ ① 경험 데이터 생성
          │   E_t = (S_t, A_t, R_t, S_{t+1})
          ▼
   ┌────────────────────────────────────┐
   │      Replay Buffer (deque)         │
   │   ┌────────────────────────────┐   │
   │   │ (S_0, A_0, R_0, S_1)       │   │  ← FIFO: maxlen 초과 시
   │   │ (S_1, A_1, R_1, S_2)       │   │     가장 오래된 것부터 삭제
   │   │  ...                       │   │
   │   └────────────────────────────┘   │
   └──────┬─────────────────────────────┘
          │
          │ ② 무작위 추출 (Random Sampling)
          ▼
   ┌────────────────────────────────────┐
   │      Mini-batch (size = 32)        │
   └──────┬─────────────────────────────┘
          │
          │ ③ Gradient Descent로 신경망 학습
          ▼
   ┌────────────────────────────────────┐
   │       Q-Network 가중치 업데이트     │
   └────────────────────────────────────┘
```

### 3.3 `ReplayBuffer` 클래스 구현

```python
from collections import deque
import random
import numpy as np
import gym

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        # deque: 선입선출, maxlen 초과 시 가장 오래된 데이터 자동 삭제
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state      = np.stack([x[0] for x in data])
        action     = np.array([x[1] for x in data])
        reward     = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done       = np.array([x[4] for x in data]).astype(np.int32)

        return state, action, reward, next_state, done
```

### 3.4 사용 예시 — 데이터 수집 & 미니배치 추출

```python
env = gym.make('CartPole-v0', render_mode='human')
replay_buffer = ReplayBuffer(buffer_size=10000, batch_size=32)

for episode in range(10):
    state = env.reset()[0]
    done = False
    while not done:
        action = 0
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state

state, action, reward, next_state, done = replay_buffer.get_batch()
print(state.shape)       # (32, 4)
print(action.shape)      # (32,)
```

> 📎 **실습**: 슬라이드 **실습 #1** — [`Replay_buffer.py`](./Replay_buffer.py)

---

## 4. Target Network (목표 신경망)

### 4.1 필요성: "정답 레이블이 계속 바뀐다"는 문제

| 학습 패러다임 | 정답 레이블 |
|--------------|------------|
| **Supervised Learning** | 학습 데이터에 **영구적으로 고정된** 레이블 |
| **Q-Network** | 목표값 $= R_t + \gamma \max_a Q(S_{t+1}, a)$ → **$Q$가 갱신될 때마다 정답 레이블도 함께 변경** |

> ⚠️ **문제점**: 학습 도중 정답 레이블이 계속 변동 → 신경망 학습이 **매우 불안정**해지고 수렴이 어려움.

### 4.2 해결책: Target Network 메커니즘

원본 신경망 `qnet`과 **구조가 동일한** `qnet_target`을 두고 다음 규칙으로 운영:

| 신경망 | 역할 | 가중치 갱신 |
|--------|------|------------|
| **`qnet`** (원본) | Q값 출력 / 매 step 학습 | 매 step **Gradient Descent** |
| **`qnet_target`** (목표) | **TD Target 계산용** | 평소 **고정**, 주기적으로 `qnet`에서 **`copy.deepcopy`** |

**TD Target (Target Network 적용):**

$$T = R_t + \gamma \max_a Q_{\theta^-}(S_{t+1}, a)$$

$\theta^-$ 는 **고정된 target network** 가중치입니다.

> 💡 **Key Idea**: 정답 레이블 변동을 억제 → **신경망 학습 안정화** (수렴 가속화).

### 4.3 `DQNAgent` 클래스 구현

```python
import copy
import numpy as np
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L


class QNet(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(128)
        self.l3 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DQNAgent:
    def __init__(self):
        self.gamma       = 0.98
        self.lr          = 0.0005
        self.epsilon     = 0.1
        self.buffer_size = 10000
        self.batch_size  = 32
        self.action_size = 2

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet          = QNet(self.action_size)
        self.qnet_target   = QNet(self.action_size)
        self.optimizer     = optimizers.Adam(self.lr)
        self.optimizer.setup(self.qnet)  # ※ 가중치 업데이트는 qnet만!

    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        state = state[np.newaxis, :]
        qs = self.qnet(state)
        return qs.data.argmax()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()

        qs = self.qnet(state)
        q  = qs[np.arange(self.batch_size), action]

        next_qs = self.qnet_target(next_state)
        next_q  = next_qs.max(axis=1)
        next_q.unchain()

        target = reward + (1 - done) * self.gamma * next_q

        loss = F.mean_squared_error(q, target)
        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()
```

**미니배치 텐서 shape (CartPole, batch=32):**

| 변수 | shape |
|------|-------|
| `state` | (32, 4) |
| `action` | (32,) |
| `qs` | (32, 2) |
| `q` | (32,) |

### 4.4 DQN 메인 학습 루프

```python
episodes      = 300
sync_interval = 20

env   = gym.make('CartPole-v0', render_mode='rgb_array')
agent = DQNAgent()
reward_history = []

for episode in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    if episode % sync_interval == 0:
        agent.sync_qnet()

    reward_history.append(total_reward)
```

> 💡 **실험 관찰**: 단일 실험 보상 그래프는 noisy 하지만, **100회 평균**을 내면 약 100 에피소드 부근부터 보상이 상승해 **140** 수준에서 안정화되는 패턴이 관찰됩니다.

> 📎 **실습**: 슬라이드 **실습 #2** — [`dqn2.py`](./dqn2.py)  
> 📎 **퀴즈 Q1**: [`quiz_q1_mountain_car_dqn.py`](./quiz_q1_mountain_car_dqn.py)

---

## 5. DQN 알고리즘 확장 (Advanced Techniques)

### 5.1 Double DQN

#### 문제: Overestimation (과대평가)

기본 DQN TD Target:

$$\text{TD Target} = R_t + \gamma \max_a Q_\theta(S_{t+1}, a)$$

**Estimation of TD Target:**

- Original network: $\theta \rightarrow R_t + \gamma \max_a Q_\theta(S_{t+1}, a)$
- Target network: $\theta' \rightarrow R_t + \gamma \max_a Q_{\theta'}(S_{t+1}, a)$

**직관적 예시:**

- 참값이 모두 0: $q(s,a_0)=q(s,a_1)=q(s,a_2)=q(s,a_3)=0$
- 이론적 기댓값: $\mathbb{E}[\max_a q(s,a)] = 0$
- $Q$에 정규분포 노이즈가 있으면: $\mathbb{E}[\max_a Q(s,a)] > 0$

```python
import numpy as np
import matplotlib.pyplot as plt

samples = 1000
action_size = 4
Qs = []

for _ in range(samples):
    Q = np.random.randn(action_size)
    Qs.append(Q.max())

plt.hist(Qs, bins=16)
plt.axvline(x=0, color='red')
plt.axvline(x=np.array(Qs).mean(), color='cyan')
plt.show()
```

> ⚠️ **핵심**: $\max$ 연산은 **가장 큰 노이즈**를 선택 → Q값 **과대평가** 편향.

#### 해결: 행동 선택과 가치 평가의 분리

- **최대 행동 선택**: $Q_\theta$ 사용 → $a^* = \arg\max_a Q_\theta(S_{t+1}, a)$
- **선택 행동의 가치**: $Q_{\theta'}$ 사용

$$\text{Double DQN Target} = R_t + \gamma \, Q_{\theta'}\!\left(S_{t+1},\ \arg\max_a Q_\theta(S_{t+1}, a)\right)$$

> 📖 V. Hasselt et al., *"Deep Reinforcement Learning with Double Q-learning"*, **AAAI 2016**.

---

### 5.2 Prioritized Experience Replay (PER)

#### 개념

균등 무작위 샘플링 대신, **TD Error $\delta_t$가 큰 경험**에 더 높은 우선순위:

$$\delta_t = R_t + \gamma \max_a Q_{\theta'}(S_{t+1}, a) - Q_\theta(S_t, A_t)$$

| $\delta_t$ | 해석 |
|------------|------|
| **큼** | 예측 오차 큼 → 학습 가치 **높음** |
| **작음** | 예측 거의 맞음 → 학습 가치 **낮음** |

확장 경험: $E_t = (S_t, A_t, R_t, S_{t+1}, \delta_t)$

**선택 확률:**

$$p_i = \frac{\delta_i}{\sum_{k=0}^{N} \delta_k}$$

> 📖 T. Schaul et al., *"Prioritized Experience Replay"*, **arXiv 2015**.

---

### 5.3 Dueling DQN

#### Advantage Function

$$A_\pi(s, a) = Q_\pi(s, a) - V_\pi(s) \quad \Leftrightarrow \quad Q_\pi(s, a) = V_\pi(s) + A_\pi(s, a)$$

| 기호 | 의미 |
|------|------|
| $Q_\pi(s,a)$ | 상태 $s$에서 행동 $a$ 후 정책 $\pi$ 따를 때 기대 수익 |
| $V_\pi(s)$ | 상태 $s$에서 정책 $\pi$ 따를 때 기대 수익 |
| $A_\pi(s,a)$ | 행동 $a$가 평균 대비 얼마나 유리한지 |

#### 네트워크 구조 비교

**일반 DQN:**

```
  Input (s) ──▶ [공유 백본] ──▶ Q(s,a₁), Q(s,a₂), …, Q(s,aₙ)
```

**Dueling DQN:**

```
                    ┌──▶ V(s)
  Input (s) ──▶ [공유 백본] ─┤
                    └──▶ A(s,a₁), …, A(s,aₙ)
                              │
                              ▼
                    Q(s,a) = V(s) + A(s,a)
```

> 💡 **어떤 행동을 해도 결과가 거의 같을 때** (상태 가치만 중요): 일반 DQN은 모든 $Q(s,a)$를 따로 학습해 진행이 느리고, Dueling DQN은 $V(s)$ 한 줄기로 빠르게 학습 가능.  
> 📝 *Dueling* = *competing* (두 분기가 경쟁하듯 분리).

> 📖 Z. Wang et al., *"Dueling Network Architectures for Deep Reinforcement Learning"*, **ICML 2016**.

---

## 6. 퀴즈 및 실습 과제

### (Q1) Mountain Car 문제에 DQN 적용하기

**과제**: DQN을 **Mountain Car**에 적용하고, **Hyper-parameter를 변경**하여 **최대 total reward**를 갖는 policy를 찾을 것.

#### Mountain Car 환경 명세

| 항목 | 내용 |
|------|------|
| **State (Observation)** | `[position (-1.2 ~ 0.6), velocity (-0.07 ~ 0.07)]` |
| **Action** | `0`: 왼쪽 가속 / `1`: 가속 안 함 / `2`: 오른쪽 가속 |
| **Reward** | 매 timestep **-1** (목표: 깃발에 빨리 도달) |
| **Starting state** | Position: `[-0.6, -0.4]` 균등 무작위 |
| **Termination** | 카트 위치 ≥ **0.5** (깃발 도달) |
| **Truncation** | 에피소드 길이 ≥ **200** |

🔗 <https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/mountain_car.py>

#### 튜닝 대상 Hyper-parameter (기본값 예시)

| 변수 | 의미 | 기본값 (예) |
|------|------|------------|
| `gamma` | 할인율 | 0.98 |
| `lr` | 학습률 | 0.0005 |
| `epsilon` | ε-greedy 탐색 확률 | 0.05 |
| `buffer_size` | Replay buffer 크기 | 100000 |
| `batch_size` | Mini-batch 크기 | 32 |
| `episodes` | 학습 에피소드 수 | 300 |
| `sync_interval` | Target network 동기화 주기 | 20 |
| 신경망 구조 | 은닉층 수·노드 수 | (자유) |

#### 제출 요건 (PPT)

1. **프로그램 소스코드** (전체 `.py`)
2. **최적 hyperparameter** 조합
3. **Episode 별 total reward graph** (matplotlib)
4. **최대 total reward** 및 **해당 policy 실행 동영상**

> ⚠️ **Mountain Car 특수성**: 보상이 항상 **-1**이라 탐험이 부족하면 깃발에 거의 도달하지 못합니다. `epsilon` 스케줄링, reward shaping, episode 수 증가 등을 적극 고려하세요.

---

## 7. 참고 문헌

- V. Hasselt et al., *"Deep Reinforcement Learning with Double Q-learning"*, **AAAI 2016**
- T. Schaul et al., *"Prioritized Experience Replay"*, **arXiv 2015**
- Z. Wang et al., *"Dueling Network Architectures for Deep Reinforcement Learning"*, **ICML 2016**
