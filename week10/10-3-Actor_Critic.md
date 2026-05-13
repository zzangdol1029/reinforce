# Actor-Critic — 아키텍처와 DeZero 구현 해설

> 강의 자료 `10-Policy Gradient Method.pdf` **19~23페이지**  
> [← 메인 요약으로 돌아가기](./10-Policy_Gradient_Method.md#6-actor-critic)

---

## 1. 구조 한 장 요약

| 구성 요소 | 파라미터 | 출력 | 하는 일 |
|----------|----------|------|--------|
| **Actor** | $\theta$ | $\pi_\theta(a\mid s)$ | 행동 **확률 분포**를 내고 행동을 샘플링 |
| **Critic** | $w$ | $V_w(s)$ | 상태 **기대 가치**를 근사 (베이스라인 역할 포함) |

**하이브리드**라 불리는 이유:

| 만 Actor만 | 만 Critic만 |
|------------|-------------|
| Policy gradient 계열(REINFORCE 등) | Value-based (DQN 등) |
| **Actor+Critic 둘 다** 학습하면, PG의 높은 분산을 줄이면서 정책을 직접 고칠 수 있습니다. |

---

## 2. 슬라이드 수식 흐름 (MC $\to$ TD)

### 2.1 베이스라인 포함

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_t \bigl(G_t - b(S_t)\bigr) \nabla_\theta \log\pi_\theta(A_t\mid S_t)\right]$$

### 2.2 Critic 로 베이스라인 교체 ($b\to V_w$)

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_t \bigl(G_t - V_w(S_t)\bigr) \nabla_\theta \log\pi_\theta(A_t\mid S_t)\right]$$

**(Monte Carlo return $G_t$ 를 쓰면)** 에피소드가 끝나야 하는 면이 남습니다.

### 2.3 TD 로 $G_t$ 를 1-step 으로 치환 (슬라이드 형태)

**부트스트랩**으로 다음을 정의하면:

$$\delta^{(1\text{-step TD})}_{t}= R_t+\gamma V_w(S_{t+1})-V_w(S_t)$$

그리고:

$$\boxed{\nabla_\theta J(\theta) \approx \mathbb{E}\left[\sum_t \bigl(R_t+\gamma V_w(S_{t+1})-V_w(S_t)\bigr)\nabla_\theta \log\pi_\theta(A_t\mid S_t)\right]}$$

| 기호 | 의미 |
|------|------|
| $R_t+\gamma V_w(S_{t+1})$ | **TD target** (실제 즉시보상 + 할인된 다음 상태 가치) |
| $V_w(S_t)$ | 현재 상태에 대한 예측 |
| $\delta_t$ | **TD error**(예측 오차): target − prediction |

직관적으로:

> Critic 이 “평균적으로 이 정도 받을 거야” 라고 할 때, 실제 다음 보상까지 합산한 게 그보다 **얼마나 좋거나 나쁜지** 가 $\delta_t$ 입니다. Actor 는 그 크기만큼 **그 행동의 로그 확률을 키우거나 줄입니다.**

---

## 3. Critic 과 Actor 의 손실(역할 분리)

### 3.1 Critic 학습 목표 — 가깝게 만들기

1-step 타깃과의 **제곱 오차**(회귀):

$$\mathcal{L}_V=\frac{1}{2}\mathbb{E}\bigl[\bigl(R_t+\gamma V_w(S_{t+1})-V_w(S_t)\bigr)^2\bigr]$$

실제 구현에서는 `mean_squared_error` 로 배치 평균을 씁니다.

### 3.2 Actor 학습 신호 — policy gradient 형태

**스칼라**로는 자주 다음과 같습니다:

$$\mathcal{L}_\pi = -\,\delta_t\,\log \pi_\theta(A_t \mid S_t)$$

단, 여기서 **$\delta_t$ 는 역전파(backprop)** 에서 파라미터 $\theta$ 쪽 경로로 흘리지 않는 것이 일반적입니다(가중치처럼 취급 → REINFORCE/actor-critic의 전형 패턴).

> 구현 디테일에서 `detach()` / `.unchain()` 으로 **$\delta_t$ 에서 gradient를 차단**하는 이유입니다.

---

## 4. DeZero 스타일 코드 — 라인별 매핑

아래 코드는 교재 패턴(**PolicyNet**, **ValueNet**, **Agent**)을 기준으로 한 **교육용 예시**입니다. 실습 파일(`Actor_critic2.py` 등)과 줄 순서만 다를 수 있으니, 아래 표의 **수식 매핑**이 일치하는지 보면 됩니다.

```python
import numpy as np
import dezero.functions as F
import dezero.optimizers as optimizers
from dezero import Model
import dezero.layers as L

class PolicyNet(Model):
    def __init__(self, action_size=2):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x))           # 수식: π_θ(a|s)
        return x

class ValueNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)                        # 수식: V_w(s)
        return x

class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v = 0.0005
        self.pi = PolicyNet(action_size=2)
        self.v = ValueNet()
        self.optimizer_pi = optimizers.Adam(self.lr_pi)
        self.optimizer_v = optimizers.Adam(self.lr_v)
        self.optimizer_pi.setup(self.pi)
        self.optimizer_v.setup(self.v)

    def update(self, state, action_prob, reward, next_state, done):
        ns = next_state[np.newaxis, :]      # 배치 차원 (1, ...)
        s = state[np.newaxis, :]

        # 수식: R_t + γ · V_w(S_{t+1}) · 𝕀(비종료)
        next_v = self.v(ns)
        target = reward + self.gamma * next_v * (0.0 if done else 1.0)

        target.unchain()                    # 타깃 쪽 역전파 차단 — critic 의 “목표 레이블” 안정화
        v = self.v(s)

        # 수식: L_V = mean( (target − V_w(S_t))^2 )
        loss_v = F.mean_squared_error(v, target)

        # 수식: δ_t = target − V_w(S_t)  (= R_t + γV(S_{t+1}) − V(S_t))
        delta = target - v
        delta.unchain()                     # actor loss에서는 δ를 “상수 계수”로만 사용

        # 수식: L_π = − δ_t · log π_θ(A_t | S_t)
        loss_pi = -F.log(action_prob) * delta

        self.v.cleargrads()
        loss_v.backward()
        self.optimizer_v.update()

        self.pi.cleargrads()
        loss_pi.backward()
        self.optimizer_pi.update()
```

### 4.1 반드시 맞춰야 하는 부호/`unchain` 규칙

| 줄/개념 | 수식 대응 | 왜 필요한가 |
|---------|-----------|-------------|
| `target = R + γ * V(next) * (0 if done else 1)` | $R_t+\gamma V_w(S_{t+1})$ (종료 시 다음 가치 0) | 1-step TD target |
| `target.unchain()` | 학습 신호 분리 | 많은 교재 버전에서 **타깃을 고정**(또는 별도 target net)해서 critic 붕 뜸 방지 |
| `loss_v = mse(V(s), target)` | $\mathcal{L}_V = (R_t+\gamma V_w(S_{t+1})-V_w)^2$ 형태 | 가치 회귀 |
| `delta = target - v` 후 `delta.unchain()` | $\delta_t$ 는 로그 확률의 계수만 | **Actor** 업데이트가 $\delta$ 를 미분해서 $\theta,\,w$ 가 꼬이지 않도록 막음 |
| `loss_pi = -log(prob) * delta` | $-\delta_t \log\pi_\theta$ | policy gradient(가중 로그 우도 형태) |

> ⚠️ **실무 팁:** Critic 과 Actor 에 **각각 학습률**을 두는 것이 흔합니다. Critic 이 너무 빠르면 target 이 너무 흔들리고, Actor 가 너무 빠르면 softmax 가 붕 뜰 수 있습니다.

---

## 5. 학습 순서 차이와 메모리

| 방법 | 언제 `update()`? | 메모리 |
|------|------------------|--------|
| REINFORCE | 에피소드 끝, 궤적 전체 이용 | 에피소드 버퍼 |
| Actor-Critic (위 TD) | **매 스텝** | 현재 줄(line) 또는 짧은 n-step 버퍼 |

---

## 6. 간단 디버그 체크리스트

1. **확률 합은 1** 인가 (`softmax` 후 액션 샘플링).  
2. `log(action_prob)` 는 **실제 선택한 행동** 에 대한 로그 확률인가.  
3. `delta` 를 actor loss 와 어떻게 연결했는지: **두 번 미분되어 이상해지면** detach/unchain 규칙을 의심.  
4. `done=True` 에서 다음 가치항을 **0** 으로 막았는지(`(1-done)` 패턴).

---

## 관련 문서

- [전체 요약](./10-Policy_Gradient_Method.md)
- [PG 수식 증명](./10-1-Policy_Gradient_Math.md)
- [Baseline 직관](./10-2-Baseline_Intuition.md)
