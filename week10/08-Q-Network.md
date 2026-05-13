# Q-Network

> Prof. Tae-Hyoung Park  
> Dept. of Intelligent Systems & Robotics, CBNU

---

## 목차

1. [DeZero 소개](#1-dezero-소개)
   - 1.1 Deep Learning Framework
   - 1.2 클래스 구조 (Variable / Function)
   - 1.3 Gradient 계산 — Rosenbrock 함수
   - 1.4 Gradient Descent
2. [회귀 응용 (Regression)](#2-회귀-응용-regression)
   - 2.1 Linear Regression
   - 2.2 실습 #1 — `dezero3.py` ($y = 2x + 5$)
   - 2.3 Nonlinear Regression (MLP)
   - 2.4 실습 #2 — `dezero4.py` ($y = \sin(2\pi x)$)
3. [강화학습 복습](#3-강화학습-복습)
   - 3.1 Policy Evaluation & Update
   - 3.2 Q-Learning 요약
4. [Q-Network](#4-q-network)
   - 4.1 Q-Learning의 한계
   - 4.2 Q-Network의 핵심 아이디어
   - 4.3 One-Hot Vector
   - 4.4 QNet 구조
   - 4.5 Q Function 의 학습 — Target $T$
   - 4.6 Q-Learning vs Q-Network 비교
   - 4.7 실습 #3 — `q_learning_nn.py`
5. [퀴즈 (Quiz)](#5-퀴즈-quiz)
6. [요약](#6-요약)

---

## 1. DeZero 소개

### 1.1 Deep Learning Framework

| Framework | 특징 |
|-----------|------|
| **PyTorch** | 동적 계산 그래프, 연구·교육 표준 |
| **TensorFlow** | 정적/동적 모두 지원, 산업 배포 강점 |
| **DeZero** | PyTorch 와 거의 동일한 API. *처음부터 만드는 딥러닝 3* 의 학습용 프레임워크 |

설치:

```bash
$ pip install dezero
$ pip install numpy==1.23.0   # DeZero는 numpy 1.23.x 와 호환됩니다
```

> 💡 **왜 DeZero 인가?**  
> PyTorch 의 `autograd`, `nn.Module` 과 거의 동일한 인터페이스를 **2,000줄 미만의 순수 파이썬 코드**로 구현했기 때문에, "프레임워크가 미분과 학습을 어떻게 수행하는가?" 를 *코드 한 줄까지* 들여다볼 수 있습니다.

### 1.2 클래스 구조 (Variable / Function)

```
                 ┌────────────────────────────┐
                 │   Variable (np.ndarray)    │
                 │   - data, grad, creator    │
                 │   - backward()             │
                 └──────────────┬─────────────┘
                                │
                  ┌─────────────┴─────────────┐
                  │                           │
            ┌─────▼──────┐              ┌─────▼──────┐
            │  Function  │              │   Layer    │
            │  forward() │              │  (params)  │
            │  backward()│              └─────┬──────┘
            └────────────┘                    │
                                       ┌──────▼───────┐
                                       │    Model     │
                                       │  (MLP, …)    │
                                       └──────────────┘
```

#### Variable Class

- `numpy.ndarray` 를 감싸는 텐서 클래스
- 핵심 메서드: **`backward()`** — 자동 미분

```python
import numpy as np
from dezero import Variable

x = Variable(np.array(5.0))
y = 3 * x ** 2          # y = 3x²
y.backward()            # 역전파
print(x.grad)           # dy/dx = 6x = 30
```

수식으로 표현하면:

$$y = 3x^2,\qquad \left.\frac{dy}{dx}\right|_{x=5} = 6x\bigl|_{x=5} = 30$$

#### Function Class

- `Add`, `Mul`, `Exp`, `MatMul`, `Sin` … 등 Variable 에 대한 **연산자**
- 가상함수 `forward()` / `backward()` 를 오버라이드

행렬곱 예:

$$\begin{bmatrix}1 & 2\\ 3 & 4\end{bmatrix}\begin{bmatrix}5 & 6\\ 7 & 8\end{bmatrix} = \begin{bmatrix}19 & 22\\ 43 & 50\end{bmatrix}$$

```python
from dezero import Variable
import dezero.functions as F

a = Variable(np.array([[1, 2], [3, 4]]))
b = Variable(np.array([[5, 6], [7, 8]]))
c = F.matmul(a, b)
print(c.data)   # [[19 22] [43 50]]
```

### 1.3 Gradient 계산 — Rosenbrock 함수

Rosenbrock 함수는 최적화 알고리즘의 **벤치마크 문제**입니다.

$$y = 100\,(x_1 - x_0^{\,2})^2 + (x_0 - 1)^2$$

- 전역 최소: $(x_0^*, x_1^*) = (1, 1)$, $y^* = 0$
- 좁고 휘어진 골짜기 → 1차 미분만으로는 최적화가 까다로움

해석적 그래디언트:

$$\frac{\partial y}{\partial x_0} = -400\,x_0(x_1 - x_0^{\,2}) + 2(x_0 - 1)$$

$$\frac{\partial y}{\partial x_1} = 200\,(x_1 - x_0^{\,2})$$

$(x_0, x_1) = (0, 2)$ 에서의 값:

$$\left.\frac{\partial y}{\partial x_0}\right|_{(0,2)} = -2,\qquad \left.\frac{\partial y}{\partial x_1}\right|_{(0,2)} = 400$$

```python
def rosenbrock(x0, x1):
    return 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
y = rosenbrock(x0, x1)
y.backward()
print(x0.grad, x1.grad)   # -2.0, 400.0
```

### 1.4 Gradient Descent

$$x_0^{k+1} = x_0^{k} - \alpha\,\frac{\partial y}{\partial x_0}(x_0^{k}, x_1^{k})$$

$$x_1^{k+1} = x_1^{k} - \alpha\,\frac{\partial y}{\partial x_1}(x_0^{k}, x_1^{k})$$

| 기호 | 의미 |
|------|------|
| $\alpha$ | learning rate (학습률) |
| $k$ | 반복(iteration) 인덱스 |

> ⚠️ **주의:** 매 반복마다 `x.cleargrad()` (또는 `optimizer.zero_grad()`) 로 그래디언트를 0 으로 초기화해야 합니다. DeZero/PyTorch 의 `grad` 는 **누적(accumulate)** 되기 때문입니다.

---

## 2. 회귀 응용 (Regression)

### 2.1 Linear Regression

**모델**

$$y = Wx + b$$

**데이터**: $(x_1, y_1), \dots, (x_N, y_N)$

**손실 (Mean Squared Error)**

$$L(W, b) = \frac{1}{N}\sum_{n=1}^{N}\bigl(W x_n + b - y_n\bigr)^2$$

**목적**

$$(W^*, b^*) = \arg\min_{(W, b)} L$$

**Gradient Descent**

$$W^{k+1} = W^{k} - \alpha\,\frac{\partial L}{\partial W}(x_k, y_k),\qquad b^{k+1} = b^{k} - \alpha\,\frac{\partial L}{\partial b}(x_k, y_k)$$

### 2.2 실습 #1 — `dezero3.py` ($y = 2x + 5$)

목표: 노이즈가 섞인 $y = 2x + 5$ 데이터로부터 $W \approx 2, b \approx 5$ 를 학습.

```python
# week10/dezero3.py 의 핵심 흐름
import numpy as np
from dezero import Variable
import dezero.functions as F

np.random.seed(0)
x = np.random.rand(100, 1)
y = 2 * x + 5 + np.random.rand(100, 1)   # 노이즈 포함 데이터

W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))

def predict(x):
    return F.matmul(x, W) + b

lr, iters = 0.1, 100
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    W.cleargrad(); b.cleargrad()
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data

print(W, b)
```

> 📄 전체 코드: [`dezero3.py`](./dezero3.py)

### 2.3 Nonlinear Regression (MLP)

**모델**: Multi-Layer Perceptron (다층 퍼셉트론)

$$h = \sigma\,(W^{(1)} x + b^{(1)}),\qquad \hat{y} = W^{(2)} h + b^{(2)}$$

- $\sigma(\cdot)$: 비선형 활성화 (sigmoid, tanh, ReLU …)
- *Affine transformation* 의 반복 + 비선형 함수 → **임의의 함수**를 근사 가능 (Universal Approximation Theorem)

DeZero 에서는 다음 두 가지 방식을 선택할 수 있습니다.

| 방식 | 장점 |
|------|------|
| `Layer` 클래스를 직접 조합 | 내부 구조를 이해하기 좋음 |
| `models.MLP([10, 1])` 와 `optimizers.SGD` 사용 | PyTorch 와 거의 동일한 간결한 API |

### 2.4 실습 #2 — `dezero4.py` ($y = \sin(2\pi x)$)

```python
# week10/dezero4.py 의 핵심 흐름
import numpy as np
from dezero import Variable
import dezero.functions as F
from dezero.models import MLP
from dezero.optimizers import SGD

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1) * 0.1

model = MLP((10, 1))
opt = SGD(lr=0.2).setup(model)

for i in range(10000):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()
    opt.update()
```

> 📄 전체 코드: [`dezero4.py`](./dezero4.py)

---

## 3. 강화학습 복습

### 3.1 Policy Evaluation & Update

```
   ┌────────────┐  return G_t   ┌────────────┐
   │   Value    │ ◀──────────── │   Policy   │
   │  V(s) /    │               │  π(a | s)  │
   │  Q(s, a)   │ ────────────▶ │            │
   └────────────┘   (Update)    └────────────┘
```

- **Evaluation**: 상태(또는 상태-행동)의 *가치 함수* 를 계산
- **Update**: 가치 함수에서 *최적 행동* 을 도출 → 예: $\varepsilon$-greedy

### 3.2 Q-Learning 요약

| Evaluation Method | 계산 범위 | 모델 |
|-------------------|-----------|------|
| **DP** (Dynamic Programming) | 모든 $(s, a)$ | Bellman's Equation |
| **MC** (Monte Carlo) | 에피소드에 등장한 $(s, a)$ | 없음 (sample mean) |
| **TD / Q-Learning** | 진행 중인 $(s, a)$ | Bellman's *Optimality* Equation |

**Q-Learning 알고리즘**

1. Start state $S_0$ → goal state $S_N$ 까지 진행하며 매 step 마다 $Q(S_t, A_t)$ 와 정책 $\pi$ 를 update.

    - Evaluation (Bellman's optimality):
      $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\bigl[R_t + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)\bigr]$$
    - Update ($\varepsilon$-greedy):
      $$\pi(a \mid s) = \begin{cases} 1 - \varepsilon & a^* = \arg\max_a Q(s, a) \\[4pt] \varepsilon / |A| & \text{otherwise} \end{cases}$$

2. $K$ 개의 에피소드에 대해 1) 반복.

---

## 4. Q-Network

### 4.1 Q-Learning의 한계

Q-Learning 은 **모든 $(s, a)$ 쌍에 대한 Q 값**을 dictionary 나 table 로 저장합니다.

| 문제 | 상태 수 |
|------|--------|
| 3 × 4 Grid World | $12 \times 4 = 48$ |
| 체스(Chess) 보드 | $\approx 10^{123}$ |
| Atari 2600 픽셀 입력 | $\approx 256^{84 \times 84 \times 4}$ |

> ❌ **상태 수가 폭발하면**
> - 테이블/딕셔너리로 관리 불가
> - 막대한 수의 $(s, a)$ 에 대해 **독립적으로 평가/업데이트** 불가능
> - 비슷한 상태들 간에 **지식 일반화(generalization)** 도 불가능

### 4.2 Q-Network의 핵심 아이디어

> 🎯 **Q-Network = Q-Learning + Neural Network (regression model)**  
> 모든 $(s, a)$ 에 대한 $Q$ 값을 **신경망 $Q_\theta$ 로 근사**합니다.

| 장점 |
|------|
| 큰/연속 상태 공간에 적용 가능 |
| 비슷한 상태 간 **지식 일반화** |
| 입력만 바뀌면 동일한 코드로 Atari, 로봇 제어 등 다양한 문제에 적용 가능 |

### 4.3 One-Hot Vector

신경망에 *범주형(discrete) 상태* 를 넣을 때는 **One-Hot 인코딩**을 사용합니다.

- 여러 원소 중 **하나만 `1`**, 나머지는 `0`
- 범주 간 임의의 거리·순서 관계를 만들지 않음

**예 1**: 옷 사이즈 `S, M, L`

$$\text{S} \to (1, 0, 0),\quad \text{M} \to (0, 1, 0),\quad \text{L} \to (0, 0, 1)$$

**예 2**: 3 × 4 Grid World 에서 agent 위치 $(y, x)$, 총 12 상태

$$\underbrace{(0,0)}_{\text{index }0} \to (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)$$

$$\underbrace{(2,3)}_{\text{index }11} \to (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)$$

### 4.4 QNet 구조

이산 행동 공간에서는 **모든 행동의 Q 값을 한 번에 출력**하는 구조가 표준입니다 — 그래야 $\max_a Q(s, a)$ 계산이 *한 번의 forward pass* 로 끝나기 때문입니다.

```
                  ┌─────────────────────┐
   state s ─────▶ │        QNet         │ ─────▶  Q(s, a₁)
   (one-hot)      │  Linear(12 → 100)   │ ─────▶  Q(s, a₂)
                  │  ReLU               │ ─────▶  Q(s, a₃)
                  │  Linear(100 →  4)   │ ─────▶  Q(s, a₄)
                  └─────────────────────┘
```

| 항목 | 내용 |
|------|------|
| 입력 | one-hot state $\in \mathbb{R}^{|S|}$ |
| 출력 | 모든 행동의 Q 값 $\in \mathbb{R}^{|A|}$ |
| 손실 | Mean Squared Error |
| 최적화 | SGD / Adam |

```python
from dezero import Layer
from dezero.layers import Linear
import dezero.functions as F

class QNet(Layer):
    def __init__(self, action_size: int, hidden: int = 100):
        super().__init__()
        self.l1 = Linear(hidden)
        self.l2 = Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        return self.l2(x)
```

### 4.5 Q Function 의 학습 — Target $T$

#### Update equation

Q-Learning 의 **Bellman optimality update** 를 신경망의 *라벨(target)* 로 재해석합니다.

$$\boxed{\,T \;=\; R_t \;+\; \gamma\,\max_a Q\bigl(S_{t+1},\,a\bigr)\,}$$

| 기호 | 의미 |
|------|------|
| $T$ | 신경망의 **정답(label)** — Q 함수가 닮아가야 할 *목표값* |
| $R_t$ | 현재 step 의 즉시 보상 |
| $\gamma$ | discount factor $\in [0, 1]$ |
| $\max_a Q(S_{t+1}, a)$ | 다음 상태에서 *얻을 수 있는 최대 Q 값* (bootstrap) |

#### 학습 절차

1. 환경에서 $(S_t, A_t, R_t, S_{t+1})$ 을 얻는다.
2. **현재 Q 값**: $\hat{Q} = Q_\theta(S_t, A_t)$ — 신경망 forward.
3. **목표 Q 값**:  
   - 종료 상태(`done=True`) 이면 $T = R_t$  
   - 아니면 $T = R_t + \gamma \,\max_a Q_\theta(S_{t+1}, a)$  
   - 단, $T$ 의 그래디언트는 **차단** (target = constant 로 취급)
4. **손실**: $L = (T - \hat{Q})^2$
5. $\theta \leftarrow \theta - \alpha \,\nabla_\theta L$

> ⚠️ **TD-target 의 그래디언트 차단**  
> 만약 $T$ 에 대해서도 $\theta$ 의 그래디언트를 전파하면, "정답 자체가 움직이는 학습" 이 되어 **불안정**해집니다. DeZero/PyTorch 의 `.data` (또는 `.detach()`) 로 상수처럼 취급해야 합니다.

### 4.6 Q-Learning vs Q-Network 비교

```
                Q-Learning                          Q-Network

   ┌───┬───┬───┬───┐                       ┌──────────────────────┐
   │ Q │ Q │ Q │ Q │                       │                      │
   ├───┼───┼───┼───┤            s  ─────▶  │   Neural Network     │ ─▶ Q(s, a₁..a_n)
   │ Q │ Q │ Q │ Q │                       │      Q_θ(s, ·)       │
   ├───┼───┼───┼───┤                       │                      │
   │ Q │ Q │ Q │ Q │                       └──────────────────────┘
   └───┴───┴───┴───┘
        Table                                       Network
   (s, a) → 한 칸                              θ ∈ ℝᴾ — 모든 (s, a)
   independent 업데이트                         공유 파라미터 → 일반화
```

| 항목 | Q-Learning | Q-Network |
|------|-----------|-----------|
| 기본 개념 | **테이블 기반** Q-value 업데이트 | **신경망 기반** Q-value 업데이트 |
| Q 함수 표현 | $(s, a)$ 쌍을 테이블로 저장 (작은/이산 상태에 적합) | 신경망(regression) 으로 Q 근사 (큰/연속 상태도 가능) |
| Update 방식 | $Q(S_t, A_t) \leftarrow Q + \alpha[T - Q]$ | 신경망 출력 $=Q_\theta(S_t, A_t)$, 신경망 목표 $T = R_t + \gamma\,\max_a Q_\theta(S_{t+1}, a)$, loss 기반 학습 |
| 행동 선택 | $\varepsilon$-greedy 등 간단 전략 | $\varepsilon$-greedy 등 간단 전략 |
| 문제점 | 크거나 연속 상태 공간에 적용 어려움 | 신경망 성능/하이퍼파라미터에 결과 의존 |
| 적용 예 | Grid World, 간단한 게임 | Atari 게임, 로봇 제어, 자율주행 등 |

### 4.7 실습 #3 — `q_learning_nn.py`

3 × 4 Grid World 에서 다음의 흐름으로 동작합니다.

**학습 단계**

```
[reset env]
   │
   ▼
[ state (y, x) ──▶ one-hot ──▶ QNet ──▶ Q(s, ·) ]
   │
   ├── ε-greedy → action
   │
   ▼
[env.step] → next_state, reward, done
   │
   ▼
[ T = r + γ · max_a Q(s', a)  (or r if done) ]
   │
   ▼
[ loss = (T − Q(s, a))²  →  backward → step ]
```

**추론(inference) 단계**

학습된 QNet 으로 모든 상태의 $\arg\max_a Q(s, a)$ 를 출력하여 정책을 시각화합니다.

> 📄 전체 코드: [`q_learning_nn.py`](./q_learning_nn.py) — Q-Network 학습/추론  
> 📄 환경 코드: [`grid_world.py`](./grid_world.py) — 3×4 / 5×5 Grid World

---

## 5. 퀴즈 (Quiz)

### Q1 — Optimizer 비교

실습 #2 의 SGD 대신 다음 optimizer 를 적용하여 학습 곡선(loss)을 비교하시오.

- `MomentumSGD`
- `AdaGrad`
- `Adam`

### Q2 — 더 어려운 함수 적합

$$y = \sin(4\pi x),\qquad 0 \le x \le 1$$

위 함수에 대해 loss 가 최소화되도록 신경망 구조와 하이퍼파라미터를 조정하고 결과를 출력하시오.

> 💡 **힌트:** sin 의 주기가 짧아질수록 더 많은 은닉 뉴런 (또는 더 깊은 네트워크), 더 큰 학습 데이터, 더 작은 학습률이 필요합니다.

### Q3 — 5 × 5 Grid World

다음과 같이 보상이 배치된 5 × 5 Grid World 에 Q-Network 를 적용하여,

- 모든 셀의 $\max_a Q(s, a)$ 테이블
- $\arg\max_a Q(s, a)$ 로부터 도출된 정책

을 출력하시오.

```
   +---+---+---+---+---+
   |   |   |   |   |+1 |   ← Goal
   +---+---+---+---+---+
   |   |   |   |-1 |   |
   +---+---+---+---+---+
   |   |-1 |   |   |   |
   +---+---+---+---+---+
   |   |   |   |   |   |
   +---+---+---+---+---+
   |Sₛ |   |   |   |   |   ← Start
   +---+---+---+---+---+
```

> ⚙️ **튜닝 포인트**: hidden size, learning rate, 에피소드 수, $\varepsilon$ schedule, $\gamma$.

---

## 6. 요약

### Q-Learning vs Q-Network

| 항목 | Q-Learning | Q-Network |
|------|-----------|-----------|
| **기본 개념** | 테이블 기반 Q-value 업데이트 | 신경망 기반 Q-value 업데이트 |
| **Q 함수 표현** | $(s, a)$ 테이블 (작은/이산 상태에 적합) | 신경망 regression 모델 (큰/연속 상태) |
| **업데이트** | $Q \leftarrow Q + \alpha[R + \gamma\max Q' - Q]$ | $\text{loss} = (T - Q_\theta)^2$, 경사하강 |
| **타겟 $T$** | — (직접 갱신) | $T = R + \gamma\,\max_a Q_\theta(S', a)$ |
| **일반화** | 없음 (각 셀이 독립) | **있음** — 비슷한 상태는 비슷한 출력 |
| **행동 선택** | $\varepsilon$-greedy | $\varepsilon$-greedy |
| **약점** | 큰/연속 상태에 적용 곤란 | 신경망 학습 안정성에 의존 |
| **대표 응용** | Grid World, Tic-Tac-Toe | Atari (DQN), 로봇 제어, 자율주행 |

### 한 줄 정리

> **Q-Network 는 "테이블에 다 못 적는 거대한 Q 함수를, 신경망 한 개로 압축·일반화한 것"** 이다.  
> 이로써 강화학습은 비로소 *고차원(state) 문제* 로 확장될 수 있었고, 이후 **DQN → Double DQN → Dueling DQN → Rainbow** 로 발전해 갑니다.

---

## 참고 문헌

- 사이토 고키, *밑바닥부터 시작하는 딥러닝 4 — 강화학습 편*, 한빛미디어.
- Mnih et al., *Playing Atari with Deep Reinforcement Learning*, DeepMind, 2013.
- Mnih et al., *Human-level control through deep reinforcement learning*, **Nature**, 2015.
