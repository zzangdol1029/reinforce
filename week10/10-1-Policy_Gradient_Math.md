# Policy Gradient — 수식 증명 (로그 기울기 트릭)

> 강의 자료 `10-Policy Gradient Method.pdf` **4~5페이지** 수식의 **단계별 유도**  
> [← 메인 요약으로 돌아가기](./10-Policy_Gradient_Method.md#2-policy-gradient의-수학적-유도)

---

## 1. 목표: 무엇을 증명하는가?

**목적 함수** (에피소드가 끝날 때까지의 할인 수익의 기댓값):

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\bigl[\,G(\tau)\,\bigr]$$

우리가 구하고 싶은 것은 **$J(\theta)$ 를 $\theta$ 로 미분한 것** — 즉 **정책 경사(policy gradient)**:

$$\nabla_\theta J(\theta)$$

실제 학습에서는 **gradient ascent**:

$$\theta \leftarrow \theta + \alpha\, \nabla_\theta J(\theta)$$

| 기호 | 의미 |
|------|------|
| $\tau$ | 궤적(trajectory), 예: $\tau=(S_0,A_0,R_0,S_1,\ldots,S_T,R_T,S_{T+1})$ |
| $\pi_\theta$ | 파라미터 $\theta$ 를 갖는 **정책** (예: softmax 정책 네트워크) |
| $G(\tau)$ | 해당 궤적에서 받은 **총(할인) 수익** return |
| $\mathbb{E}_{\tau\sim\pi_\theta}[\cdot]$ | 정책 $\pi_\theta$ 로 에이전트를 굴린 뒤 얻은 궤적에 대한 **기댓값** |

> 💡 관문은 한 가지입니다. **분포 자체가 $\theta$ 에 의존**하므로, “기댓값의 미분”을 평소처럼 안쪽으로 들이밀 수 없습니다. 그래서 **확률 밀도 $p(\tau\mid\theta)$** 를 드러낸 다음 **로그 미분 트릭**을 씁니다.

---

## 2. 로그 미분 트릭(Log-Derivative Trick) 자체

아주 작은 분식 미분 규칙입니다.

$$\nabla_\theta \log f(\theta) = \frac{\nabla_\theta f(\theta)}{f(\theta)}$$

양변에 $f(\theta)$ 를 곱하면 자주 쓰이는 형태가 됩니다:

$$\boxed{\nabla_\theta f(\theta) = f(\theta)\,\nabla_\theta \log f(\theta)}$$

**직관:** “미분값”을 **“값 × 로그값의 미분”** 로 바꾸면, 나중에 기댓값 안으로 들어올 때 **샘플로 추정 가능한 형태**(score-function 형태)로 정리하기 쉽습니다.

---

## 3. 궤적의 확률 $p(\tau \mid \theta)$

**(일반적인) MDP** 에서 한 궤적은 다음 순서로 생성됩니다.

1. 시작 상태 $S_0 \sim \mu$
2. 각 $t$ 에 대해 $A_t \sim \pi_\theta(\cdot \mid S_t)$
3. 환경이 다음 상태와 보상 $(S_{t+1}, R_t)$ 을 전이 규칙에 따라 발생

한 궤적의 결합 분포를 **기호 하나**로 줄이면:

$$p(\tau \mid \theta) = \mathbb{P}(S_0)\,\prod_{t=0}^{T-1} \pi_\theta(A_t \mid S_t)\, p(S_{t+1}, R_t \mid S_t, A_t)$$

**(마지막 스텝을 어떻게 정의했는지)** 에 따라 곱 인덱스가 $T$ 까지로 보이거나 $T{-}1$ 까지로 보일 수 있습니다. 증명의 본질은 변하지 않습니다.

**중요한 사실:**

> 환경의 전이 $p(s',r \mid s,a)$ 는 **정책 파라미터 $\theta$ 와 무관**합니다.

따라서 $\theta$ 에 대한 미분 시 **환경 항과 초기 상태 항은 미분 결과 0** 이 되어 사라지고, 남는 것은 정책의 곱 항입니다.

| 항 | $\nabla_\theta$ 에 대한 일반적인 기여 |
|----|--------------------------------------|
| $\mathbb{P}(S_0)$ | 0 |
| $\prod_t p(S_{t+1},R_t\mid S_t,A_t)$ | 0 |
| $\prod_t \pi_\theta(A_t\mid S_t)$ | **0이 아님** |

---

## 4. 목적 함수를 적분형으로 쓰기

이산 궤적(유한 개의 $\tau$)이면 합으로, 연속이라면 다음과 같이 씁니다.

$$J(\theta) = \int p(\tau \mid \theta)\, G(\tau)\, d\tau$$

이산 합 버전에서는 $J(\theta)=\sum_\tau p(\tau\mid\theta)\, G(\tau)$ 로 읽으면 됩니다.

미분하면 (적분–미분 교환 조건 아래):

$$\nabla_\theta J(\theta) = \int \nabla_\theta p(\tau \mid \theta)\, G(\tau)\, d\tau$$

---

## 5. 핵심 한 줄: $p$ 대신 $\log p$ 로 바꾸기

$$\nabla_\theta p(\tau\mid\theta) = p(\tau\mid\theta)\,\nabla_\theta \log p(\tau\mid\theta)$$

적분에 대입하면:

$$\nabla_\theta J(\theta) = \int p(\tau\mid\theta)\, G(\tau)\, \nabla_\theta \log p(\tau\mid\theta)\, d\tau$$

즉:

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\bigl[\,G(\tau)\,\nabla_\theta \log p(\tau\mid\theta)\,\bigr]}$$

---

## 6. $\log p(\tau\mid\theta)$ 를 풀어 쓰고 미분하기

$$\log p(\tau\mid\theta) = \log \mathbb{P}(S_0) + \sum_{t=0}^{T-1} \Bigl(\log \pi_\theta(A_t\mid S_t) + \log p(S_{t+1},R_t\mid S_t,A_t)\Bigr)$$

$\theta$ 로 미분하면 **$\theta$ 와 무관한 항은 0**:

$$\nabla_\theta \log p(\tau\mid\theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(A_t\mid S_t)$$

따라서:

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\left[\,G(\tau)\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(A_t\mid S_t)\right]}$$

슬라이드처럼 **합안의 순서만 바꾼** 형태와 동치입니다:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T} G(\tau)\,\nabla_\theta \log \pi_\theta(A_t\mid S_t)\right]$$

---

## 7. 왜 구현에서는 $\nabla_\theta \log \pi_\theta$ 만 보면 되는가?

- 정책 네트워크가 $\pi_\theta(\cdot\mid s)$ 를 출력하므로 **$\log\pi$ 및 그 기울기**는 자동미분으로 바로 계산 가능합니다.
- 환경 전이를 몰라도, **실제로 관측한 $(S_t,A_t)$** 만으로 학습 신호를 구성합니다.

---

## 8. Monte Carlo 추정

**$n$개 궤적:**

$$\nabla_\theta J(\theta) \approx \frac{1}{n}\sum_{i=1}^{n} G(\tau^{(i)})\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta\bigl(A_t^{(i)}\mid S_t^{(i)}\bigr)$$

**1샘플($n=1$):**

$$\nabla_\theta J(\theta) \approx G(\tau)\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(A_t\mid S_t)$$

---

## 9. 코드와 연결 (`loss += -log(prob) * G`)

한 궤적에 대한 MC 추정:

$$\widehat{\nabla_\theta J} \;=\; \sum_{t=0}^{T} G(\tau)\,\nabla_\theta \log \pi_\theta(A_t\mid S_t)$$

**minimize** 용 스칼라:

$$\mathcal{L} = -\sum_{t=0}^{T} G(\tau)\,\log \pi_\theta(A_t\mid S_t)$$

`loss.backward()` 는 $\nabla_\theta \mathcal{L} = -\widehat{\nabla_\theta J}$ 에 해당합니다.

> 구현에서는 옵티마이저 기본 업데이트가 “loss를 줄이는 방향”이므로, **부호 설계**(예: 학습률 부호, `loss = -(...)`)까지 포함해 ascent가 되는지 한 번 검증하는 것이 좋습니다.

---

## 10. 다음 단계: REINFORCE

여기까지는 가중치에 **전체 궤적 수익 $G(\tau)$** 를 쓴 형태입니다. **REINFORCE**는 시간 $t$ 이후 보상만 쓴 $G_t$ 로 바꿔 **분산을 줄입니다**.

- [← REINFORCE (메인 요약)](./10-Policy_Gradient_Method.md#4-reinforce)

---

## 관련 문서

- [전체 요약](./10-Policy_Gradient_Method.md)
- [Baseline 분산 직관](./10-2-Baseline_Intuition.md)
- [Actor-Critic 코드 해설](./10-3-Actor_Critic.md)
