# Baseline & 분산 감소 원리 — 직관과 수식

> 강의 자료 `10-Policy Gradient Method.pdf` **16~18페이지** (시험 성적 비유 ~ $V_{\pi_\theta}$ 베이스라인)  
> [← 메인 요약으로 돌아가기](./10-Policy_Gradient_Method.md#5-baseline)

---

## 1. 핵심 질문

REINFORCE는 이미 다음과 같은 형태로 정책을 업데이트합니다:

$$\mathbb{E}\left[\sum_t G_t\, \nabla_\theta \log \pi_\theta(A_t \mid S_t)\right]$$

그럼에도 불구하고 학습 신호에는 **무작위성(환경의 확률, 정책의 샘플링)** 이 크게 남습니다. 슬라이드는 다음 직관으로 설명합니다.

> 📌 베이스라인(baseline) $b(S_t)$ 를 빼 주면 **분산(variance)은 줄이되**, 어떤 조건에서는 **평균(=참되는 기울기 방향)** 을 바꾸지 않는다.

이 문서에서는 (1) **시험 성적 비유**로 분산 감소를 직관적으로 잡고, (2) **수학적으로 무엇이 “평균 불변 · 분산 감소”인지**를 정리합니다.

---

## 2. 슬라이드의 시험 성적 비유 (정리본)

학생 세 명의 **실제 시험 점수**(실측값)가 있다고 합시다:

| 학생 | 실제 점수 |
|------|----------|
| A | 90 |
| B | 70 |
| C | 50 |

**(과거 평균 같은) 예측값**까지 같이 적으면 다음과 같은 구조입니다.

### ① 원본 데이터

슬라이드 예시에서는 **예측(평균치)** 이 각각 대략 85, 75, 55 로 주어져, 실제값의 **분산**이 크게 보입니다.

예를 들어 (슬라이드 수치에 맞춘 전형적인 구성):

| 학생 | 실제값 $x$ | 예측값 $\hat{x}$ |
|------|-----------|----------------|
| A | 90 | 85 |
| B | 70 | 75 |
| C | 50 | 55 |

### ② “편차”로 바꾸기

$$\tilde{x} = x - \hat{x}$$

| 학생 | $x-\hat{x}$ |
|------|------------|
| A | +5 |
| B | -5 |
| C | -5 |

슬라이드에서는 **실제값의 분산**(약 466.667) 보다 **편차의 분산**(약 32.667)이 훨씬 작다는 그림입니다.

### 직관 한 줄 요약

> 학습 신호 전체 크기 자체가 아니라, **변동폭(noise 크기)** 를 줄이면 업데이트가 덜 들쭉날쭉해져 학습이 쉬워진다는 이야기입니다.

**(주의)** 베이스라인의 수학 증명은 “분산 줄이면 좋은 이유”와는 별개로, 다음 절처럼 **기울기의 기댓값이 유지된다**는 명제가 핵심입니다.

---

## 3. Policy Gradient 에서 왜 빼도 되나? ($b(S_t)$가 ‘올바른’ 베이스라인이라면)

REINFORCE:

$$\mathbb{E}\Bigl[\sum_t G_t\, \nabla_\theta \log \pi_\theta(A_t\mid S_t)\Bigr]$$

여기서 다음을 빼더라도 **동일**(기대값 의미에서 0항 추가)하길 원합니다:

$$\mathbb{E}\Bigl[\sum_t \bigl(G_t - b(S_t)\bigr)\, \nabla_\theta \log \pi_\theta(A_t\mid S_t)\Bigr]$$

두 식의 차이가 0이라는 것은, 각 시간 $t$ 에 대해:

$$\mathbb{E}\bigl[\, - b(S_t)\, \nabla_\theta \log \pi_\theta(A_t \mid S_t)\,\bigr] = 0$$

가 성립하면 됩니다.

### 조건

**$b(S_t)$ 가 행동 $A_t$ 와 조건부 독립**이라고 합시다. 즉 상태 $S_t=s$ 로 고정해 두면:

$$b(S_t)=b(s)$$

는 확률 변수 $A_t$ 와 무관한 상수처럼 취급됩니다.

### 핵등식: softmax 정책에서의 확률 합 규칙

아무 상태 $s$ 에서든:

$$\sum_{a\in\mathcal{A}} \pi_\theta(a \mid s) = 1$$

양변을 $\theta$ 로 미분하면:

$$\sum_{a\in\mathcal{A}} \nabla_\theta \pi_\theta(a\mid s) = 0$$

로그 미분 형태로 쓰면:

$$\mathbb{E}_{a\sim\pi_\theta(\cdot\mid s)}\bigl[\nabla_\theta \log \pi_\theta(a\mid s)\bigr]
= \sum_a \pi_\theta(a\mid s)\,\nabla_\theta \log \pi_\theta(a\mid s)=0$$

따라서 **상태 $S_t=s$ 에서**:

$$\mathbb{E}_{A_t\sim\pi_\theta(\cdot|s)}\bigl[b(s)\,\nabla_\theta \log\pi_\theta(A_t\mid s)\bigr]=b(s)\cdot 0=0$$

이것을 $S_t$ 에 대해 바깥쪽 기댓값으로 감싸도 0입니다.

$$\boxed{\mathbb{E}\bigl[b(S_t)\,\nabla_\theta \log \pi_\theta(A_t\mid S_t)\bigr]=0}$$

**(요지)** 상태에만 의존하는 어떤 함수 $b(S_t)$ 를 빼도, 평균적인 정책 경사 성질 자체가 유지되는 방향입니다. (증명은 정책이 모든 $a$ 에 대해 매끈하고 합 규칙이 성립하면 됩니다.)

---

## 4. “그럼 기울기는 그대로인데 왜 학습에 좋지?” — 분산 감소

한 스텝 업데이트 벡터를:

$$Z_t(G_t)=G_t\, \nabla_\theta \log \pi_\theta(A_t \mid S_t)$$

이라 하고 베이스라인 버전은:

$$\tilde Z_t(G_t)=\bigl(G_t-b(S_t)\bigr)\nabla_\theta \log \pi_\theta(A_t \mid S_t)$$

**기댓값이 같더라도** 분산/trace는 달라질 수 있습니다. 좋은 $b(\cdot)$ 는 $G_t$ 의 변동분 중 **설명 가능한 상태 의존 부분**을 잘 빼내서:

> $\|\mathrm{Var}(\tilde Z)\|$ 가 작아져 **같은 샘플 예산으로 더 안정적인 추정치**가 됩니다.

직관적으로 $b(S)\approx\mathbb{E}[G_t\mid S_t{=}S]$ 가 비슷할수록,  
$G_t-b(S)$ 는 “이 상태에서는 평균보다 좋았다/나빴따” 같은 **비교 신호**에 가까워져 흔들림이 줄어듭니다.

---

## 5. 좋은 baseline 의 선택과 $V$

슬라이드에서는:

$$b(S_t) \approx V_{\pi_\theta}(S_t)$$

즉 해당 상태에서 **정책 $\pi_\theta$ 로 얻게 될 평균적인 수익**을 빼 준 것과 같은 역할이라고 안내합니다.

| 후보 baseline | 장단 |
|---------------|-----|
| 상수 | 구현 간단하지만 상태별 크기 차이 조정이 약함 |
| 과거 에피소드 평균 reward | 간단히 분산 줄이기에 도움이 될 수 있음 |
| **$V_{\pi_\theta}(s)$** | 상태별 평균 수익에 가까워 **신호 정제**가 잘 되는 편 |

> 다음 단계: $V_{\pi_\theta}$ 를 정확히 모르므로 **신경망으로 학습**(Critic) → Actor-Critic

- [← Actor-Critic (메인 요약)](./10-Policy_Gradient_Method.md#6-actor-critic)
- [← 수식 증명 (PG)](./10-1-Policy_Gradient_Math.md)

---

## 6. 흔한 오해 두 가지

1. **“베이스라인을 빼면 보상 크기 자체가 작아져서 학습이 느려진다?”**  
   기울기 **기대값**(방향성) 관점에서는 올바른 베이스라인 추가는 무해합니다. 느려짐/빨라짐은 **분산과 스텝 크기**, 그리고 $b$ 근사 오차 문제로 나타납니다.

2. **“아무거나 빼도 되나?”**  
   상태에만 의존하는 $b(S_t)$ 라는 조건과, 정확히는 증명에 쓰이는 확률 합 규칙이 중요합니다. **행동에 의존하는 잘못된 빼기**는 기대 기울기를 바꿀 수 있습니다.

---

## 관련 문서

- [전체 요약](./10-Policy_Gradient_Method.md)
- [수식 증명 (PG)](./10-1-Policy_Gradient_Math.md)
- [Actor-Critic 코드 해설](./10-3-Actor_Critic.md)
