# Week 2: Markov Decision Process & Bellman Equation

**03-Markov Decision Process.pdf**, **4.pdf** 기반 실습입니다.

## 실습 목록

| 파일 | 설명 |
|------|------|
| `grid_world_1x2.py` | 1x2 Grid World MDP: 전이 테이블, 4가지 정책(μ1~μ4), Policy Evaluation, Optimal = μ2 |
| `grid_world_3x4.py` | 3x4 Grid World (Episodic): terminal state, 랜덤 정책에 대한 v_π |
| `bellman.py` | Bellman Expectation / Optimality 방정식, Value Iteration으로 v* 및 최적 정책 |

## 실행 방법

```bash
# 실습 #1: 1x2 Grid World (전이 테이블, 4 policies, v_π)
python week2/grid_world_1x2.py

# 실습 #2: 3x4 Grid World (Episodic)
python week2/grid_world_3x4.py

# 실습 #3: Bellman Equation (v_π, v*, 최적 정책)
python week2/bellman.py
```

## 주요 개념 (03-MDP)

- **MDP**: (S, A, T, r, γ) — 상태, 행동, 전이확률, 보상, 할인율
- **Episodic vs Continuing**: terminal state 유무
- **Return**: G_t = R_{t+1} + γ R_{t+2} + …
- **Value**: v_π(s), q_π(s,a) / v*(s), q*(s,a), 최적 정책 π*

## 주요 개념 (4.pdf – Bellman)

- **Bellman Expectation**: v_π(s) = Σ_a π(a|s)[ R(s,a) + γ Σ_{s'} P(s'|s,a) v_π(s') ]
- **Bellman Optimality**: v*(s) = max_a [ R(s,a) + γ Σ_{s'} P(s'|s,a) v*(s') ]
- **Prediction**: 주어진 π에 대한 v_π 계산
- **Control**: v* 및 최적 정책 찾기 (Value Iteration 등)
