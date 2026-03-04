# Week 1: Bandit Problem (밴딧 문제)

강화학습 기초 - Multi-Armed Bandit 문제 실습 소스입니다.

## 실습 목록

| 파일 | 설명 |
|------|------|
| `bandit.py` | **실습 #1** - 기본 Bandit: SlotMachine, Agent, ε-greedy 정책 |
| `bandit_avg.py` | **실습 #2** - Sample Average를 이용한 Action Value 추정 |
| `non_stationary.py` | **실습 #3** - Non-Stationary Bandit, AlphaAgent |

## 실행 방법

```bash
# 의존성 설치
pip install -r requirements.txt

# 실습 #1 실행 (매번 다른 결과)
python week1/bandit.py

# 실습 #2 실행 (그래프 출력)
python week1/bandit_avg.py

# 실습 #3 실행 (Non-stationary 비교 그래프)
python week1/non_stationary.py
```

## 주요 개념

- **Action Value Q(a)**: 행동 a의 기댓값
- **ε-greedy**: exploitation(활용) + exploration(탐색) 균형
- **Incremental equation**: Q_{n+1} = Q_n + (1/n)(R_n - Q_n)
- **Constant α**: Non-stationary 환경에서 최근 보상에 높은 가중치
