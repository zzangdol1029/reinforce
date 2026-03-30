"""
Practice #3 non_stationary.py
비정상(Non-Stationary) Bandit에서 Sample Average vs Constant Alpha 비교

- Stationary: 슬롯머신 승률 고정 → Sample Average(1/n) 적합
- Non-Stationary: 승률이 매 스텝 변함 → Constant Alpha(α)가 최근 보상에 높은 가중치
- [Fig 1-20] 표본 평균과 고정값 α에 의한 갱신 비교
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from bandit1 import Agent  # Sample Average 방식 Agent


class NonStatBandit:
    """
    비정상(Non-Stationary) Bandit 환경
    - 매 play() 호출 시 모든 팔의 승률에 가우시안 노이즈를 더해 변동
    - 최적의 팔이 시간에 따라 바뀌므로, 최근 보상에 더 큰 가중치를 두는 방식이 유리
    """

    def __init__(self, arms=10):
        """
        Args:
            arms: 팔(슬롯머신) 개수
        """
        self.arms = arms
        self.rates = np.random.rand(arms)  # 초기 승률 [0, 1)

    def play(self, arm):
        """
        선택한 팔을 당겨 보상 반환. 호출 시 모든 rates에 노이즈 추가.
        Args:
            arm: 선택한 팔 인덱스
        Returns:
            1 또는 0 (승률에 따른 베르누이 시행)
        """
        rate = self.rates[arm]  # 현재 보상 산출 전 rate 사용
        # 노이즈 추가: 모든 팔의 승률에 N(0, 0.1²) 더함 → 비정상 환경
        self.rates += 0.1 * np.random.randn(self.arms)
        if np.random.rand() < rate:
            return 1
        return 0


class AlphaAgent:
    """
    상수 step-size(α)를 사용하는 에이전트
    - 갱신식: Q_{n+1} = Q_n + α * (R_n - Q_n)
    - α가 크면 최근 보상에 더 큰 가중치 (non-stationary에 유리)
    - α가 작으면 과거 보상도 오래 반영 (stationary에 가깝게 동작)
    """

    def __init__(self, epsilon, alpha, actions=10):
        """
        Args:
            epsilon: ε-greedy 탐색 확률
            alpha: 고정 step-size. 0 < α ≤ 1. 여기서 0.8 사용.
            actions: 행동(팔) 개수
        """
        self.Qs = np.zeros(actions)  # Q-value 추정
        self.epsilon = epsilon
        self.alpha = alpha  # 고정값 α (Sample Average는 1/n 사용)

    def update(self, action, reward):
        """
        상수 α로 Q-value 갱신
        수식: Q += α * (R - Q)  →  최근 보상에 α 비율로 반영
        """
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha

    def get_action(self):
        """ε-greedy로 행동 선택"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.Qs))
        return np.argmax(self.Qs)


# ========== 실험 파라미터 ==========
runs = 200  # 각 에이전트 타입별 200회 실험
steps = 1000  # 실험당 스텝
epsilon = 0.1  # ε-greedy
alpha = 0.8  # AlphaAgent의 step-size (높을수록 최근 보상 중시)
agent_types = ["sample average", "alpha const update"]  # 비교할 두 방식
results = {}  # agent_type → 스텝별 평균 승률 배열

# ========== 두 에이전트 타입별로 실험 수행 ==========
for agent_type in agent_types:
    all_rates = np.zeros((runs, steps))

    for run in range(runs):
        # Sample Average: bandit1.Agent (Q += (R-Q)/n)
        # Alpha Const: AlphaAgent (Q += α*(R-Q))
        agent = Agent(epsilon) if agent_type == "sample average" else AlphaAgent(epsilon, alpha)
        bandit = NonStatBandit()  # 매 run마다 새 Bandit, 매 step마다 rates 변동
        total_reward = 0
        rates = []

        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)  # play 시 rates에 노이즈 추가됨
            agent.update(action, reward)

            total_reward += reward
            rates.append(total_reward / (step + 1))  # 스텝별 승률

        all_rates[run] = rates

    avg_rates = np.average(all_rates, axis=0)  # 200회 평균
    results[agent_type] = avg_rates

# ========== [Fig 1-20] 그래프 출력 ==========
# 표본 평균(Sample Average) vs 고정 α(Constant Alpha) 갱신 비교
plt.figure()
plt.ylabel("Average Rates")
plt.xlabel("Steps")
for key, avg_rates in results.items():
    plt.plot(avg_rates, label=key)
plt.legend()
plt.show()
