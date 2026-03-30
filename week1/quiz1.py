"""
Quiz: Write a Bandit program that outputs the following graph
[Fig 1-17] Results of changing epsilon values for epsilon-greedy policy

- ε 값에 따른 ε-greedy 정책 성능 비교
- ε=0.01: 탐색 적음 → 수렴 느림, 장기 성능 낮음
- ε=0.1: 균형 잡힌 탐색 → 일반적으로 좋은 성능
- ε=0.3: 탐색 많음 → 초기 수렴 빠르나 장기 성능 상대적으로 낮음
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from bandit1 import Bandit, Agent

# ========== 실험 파라미터 ==========
runs = 200  # 각 ε별 200회 독립 실험
steps = 1000  # 실험당 1000 스텝
# 비교할 ε 값들. 순서에 따라 색상: 0.1(파랑), 0.3(주황), 0.01(초록)
epsilons = [0.1, 0.3, 0.01]

plt.figure(figsize=(8, 6))

# ========== 각 ε 값에 대해 실험 및 그래프 그리기 ==========
for epsilon in epsilons:
    # 이 ε에 대한 200회 실험의 스텝별 승률 저장
    all_rates = np.zeros((runs, steps))

    for run in range(runs):
        bandit = Bandit()
        agent = Agent(epsilon)  # 해당 ε으로 에이전트 생성
        total_reward = 0
        rates = []

        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            rates.append(total_reward / (step + 1))  # 스텝별 승률

        all_rates[run] = rates

    # 200회 실험의 스텝별 평균 승률
    avg_rates = np.average(all_rates, axis=0)
    plt.plot(avg_rates, label=str(epsilon))  # 범례에 ε 값 표시

# ========== [Fig 1-17] 그래프 꾸미기 ==========
plt.ylabel("Rates")  # y축: 승률
plt.xlabel("Steps")  # x축: 스텝
plt.title("[Fig 1-17] Results of changing epsilon values for epsilon-greedy policy")
plt.legend(loc="upper left")  # 범례: 각 ε 값
plt.grid(True, alpha=0.3)  # 격자선 (가독성)
plt.tight_layout()
plt.show()
