"""
Practice #2 bandit_avg.py
200회 실험 후 스텝별 평균 승률을 시각화

- 동일한 ε-greedy 정책으로 200번 독립 실험 수행
- 각 스텝에서 200회 실험의 승률을 평균내어 노이즈 감소
- [Fig 1-16] 단계별 승률 그래프 출력
"""

import sys
from pathlib import Path

# bandit1 모듈 import를 위해 week1 디렉토리를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from bandit1 import Bandit, Agent

# ========== 실험 파라미터 ==========
runs = 200  # 독립 실험 반복 횟수 (많을수록 평균 곡선이 부드러워짐)
steps = 1000  # 실험당 플레이(스텝) 횟수
epsilon = 0.1  # ε-greedy의 ε 값 (탐색 확률 10%)

# ========== 데이터 저장소 ==========
# all_rates[r][s] = run r의 step s 시점에서의 승률 (total_reward / (s+1))
# Shape: (200, 1000) = (runs, steps)
all_rates = np.zeros((runs, steps))

# ========== 200회 실험 수행 ==========
for run in range(runs):
    bandit = Bandit()  # 매 run마다 새로운 무작위 Bandit (승률이 매번 다름)
    agent = Agent(epsilon)
    total_reward = 0
    rates = []  # 이번 run의 스텝별 승률

    for step in range(steps):
        action = agent.get_action()  # ε-greedy로 행동 선택
        reward = bandit.play(action)  # 보상 획득 (0 또는 1)
        agent.update(action, reward)  # Q-value 샘플 평균 업데이트

        total_reward += reward
        # step 시점까지의 평균 보상 = 누적보상/스텝수 = 승률
        rates.append(total_reward / (step + 1))

    all_rates[run] = rates  # 이 run의 스텝별 승률 저장

# ========== 스텝별 평균 계산 ==========
# axis=0: run 축으로 평균 → 각 step에서 200개 run의 승률 평균
# avg_rates[s] = 200회 실험의 step s 시점 평균 승률
avg_rates = np.average(all_rates, axis=0)

# ========== [Fig 1-16] 그래프 출력 ==========
plt.ylabel("Rates")  # y축: 승률 (평균 보상)
plt.xlabel("Steps")  # x축: 스텝
plt.plot(avg_rates)  # 200회 실험 평균 승률 곡선
plt.title("[Fig 1-16] Win Rate by Step (Average after 200 experiments)")
plt.show()
