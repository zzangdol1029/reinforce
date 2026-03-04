"""
실습 #3: Non-Stationary Bandit Problem
- Stationary: 슬롯머신 승률이 고정, 과거/현재 보상에 동일 가중치
- Non-Stationary: 슬롯머신 승률이 변경, 최근 보상에 높은 가중치
- AlphaAgent: 상수 step-size(α) 사용 - Q_{n+1} = Q_n + α * (R_n - Q_n)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import random
import numpy as np
import matplotlib.pyplot as plt


class NonStatBandit:
    """Non-Stationary 슬롯머신
    - 승률(mean)이 매 스텝마다 랜덤 워크로 변경됨
    """

    def __init__(self, n_arms: int = 10, std: float = 1.0, seed: int = None):
        self.n_arms = n_arms
        self.std = std
        if seed is not None:
            np.random.seed(seed)
        # 초기 q* 값
        self.q_star = np.zeros(n_arms)

    def play(self, action: int) -> float:
        """해당 action으로 플레이하여 reward 반환"""
        reward = np.random.normal(self.q_star[action], self.std)
        # Non-stationary: 모든 arm의 q*에 랜덤 워크 적용
        self.q_star += np.random.normal(0, 0.01, self.n_arms)
        return reward


class AlphaAgent:
    """상수 step-size(α)를 사용하는 에이전트
    - Sample average: Q_{n+1} = Q_n + (1/n)*(R_n - Q_n) → stationary에 적합
    - Constant step-size: Q_{n+1} = Q_n + α*(R_n - Q_n) → non-stationary에 적합
    - α가 높을수록 최근 보상에 더 큰 가중치
    """

    def __init__(self, n_arms: int, epsilon: float, alpha: float = 0.1):
        """
        Args:
            n_arms: 슬롯머신 개수
            epsilon: exploration 확률
            alpha: step-size (학습률), 0 < α ≤ 1
        """
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.alpha = alpha
        self.q_values = np.zeros(n_arms)

    def select_action(self) -> int:
        """ε-greedy로 action 선택"""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_arms - 1)
        else:
            max_q = np.max(self.q_values)
            best_arms = np.where(self.q_values == max_q)[0]
            return int(np.random.choice(best_arms))

    def update(self, action: int, reward: float):
        """상수 step-size로 업데이트: Q += α * (R - Q)"""
        self.q_values[action] += self.alpha * (reward - self.q_values[action])


def run_non_stationary(n_runs: int = 2000, n_plays: int = 10000):
    """Non-stationary bandit 실험
    - Sample average (α = 1/n) vs Constant α = 0.1 비교
    """
    n_arms = 10
    epsilon = 0.1

    # Sample average agent (bandit.py의 Agent 사용)
    from bandit import Agent as SampleAvgAgent

    reward_sa = np.zeros((n_runs, n_plays))  # Sample Average
    reward_ca = np.zeros((n_runs, n_plays))  # Constant Alpha

    for run in range(n_runs):
        # 매 run마다 새로운 bandit
        bandit_sa = NonStatBandit(n_arms=n_arms, seed=run)
        bandit_ca = NonStatBandit(n_arms=n_arms, seed=run)

        agent_sa = SampleAvgAgent(n_arms=n_arms, epsilon=epsilon)
        agent_ca = AlphaAgent(n_arms=n_arms, epsilon=epsilon, alpha=0.1)

        for step in range(n_plays):
            # Sample Average
            action_sa = agent_sa.select_action()
            reward_s = bandit_sa.play(action_sa)
            agent_sa.update(action_sa, reward_s)
            reward_sa[run, step] = reward_s

            # Constant Alpha
            action_ca = agent_ca.select_action()
            reward_c = bandit_ca.play(action_ca)
            agent_ca.update(action_ca, reward_c)
            reward_ca[run, step] = reward_c

    # 평균
    avg_reward_sa = np.mean(reward_sa, axis=0)
    avg_reward_ca = np.mean(reward_ca, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(avg_reward_sa, label="Sample Average (α=1/n)", color="blue")
    plt.plot(avg_reward_ca, label="Constant α=0.1", color="red")
    plt.xlabel("Plays")
    plt.ylabel("Average Reward")
    plt.title("실습 #3: Non-Stationary Bandit\nSample Average vs Constant Step-Size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = Path(__file__).parent / "non_stationary_result.png"
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"그래프가 {output_path}로 저장되었습니다.")


if __name__ == "__main__":
    print("=== 실습 #3: Non-Stationary Bandit Problem ===\n")
    print("Sample Average vs Constant α 비교")
    print("- Sample Average: 과거/현재 보상 동등 가중치 (stationary에 적합)")
    print("- Constant α: 최근 보상에 높은 가중치 (non-stationary에 적합)\n")

    run_non_stationary(n_runs=2000, n_plays=10000)
