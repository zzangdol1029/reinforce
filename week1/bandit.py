"""
실습 #1: Multi-Armed Bandit Problem
- Slot Machine (환경)과 Agent 구현
- ε-greedy 정책: exploitation & exploration
- Random number generation → 실행 시마다 다른 결과
"""

import random
import numpy as np


class SlotMachine:
    """슬롯머신 (Environment)
    - 각 슬롯머신은 고유한 확률분포(승률)를 가짐
    - play 시 reward(코인 개수) 반환
    """

    def __init__(self, mean: float, std: float = 1.0, seed: int = None):
        """
        Args:
            mean: 해당 슬롯머신의 기댓값 (실제 Action Value)
            std: 보상의 표준편차
            seed: 재현성을 위한 시드 (None이면 매번 다른 결과)
        """
        self.mean = mean
        self.std = std
        if seed is not None:
            np.random.seed(seed)

    def play(self) -> float:
        """슬롯머신을 플레이하여 reward(코인) 획득"""
        return np.random.normal(self.mean, self.std)


class Agent:
    """에이전트 (Player)
    - ε-greedy 정책으로 action 선택
    - exploitation: 현재 가장 좋은 슬롯머신 선택 (greedy)
    - exploration: 무작위로 슬롯머신 선택
    """

    def __init__(self, n_arms: int, epsilon: float = 0.1):
        """
        Args:
            n_arms: 슬롯머신(팔) 개수
            epsilon: exploration 확률 (0~1)
                     ε=0 → greedy (exploitation만)
                     ε=1 → random (exploration만)
        """
        self.n_arms = n_arms
        self.epsilon = epsilon
        # 각 action의 추정 가치 (Q-values)
        self.q_values = [0.0] * n_arms
        # 각 action의 선택 횟수
        self.action_counts = [0] * n_arms

    def select_action(self) -> int:
        """ε-greedy로 action(슬롯머신) 선택"""
        if random.random() < self.epsilon:
            # Exploration: 무작위 선택
            return random.randint(0, self.n_arms - 1)
        else:
            # Exploitation: 현재 가장 좋은 팔 선택 (greedy)
            max_q = max(self.q_values)
            best_arms = [i for i, q in enumerate(self.q_values) if q == max_q]
            return random.choice(best_arms)

    def update(self, action: int, reward: float):
        """Action value 업데이트 (샘플 평균)
        Q_{n+1} = Q_n + (1/n) * (R_n - Q_n)
        """
        n = self.action_counts[action] + 1
        self.q_values[action] += (reward - self.q_values[action]) / n
        self.action_counts[action] = n


def run_bandit(n_arms: int = 10, n_plays: int = 1000, epsilon: float = 0.1):
    """Bandit 실험 실행"""
    # Slot machines 생성 (각각 다른 mean을 가짐)
    np.random.seed()  # 매 실행마다 다른 결과
    slot_machines = [SlotMachine(mean=np.random.randn()) for _ in range(n_arms)]

    agent = Agent(n_arms=n_arms, epsilon=epsilon)

    total_reward = 0
    rewards_history = []

    for step in range(n_plays):
        # 1. Action 선택
        action = agent.select_action()

        # 2. Environment에서 reward 얻기
        reward = slot_machines[action].play()
        total_reward += reward
        rewards_history.append(reward)

        # 3. Agent 업데이트
        agent.update(action, reward)

    return total_reward, rewards_history, agent, slot_machines


if __name__ == "__main__":
    print("=== 실습 #1: Multi-Armed Bandit ===\n")
    print("ε-greedy 정책으로 1000회 플레이 (실행 시마다 다른 결과)\n")

    total_reward, rewards, agent, machines = run_bandit(
        n_arms=10, n_plays=1000, epsilon=0.1
    )

    print(f"총 획득 코인: {total_reward:.2f}")
    print(f"평균 보상: {total_reward / len(rewards):.2f}")
    print(f"\n실제 Action Values (슬롯머신 기댓값):")
    for i, m in enumerate(machines):
        print(f"  Arm #{i+1}: {m.mean:.3f}")
    print(f"\n추정된 Action Values (Q-values):")
    for i, q in enumerate(agent.q_values):
        print(f"  Arm #{i+1}: {q:.3f} (선택 횟수: {agent.action_counts[i]})")
