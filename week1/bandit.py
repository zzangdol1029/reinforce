"""
Practice #1: Multi-Armed Bandit Problem
- Slot Machine (Environment) and Agent implementation
- epsilon-greedy policy: exploitation & exploration
- Random number generation -> different results each run
"""

import random
import numpy as np


class SlotMachine:
    """Slot Machine (Environment)
    - Each slot machine has its own probability distribution (win rate)
    - Returns reward (coin count) on play
    """

    def __init__(self, mean: float, std: float = 1.0, seed: int = None):
        """
        Args:
            mean: Expected value of this slot machine (true Action Value)
            std: Standard deviation of reward
            seed: Seed for reproducibility (None for different results each run)
        """
        self.mean = mean
        self.std = std
        if seed is not None:
            np.random.seed(seed)

    def play(self) -> float:
        """Play the slot machine and get reward (coins)"""
        return np.random.normal(self.mean, self.std)


class Agent:
    """Agent (Player)
    - Selects action using epsilon-greedy policy
    - exploitation: choose current best slot machine (greedy)
    - exploration: choose slot machine at random
    """

    def __init__(self, n_arms: int, epsilon: float = 0.1):
        """
        Args:
            n_arms: Number of slot machines (arms)
            epsilon: Exploration probability (0~1)
                     epsilon=0 -> greedy (exploitation only)
                     epsilon=1 -> random (exploration only)
        """
        self.n_arms = n_arms
        self.epsilon = epsilon
        # Estimated value for each action (Q-values)
        self.q_values = [0.0] * n_arms
        # Selection count for each action
        self.action_counts = [0] * n_arms

    def select_action(self) -> int:
        """Select action (slot machine) using epsilon-greedy"""
        if random.random() < self.epsilon:
            # Exploration: random selection
            return random.randint(0, self.n_arms - 1)
        else:
            # Exploitation: choose current best arm (greedy)
            max_q = max(self.q_values)
            best_arms = [i for i, q in enumerate(self.q_values) if q == max_q]
            return random.choice(best_arms)

    def update(self, action: int, reward: float):
        """Update action value (sample average)
        Q_{n+1} = Q_n + (1/n) * (R_n - Q_n)
        """
        n = self.action_counts[action] + 1
        self.q_values[action] += (reward - self.q_values[action]) / n
        self.action_counts[action] = n


def run_bandit(n_arms: int = 10, n_plays: int = 1000, epsilon: float = 0.1):
    """Run bandit experiment"""
    # Create slot machines (each with different mean)
    np.random.seed()  # Different results each run
    slot_machines = [SlotMachine(mean=np.random.randn()) for _ in range(n_arms)]

    agent = Agent(n_arms=n_arms, epsilon=epsilon)

    total_reward = 0
    rewards_history = []

    for step in range(n_plays):
        # 1. Select action
        action = agent.select_action()

        # 2. Get reward from environment
        reward = slot_machines[action].play()
        total_reward += reward
        rewards_history.append(reward)

        # 3. Update agent
        agent.update(action, reward)

    return total_reward, rewards_history, agent, slot_machines


if __name__ == "__main__":
    print("=== Practice #1: Multi-Armed Bandit ===\n")
    print("1000 plays with epsilon-greedy policy (different results each run)\n")

    total_reward, rewards, agent, machines = run_bandit(
        n_arms=10, n_plays=1000, epsilon=0.1
    )

    print(f"Total coins earned: {total_reward:.2f}")
    print(f"Average reward: {total_reward / len(rewards):.2f}")
    print(f"\nTrue Action Values (slot machine expected values):")
    for i, m in enumerate(machines):
        print(f"  Arm #{i+1}: {m.mean:.3f}")
    print(f"\nEstimated Action Values (Q-values):")
    for i, q in enumerate(agent.q_values):
        print(f"  Arm #{i+1}: {q:.3f} (selection count: {agent.action_counts[i]})")
