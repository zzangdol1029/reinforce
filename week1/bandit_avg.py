"""
실습 #2: Bandit with Sample Average
- Action Value 추정: Incremental equation 사용
- Q_{n+1} = Q_n + (1/n) * (R_n - Q_n)
- 1/n: learning rate (학습률)
- 여러 실행의 평균 성능을 그래프로 시각화
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import random
import numpy as np
import matplotlib.pyplot as plt
from bandit import SlotMachine, Agent


def run_single_bandit(n_arms: int, n_plays: int, epsilon: float, seed: int = None):
    """단일 Bandit 실행 - 평균 보상 히스토리 반환"""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    slot_machines = [SlotMachine(mean=np.random.randn(), seed=seed) for _ in range(n_arms)]
    agent = Agent(n_arms=n_arms, epsilon=epsilon)

    rewards_history = []
    for step in range(n_plays):
        action = agent.select_action()
        reward = slot_machines[action].play()
        agent.update(action, reward)
        rewards_history.append(reward)

    return rewards_history


def run_bandit_average(n_runs: int = 2000, n_plays: int = 1000, n_arms: int = 10):
    """여러 번 실행하여 평균 성능 측정"""
    epsilons = [0.0, 0.01, 0.1]  # greedy, 약한 exploration, 일반적 ε-greedy

    plt.figure(figsize=(12, 5))

    # 각 ε에 대한 평균 보상 그래프
    for epsilon in epsilons:
        all_rewards = np.zeros((n_runs, n_plays))
        for run in range(n_runs):
            rewards = run_single_bandit(n_arms, n_plays, epsilon, seed=run)
            all_rewards[run] = rewards

        # 각 스텝별 평균 보상
        avg_rewards = np.mean(all_rewards, axis=0)
        # 각 스텝별 optimal action 선택 비율
        # (가장 높은 mean을 가진 슬롯머신 선택 비율은 별도 계산 필요)

        plt.subplot(1, 2, 1)
        plt.plot(avg_rewards, label=f"ε = {epsilon}")
        plt.subplot(1, 2, 2)
        plt.plot(avg_rewards, label=f"ε = {epsilon}")

    plt.subplot(1, 2, 1)
    plt.xlabel("Plays")
    plt.ylabel("Average Reward")
    plt.title("실습 #2: Average Reward over 2000 runs")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.xlabel("Plays")
    plt.ylabel("Average Reward")
    plt.title("실습 #2: Average Reward (확대)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(__file__).parent / "bandit_avg_result.png"
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"그래프가 {output_path}로 저장되었습니다.")


def demo_single_run():
    """단일 실행 데모 - Action Value 추정 과정 시각화"""
    print("=== 실습 #2: Sample Average를 이용한 Action Value 추정 ===\n")
    print("두 대의 슬롯머신 예시:")
    print("  - Arm A: 3번 플레이 → 보상 0, 1, 5 → Q(a) = (0+1+5)/3 = 2")
    print("  - Arm B: 3번 플레이 → 보상 1, 0, 0 → Q(b) = (1+0+0)/3 = 0.333\n")

    np.random.seed(42)
    slot_machines = [
        SlotMachine(mean=1.5, std=1.0, seed=42),  # Arm 1
        SlotMachine(mean=0.5, std=1.0, seed=43),  # Arm 2
    ]
    agent = Agent(n_arms=2, epsilon=0.1)

    for step in range(10):
        action = agent.select_action()
        reward = slot_machines[action].play()
        agent.update(action, reward)
        arm_name = "A" if action == 0 else "B"
        print(f"Step {step+1}: Arm {arm_name} 선택 → Reward={reward:.2f} | Q(A)={agent.q_values[0]:.3f}, Q(B)={agent.q_values[1]:.3f}")


if __name__ == "__main__":
    # 단일 실행 데모
    demo_single_run()

    print("\n" + "=" * 50)
    print("여러 ε 값에 대한 평균 성능 그래프 생성...")
    run_bandit_average(n_runs=2000, n_plays=1000)
