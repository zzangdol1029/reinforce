"""
실습 #1 bandit.py
Multi-Armed Bandit 문제 - ε-탐욕 정책 구현

- Multi-Armed Bandit: 여러 개의 슬롯머신(팔) 중 최적의 것을 찾는 문제
- ε-greedy: 확률 ε로 탐색(랜덤), (1-ε)로 활용(최선 선택)
- Sample Average: Q_{n+1} = Q_n + (1/n) * (R_n - Q_n) 으로 가치 추정
"""

import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    """
    슬롯머신 환경 (Environment)
    - Stationary: 각 팔의 승률이 고정되어 있음
    - 0~1 사이의 승률을 가진 여러 개의 팔로 구성
    """

    def __init__(self, arms=10):
        """
        Args:
            arms: 슬롯머신(팔) 개수. 기본값 10.
        """
        # 슬롯머신 각각의 승률을 [0, 1) 구간의 균등분포로 무작위 설정
        self.rates = np.random.rand(arms)

    def play(self, arm):
        """
        선택한 팔을 당겨 보상 반환
        Args:
            arm: 선택한 팔의 인덱스 (0 ~ arms-1)
        Returns:
            1: 승리 (확률=self.rates[arm])
            0: 패배 (확률=1-self.rates[arm])
        """
        # np.random.rand() < rate 이면 1 반환 (승률에 따른 베르누이 시행)
        if self.rates[arm] > np.random.rand():
            return 1
        return 0


class Agent:
    """
    ε-greedy 정책을 사용하는 학습 에이전트
    - Sample Average(샘플 평균) 방식으로 Q-value 추정
    - Incremental formula: 저장 공간 O(1)로 효율적 업데이트
    """

    def __init__(self, epsilon, action_size=10):
        """
        Args:
            epsilon: 탐색 확률. 0~1. ε만큼 무작위, (1-ε)만큼 탐욕적 선택.
            action_size: 행동(팔) 개수. Bandit의 arms와 일치해야 함.
        """
        self.epsilon = epsilon  # 무작위로 행동할 확률 (exploration)
        self.Qs = np.zeros(action_size)  # Q(a): 각 행동의 추정 가치 (Action Value)
        self.ns = np.zeros(action_size)  # n(a): 각 행동이 선택된 누적 횟수

    def update(self, action, reward):
        """
        Sample Average 방식으로 Q-value 갱신
        수식: Q_{n+1} = Q_n + (1/n) * (R_n - Q_n)
        - 1/n이 학습률(learning rate) 역할
        - 새 보상과 추정치의 오차에 비례해 업데이트
        """
        self.ns[action] += 1  # 해당 행동 선택 횟수 증가
        # 증분 평균 공식: 기존 Q에 (보상-기존Q)/n 을 더함
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        """
        ε-greedy 정책으로 행동 선택
        Returns:
            선택한 팔의 인덱스
        """
        if np.random.rand() < self.epsilon:
            # Exploration: epsilon 확률로 무작위 행동 (탐색)
            return np.random.randint(len(self.Qs))
        else:
            # Exploitation: (1-epsilon) 확률로 현재 최고 Q값의 행동 (활용)
            return np.argmax(self.Qs)


def run_single(steps: int, epsilon: float):
    """
    단일 Bandit 실험 1회 실행
    Args:
        steps: 총 플레이(스텝) 횟수
        epsilon: ε-greedy의 ε 값
    Returns:
        total_reward: 총 획득 보상
        total_rewards: 스텝별 누적 보상 리스트 (길이=steps)
        rates: 스텝별 평균 보상(=승률) 리스트. rate[t] = total_reward/(t+1)
    """
    bandit = Bandit()  # 매 실행마다 새로운 무작위 Bandit 생성
    agent = Agent(epsilon)
    total_reward = 0
    total_rewards = []  # 각 스텝 시점의 누적 보상
    rates = []  # 각 스텝 시점까지의 평균 보상 (승률)

    for step in range(steps):
        action = agent.get_action()  # ε-greedy로 행동 선택
        reward = bandit.play(action)  # Bandit에서 보상 획득
        agent.update(action, reward)  # Q-value 업데이트

        total_reward += reward
        total_rewards.append(total_reward)  # 누적 보상 기록
        rates.append(total_reward / (step + 1))  # 현재까지 평균 = 승률
    return total_reward, total_rewards, rates


if __name__ == "__main__":
    steps = 1000  # 실험당 스텝 수
    epsilon = 0.1  # ε-greedy 파라미터 (10% 확률로 탐색)
    n_runs = 10  # 10회 실행 결과를 별도 그래프로 표시

    # === 1. 단일 실행 결과 ===
    total_reward, total_rewards, rates = run_single(steps, epsilon)
    print(f"Single run - Total reward: {total_reward}")

    # [Fig 1-12] 단계별 총 보상, [Fig 1-13] 단계별 승률
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)  # 왼쪽: 총 보상
    plt.plot(total_rewards)
    plt.ylabel("Total reward")
    plt.xlabel("Steps")
    plt.title("[Fig 1-12] Total Reward by Step (Single Run)")

    plt.subplot(1, 2, 2)  # 오른쪽: 승률
    plt.plot(rates)
    plt.ylabel("Rates")
    plt.xlabel("Steps")
    plt.title("[Fig 1-13] Win Rate by Step (Single Run)")

    plt.tight_layout()
    plt.show()

    # === 2. 10회 실행 - 각 run을 서로 다른 색으로 표시 ===
    colors = plt.cm.tab10(np.linspace(0, 1, n_runs))  # 10가지 색상
    all_total_rewards = []
    all_rates = []

    for run in range(n_runs):
        _, tr, r = run_single(steps, epsilon)
        all_total_rewards.append(tr)
        all_rates.append(r)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)  # 왼쪽: 10회 총 보상 곡선 (각기 다른 색)
    for run in range(n_runs):
        plt.plot(all_total_rewards[run], color=colors[run], label=f"Run {run + 1}")
    plt.ylabel("Total reward")
    plt.xlabel("Steps")
    plt.title("[Fig] Total Reward by Step (10 Runs)")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)  # 오른쪽: 10회 승률 곡선 (각기 다른 색)
    for run in range(n_runs):
        plt.plot(all_rates[run], color=colors[run], label=f"Run {run + 1}")
    plt.ylabel("Rates")
    plt.xlabel("Steps")
    plt.title("[Fig] Win Rate by Step (10 Runs)")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
