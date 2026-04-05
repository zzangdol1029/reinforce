# 실습 #2 mc_control.py
# Monte Carlo Method - Policy Control (ε-greedy)
#
# 목표
# - Monte Carlo(에피소드 샘플)로부터 Q(s,a)를 추정
# - Q로부터 ε-greedy 정책 π를 계속 개선하여(Policy Improvement)
#   결국 더 좋은 정책으로 수렴하게 만든다.
#
# 핵심 아이디어
# 1) 한 에피소드(trajectory) 동안 (s, a, r)을 전부 저장
# 2) 에피소드가 끝나면 뒤에서부터 G를 계산(returns)
# 3) 방문한 (s,a)에 대해 Q(s,a)를 갱신
# 4) 갱신된 Q를 기준으로 해당 state의 정책 π(·|s)를 ε-greedy로 다시 만든다

import common.mpl_window  # noqa: F401 — GridWorld보다 먼저: matplotlib 창 표시

from collections import defaultdict

import numpy as np

from common.gridworld import GridWorld


def greedy_probs(Q, state, epsilon=0.0, action_size=4):
    """
    ε-greedy 정책 확률 분포를 만들어 반환.

    Args:
        Q: Q(s,a) 저장소. dict 형태를 가정: Q[(state, action)] -> float
        state: 현재 상태 (h, w)
        epsilon: 탐색률 ε (0~1)
        action_size: 행동 개수 (GridWorld는 4)

    Returns:
        action_probs: {action: prob, ...} 형태의 확률 분포

    동작:
    - base_prob = ε / action_size 를 모든 행동에 뿌림 (탐색용)
    - Q가 최대인 행동(max_action)에 (1-ε)를 추가로 더함 (탐욕 행동)
    """
    # state에서 각 action에 대한 Q값을 리스트로 구성
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = int(np.argmax(qs))

    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}
    action_probs[max_action] += (1.0 - epsilon)
    return action_probs


class McAgent:
    """
    Monte Carlo Control 에이전트.

    주요 멤버:
    - gamma: 할인율
    - epsilon: ε-greedy의 ε (탐색률)
    - alpha: Q 업데이트에 사용하는 고정 step-size (상수 학습률)
    - pi[state]: 상태별 정책(행동 확률 분포)
    - Q[(state, action)]: 행동가치 함수
    - cnts[(state, action)]: 방문 횟수(슬라이드에는 남아있지만, 여기서는 alpha 업데이트를 사용)
    - memory: 한 에피소드 trajectory (state, action, reward) 기록
    """

    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 0.1  # (첫 번째 개선) ε-탐욕 정책의 ε
        self.alpha = 0.1    # (두 번째 개선) Q 함수 갱신 시 고정값 α
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)  # 초기 정책(균등)

        # Q는 (state, action)을 key로 가지는 dict 형태로 사용
        self.Q = defaultdict(lambda: 0.0)
        self.cnts = defaultdict(lambda: 0)  # (state, action) 방문 횟수

        self.memory = []

    def get_action(self, state):
        """
        현재 정책 π(·|s)로부터 행동을 샘플링하여 반환.
        """
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return int(np.random.choice(actions, p=probs))

    def add(self, state, action, reward):
        """한 스텝 (s, a, r) 기록."""
        self.memory.append((state, action, reward))

    def reset(self):
        """새 에피소드 시작 시 memory 초기화."""
        self.memory.clear()

    def update(self):
        """
        에피소드가 끝난 뒤 한 번 호출.

        1) 뒤에서부터 G 계산:
           G <- gamma * G + reward
        2) (state, action)마다 Q 업데이트:
           - 슬라이드 버전: 고정 α 업데이트
             Q <- Q + α (G - Q)
           - (참고) sample average라면: Q <- Q + (G-Q)/N
        3) 갱신된 Q로부터 현재 state의 정책 π(·|s)를 ε-greedy로 업데이트
        """
        G = 0.0
        for (state, action, reward) in reversed(self.memory):
            G = self.gamma * G + reward

            key = (state, action)
            self.cnts[key] += 1

            # (슬라이드) Q 함수 업데이트: 고정 α
            self.Q[key] += (G - self.Q[key]) * self.alpha

            # (슬라이드) 현재 state의 정책을 ε-greedy로 개선
            self.pi[state] = greedy_probs(self.Q, state, self.epsilon, self.action_size)


if __name__ == "__main__":
    env = GridWorld()
    agent = McAgent()

    episodes = 10000
    for episode in range(episodes):
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.add(state, action, reward)

            if done:
                agent.update()
                break

            state = next_state

    # 슬라이드와 동일: Q 마름모 시각화 + greedy policy 화살표 (한 창, 2패널)
    from common.gridworld_render import Renderer
    renderer = Renderer(env.reward_map, env.goal_state, env.wall_state)
    renderer.render_q_and_policy(agent.Q)

