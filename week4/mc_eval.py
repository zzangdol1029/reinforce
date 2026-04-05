# 실습 #1 mc_eval.py
# Monte Carlo Method - Policy Evaluation
#
# 목표
# - 랜덤 정책(무작위 행동)을 따르는 에이전트를 여러 에피소드 실행
# - 각 에피소드에서 (state, action, reward) 시퀀스를 저장
# - 에피소드가 끝난 뒤, Monte Carlo(에피소드 샘플)로부터 상태가치 V(s)를 추정
#   V(s) <- V(s) + (G - V(s)) / N(s)  (증분 평균 / sample average)

import common.mpl_window  # noqa: F401 — GridWorld보다 먼저: matplotlib 창 표시

from collections import defaultdict

import numpy as np

from common.gridworld import GridWorld


class RandomAgent:
    """
    무작위(Random) 정책으로 행동하는 에이전트.

    - pi[s] = {a0:p0, a1:p1, ...} 형태의 정책(확률 분포)
      여기서는 모든 상태에서 4개 행동을 균등확률(0.25)로 선택.
    - memory: 한 에피소드 동안의 (state, action, reward) 기록
    - V[s]: 상태 가치 추정치
    - cnts[s]: 상태 방문(또는 업데이트) 횟수 (증분 평균용)
    """

    def __init__(self):
        self.gamma = 0.9
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)  # 정책
        self.V = defaultdict(lambda: 0.0)  # 가치 함수 V(s)
        self.cnts = defaultdict(lambda: 0)  # 상태별 카운트 N(s)

        # 한 에피소드의 trajectory 기록 공간
        self.memory = []

    def get_action(self, state):
        """
        현재 상태 state에서 정책 pi에 따라 행동 샘플링.
        - actions: 가능한 행동들
        - probs: 각 행동 확률
        """
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return int(np.random.choice(actions, p=probs))

    def add(self, state, action, reward):
        """(state, action, reward) 한 스텝 데이터를 memory에 저장."""
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        """새 에피소드를 시작할 때, 이전 에피소드 기록을 비움."""
        self.memory.clear()

    def eval(self):
        """
        Monte Carlo로 V(s)를 업데이트.

        - 에피소드 끝에서부터 역순으로 G를 누적:
            G_t = r_{t+1} + gamma * G_{t+1}
        - 각 state에 대해 증분 평균(sample average)으로 V(s) 갱신:
            V <- V + (G - V) / N
        """
        G = 0.0
        for (state, action, reward) in reversed(self.memory):  # 역방향으로(reversed) 따라가기
            G = self.gamma * G + reward
            self.cnts[state] += 1
            self.V[state] += (G - self.V[state]) / self.cnts[state]


# 슬라이드 오른쪽: 환경 생성 후 학습 끝에 V(s)를 격자(네모) 히트맵으로 표시 — env.render_v(agent.V)
if __name__ == "__main__":
    env = GridWorld()
    agent = RandomAgent()

    episodes = 1000
    for episode in range(episodes):  # 에피소드 1000번 수행
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_action(state)  # 행동 선택
            next_state, reward, done = env.step(action)  # 행동 수행

            agent.add(state, action, reward)  # (상태, 행동, 보상) 저장

            if done:  # 목표에 도달 시
                agent.eval()  # 몬테카를로법으로 가치 함수 갱신
                break  # 다음 에피소드 시작

            state = next_state

    # 슬라이드와 동일: V(s)만 색 격자로 표시 (정책 화살표 없음)
    env.render_v(agent.V)

