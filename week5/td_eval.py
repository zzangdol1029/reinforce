"""
Policy Evaluation — 실습 #1 td_eval.py
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "week3"))

import matplotlib

if sys.platform == "darwin":
    matplotlib.use("macosx", force=True)
else:
    for _name in ("TkAgg", "Qt5Agg", "QtAgg"):
        try:
            matplotlib.use(_name, force=True)
            break
        except Exception:
            continue

from collections import defaultdict

import numpy as np

from gridworld import GridWorld


class TdAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.01
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.V = defaultdict(lambda: 0)

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def eval(self, state, reward, next_state, done):
        # 목표 지점의 가치 함수는 0
        next_V = 0 if done else self.V[next_state]
        target = reward + self.gamma * next_V
        self.V[state] += (target - self.V[state]) * self.alpha


if __name__ == "__main__":
    env = GridWorld()
    agent = TdAgent()

    episodes = 1000
    for episode in range(episodes):
        state = env.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.eval(state, reward, next_state, done)

            if done:
                break
            state = next_state

    env.render_v(agent.V)
