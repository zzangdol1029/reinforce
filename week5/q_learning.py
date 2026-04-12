"""
실습 #3 q_learning.py — Q-Learning
"""

import importlib.util
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

# week3의 render_q는 텍스트만 지원 — 슬라이드 2패널(Q+greedy)은 week4 Renderer 사용
_week4_render = _root / "week4" / "common" / "gridworld_render.py"
_spec = importlib.util.spec_from_file_location("week4_gridworld_render", _week4_render)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load Renderer from {_week4_render}")
_gw4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gw4)
Renderer = _gw4.Renderer


def greedy_probs(Q, state, epsilon=0.0, action_size=4):
    """ε-greedy 정책 확률 분포."""
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = int(np.argmax(qs))

    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}
    action_probs[max_action] += 1.0 - epsilon
    return action_probs


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.b = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state):
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return int(np.random.choice(actions, p=probs))

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q_max = 0.0
        else:
            next_q_max = max(
                self.Q[next_state, a] for a in range(self.action_size)
            )

        target = reward + self.gamma * next_q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        self.b[state] = greedy_probs(self.Q, state, self.epsilon)


if __name__ == "__main__":
    env = GridWorld()
    agent = QLearningAgent()

    episodes = 10000
    for episode in range(episodes):
        state = env.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, next_state, done)

            if done:
                break
            state = next_state

    renderer = Renderer(env.reward_map, env.goal_state, env.wall_state)
    renderer.render_q_and_policy(agent.Q)
