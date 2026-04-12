"""
실습 #2 sarsa.py — SARSA (State-Action-Reward-State-Action)

시각화: week4 Renderer — Q(s,a) 마름모 + greedy policy (교재 2패널 스타일)
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

import matplotlib.pyplot as plt

from collections import defaultdict, deque

import numpy as np

from gridworld import GridWorld

_week4_render = _root / "week4" / "common" / "gridworld_render.py"
_spec = importlib.util.spec_from_file_location("week4_gridworld_render", _week4_render)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load Renderer from {_week4_render}")
_gw4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gw4)
Renderer = _gw4.Renderer


def greedy_probs(Q, state, epsilon=0.0, action_size=4):
    """ε-greedy 정책 확률 분포. (week4 mc_control.py와 동일)"""
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = int(np.argmax(qs))

    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}
    action_probs[max_action] += 1.0 - epsilon
    return action_probs


class SarsaAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.memory = deque(maxlen=2)

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return int(np.random.choice(actions, p=probs))

    def reset(self):
        self.memory.clear()

    def update(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return

        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]

        next_q = 0 if done else self.Q[next_state, next_action]

        target = reward + self.gamma * next_q
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        self.pi[state] = greedy_probs(self.Q, state, self.epsilon)


if __name__ == "__main__":
    env = GridWorld()
    agent = SarsaAgent()

    episodes = 10000
    for episode in range(episodes):
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, done)

            if done:
                agent.update(next_state, None, None, None)
                break
            state = next_state

    # 3×4: 함정 (1,3) — greedy 화살표·Q 패널에서 목표/함정 처리
    trap_states = frozenset([(1, 3)])
    renderer = Renderer(
        env.reward_map,
        env.goal_state,
        env.wall_state,
        trap_states=trap_states,
    )
    _gw4._module._configure_matplotlib_korean_font()

    fig, (ax_q, ax_pi) = plt.subplots(1, 2, figsize=(13, 5.0))
    renderer._draw_q_diamond(agent.Q, ax_q)
    renderer._draw_greedy_policy(agent.Q, ax_pi)
    ax_q.set_title("Q 함수 시각화", fontsize=11)
    ax_pi.set_title("Q 함수에 대한 greedy policy", fontsize=11)
    fig.text(
        0.5,
        0.02,
        "폭탄에서 멀어지는 행동",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.show()
