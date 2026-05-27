"""
실습 #1 — Simple Policy Gradient (CartPole)
PDF: 10-Policy Gradient Method — slides 6~10

∇_θ J(θ) ≈ Σ_t G(τ) ∇_θ log π_θ(A_t|S_t)
loss = − Σ_t G(τ) log π_θ(A_t|S_t)

실행:
  python Simple_pg2.py
  python Simple_pg2.py --plot
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

try:
    import gym
except ModuleNotFoundError:
    import gymnasium as gym

from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L


class Policy(Model):
    """Policy Network (π_θ): 2-layer NN, CartPole classification."""

    def __init__(self, action_size: int = 2):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x))
        return x


class Agent:
    def __init__(self, action_size: int = 2, gamma: float = 0.98, lr: float = 0.0002):
        self.gamma = gamma
        self.action_size = action_size
        self.memory: list[tuple[float, object]] = []

        self.pi = Policy(action_size)
        self.optimizer = optimizers.Adam(lr)
        self.optimizer.setup(self.pi)

    def get_action(self, state: np.ndarray) -> tuple[int, object]:
        state = state[np.newaxis, :]
        probs = self.pi(state)
        probs = probs[0]
        action = int(np.random.choice(self.action_size, p=probs.data))
        return action, probs[action]

    def add(self, reward: float, prob) -> None:
        self.memory.append((reward, prob))

    def update(self) -> None:
        self.pi.cleargrads()

        g_tau = 0.0
        for reward, _ in reversed(self.memory):
            g_tau = float(reward + self.gamma * g_tau)

        loss = 0
        for _, prob in self.memory:
            loss += -F.log(prob) * g_tau

        loss.backward()
        self.optimizer.update()
        self.memory.clear()


def moving_average(xs: list[float], window: int) -> np.ndarray:
    if not xs:
        return np.array([])
    arr = np.asarray(xs, dtype=np.float64)
    out = np.empty_like(arr)
    cum = np.cumsum(np.insert(arr, 0, 0.0))
    for i in range(len(arr)):
        j = max(0, i - window + 1)
        out[i] = (cum[i + 1] - cum[j]) / (i - j + 1)
    return out


def reset_env(env):
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        next_state, reward, terminated, truncated, _ = out
        done = bool(terminated or truncated)
    else:
        next_state, reward, done, _ = out
    return next_state, float(reward), done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--lr", type=float, default=0.0002)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    env = gym.make("CartPole-v0")
    agent = Agent(gamma=args.gamma, lr=args.lr)
    reward_history: list[float] = []

    for episode in range(args.episodes):
        state = reset_env(env)
        done = False
        total_reward = 0.0

        while not done:
            action, prob = agent.get_action(np.asarray(state, dtype=np.float32))
            next_state, reward, done = step_env(env, action)
            agent.add(reward, prob)
            state = next_state
            total_reward += reward

        agent.update()
        reward_history.append(total_reward)

        if episode % 100 == 0:
            print(f"episode :{episode}, total reward : {total_reward}")

    if args.plot:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(reward_history)
        axes[0].set_xlabel("episode")
        axes[0].set_ylabel("total reward")
        axes[0].set_title("Episode – total reward")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(moving_average(reward_history, 100))
        axes[1].set_xlabel("episode")
        axes[1].set_ylabel("total reward")
        axes[1].set_title("Episode – total reward (100 회 평균)")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        plt.show()

    env.close()


if __name__ == "__main__":
    main()
