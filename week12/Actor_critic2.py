"""
실습 #3 — Actor-Critic (CartPole, 1-step TD)
PDF: 10-Policy Gradient Method — slides 19~26

loss_v = (R_t + γ V_w(S_{t+1}) − V_w(S_t))^2
loss_pi = −(R_t + γ V_w(S_{t+1}) − V_w(S_t)) · log π_θ(A_t|S_t)

실행:
  python Actor_critic2.py
  python Actor_critic2.py --plot
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


class PolicyNet(Model):
    """Actor — policy network π_θ(a|s)."""

    def __init__(self, action_size: int = 2):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x))
        return x


class ValueNet(Model):
    """Critic — value network V_w(s)."""

    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class Agent:
    def __init__(
        self,
        action_size: int = 2,
        gamma: float = 0.98,
        lr_pi: float = 0.0002,
        lr_v: float = 0.0005,
    ):
        self.gamma = gamma
        self.action_size = action_size

        self.pi = PolicyNet(action_size)
        self.v = ValueNet()
        self.optimizer_pi = optimizers.Adam(lr_pi)
        self.optimizer_v = optimizers.Adam(lr_v)
        self.optimizer_pi.setup(self.pi)
        self.optimizer_v.setup(self.v)

    def get_action(self, state: np.ndarray) -> tuple[int, object]:
        state = state[np.newaxis, :]
        probs = self.pi(state)
        probs = probs[0]
        action = int(np.random.choice(self.action_size, p=probs.data))
        return action, probs[action]

    def update(self, state, action_prob, reward, next_state, done) -> None:
        next_state = next_state[np.newaxis, :]
        target = reward + self.gamma * self.v(next_state) * (1 - done)
        target.unchain()

        v = self.v(state[np.newaxis, :])
        loss_v = F.mean_squared_error(v, target)

        delta = target - v
        delta.unchain()
        loss_pi = -F.log(action_prob) * delta

        self.v.cleargrads()
        self.pi.cleargrads()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.update()
        self.optimizer_pi.update()


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
        dead = int(terminated)
    else:
        next_state, reward, done, info = out
        truncated = bool(info.get("TimeLimit.truncated", False)) if isinstance(info, dict) else False
        dead = int(bool(done and not truncated))
    return next_state, float(reward), done, dead


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--lr_pi", type=float, default=0.0002)
    ap.add_argument("--lr_v", type=float, default=0.0005)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    env = gym.make("CartPole-v0")
    agent = Agent(gamma=args.gamma, lr_pi=args.lr_pi, lr_v=args.lr_v)
    reward_history: list[float] = []

    for episode in range(args.episodes):
        state = reset_env(env)
        done = False
        total_reward = 0.0

        while not done:
            action, prob = agent.get_action(np.asarray(state, dtype=np.float32))
            next_state, reward, done, dead = step_env(env, action)
            agent.update(state, prob, reward, next_state, dead)
            state = next_state
            total_reward += reward

        reward_history.append(total_reward)

        if episode % 100 == 0:
            print(f"episode :{episode}, total reward : {total_reward}")

    if args.plot:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(reward_history)
        axes[0].set_xlabel("episode")
        axes[0].set_ylabel("total reward")
        axes[0].set_title("Actor-Critic — Episode – total reward")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(moving_average(reward_history, 100))
        axes[1].set_xlabel("episode")
        axes[1].set_ylabel("total reward")
        axes[1].set_title("(100 회 평균)")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        plt.show()

    env.close()


if __name__ == "__main__":
    main()
