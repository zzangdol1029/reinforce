"""
Quiz Q1 (PDF slide 28) — Mountain Car + Actor-Critic
=====================================================
(Q1) Actor-Critic 을 Mountain Car 문제에 적용하고 Hyper-parameter 를
     변경하여 최대 total reward policy 를 결정하라.

제출 (PPT):
  1) 프로그램 소스
  2) 최적 hyperparameter
  3) Episode 별 total reward graph
  4) 최대 total reward 값 및 해당 policy 적용 시의 동영상

Mountain Car (PDF slide 27):
  - Reward: -1 for each time step
  - Starting position: uniform in [-0.6, -0.4]
  - Episode ends: position >= 0.5 or length >= 200

실행:
  python quiz_q1_mountain_car_actor_critic.py
  python quiz_q1_mountain_car_actor_critic.py --play
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L

OUT_DIR = Path(__file__).resolve().parent / "results_quiz_q1_mountain_car_ac"


class PolicyNet(Model):
    """Actor — policy network π_θ(a|s)."""

    def __init__(self, action_size: int, hidden: int = 128):
        super().__init__()
        self.l1 = L.Linear(hidden)
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x))
        return x


class ValueNet(Model):
    """Critic — value network V_w(s)."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.l1 = L.Linear(hidden)
        self.l2 = L.Linear(1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class Agent:
    def __init__(
        self,
        action_size: int,
        *,
        gamma: float,
        lr_pi: float,
        lr_v: float,
        hidden: int,
    ):
        self.gamma = gamma
        self.action_size = action_size

        self.pi = PolicyNet(action_size, hidden)
        self.v = ValueNet(hidden)
        self.optimizer_pi = optimizers.Adam(lr_pi)
        self.optimizer_v = optimizers.Adam(lr_v)
        self.optimizer_pi.setup(self.pi)
        self.optimizer_v.setup(self.v)

    def get_action(self, state: np.ndarray) -> tuple[int, object]:
        state = state[np.newaxis, :].astype(np.float32)
        probs = self.pi(state)
        probs = probs[0]
        action = int(np.random.choice(self.action_size, p=probs.data))
        return action, probs[action]

    def update(self, state, action_prob, reward, next_state, done) -> None:
        next_state = next_state[np.newaxis, :].astype(np.float32)
        target = reward + self.gamma * self.v(next_state) * (1 - done)
        target.unchain()

        v = self.v(state[np.newaxis, :].astype(np.float32))
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


def train(agent, env, episodes) -> list[float]:
    reward_history: list[float] = []
    for episode in range(episodes):
        state, _ = env.reset()
        state = np.asarray(state, dtype=np.float32)
        done = False
        total_reward = 0.0

        while not done:
            action, prob = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            dead = int(terminated)
            ns = np.asarray(next_state, dtype=np.float32)
            agent.update(state, prob, float(reward), ns, dead)
            state = ns
            total_reward += float(reward)

        reward_history.append(total_reward)
        if episode % 10 == 0:
            print(f"episode :{episode}, total reward : {total_reward}")
    return reward_history


def play_greedy(agent, env) -> float:
    state, _ = env.reset()
    state = np.asarray(state, dtype=np.float32)
    done = False
    total_reward = 0.0
    while not done:
        action, _ = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        state = np.asarray(next_state, dtype=np.float32)
        total_reward += float(reward)
        env.render()
    return total_reward


def save_reward_plot(reward_history: list[float], path: Path) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.xlabel("episode")
    plt.ylabel("total reward")
    plt.plot(range(len(reward_history)), reward_history)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"저장: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Quiz Q1: Mountain Car Actor-Critic")
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--lr_pi", type=float, default=0.0002)
    ap.add_argument("--lr_v", type=float, default=0.0005)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--play", action="store_true", help="학습 후 greedy 시연 (render)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    action_size = int(env.action_space.n)

    agent = Agent(
        action_size,
        gamma=args.gamma,
        lr_pi=args.lr_pi,
        lr_v=args.lr_v,
        hidden=args.hidden,
    )

    print("Hyper-parameters:", vars(args))
    reward_history = train(agent, env, args.episodes)
    best = max(reward_history)
    best_ep = int(np.argmax(reward_history))
    print(f"\n최대 total reward: {best} (episode {best_ep})")

    plot_path = OUT_DIR / "episode_total_reward.png"
    if not args.no_plot:
        save_reward_plot(reward_history, plot_path)

    hp_path = OUT_DIR / "hyperparameters.txt"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hp_path.write_text(
        "\n".join(
            [
                "Quiz Q1 — Mountain Car Actor-Critic",
                "",
                *[f"{k}: {v}" for k, v in vars(args).items()],
                "",
                f"max_total_reward: {best}",
                f"max_reward_episode: {best_ep}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"저장: {hp_path}")

    if args.play:
        env2 = gym.make("MountainCar-v0", render_mode="human")
        tr = play_greedy(agent, env2)
        print("Total Reward:", tr)
        env2.close()

    env.close()


if __name__ == "__main__":
    main()
