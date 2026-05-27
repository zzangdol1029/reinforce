"""
Quiz Q1 (PDF slide 28) — Mountain Car + DQN
===========================================
(Q1) DQN 을 Mountain Car 에 적용하고 Hyper-parameter 를 변경하여
     최대 total reward policy 를 찾을 것.

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
  python quiz_q1_mountain_car_dqn.py
  python quiz_q1_mountain_car_dqn.py --play
"""
from __future__ import annotations

import argparse
import copy
from collections import deque
from pathlib import Path
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L

OUT_DIR = Path(__file__).resolve().parent / "results_quiz_q1_mountain_car_dqn"


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)
        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.int32)
        return state, action, reward, next_state, done


class QNet(Model):
    def __init__(self, action_size, hidden1: int = 128, hidden2: int = 128):
        super().__init__()
        self.l1 = L.Linear(hidden1)
        self.l2 = L.Linear(hidden2)
        self.l3 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DQNAgent:
    def __init__(
        self,
        *,
        action_size: int,
        gamma: float,
        lr: float,
        epsilon: float,
        buffer_size: int,
        batch_size: int,
        hidden1: int,
        hidden2: int,
    ):
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.action_size = action_size

        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        self.qnet = QNet(action_size, hidden1, hidden2)
        self.qnet_target = QNet(action_size, hidden1, hidden2)
        self.optimizer = optimizers.Adam(lr)
        self.optimizer.setup(self.qnet)

    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)

    def get_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(self.action_size))
        state = state[np.newaxis, :].astype(np.float32)
        qs = self.qnet(state)
        return int(qs.data.argmax())

    def update(self, state, action, reward, next_state, done) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(state)
        q = qs[np.arange(self.batch_size), action]

        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(axis=1)
        next_q.unchain()
        target = reward + (1 - done) * self.gamma * next_q

        loss = F.mean_squared_error(q, target)
        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()


def train(agent, env, episodes, sync_interval) -> list[float]:
    reward_history: list[float] = []
    for episode in range(episodes):
        state, _ = env.reset()
        state = np.asarray(state, dtype=np.float32)
        done = False
        total_reward = 0.0

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ns = np.asarray(next_state, dtype=np.float32)
            agent.update(state, action, float(reward), ns, done)
            state = ns
            total_reward += float(reward)

        if episode % sync_interval == 0:
            agent.sync_qnet()

        reward_history.append(total_reward)
        if episode % 10 == 0:
            print(f"episode :{episode}, total reward : {total_reward}")
    return reward_history


def play_greedy(agent, env) -> float:
    agent.epsilon = 0.0
    state, _ = env.reset()
    state = np.asarray(state, dtype=np.float32)
    done = False
    total_reward = 0.0
    while not done:
        action = agent.get_action(state)
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
    print(f"저장: {path}")
    plt.show()
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Quiz Q1: Mountain Car DQN")
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--lr", type=float, default=0.0005)
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--buffer-size", type=int, default=100_000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--sync-interval", type=int, default=20)
    ap.add_argument("--hidden1", type=int, default=128)
    ap.add_argument("--hidden2", type=int, default=128)
    ap.add_argument("--play", action="store_true", help="학습 후 greedy 시연 (render)")
    args = ap.parse_args()

    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    action_size = int(env.action_space.n)

    agent = DQNAgent(
        action_size=action_size,
        gamma=args.gamma,
        lr=args.lr,
        epsilon=args.epsilon,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
    )

    print("Hyper-parameters:", vars(args))
    reward_history = train(agent, env, args.episodes, args.sync_interval)
    best = max(reward_history)
    best_ep = int(np.argmax(reward_history))
    print(f"\n최대 total reward: {best} (episode {best_ep})")

    plot_path = OUT_DIR / "episode_total_reward.png"
    save_reward_plot(reward_history, plot_path)

    hp_path = OUT_DIR / "hyperparameters.txt"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hp_path.write_text(
        "\n".join(
            [
                "Quiz Q1 — Mountain Car DQN",
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
