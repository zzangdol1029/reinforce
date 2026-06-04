"""
DQN / PPO / SAC 학습 스크립트 (Stable-Baselines3)
=================================================
사용법:
    python train.py --algo DQN   --timesteps 150000
    python train.py --algo PPO   --timesteps 150000
    python train.py --algo SAC   --timesteps 150000
    python train.py --algo all   --timesteps 150000

- DQN, PPO : Discrete 행동공간(LoadBalancerEnv continuous=False)
- SAC      : 연속 행동공간(continuous=True), 인스턴스별 가중치를 출력 -> argmax 라우팅
모델은 models/<ALGO>.zip 에 저장된다.
"""
from __future__ import annotations

import os
import argparse

from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.monitor import Monitor

from envs.load_balancer_env import LoadBalancerEnv
from config import ENV_KWARGS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def make_env(continuous: bool):
    kwargs = dict(ENV_KWARGS)
    kwargs["continuous"] = continuous
    return Monitor(LoadBalancerEnv(**kwargs))


def train_dqn(timesteps: int):
    env = make_env(continuous=False)
    model = DQN(
        "MlpPolicy", env,
        learning_rate=5e-4, buffer_size=50_000, learning_starts=1000,
        batch_size=128, gamma=0.99, train_freq=4, target_update_interval=500,
        exploration_fraction=0.2, exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[128, 128]), verbose=0, seed=0,
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)
    model.save(os.path.join(MODELS_DIR, "DQN"))
    print("[DQN] saved")


def train_ppo(timesteps: int):
    env = make_env(continuous=False)
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4, n_steps=2048, batch_size=128, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
        policy_kwargs=dict(net_arch=[128, 128]), verbose=0, seed=0,
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)
    model.save(os.path.join(MODELS_DIR, "PPO"))
    print("[PPO] saved")


def train_sac(timesteps: int):
    env = make_env(continuous=True)
    model = SAC(
        "MlpPolicy", env,
        learning_rate=3e-4, buffer_size=50_000, learning_starts=1000,
        batch_size=256, gamma=0.99, tau=0.005, train_freq=1,
        policy_kwargs=dict(net_arch=[128, 128]), verbose=0, seed=0,
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)
    model.save(os.path.join(MODELS_DIR, "SAC"))
    print("[SAC] saved")


TRAINERS = {"DQN": train_dqn, "PPO": train_ppo, "SAC": train_sac}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["DQN", "PPO", "SAC", "all"], default="all")
    p.add_argument("--timesteps", type=int, default=150_000)
    args = p.parse_args()

    algos = ["DQN", "PPO", "SAC"] if args.algo == "all" else [args.algo]
    for a in algos:
        print(f"=== Training {a} for {args.timesteps} steps ===")
        TRAINERS[a](args.timesteps)
