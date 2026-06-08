"""
DQN / PPO / SAC 학습 (통합 admission control, Stable-Baselines3)
==============================================================
다차원 행동공간(라우트별 ΔL)에 따른 알고리즘 매핑:
  DQN : action_mode="flat"  -> Discrete(7^N)        (이산, 조합 폭증 - 한계 노출)
  PPO : action_mode="multi" -> MultiDiscrete([7]*N) (라우트별 이산, 자연스러움)
  SAC : action_mode="box"   -> Box(N)               (연속 벡터, 자연스러움)

사용법:
    python train.py --algo all --timesteps 300000
    python train.py --algo PPO --timesteps 300000
모델: models/<ALGO>.zip
"""
from __future__ import annotations
import os, argparse
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.monitor import Monitor
from envs.admission_env import AdmissionEnv
from config import ENV_KWARGS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# 알고리즘별 행동 모드
ACTION_MODE = {"DQN": "flat", "PPO": "multi", "SAC": "box"}


def make_env(mode):
    return Monitor(AdmissionEnv(**ENV_KWARGS, action_mode=mode))


def train_one(algo, timesteps):
    env = make_env(ACTION_MODE[algo])
    common = dict(policy="MlpPolicy", env=env, gamma=0.99, verbose=0, seed=0,
                  policy_kwargs=dict(net_arch=[256, 256]))
    if algo == "DQN":
        model = DQN(learning_rate=5e-4, buffer_size=200_000, learning_starts=5000, batch_size=128,
                    train_freq=4, target_update_interval=2000, exploration_fraction=0.3,
                    exploration_final_eps=0.05, **common)
    elif algo == "PPO":
        model = PPO(learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10,
                    gae_lambda=0.95, ent_coef=0.01, **common)
    else:  # SAC
        model = SAC(learning_rate=3e-4, buffer_size=200_000, learning_starts=5000, batch_size=256,
                    tau=0.005, train_freq=1, **common)
    model.learn(total_timesteps=timesteps)
    model.save(os.path.join(MODELS_DIR, algo))
    print(f"[{algo}] saved -> models/{algo}.zip")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["DQN", "PPO", "SAC", "all"], default="all")
    p.add_argument("--timesteps", type=int, default=300_000)
    args = p.parse_args()
    for a in (["DQN", "PPO", "SAC"] if args.algo == "all" else [args.algo]):
        print(f"=== Training {a} ({ACTION_MODE[a]}) for {args.timesteps} steps ===")
        train_one(a, args.timesteps)
