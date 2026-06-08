"""간단 하이퍼파라미터 탐색 (grid search).

config.py의 권장값이 출발점이고, 이 스크립트는 그 주변을 짧은 학습으로
훑어서 검증 보상 기준 순위를 출력한다. 본 학습 전에 환경/머신에 맞는
설정을 확인하는 용도.

    python hp_search.py --env ev-container --algo dqn --episodes 80
    python hp_search.py --env ev-container --algo ppo --episodes 80

주의: 탐색은 '검증 시드(VAL_SEED)' 성능으로 비교한다 — 최종 평가
시드(EVAL_SEED)는 절대 사용하지 않는다 (선택 과정 자체의 과적합 방지).
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np

import config as C
from train import make_env, validate
from agents import DQNAgent, PPOAgent

# 탐색 공간 — 권장값(config.py) 주변의 합리적 이웃
GRID_DQN = dict(
    lr=[1e-3, 5e-4, 2.5e-4],
    batch_size=[64, 128],
    hidden=[64, 128],
)
GRID_PPO = dict(
    lr=[1e-3, 3e-4, 1e-4],
    rollout_steps=[2_048, 4_096],
    ent_coef=[0.01, 0.03],
)


def short_train_dqn(env_name: str, cfg: dict, episodes: int, seed: int) -> float:
    env = make_env(env_name, seed)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n,
                     episodes * C.EPISODE_STEPS, cfg, seed=seed)
    for ep in range(episodes):
        s, _ = env.reset(seed=seed * 10_000 + ep)
        done = False
        while not done:
            a = agent.act(s)
            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            agent.remember(s, a, r, s2, done)
            agent.learn()
            s = s2
    return validate(env_name, agent)


def short_train_ppo(env_name: str, cfg: dict, episodes: int, seed: int) -> float:
    env = make_env(env_name, seed)
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.n,
                     cfg, seed=seed)
    rollout = 0
    for ep in range(episodes):
        s, _ = env.reset(seed=seed * 10_000 + ep)
        done = False
        while not done:
            a, logp, v = agent.act(s)
            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            agent.remember(s, a, r, done, logp, v)
            rollout += 1
            if rollout >= cfg["rollout_steps"]:
                agent.update(s2)
                rollout = 0
            s = s2
    return validate(env_name, agent)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", required=True)
    ap.add_argument("--algo", choices=["dqn", "ppo"], required=True)
    ap.add_argument("--episodes", type=int, default=80,
                    help="조합당 학습 에피소드 (짧게)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = dict(C.DQN if args.algo == "dqn" else C.PPO)
    grid = GRID_DQN if args.algo == "dqn" else GRID_PPO
    fn = short_train_dqn if args.algo == "dqn" else short_train_ppo

    keys = list(grid.keys())
    results = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        cfg = dict(base)
        cfg.update(dict(zip(keys, combo)))
        t0 = time.time()
        score = fn(args.env, cfg, args.episodes, args.seed)
        results.append((score, dict(zip(keys, combo))))
        print(f"val={score:9.2f}  {dict(zip(keys, combo))}  ({time.time()-t0:.0f}s)")

    results.sort(key=lambda x: -x[0])
    print("\n=== 순위 (검증 보상 기준) ===")
    for rank, (score, cfg) in enumerate(results, 1):
        print(f"{rank:2d}. val={score:9.2f}  {cfg}")
    print("\n1위 설정을 config.py에 반영한 뒤 본 학습(train.py)을 실행하세요.")
