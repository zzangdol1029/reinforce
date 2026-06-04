"""
평가 · 비교 · 시각화
====================
학습된 RL 정책(DQN/PPO/SAC)과 베이스라인을 동일한 시나리오/시드에서 평가하고
비교 표(CSV)와 그래프(PNG)를 생성한다.

사용법:
    python evaluate.py --episodes 30
"""
from __future__ import annotations

import os
import argparse
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import DQN, PPO, SAC

from envs.load_balancer_env import LoadBalancerEnv
from baselines import BASELINES, evaluate_baseline, _mean_metrics
from config import ENV_KWARGS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RL_SPECS = [
    ("DQN", DQN, False),
    ("PPO", PPO, False),
    ("SAC", SAC, True),
]


def evaluate_rl(name, model, continuous, n_episodes, base_seed=1000):
    agg = []
    for ep in range(n_episodes):
        kwargs = dict(ENV_KWARGS); kwargs["continuous"] = continuous
        env = LoadBalancerEnv(**kwargs)
        obs, _ = env.reset(seed=base_seed + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
        agg.append(env.metrics())
    return _mean_metrics(agg)


def main(n_episodes: int):
    rows = {}  # name -> metrics

    # 베이스라인
    for pol in BASELINES:
        rows[pol.name] = evaluate_baseline(pol, ENV_KWARGS, n_episodes=n_episodes)
        print(f"[baseline] {pol.name:12s} {rows[pol.name]}")

    # RL 정책 (모델이 있으면)
    for name, cls, cont in RL_SPECS:
        path = os.path.join(MODELS_DIR, f"{name}.zip")
        if not os.path.exists(path):
            print(f"[skip] {name} 모델 없음 ({path}). train.py 먼저 실행.")
            continue
        model = cls.load(path)
        rows[name] = evaluate_rl(name, model, cont, n_episodes)
        print(f"[RL]       {name:12s} {rows[name]}")

    _save_csv(rows)
    _plot(rows)
    print(f"\n결과 저장: {RESULTS_DIR}")


def _save_csv(rows: dict):
    metrics = ["mean_latency", "p95_latency", "throughput", "sla_violation_rate", "load_imbalance"]
    path = os.path.join(RESULTS_DIR, "comparison.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy"] + metrics)
        for name, m in rows.items():
            w.writerow([name] + [round(m[k], 4) for k in metrics])


def _plot(rows: dict):
    names = list(rows.keys())
    # RL은 굵게 강조
    rl_set = {"DQN", "PPO", "SAC"}
    colors = ["#d62728" if n in rl_set else "#7f9cb0" for n in names]

    # 한글 폰트가 없는 환경을 고려해 그래프 라벨은 영문 사용
    panels = [
        ("mean_latency", "Mean latency (s)  (lower better)"),
        ("p95_latency", "p95 latency (s)  (lower better)"),
        ("sla_violation_rate", "SLA violation rate  (lower better)"),
        ("load_imbalance", "Load imbalance (std)  (lower better)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (key, title) in zip(axes.ravel(), panels):
        vals = [rows[n][key] for n in names]
        ax.bar(names, vals, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("MSA Gateway Load Balancing: RL (red) vs Baselines (gray)", fontsize=14)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "comparison.png")
    fig.savefig(out, dpi=130)
    print(f"그래프: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=30)
    args = p.parse_args()
    main(args.episodes)
