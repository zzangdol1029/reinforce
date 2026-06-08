"""설명용 그림 생성기 (PPT 슬라이드 6·8에 들어가는 그림).

    python make_figures.py

생성물:
    results/traffic_sample.png    트래픽 생성기 샘플 (일일 사이클 + 버스트)
    results/capacity_curve.png    thread 비단조 용량 곡선
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from envs.traffic import make_traffic
from envs import ThreadPoolEnv

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def fig_traffic():
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for i, seed in enumerate([3, 17, 42]):
        rng = np.random.default_rng(seed)
        lam = make_traffic(288, 80.0, rng)
        ax.plot(lam, lw=1.4, alpha=0.85, label=f"episode {i+1}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Arrival rate (req/s)")
    ax.set_title("Generated traffic patterns (daily cycle + random bursts)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "traffic_sample.png"), dpi=150)
    plt.close(fig)


def fig_capacity():
    env = ThreadPoolEnv()
    cs = np.arange(4, 65)
    caps = [env.capacity(int(c)) for c in cs]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(cs, caps, lw=2.5, color="#1E2761")
    ax.axvline(32, color="#C55A11", ls="--", lw=1.5)
    ax.annotate("c_knee=32\n(context switching begins)", xy=(32, 500),
                xytext=(38, 350), fontsize=9, color="#C55A11",
                arrowprops=dict(arrowstyle="->", color="#C55A11"))
    imax = int(np.argmax(caps))
    ax.scatter([cs[imax]], [caps[imax]], color="#C55A11", zorder=5)
    ax.annotate(f"peak capacity @ c={cs[imax]}", xy=(cs[imax], caps[imax]),
                xytext=(10, 820), fontsize=9, arrowprops=dict(arrowstyle="->"))
    ax.set_xlabel("Worker threads (c)")
    ax.set_ylabel("Capacity (req/s)")
    ax.set_title("Non-monotonic capacity: more threads != more throughput")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "capacity_curve.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    fig_traffic()
    fig_capacity()
    print("저장: results/traffic_sample.png, results/capacity_curve.png")
