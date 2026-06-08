"""정책 거동 애니메이션 GIF 생성기 (PPT 삽입용).

    python make_gif.py --env threadpool --algo dqn
    python make_gif.py --env ev-threadpool --algo ppo --fps 20

평가 트래픽 1 에피소드에서 정책을 실행하고, 트래픽/자원량/지연이
시간에 따라 그려지는 GIF를 results/<env>_<algo>_anim.gif 로 저장한다.
PowerPoint(M365)에 삽입하면 슬라이드에서 자동 재생된다.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import config as C
from train import make_env, ENV_CHOICES
from evaluate import load_agents, run_episode
from baselines import ThresholdAutoscaler

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", choices=ENV_CHOICES, default="threadpool")
    ap.add_argument("--algo", choices=["dqn", "ppo", "rule"], default="dqn")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--stride", type=int, default=2,
                    help="N step마다 1프레임 (GIF 용량 조절)")
    args = ap.parse_args()

    env = make_env(args.env)
    if args.algo == "rule":
        policy, label = ThresholdAutoscaler(), "Rule-based"
    else:
        agents = load_agents(args.env)
        label = args.algo.upper()
        if label not in agents:
            raise FileNotFoundError(f"results/에 {args.env}_{args.algo} 가중치 없음 — train.py 먼저 실행")
        policy = agents[label]

    # 평가 트래픽 1 에피소드 기록
    _, tr = run_episode(env, policy, seed=C.EVAL_SEED, record=True)
    lam = np.array(tr["lam"]); cs = np.array(tr["c"]); lat = np.array(tr["latency"]) * 1000
    sla = (env.sla if args.env.startswith("ev-") else C.SLA_LATENCY) * 1000
    T = len(lam)
    frames = list(range(2, T, args.stride))

    fig, axes = plt.subplots(3, 1, figsize=(8, 6.4), sharex=True)
    fig.suptitle(f"{label} policy — {args.env}", fontsize=13)
    lines = []
    for ax, (data, color, ylab) in zip(axes, [
            (lam, "#555555", "Traffic (req/s)"),
            (cs, "#1E2761", "Resources (c)"),
            (lat, "#C55A11", "Latency (ms)")]):
        ln, = ax.plot([], [], color=color, lw=1.8)
        lines.append(ln)
        ax.set_xlim(0, T)
        ax.set_ylim(0, float(data.max()) * 1.15)
        ax.set_ylabel(ylab, fontsize=10)
        ax.grid(alpha=0.3)
    axes[2].axhline(sla, color="red", ls="--", lw=1)
    axes[2].set_ylim(0, sla * 3)
    axes[2].set_xlabel("Step")
    head = axes[0].axvline(0, color="#C55A11", lw=1, alpha=0.6)

    def update(f):
        x = np.arange(f)
        for ln, data in zip(lines, (lam, cs, lat)):
            ln.set_data(x, data[:f])
        head.set_xdata([f, f])
        return lines + [head]

    anim = FuncAnimation(fig, update, frames=frames, blit=True)
    out = os.path.join(RESULTS, f"{args.env}_{args.algo}_anim.gif")
    anim.save(out, writer=PillowWriter(fps=args.fps), dpi=80)
    plt.close(fig)
    print(f"저장: {out}  ({len(frames)} 프레임)")


if __name__ == "__main__":
    main()
