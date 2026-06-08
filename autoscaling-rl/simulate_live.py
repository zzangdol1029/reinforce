"""실시간 시뮬레이션 뷰어 — 정책이 스케일링하는 모습을 라이브로 관찰.

    python simulate_live.py --env threadpool --algo dqn        # 학습 후
    python simulate_live.py --env threadpool --algo rule       # 학습 전에도 가능
    python simulate_live.py --env ev-threadpool --algo ppo --interval 0.03

matplotlib 창이 열리고 트래픽 / 자원량 / 응답지연(SLA 점선)이
한 step씩 실시간으로 그려진다. 창을 닫으면 종료.
--save 를 주면 창 대신 results/<env>_<algo>_live.png 로 최종 화면을 저장.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import config as C
from train import make_env, ENV_CHOICES
from baselines import ThresholdAutoscaler, StaticPolicy

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def get_policy(env_name: str, algo: str):
    if algo == "rule":
        return ThresholdAutoscaler(), "Rule-based"
    if algo == "static":
        return StaticPolicy(), "Static"
    from evaluate import load_agents          # 무거운 import는 필요할 때만
    agents = load_agents(env_name)
    label = algo.upper()
    if label not in agents:
        raise SystemExit(
            f"results/에 {env_name}_{algo} 가중치가 없습니다.\n"
            f"먼저:  python train.py --env {env_name} --algo {algo}\n"
            f"또는 학습 없이 보려면:  --algo rule")
    return agents[label], label


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", choices=ENV_CHOICES, default="threadpool")
    ap.add_argument("--algo", choices=["dqn", "ppo", "rule", "static"], default="rule")
    ap.add_argument("--seed", type=int, default=C.EVAL_SEED)
    ap.add_argument("--interval", type=float, default=0.05,
                    help="step 간 대기 시간(초) — 작을수록 빠른 재생")
    ap.add_argument("--save", action="store_true",
                    help="창을 띄우지 않고 최종 그림만 저장 (원격/서버 환경용)")
    args = ap.parse_args()

    if args.save:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    policy, label = get_policy(args.env, args.algo)
    env = make_env(args.env)
    is_event = args.env.startswith("ev-")
    sla_ms = (env.sla if is_event else C.SLA_LATENCY) * 1000

    # ---- figure 구성 ---------------------------------------------------
    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    fig.suptitle(f"{label} policy — {args.env} (live)", fontsize=13)
    T = C.EPISODE_STEPS
    labels = ["Traffic (req/s)",
              "Instances" if "container" in args.env else "Worker threads",
              ("p95 " if is_event else "") + "Latency (ms)"]
    colors = ["#555555", "#1E2761", "#C55A11"]
    lines = []
    for ax, ylab, color in zip(axes, labels, colors):
        ln, = ax.plot([], [], color=color, lw=1.6)
        lines.append(ln)
        ax.set_xlim(0, T)
        ax.set_ylabel(ylab, fontsize=10)
        ax.grid(alpha=0.3)
    axes[2].axhline(sla_ms, color="red", ls="--", lw=1, label="SLA")
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].set_xlabel("Step")
    status = axes[0].set_title("", fontsize=10, loc="right", color="#777777")

    # ---- 에피소드 실행 + 실시간 갱신 -------------------------------------
    s, _ = env.reset(seed=args.seed)
    if hasattr(policy, "reset"):
        policy.reset()
    lam_h, c_h, lat_h = [], [], []
    reward, viol = 0.0, 0
    done = False
    if not args.save:
        plt.ion()
        plt.show()

    while not done:
        a = policy.act(s, greedy=True)
        if isinstance(a, tuple):
            a = a[0]
        s, r, term, trunc, info = env.step(a)
        done = term or trunc
        reward += r
        viol += int(info["sla_violated"])
        lam_h.append(info["lam"])
        c_h.append(info["c"])
        lat_h.append(info["latency"] * 1000)

        x = np.arange(len(lam_h))
        for ln, data in zip(lines, (lam_h, c_h, lat_h)):
            ln.set_data(x, data)
        # y축 자동 조정
        axes[0].set_ylim(0, max(lam_h) * 1.15)
        axes[1].set_ylim(0, env.c_max * 1.1)
        axes[2].set_ylim(0, sla_ms * 3)
        status.set_text(f"step {len(lam_h)}/{T}  reward {reward:.1f}  SLA위반 {viol}")

        if not args.save:
            plt.pause(args.interval)
            if not plt.get_fignums():      # 사용자가 창을 닫으면 종료
                return

    print(f"완료: reward={reward:.1f}, SLA 위반 step={viol}/{T}")
    if args.save:
        os.makedirs(RESULTS, exist_ok=True)
        out = os.path.join(RESULTS, f"{args.env}_{args.algo}_live.png")
        fig.savefig(out, dpi=130)
        print(f"저장: {out}")
    else:
        plt.ioff()
        plt.show()                         # 끝 화면 유지


if __name__ == "__main__":
    main()
