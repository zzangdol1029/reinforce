"""평가 + 그래프 생성 스크립트.

학습된 DQN/PPO(검증 best 가중치 우선)와 룰 기반 베이스라인을
**학습/검증에 쓰지 않은 평가 트래픽**(EVAL_SEED) 20 에피소드로 비교하고,
발표용 그래프를 results/ 에 저장한다.

    python evaluate.py --env container        # v1
    python evaluate.py --env ev-container     # v2 (이산사건)

생성물:
    results/<env>_learning_curves.png   학습 곡선 + 검증 곡선
    results/<env>_behavior.png          트래픽/자원량/지연 거동 비교
    results/<env>_comparison.png        종합 지표 비교 (보상/SLA/비용)
    results/<env>_summary.csv           수치 요약 (표)
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12})  # 발표 슬라이드 가독성

import config as C
from agents import DQNAgent, PPOAgent
from baselines import ThresholdAutoscaler, StaticPolicy
from train import make_env, ENV_CHOICES

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ------------------------------------------------------------------ rollout
def run_episode(env, policy, seed: int, record: bool = False):
    """정책 1개를 1 에피소드 실행하고 지표(및 선택적으로 시계열)를 반환."""
    s, _ = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    total_r, n_viol, cost_sum, lat_sum, n = 0.0, 0, 0.0, 0.0, 0
    trace = {"lam": [], "c": [], "latency": []} if record else None
    done = False
    while not done:
        a = policy.act(s, greedy=True)
        if isinstance(a, tuple):          # PPO.act는 (a, logp, v) 반환
            a = a[0]
        s, r, term, trunc, info = env.step(a)
        done = term or trunc
        total_r += r
        n_viol += int(info["sla_violated"])
        cost_sum += info["cost"]
        lat_sum += info["latency"]
        n += 1
        if record:
            trace["lam"].append(info["lam"])
            trace["c"].append(info["c"])
            trace["latency"].append(info["latency"])
    metrics = dict(reward=total_r, sla_rate=n_viol / n,
                   mean_cost=cost_sum / n, mean_latency=lat_sum / n)
    return metrics, trace


def evaluate_policy(env_name: str, policy, episodes: int):
    env = make_env(env_name)
    ms = [run_episode(env, policy, seed=C.EVAL_SEED + i)[0] for i in range(episodes)]
    return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}


# ------------------------------------------------------------------ 모델 로드
def load_agents(env_name: str):
    """저장된 가중치에서 에이전트 복원. 검증 best 가 있으면 그것을 사용."""
    env = make_env(env_name)
    sdim, n_act = env.observation_space.shape[0], env.action_space.n
    agents = {}

    # DQN: <env>_dqn_best.npz 우선, 없으면 <env>_dqn_w.npz
    for tag in ("best", "w"):
        p = os.path.join(RESULTS, f"{env_name}_dqn_{tag}.npz")
        if os.path.exists(p):
            dqn = DQNAgent(sdim, n_act, 1, C.DQN)
            dqn.load(p)
            agents["DQN"] = dqn
            break

    # PPO: 가중치가 .actor.npz / .critic.npz 두 파일로 저장됨
    for tag in ("best", "w"):
        base = os.path.join(RESULTS, f"{env_name}_ppo_{tag}.npz")
        if os.path.exists(base + ".actor.npz"):
            ppo = PPOAgent(sdim, n_act, C.PPO)
            ppo.load(base)
            agents["PPO"] = ppo
            break
    return agents


# ------------------------------------------------------------------ plots
def plot_learning_curves(env_name: str):
    """학습 곡선(+검증 곡선). 검증 곡선은 best 체크포인트 선택의 근거 자료."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for algo, color in (("dqn", "#1E2761"), ("ppo", "#C55A11")):
        cpath = os.path.join(RESULTS, f"{env_name}_{algo}_curve.npy")
        if not os.path.exists(cpath):
            continue
        curve = np.load(cpath)
        smooth = np.convolve(curve, np.ones(10) / 10, mode="valid")
        ax.plot(curve, alpha=0.2, color=color)
        ax.plot(np.arange(len(smooth)) + 9, smooth, label=algo.upper(),
                color=color, lw=2)
        vpath = os.path.join(RESULTS, f"{env_name}_{algo}_val.npy")
        if os.path.exists(vpath):
            val = np.load(vpath)
            if val.ndim == 2 and len(val):
                ax.plot(val[:, 0], val[:, 1], "--", color=color, lw=1.5,
                        label=f"{algo.upper()} (validation)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")
    ax.set_title(f"Learning curves — {env_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, f"{env_name}_learning_curves.png"), dpi=150)
    plt.close(fig)


def plot_behavior(env_name: str, policies: dict):
    """같은 트래픽 에피소드에서 정책별 c(t)와 latency(t)를 겹쳐 그린다."""
    env = make_env(env_name)
    traces = {}
    for name, pol in policies.items():
        _, tr = run_episode(env, pol, seed=C.EVAL_SEED, record=True)
        traces[name] = tr

    steps = np.arange(len(next(iter(traces.values()))["lam"]))
    lam = traces[next(iter(traces))]["lam"]
    is_event = env_name.startswith("ev-")
    sla = env.sla if is_event else C.SLA_LATENCY
    lat_label = "p95 latency (ms)" if is_event else "Latency (ms)"

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(steps, lam, color="#555555")
    axes[0].set_ylabel("Traffic (req/s)")
    axes[0].set_title(f"Policy behavior on identical traffic — {env_name}")
    axes[0].grid(alpha=0.3)

    colors = {"DQN": "#1E2761", "PPO": "#C55A11", "Rule-based": "#2C7A2C",
              "Static": "#999999"}
    for name, tr in traces.items():
        axes[1].plot(steps, tr["c"], label=name, color=colors.get(name), lw=1.8)
    ylabel = "Instances" if "container" in env_name else "Worker threads"
    axes[1].set_ylabel(ylabel)
    axes[1].legend(ncol=4, fontsize=11)
    axes[1].grid(alpha=0.3)

    for name, tr in traces.items():
        axes[2].plot(steps, np.array(tr["latency"]) * 1000,
                     label=name, color=colors.get(name), lw=1.5)
    axes[2].axhline(sla * 1000, color="red", ls="--", lw=1, label="SLA")
    axes[2].set_ylabel(lat_label)
    axes[2].set_xlabel("Step")
    axes[2].set_ylim(0, sla * 1000 * 3)
    axes[2].legend(ncol=5, fontsize=11)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, f"{env_name}_behavior.png"), dpi=150)
    plt.close(fig)


def plot_comparison(env_name: str, summary: dict):
    names = list(summary.keys())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    panels = [("reward", "Mean episode reward", 1.0),
              ("sla_rate", "SLA violation rate (%)", 100.0),
              ("mean_cost", "Mean resource cost", 1.0)]
    bar_colors = ["#1E2761", "#C55A11", "#2C7A2C", "#999999"][: len(names)]
    for ax, (key, title, scale) in zip(axes, panels):
        vals = [summary[n][key] * scale for n in names]
        ax.bar(names, vals, color=bar_colors)
        ax.set_title(title, fontsize=12)
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis="x", labelsize=11)
    fig.suptitle(f"Policy comparison — {env_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, f"{env_name}_comparison.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", choices=ENV_CHOICES, required=True)
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    policies = load_agents(args.env)
    policies["Rule-based"] = ThresholdAutoscaler()
    policies["Static"] = StaticPolicy()

    summary = {}
    for name, pol in policies.items():
        summary[name] = evaluate_policy(args.env, pol, C.EVAL_EPISODES)
        m = summary[name]
        print(f"{name:12s} reward={m['reward']:8.2f}  SLA위반={m['sla_rate']*100:5.1f}%  "
              f"비용={m['mean_cost']:.3f}  평균지연={m['mean_latency']*1000:.0f}ms")

    with open(os.path.join(RESULTS, f"{args.env}_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "reward", "sla_rate", "mean_cost", "mean_latency_ms"])
        for name, m in summary.items():
            w.writerow([name, f"{m['reward']:.2f}", f"{m['sla_rate']:.4f}",
                        f"{m['mean_cost']:.4f}", f"{m['mean_latency']*1000:.1f}"])

    plot_learning_curves(args.env)
    plot_behavior(args.env, policies)
    plot_comparison(args.env, summary)
    print(f"plots saved to results/{args.env}_*.png")
