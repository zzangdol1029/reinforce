"""
평가·비교·시각화 (통합 admission control)
==========================================
베이스라인(독립 제어) + RL(DQN/PPO/SAC, 통합)을 동일 시나리오에서 평가.
산출물:
  results/comparison.csv          정책별 수치
  results/comparison.png          보상/SLA위반/처리량/DB과부하 막대
  results/timeseries.png          숨은 C_db vs db_util, 라우트별 L, 우선순위 보호
사용법:
    python evaluate.py --episodes 20
"""
from __future__ import annotations
import os, argparse, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO, SAC
from envs.admission_env import AdmissionEnv
from baselines import BASELINES, evaluate_policy
from config import ENV_KWARGS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RL_SPECS = [("DQN", DQN, "flat"), ("PPO", PPO, "multi"), ("SAC", SAC, "box")]


def evaluate_rl(model, mode, n_episodes, base_seed=3000):
    env = AdmissionEnv(**ENV_KWARGS, action_mode=mode)
    agg = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        done, R = False, 0.0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a); R += r
            done = term or trunc
        m = env.metrics(); m["reward"] = R; agg.append(m)
    return {k: float(np.mean([d[k] for d in agg])) for k in agg[0]}


def main(n_episodes):
    rows = {}
    env = AdmissionEnv(**ENV_KWARGS, action_mode="multi")
    for pol in BASELINES:
        rows[pol.name] = evaluate_policy(pol, env, n_episodes=n_episodes)
        print(f"[baseline] {pol.name:14s} reward={rows[pol.name]['reward']:.0f}")
    for name, cls, mode in RL_SPECS:
        path = os.path.join(MODELS_DIR, f"{name}.zip")
        if not os.path.exists(path):
            print(f"[skip] {name} 모델 없음. train.py 먼저 실행."); continue
        rows[name] = evaluate_rl(cls.load(path), mode, n_episodes)
        print(f"[RL]       {name:14s} reward={rows[name]['reward']:.0f}")
    _save_csv(rows); _plot_bars(rows); _plot_timeseries(rows)
    print(f"\n결과 저장: {RESULTS_DIR}")


def _save_csv(rows):
    cols = ["reward", "sla_violation_rate", "mean_latency", "throughput", "rejected", "db_util", "prio_good"]
    with open(os.path.join(RESULTS_DIR, "comparison.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["policy"] + cols)
        for n, m in rows.items():
            w.writerow([n] + [round(m[k], 3) for k in cols])


def _plot_bars(rows):
    names = list(rows.keys()); rl = {"DQN", "PPO", "SAC"}
    colors = ["#d62728" if n in rl else "#7f9cb0" for n in names]
    panels = [("reward", "Avg reward (higher better)"),
              ("sla_violation_rate", "SLA violation rate (lower better)"),
              ("throughput", "Throughput (higher better)"),
              ("db_util", "Shared-DB utilization (>1 = overload)")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (k, t) in zip(axes.ravel(), panels):
        ax.bar(names, [rows[n][k] for n in names], color=colors)
        ax.set_title(t); ax.tick_params(axis="x", rotation=30); ax.grid(axis="y", alpha=0.3)
        if k == "db_util":
            ax.axhline(1.0, color="red", ls="--", lw=1)
    fig.suptitle("Joint admission control over shared DB: RL (red) vs Baselines (gray)", fontsize=14)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "comparison.png"), dpi=130)
    print("그래프:", os.path.join(RESULTS_DIR, "comparison.png"))


def _plot_timeseries(rows):
    # 사용 가능한 첫 RL 모델, 없으면 Oracle
    best, mode, label = None, "multi", None
    for name, cls, md in RL_SPECS:
        p = os.path.join(MODELS_DIR, f"{name}.zip")
        if os.path.exists(p):
            m = cls.load(p); best = lambda env, obs, _m=m: _m.predict(obs, deterministic=True)[0]
            mode, label = md, name; break
    if best is None:
        from baselines import OracleJoint
        orc = OracleJoint(); best = lambda env, obs: orc.act(env, obs); label = "Oracle-Joint(ref)"

    env = AdmissionEnv(**ENV_KWARGS, action_mode=mode); obs, _ = env.reset(seed=4242)
    cdb_log, util_log = [], []
    L_log = [[] for _ in range(env.N)]
    lat_ratio = [[] for _ in range(env.N)]
    names = [r["name"] for r in env.routes]; sla = np.array([r["sla"] for r in env.routes])
    done = False
    while not done:
        a = best(env, obs); obs, r, term, trunc, info = env.step(a)
        cdb_log.append(info["cdb"]); util_log.append(info["db_util"])
        for i in range(env.N):
            L_log[i].append(info["L"][i]); lat_ratio[i].append(info["latency"][i] / sla[i])
        done = term or trunc

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(cdb_log, color="#1f77b4", lw=1.4, label="hidden shared-DB capacity C_db(t)")
    ax0b = axes[0].twinx(); ax0b.plot(util_log, color="#d62728", lw=1.2, label="DB utilization")
    ax0b.axhline(1.0, color="red", ls=":", lw=1); ax0b.set_ylabel("DB util")
    axes[0].set_ylabel("C_db (HIDDEN)"); axes[0].set_title(f"{label}: tracks hidden DB capacity, holds util≈1")
    axes[0].legend(loc="upper left"); ax0b.legend(loc="upper right")
    for i in range(env.N):
        axes[1].plot(L_log[i], lw=1.3, label=f"L[{names[i]}]")
    axes[1].set_ylabel("concurrency limit L_i"); axes[1].legend(loc="upper left")
    axes[1].set_title("Per-route admission limits (priority-aware allocation)")
    for i in range(env.N):
        axes[2].plot(lat_ratio[i], lw=1.1, label=f"{names[i]} lat/SLA")
    axes[2].axhline(1.0, color="black", ls=":", lw=1); axes[2].set_ylim(0, 3)
    axes[2].set_ylabel("latency / SLA"); axes[2].set_xlabel("control step")
    axes[2].legend(loc="upper left"); axes[2].set_title("Per-route SLA compliance (<1 = within SLA)")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "timeseries.png"), dpi=130)
    print("시계열:", os.path.join(RESULTS_DIR, "timeseries.png"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    args = p.parse_args()
    main(args.episodes)
