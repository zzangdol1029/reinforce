"""
평가 · 비교 · 시각화 (포털 워커+DB풀 동시 튜닝)
================================================
학습된 RL 모델(DQN/PPO/SAC)과 베이스라인 정책을 동일한 시나리오에서 실행해
성능을 비교하고 결과를 그래프와 CSV로 저장한다.

■ 평가 방식
  - 모든 정책을 동일한 랜덤 시드 n_episodes개로 실행
  - 에피소드별 보상·SLA위반율·평균지연·평균자원량 집계 후 평균
  - RL은 deterministic=True (탐색 없이 greedy 행동)

■ 생성 결과물
  results/comparison_<portal>.csv   : 정책별 수치 비교표
  results/comparison_<portal>.png   : 4개 지표 막대 비교 그래프
  results/timeseries_<portal>.png   : 부하 대비 W·D 조절 시계열

사용법:
    python evaluate.py --portal research --episodes 20
"""
from __future__ import annotations

import os
import argparse
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")   # GUI 없는 서버 환경에서도 그래프 저장 가능
import matplotlib.pyplot as plt

from stable_baselines3 import DQN, PPO, SAC

from envs.portal_env import PortalPoolEnv
from baselines import BASELINES, evaluate_policy
from config import PORTALS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# RL 알고리즘 스펙: (이름, SB3 클래스, 연속행동 여부)
# SAC만 continuous=True (Box 행동공간), DQN/PPO는 Discrete
RL_SPECS = [("DQN", DQN, False), ("PPO", PPO, False), ("SAC", SAC, True)]


def evaluate_rl(model, portal: str, continuous: bool,
                n_episodes: int, base_seed: int = 4000) -> dict:
    """
    학습된 RL 모델을 n_episodes 에피소드 실행하고 평균 성능 지표 반환.

    매 에피소드마다 다른 시드(base_seed + ep)로 다양한 부하 패턴 테스트.
    model.predict(deterministic=True)로 탐색 없이 greedy 정책 평가.

    Args:
        model      : SB3로 학습된 모델 (DQN/PPO/SAC)
        portal     : 포털 이름 문자열 (config.PORTALS 키)
        continuous : SAC는 True, DQN/PPO는 False
        n_episodes : 평가 에피소드 수
        base_seed  : 첫 에피소드 시드

    Returns:
        dict: reward, sla_violation_rate, mean_latency, mean_workers, mean_db_conns 평균값
    """
    agg = []
    for ep in range(n_episodes):
        env = PortalPoolEnv(PORTALS[portal], continuous=continuous, seed=base_seed + ep)
        obs, _ = env.reset(seed=base_seed + ep)
        done = False
        R = 0.0
        while not done:
            # deterministic=True: ε-greedy 탐색 없이 최선 행동만 선택
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            R += r
            done = term or trunc
        m = env.metrics()
        m["reward"] = R
        agg.append(m)
    return {k: float(np.mean([m[k] for m in agg])) for k in agg[0]}


def main(portal: str, n_episodes: int) -> None:
    """
    전체 평가 파이프라인 실행.

    1. 베이스라인 5종 평가 (StaticCurrent, StaticMax, StaticLow, HillClimb, Oracle)
    2. RL 모델 평가 (models/ 에 .zip 파일이 있는 것만)
    3. CSV + 막대 그래프 + 시계열 그래프 저장
    """
    rows = {}

    # ── 베이스라인 평가 ──────────────────────────────────────────────────
    for pol in BASELINES:
        rows[pol.name] = evaluate_policy(pol, PORTALS[portal], n_episodes=n_episodes)
        print(f"[baseline] {pol.name:15s} reward={rows[pol.name]['reward']:.2f}")

    # ── RL 모델 평가 ─────────────────────────────────────────────────────
    for name, cls, cont in RL_SPECS:
        path = os.path.join(MODELS_DIR, f"{portal}_{name}.zip")
        if not os.path.exists(path):
            # 학습된 모델이 없으면 건너뜀 (train.py 먼저 실행 필요)
            print(f"[skip] {portal}_{name} 모델 없음. train.py 먼저 실행.")
            continue
        rows[name] = evaluate_rl(cls.load(path), portal, cont, n_episodes)
        print(f"[RL]       {name:15s} reward={rows[name]['reward']:.2f}")

    # ── 결과 저장 ────────────────────────────────────────────────────────
    _save_csv(rows, portal)
    _plot_bars(rows, portal)
    _plot_timeseries(rows, portal)
    print(f"\n결과 저장 완료: {RESULTS_DIR}")


def _save_csv(rows: dict, portal: str) -> None:
    """
    정책별 성능 지표를 CSV로 저장.

    컬럼: policy, reward, sla_violation_rate, mean_latency, mean_workers, mean_db_conns
    행: 각 정책(베이스라인 + RL)
    """
    cols = ["reward", "sla_violation_rate", "mean_latency", "mean_workers", "mean_db_conns"]
    path = os.path.join(RESULTS_DIR, f"comparison_{portal}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["policy"] + cols)
        for n, m in rows.items():
            w.writerow([n] + [round(m[k], 4) for k in cols])
    print(f"CSV: {path}")


def _plot_bars(rows: dict, portal: str) -> None:
    """
    4개 지표를 2×2 서브플롯으로 비교하는 막대 그래프 저장.

    RL 정책(DQN/PPO/SAC): 빨간색 (#d62728)
    베이스라인: 회색 (#7f9cb0)
    → 한 눈에 RL vs 베이스라인 비교 가능

    4개 지표:
      1. 평균 보상 (높을수록 좋음)
      2. SLA 위반율 (낮을수록 좋음)
      3. 평균 워커 수 (낮을수록 메모리 절약)
      4. 평균 DB 커넥션 수 (낮을수록 공유 DB 부담 적음)
    """
    names = list(rows.keys())
    rl_names = {"DQN", "PPO", "SAC"}
    colors = ["#d62728" if n in rl_names else "#7f9cb0" for n in names]

    panels = [
        ("reward",            "Avg reward (higher better)"),
        ("sla_violation_rate", "SLA violation rate (lower better)"),
        ("mean_workers",      "Avg worker threads (lower = less memory)"),
        ("mean_db_conns",     "Avg DB connections (lower = less contention)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (key, title) in zip(axes.ravel(), panels):
        ax.bar(names, [rows[n][key] for n in names], color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Portal '{portal}' — Worker+DB pool tuning: RL (red) vs Baselines (gray)",
        fontsize=14
    )
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f"comparison_{portal}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"막대 그래프: {path}")


def _plot_timeseries(rows: dict, portal: str) -> None:
    """
    최선 RL 모델(없으면 Oracle)이 부하에 따라 W·D를 어떻게 조절하는지 시계열로 시각화.

    ■ 그래프 구성
      왼쪽 Y축 (파란선): 초당 요청 수(RPS) — 부하 변화
      오른쪽 Y축:
        빨간 실선: 워커 수(W) 변화
        초록 실선: DB 커넥션 수(D) 변화
        빨간 점선: 현재 운영 워커 수 (정적 설정 기준선)
        초록 점선: 현재 운영 DB풀 크기 (정적 설정 기준선)

    점선(기준선) 위아래로 RL이 얼마나 탄력적으로 자원을 조절하는지 확인 가능.
    """
    # 사용 가능한 첫 번째 RL 모델 로드 (DQN → PPO → SAC 순)
    best_fn = None
    spec = None
    for name, cls, cont in RL_SPECS:
        p = os.path.join(MODELS_DIR, f"{portal}_{name}.zip")
        if os.path.exists(p):
            m = cls.load(p)
            # 람다 캡처 버그 방지: _m=m으로 기본값 바인딩
            best_fn = lambda env, obs, _m=m: _m.predict(obs, deterministic=True)[0]
            spec = (name, cont)
            break

    # RL 모델이 없으면 Oracle로 대체
    if best_fn is None:
        from baselines import Oracle
        orc = Oracle()
        best_fn = lambda env, obs: orc.act(env, obs)
        spec = ("Oracle(ref)", False)

    # 고정 시드로 에피소드 실행해 재현 가능한 시계열 생성
    env = PortalPoolEnv(PORTALS[portal], continuous=spec[1], seed=777)
    obs, _ = env.reset(seed=777)
    lam_list, W_list, D_list = [], [], []
    done = False
    while not done:
        a = best_fn(env, obs)
        obs, _, term, trunc, info = env.step(a)
        lam_list.append(info["lam"])
        W_list.append(info["W"])
        D_list.append(info["D"])
        done = term or trunc

    # 듀얼 Y축 그래프
    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax1.plot(lam_list, label="Incoming RPS", color="#1f77b4", lw=1.5)
    ax1.set_xlabel("Control step")
    ax1.set_ylabel("RPS")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()                                         # 오른쪽 Y축 공유
    ax2.plot(W_list, label="Workers (W)", color="#d62728", lw=1.3)
    ax2.plot(D_list, label="DB connections (D)", color="#2ca02c", lw=1.3)
    # 현재 운영 설정 기준선 (점선)
    ax2.axhline(env.current_w, color="#d62728", ls=":", alpha=0.5, lw=1,
                label=f"Current W={env.current_w}")
    ax2.axhline(env.current_d, color="#2ca02c", ls=":", alpha=0.5, lw=1,
                label=f"Current D={env.current_d}")
    ax2.set_ylabel("Workers (red) / DB conns (green)  |  dotted = current static config")
    ax2.legend(loc="upper right")

    ax1.set_title(
        f"Portal '{portal}' — {spec[0]}: W & D adjustment vs load\n"
        f"(dotted lines = current static config)"
    )
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f"timeseries_{portal}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"시계열 그래프: {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RL vs 베이스라인 평가·비교")
    p.add_argument("--portal", choices=list(PORTALS.keys()), default="research",
                   help="평가할 포털 (research/user/admin)")
    p.add_argument("--episodes", type=int, default=20,
                   help="평가 에피소드 수 (많을수록 정확, 기본 20)")
    args = p.parse_args()
    main(args.portal, args.episodes)
