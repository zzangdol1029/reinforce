"""
DQN / PPO / SAC 학습 (포털 워커+DB풀 동시 튜닝, Stable-Baselines3)
==================================================================
사용법:
    python train.py --portal research --algo all --timesteps 250000
    python train.py --portal user --algo PPO

- DQN, PPO : Discrete(25) 행동공간 (ΔW 5단계 × ΔD 5단계)
- SAC      : 연속 Box(2) 행동공간 (continuous=True)
모델은 models/<portal>_<ALGO>.zip 에 저장된다.
학습 곡선은 results/learning_curve_<portal>_<ALGO>.png 에 저장된다.
"""
from __future__ import annotations

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI 없는 환경에서도 그래프 저장 가능
import matplotlib.pyplot as plt

from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback

from envs.portal_env import PortalPoolEnv
from config import PORTALS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 수렴 판정: 이동평균이 최고점 대비 이 비율 이상 하락하면 과적합 경고
OVERFIT_DROP_RATIO = 0.05   # 5% 하락 시 과적합 구간으로 표시
# 수렴 판정: 이동평균 변화량이 이 값 이하로 N 에피소드 이상 지속되면 수렴으로 판정
CONVERGE_THRESHOLD = 0.5    # 이동평균 변화 임계값
CONVERGE_WINDOW = 50        # 수렴 판정 최소 지속 에피소드 수


class EpisodeLogCallback(BaseCallback):
    """
    에피소드 보상을 수집하고 콘솔 출력 + 학습 곡선 그래프 저장.

    학습 곡선에 표시되는 정보:
      - 에피소드별 원본 보상 (반투명 회색)
      - 이동평균 보상 (굵은 선)
      - 최고 성능 지점 (★ 마커)
      - 수렴 시작 지점 (▼ 마커): 이동평균 개선이 멈춘 지점
      - 과적합 구간 (빨간 음영): 최고점 이후 성능이 하락한 구간
    """

    def __init__(self, portal: str, algo: str, log_interval: int = 20, ma_window: int = 30):
        """
        Args:
            portal: 포털 이름 (파일명 생성용)
            algo: 알고리즘 이름 (파일명 생성용)
            log_interval: 콘솔에 요약을 출력할 에피소드 간격
            ma_window: 이동평균 계산 윈도우 크기
        """
        super().__init__(verbose=0)
        self.portal = portal
        self.algo = algo
        self.log_interval = log_interval
        self.ma_window = ma_window
        self._ep_rewards: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._ep_rewards.append(info["episode"]["r"])
                n = len(self._ep_rewards)
                if n % self.log_interval == 0:
                    recent = self._ep_rewards[-self.log_interval:]
                    print(
                        f"  ep {n:5d} | "
                        f"mean reward(last {self.log_interval}): {np.mean(recent):8.2f} | "
                        f"max: {np.max(recent):8.2f} | "
                        f"min: {np.min(recent):8.2f}"
                    )
        return True

    def _on_training_end(self) -> None:
        """학습 완료 후 학습 곡선 분석 및 그래프 저장."""
        if len(self._ep_rewards) < 2:
            return
        self._analyze_and_plot()

    def _compute_moving_average(self, rewards: list[float]) -> np.ndarray:
        """이동평균 계산. 초반 에피소드는 가용 데이터로만 계산."""
        ma = np.array([
            np.mean(rewards[max(0, i - self.ma_window + 1): i + 1])
            for i in range(len(rewards))
        ])
        return ma

    def _find_convergence(self, ma: np.ndarray) -> int | None:
        """
        수렴 시작 지점 탐지.
        이동평균 변화량이 CONVERGE_THRESHOLD 이하로 CONVERGE_WINDOW 에피소드 이상
        지속되는 첫 번째 지점을 반환.
        """
        changes = np.abs(np.diff(ma))
        for i in range(len(changes) - CONVERGE_WINDOW):
            window_changes = changes[i: i + CONVERGE_WINDOW]
            if np.max(window_changes) < CONVERGE_THRESHOLD:
                return i  # 수렴 시작 에피소드 인덱스
        return None

    def _find_overfit_start(self, ma: np.ndarray) -> int | None:
        """
        과적합 구간 시작 탐지.
        이동평균 최고점 이후 OVERFIT_DROP_RATIO 이상 하락한 지점을 반환.
        최고점이 마지막 에피소드에 가까우면 과적합 없음으로 판단.
        """
        peak_idx = int(np.argmax(ma))
        # 최고점이 전체의 90% 이후면 과적합 아님 (아직 학습 중)
        if peak_idx > len(ma) * 0.9:
            return None
        peak_val = ma[peak_idx]
        drop_threshold = peak_val - abs(peak_val) * OVERFIT_DROP_RATIO
        for i in range(peak_idx + 1, len(ma)):
            if ma[i] < drop_threshold:
                return i
        return None

    def _analyze_and_plot(self) -> None:
        rewards = self._ep_rewards
        episodes = np.arange(1, len(rewards) + 1)
        ma = self._compute_moving_average(rewards)

        peak_idx = int(np.argmax(ma))
        converge_idx = self._find_convergence(ma)
        overfit_start = self._find_overfit_start(ma)

        # ── 콘솔 요약 출력 ─────────────────────────────────────────────
        print(f"\n  ── 학습 분석 [{self.portal}/{self.algo}] ──")
        print(f"  총 에피소드     : {len(rewards)}")
        print(f"  최고 이동평균   : {ma[peak_idx]:.2f}  (ep {peak_idx + 1})")
        print(f"  최종 이동평균   : {ma[-1]:.2f}")
        if converge_idx is not None:
            print(f"  수렴 시작       : ep {converge_idx + 1} (이후 개선 없음)")
        else:
            print(f"  수렴 시작       : 미감지 (학습이 더 필요할 수 있음)")
        if overfit_start is not None:
            drop_pct = (ma[peak_idx] - ma[-1]) / max(abs(ma[peak_idx]), 1e-6) * 100
            print(f"  과적합 경고     : ep {overfit_start + 1} 이후 성능 하락 ({drop_pct:.1f}%↓)")
        else:
            print(f"  과적합          : 미감지")

        # ── 그래프 ─────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(12, 5))

        # 원본 에피소드 보상 (반투명)
        ax.plot(episodes, rewards, color="lightgray", alpha=0.5, lw=0.8, label="Episode reward")

        # 이동평균
        ax.plot(episodes, ma, color="#1f77b4", lw=2.0,
                label=f"Moving avg (window={self.ma_window})")

        # 최고 성능 지점 ★
        ax.scatter(peak_idx + 1, ma[peak_idx], color="gold", s=200,
                   zorder=5, marker="*", label=f"Peak (ep {peak_idx + 1}, {ma[peak_idx]:.1f})")

        # 수렴 시작 지점 ▼
        if converge_idx is not None:
            ax.axvline(converge_idx + 1, color="green", ls="--", lw=1.5, alpha=0.8)
            ax.scatter(converge_idx + 1, ma[converge_idx], color="green", s=100,
                       zorder=5, marker="v", label=f"Converged (ep {converge_idx + 1})")

        # 과적합 구간 빨간 음영
        if overfit_start is not None:
            ax.axvspan(overfit_start + 1, len(rewards), color="red", alpha=0.08,
                       label=f"Possible overfit (ep {overfit_start + 1}~)")
            ax.axvline(overfit_start + 1, color="red", ls=":", lw=1.5, alpha=0.7)

        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title(f"Learning Curve — {self.portal} / {self.algo}")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = os.path.join(RESULTS_DIR, f"learning_curve_{self.portal}_{self.algo}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"  학습 곡선 저장  : {path}")


def make_env(portal, continuous):
    return Monitor(PortalPoolEnv(PORTALS[portal], continuous=continuous))


def train_one(algo, portal, timesteps, log_interval):
    cont = (algo == "SAC")
    env = make_env(portal, cont)
    common = dict(policy="MlpPolicy", env=env, gamma=0.99, verbose=0, seed=0,
                  policy_kwargs=dict(net_arch=[128, 128]))
    if algo == "DQN":
        model = DQN(learning_rate=5e-4, buffer_size=100_000, learning_starts=3000, batch_size=128,
                    train_freq=4, target_update_interval=1000, exploration_fraction=0.3,
                    exploration_final_eps=0.05, **common)
    elif algo == "PPO":
        model = PPO(learning_rate=3e-4, n_steps=2048, batch_size=128, n_epochs=10,
                    gae_lambda=0.95, ent_coef=0.01, **common)
    else:  # SAC
        model = SAC(learning_rate=3e-4, buffer_size=100_000, learning_starts=3000, batch_size=256,
                    tau=0.005, train_freq=1, **common)

    callbacks = [
        ProgressBarCallback(),
        EpisodeLogCallback(portal=portal, algo=algo, log_interval=log_interval),
    ]
    model.learn(total_timesteps=timesteps, callback=callbacks)
    model.save(os.path.join(MODELS_DIR, f"{portal}_{algo}"))
    print(f"[{portal}/{algo}] saved")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--portal", choices=[*PORTALS.keys(), "all"], default="research")
    p.add_argument("--algo", choices=["DQN", "PPO", "SAC", "all"], default="all")
    p.add_argument("--timesteps", type=int, default=250_000)
    p.add_argument("--log-interval", type=int, default=20, help="N 에피소드마다 보상 요약 출력")
    args = p.parse_args()

    portals = list(PORTALS.keys()) if args.portal == "all" else [args.portal]
    algos = ["DQN", "PPO", "SAC"] if args.algo == "all" else [args.algo]
    total = len(portals) * len(algos)
    count = 0
    for portal in portals:
        for algo in algos:
            count += 1
            print(f"\n[{count}/{total}] Training {portal}/{algo} for {args.timesteps} steps")
            train_one(algo, portal, args.timesteps, args.log_interval)
    print(f"\n=== 전체 완료: {total}개 모델 학습 ===")
