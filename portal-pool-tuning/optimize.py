"""
하이퍼파라미터 자동 탐색 — Optuna 베이지안 최적화
====================================================
DQN / PPO / SAC 각각의 하이퍼파라미터를 Optuna TPE 샘플러로 탐색해
최적값을 찾고 결과를 저장한다.

■ 왜 Optuna인가?
  - 그리드 탐색: N개 파라미터 × M단계 = M^N번 실행 (조합 폭발)
  - 랜덤 탐색: 비효율적, 좋은 영역을 반복 샘플링 못함
  - Optuna TPE: 이전 trial 결과를 보고 좋은 영역을 집중 탐색 (베이지안)
    → 적은 trial로 더 좋은 HP 발견 가능

■ 동작 흐름
  trial마다:
    1. TPE 샘플러가 HP 조합 제안
    2. 제안된 HP로 모델 학습 (timesteps 스텝)
    3. 별도 평가 환경에서 greedy 실행 → 평균 보상 측정
    4. 결과를 DB에 저장 → 다음 trial에 반영

■ 재시작 가능
  중단해도 results/optuna_*.db 에 결과가 저장되어 있어
  동일 명령어 재실행 시 이어서 탐색

사용법:
    pip install optuna optuna-dashboard

    # research 포털, DQN, 50회 탐색
    python optimize.py --portal research --algo DQN --n-trials 50

    # 빠른 테스트 (trial당 스텝 줄이기)
    python optimize.py --portal research --algo PPO --n-trials 30 --timesteps 50000

    # 웹 대시보드로 결과 시각화
    optuna-dashboard sqlite:///results/optuna_research_DQN.db

탐색 대상 HP:
    DQN : learning_rate, batch_size, buffer_size, exploration_fraction,
          exploration_final_eps, target_update_interval, train_freq, net_arch
    PPO : learning_rate, n_steps, batch_size, n_epochs, ent_coef,
          gae_lambda, clip_range, net_arch
    SAC : learning_rate, batch_size, buffer_size, tau, ent_coef,
          train_freq, net_arch

결과물 (results/ 폴더):
    optuna_<portal>_<algo>.db        -- Optuna 스터디 DB (재시작·대시보드용)
    optuna_<portal>_<algo>_best.txt  -- 최적 HP 텍스트 요약
    optuna_<portal>_<algo>_trials.csv-- 전체 trial 결과표 (보상순 정렬)
    optuna_<portal>_<algo>_history.png -- 최적화 진행 + HP 중요도 그래프
"""
from __future__ import annotations

import os
import csv
import argparse
import warnings
warnings.filterwarnings("ignore")   # SB3/gym 버전 경고 억제

import numpy as np
import matplotlib
matplotlib.use("Agg")               # GUI 없는 서버 환경에서도 그래프 저장
import matplotlib.pyplot as plt

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)  # Optuna 내부 로그 억제

from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from envs.portal_env import PortalPoolEnv
from config import PORTALS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# trial당 기본 학습 스텝
# 짧으면 탐색이 빠르지만 노이즈가 많고, 길면 정확하지만 느림
# 권장: 빠른 탐색 50000~80000, 정밀 탐색 150000~250000
DEFAULT_TIMESTEPS = 80_000

# 평가 에피소드 수: 많을수록 안정적이지만 trial당 시간 증가
EVAL_EPISODES = 10


# ── HP 탐색 공간 정의 ──────────────────────────────────────────────────────
# trial.suggest_* 메서드로 각 파라미터의 탐색 범위와 방식 지정
# - suggest_float(..., log=True): 로그 스케일로 탐색 (학습률 등에 적합)
# - suggest_categorical: 미리 정의한 후보 중 선택
# - suggest_float: 연속 범위에서 균등 탐색

def sample_dqn_params(trial: optuna.Trial) -> dict:
    """
    DQN 하이퍼파라미터 탐색 공간 정의.

    DQN 주요 HP와 탐색 이유:
      learning_rate     : 너무 크면 발산, 너무 작으면 수렴 느림. 로그 스케일 탐색
      batch_size        : 작으면 노이즈 많음, 크면 GPU 메모리 사용↑
      buffer_size       : 크면 과거 경험 더 활용, 메모리 사용↑
      exploration_fraction: 전체 학습 중 탐색 단계 비율
      exploration_final_eps: 탐색 종료 후 최소 ε값
      target_update_interval: 타겟 네트워크 동기화 주기 (너무 잦으면 불안정)
      train_freq        : 몇 스텝마다 학습할지
      net_arch          : MLP 레이어 구조 (깊을수록 표현력↑, 과적합 위험↑)
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "buffer_size": trial.suggest_categorical("buffer_size", [10_000, 50_000, 100_000, 200_000]),
        "exploration_fraction": trial.suggest_float("exploration_fraction", 0.1, 0.5),
        "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.1),
        "target_update_interval": trial.suggest_categorical(
            "target_update_interval", [200, 500, 1000, 2000]
        ),
        "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8]),
        "net_arch": trial.suggest_categorical(
            "net_arch", ["64-64", "128-128", "256-256", "128-64"]
        ),
    }


def sample_ppo_params(trial: optuna.Trial) -> dict:
    """
    PPO 하이퍼파라미터 탐색 공간 정의.

    PPO 주요 HP와 탐색 이유:
      learning_rate : Actor-Critic 네트워크 업데이트 크기
      n_steps       : rollout 길이. 길수록 분산↓ 편향↑
      batch_size    : minibatch 크기 (n_steps의 약수여야 함)
      n_epochs      : 동일 rollout으로 몇 번 업데이트할지
      ent_coef      : 엔트로피 보너스. 크면 탐색 많아짐
      gae_lambda    : GAE 감쇠 계수. 1에 가까울수록 MC 추정
      clip_range    : PPO 클리핑 범위. 너무 크면 불안정
      net_arch      : 네트워크 구조
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096]),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "n_epochs": trial.suggest_categorical("n_epochs", [3, 5, 10, 20]),
        "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.1),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.999),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
        "net_arch": trial.suggest_categorical(
            "net_arch", ["64-64", "128-128", "256-256", "128-64"]
        ),
    }


def sample_sac_params(trial: optuna.Trial) -> dict:
    """
    SAC 하이퍼파라미터 탐색 공간 정의.

    SAC 주요 HP와 탐색 이유:
      learning_rate : Actor/Critic/Alpha 네트워크 학습률
      batch_size    : 리플레이 버퍼에서 샘플링 크기
      buffer_size   : 경험 리플레이 버퍼 크기
      tau           : 소프트 업데이트 계수 (작을수록 안정적)
      ent_coef      : 엔트로피 계수. "auto"면 자동 조정
      train_freq    : 몇 스텝마다 업데이트할지
      net_arch      : 네트워크 구조
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "buffer_size": trial.suggest_categorical("buffer_size", [10_000, 50_000, 100_000, 200_000]),
        "tau": trial.suggest_float("tau", 0.001, 0.05),
        "ent_coef": trial.suggest_categorical("ent_coef", ["auto", 0.01, 0.1, 0.5]),
        "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8]),
        "net_arch": trial.suggest_categorical(
            "net_arch", ["64-64", "128-128", "256-256", "128-64"]
        ),
    }


# 알고리즘 → HP 샘플링 함수 매핑
SAMPLERS = {
    "DQN": sample_dqn_params,
    "PPO": sample_ppo_params,
    "SAC": sample_sac_params,
}


def parse_net_arch(arch_str: str) -> list[int]:
    """'128-64' 형태 문자열을 [128, 64] 정수 리스트로 변환. SB3 net_arch 형식."""
    return [int(x) for x in arch_str.split("-")]


# ── 모델 생성 ──────────────────────────────────────────────────────────────

def make_model(algo: str, params: dict, env, seed: int = 0):
    """
    HP 딕셔너리로 SB3 모델 인스턴스 생성.

    net_arch는 policy_kwargs로 전달해야 하므로 params에서 꺼내 변환.
    나머지 HP는 알고리즘별 생성자에 **kwargs로 전달.

    Args:
        algo  : "DQN", "PPO", "SAC" 중 하나
        params: sample_*_params()가 반환한 HP 딕셔너리 (net_arch 포함)
        env   : 학습용 Gymnasium 환경
        seed  : 재현성 시드 (trial.number를 사용해 trial마다 다르게)
    """
    net_arch = parse_net_arch(params.pop("net_arch", "128-128"))
    common = dict(
        policy="MlpPolicy",
        env=env,
        gamma=0.99,            # 할인율은 고정 (포털 튜닝에서 0.99가 적합)
        verbose=0,             # 학습 중 SB3 출력 억제
        seed=seed,
        policy_kwargs=dict(net_arch=net_arch),
    )
    if algo == "DQN":
        return DQN(**common, **params)
    elif algo == "PPO":
        return PPO(**common, **params)
    else:  # SAC
        return SAC(**common, **params)


# ── Optuna 목적 함수 ───────────────────────────────────────────────────────

def make_objective(algo: str, portal_cfg: dict, timesteps: int):
    """
    Optuna가 호출할 목적 함수(클로저) 생성.

    Optuna는 기본적으로 최솟값을 찾으므로, 보상(최대화 목표)의
    부호를 반전해 반환 (-reward 최소화 = reward 최대화).

    학습·평가 환경을 분리해 일반화 성능 측정:
      - 학습 환경: 랜덤 시드 없음 (에피소드마다 다른 패턴)
      - 평가 환경: 고정 시드 (trial 간 공정한 비교)

    Args:
        algo      : "DQN", "PPO", "SAC"
        portal_cfg: PORTALS 딕셔너리 항목
        timesteps : trial당 학습 스텝 수
    """
    def objective(trial: optuna.Trial) -> float:
        params = SAMPLERS[algo](trial)                        # HP 샘플링
        is_cont = (algo == "SAC")

        # 학습 환경 (Monitor로 감싸야 episode 보상 기록됨)
        train_env = Monitor(PortalPoolEnv(portal_cfg, continuous=is_cont))
        params_copy = dict(params)                            # pop()으로 원본 변경 방지
        model = make_model(algo, params_copy, train_env, seed=trial.number)
        model.learn(total_timesteps=timesteps)
        train_env.close()

        # 평가 환경 (다른 시드로 학습 환경과 다른 시나리오 → 일반화 측정)
        eval_env = Monitor(
            PortalPoolEnv(portal_cfg, continuous=is_cont, seed=9999 + trial.number)
        )
        mean_reward, _ = evaluate_policy(
            model, eval_env, n_eval_episodes=EVAL_EPISODES, deterministic=True
        )
        eval_env.close()

        return -float(mean_reward)                            # 부호 반전 (최소화 목적)

    return objective


# ── 결과 저장 ──────────────────────────────────────────────────────────────

def save_results(study: optuna.Study, portal: str, algo: str) -> None:
    """
    탐색 완료 후 결과물 3종 저장:
      1. 최적 HP 텍스트 (_best.txt)
      2. 전체 trial 보상순 CSV (_trials.csv)
      3. 최적화 진행 + HP 중요도 그래프 (_history.png)
    """
    prefix = os.path.join(RESULTS_DIR, f"optuna_{portal}_{algo}")
    best = study.best_trial

    # ── 1. 최적 HP 텍스트 ──────────────────────────────────────────────
    lines = [
        f"포털: {portal}  알고리즘: {algo}",
        f"탐색 trial 수: {len(study.trials)}",
        f"최적 보상: {-best.value:.4f}  (trial #{best.number})",
        "",
        "── 최적 하이퍼파라미터 ──",
    ] + [f"  {k}: {v}" for k, v in best.params.items()]
    txt_path = prefix + "_best.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  최적 HP 저장: {txt_path}")

    # ── 2. 전체 trial CSV (보상 내림차순 정렬) ─────────────────────────
    csv_path = prefix + "_trials.csv"
    param_keys = list(study.trials[0].params.keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trial", "reward"] + param_keys)
        # value(=-reward) 오름차순 = reward 내림차순
        for t in sorted(study.trials, key=lambda x: x.value or float("inf")):
            if t.value is None:
                continue
            w.writerow(
                [t.number, f"{-t.value:.4f}"] + [t.params.get(k, "") for k in param_keys]
            )
    print(f"  전체 결과 CSV: {csv_path}")

    # ── 3. 그래프: 최적화 진행 + HP 중요도 ────────────────────────────
    rewards = [-t.value for t in study.trials if t.value is not None]
    best_so_far = np.maximum.accumulate(rewards)             # 누적 최고 보상
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 왼쪽: trial별 보상 산점도 + 누적 최고선 + 최적점 ★
    ax = axes[0]
    ax.scatter(range(len(rewards)), rewards, alpha=0.5, s=20,
               color="#7f9cb0", label="Trial reward")
    ax.plot(best_so_far, color="red", lw=2, label="Best so far")
    best_idx = int(np.argmax(rewards))
    ax.scatter(best_idx, rewards[best_idx], color="gold", s=200,
               zorder=5, marker="*",
               label=f"Best (trial {best_idx}, {rewards[best_idx]:.1f})")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Mean Reward")
    ax.set_title(f"Optimization History — {portal}/{algo}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 오른쪽: HP 중요도 가로 막대 그래프
    # 어떤 파라미터가 성능에 가장 큰 영향을 미치는지 파악 가능
    ax2 = axes[1]
    try:
        importances = optuna.importance.get_param_importances(study)
        top_params = list(importances.keys())[:6]             # 상위 6개만 표시
        ax2.barh(
            top_params[::-1],
            [importances[p] for p in top_params[::-1]],
            color="#1f77b4"
        )
        ax2.set_xlabel("Importance Score")
        ax2.set_title("Hyperparameter Importance\n(높을수록 성능에 큰 영향)")
        ax2.grid(True, alpha=0.3, axis="x")
    except Exception:
        # trial 수가 너무 적으면 중요도 계산 불가
        ax2.text(0.5, 0.5, "중요도 계산 불가\n(trial 수 부족, 최소 20회 이상 권장)",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=11)

    fig.suptitle(f"{portal} / {algo}  |  Best reward: {-best.value:.2f}", fontsize=13)
    fig.tight_layout()
    hist_path = prefix + "_history.png"
    fig.savefig(hist_path, dpi=130)
    plt.close(fig)
    print(f"  최적화 그래프: {hist_path}")


# ── 탐색 실행 ──────────────────────────────────────────────────────────────

def run_optimize(portal: str, algo: str, n_trials: int, timesteps: int) -> None:
    """
    단일 포털·알고리즘 조합에 대해 Optuna 탐색 실행.

    ■ TPE 샘플러 (Tree-structured Parzen Estimator)
      - 이전 trial 결과를 바탕으로 '좋은 영역'과 '나쁜 영역'을 구분
      - 좋은 영역에서 더 자주 샘플링 → 적은 trial로 효율적 탐색
      - 처음 N회(warm-up)는 랜덤 탐색 후 TPE로 전환

    ■ MedianPruner
      - 중간 성능 이하의 trial을 조기에 종료 → 탐색 시간 절약
      - 이 프로젝트에서는 단일 학습 후 평가라 효과 제한적

    ■ SQLite DB 저장
      - 중단/재시작해도 이전 결과 유지
      - optuna-dashboard로 웹 UI 시각화 가능
    """
    portal_cfg = PORTALS[portal]
    db_path = os.path.join(RESULTS_DIR, f"optuna_{portal}_{algo}.db")
    storage = f"sqlite:///{db_path}"

    print(f"\n{'=' * 60}")
    print(f"  포털: {portal}  알고리즘: {algo}")
    print(f"  탐색 횟수: {n_trials}회  /  trial당 스텝: {timesteps:,}")
    print(f"  DB 저장: {db_path}")
    print(f"{'=' * 60}")

    # 기존 스터디가 있으면 이어서 실행 (load_if_exists=True)
    study = optuna.create_study(
        study_name=f"{portal}_{algo}",
        storage=storage,
        direction="minimize",                              # -reward 최소화
        load_if_exists=True,                               # 중단 후 재시작 가능
        sampler=optuna.samplers.TPESampler(seed=0),        # 베이지안 탐색
        pruner=optuna.pruners.MedianPruner(),              # 하위 50% trial 조기 종료
    )

    already_done = len(study.trials)
    remaining = max(0, n_trials - already_done)
    if already_done > 0:
        print(f"  이전 탐색 {already_done}회 발견 → {remaining}회 추가 실행")

    objective = make_objective(algo, portal_cfg, timesteps)

    # trial을 1개씩 실행해 진행 상황을 콘솔에 출력
    for i in range(remaining):
        trial_num = already_done + i + 1
        print(f"  [{trial_num}/{n_trials}] trial 실행 중...", end=" ", flush=True)
        study.optimize(objective, n_trials=1, show_progress_bar=False)
        last = study.trials[-1]
        reward = -last.value if last.value is not None else float("nan")
        best_reward = -study.best_value
        print(f"보상: {reward:8.2f}  |  현재 최고: {best_reward:8.2f}")

    # 탐색 완료 요약
    print(f"\n── 탐색 완료 ──")
    print(f"  최적 보상: {-study.best_value:.4f}  (trial #{study.best_trial.number})")
    print(f"  최적 HP:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

    save_results(study, portal, algo)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="하이퍼파라미터 탐색 (Optuna 베이지안 최적화)")
    p.add_argument("--portal", choices=[*PORTALS.keys(), "all"], default="research",
                   help="탐색할 포털 (research/user/admin/all)")
    p.add_argument("--algo", choices=["DQN", "PPO", "SAC", "all"], default="DQN",
                   help="탐색할 알고리즘")
    p.add_argument("--n-trials", type=int, default=50,
                   help="탐색 횟수 (기본 50, 많을수록 정확)")
    p.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS,
                   help=f"trial당 학습 스텝 (기본 {DEFAULT_TIMESTEPS:,})")
    args = p.parse_args()

    portals = list(PORTALS.keys()) if args.portal == "all" else [args.portal]
    algos = ["DQN", "PPO", "SAC"] if args.algo == "all" else [args.algo]

    for portal in portals:
        for algo in algos:
            run_optimize(portal, algo, args.n_trials, args.timesteps)

    print("\n=== 전체 탐색 완료 ===")
    print("최적 HP로 학습하려면:")
    print("  results/optuna_<portal>_<algo>_best.txt 참고 후 train.py 실행")
