"""
Quiz Q3 — y = sin(6πx) 회귀
==============================
아래 RUNS 리스트에 원하는 하이퍼파라미터 조합을 추가/수정한 뒤
  python quiz_q3_sin_6pi.py
로만 실행하면 한 그래프에 모든 곡선이 그려집니다.

sin(6πx) 는 sin(4πx) 보다 주기가 짧아(구간 [0,1] 안에 3주기)
더 많은 은닉 뉴런 · 더 높은 학습률 · 더 많은 반복이 필요합니다.

--single 플래그를 주면 기존 단일 실행 모드로 동작합니다.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from mpl_korean_font import configure_korean_font

from dezero import Model
from dezero import optimizers
import dezero.layers as L
import dezero.functions as F


# ──────────────────────────────────────────────────────────────────────────────
#  ★ 여기서 비교할 하이퍼파라미터 조합을 직접 수정하세요 ★
#
#   hidden   : 은닉층 뉴런 수
#   lr       : 학습률
#   optimizer: "sgd" | "momentum" | "adagrad" | "adam"
#   iters    : 학습 반복 횟수
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Run:
    hidden: int
    lr: float
    optimizer: str
    iters: int

    def label(self) -> str:
        return f"h={self.hidden}, lr={self.lr:g}, {self.optimizer.upper()}, {self.iters}iter"


RUNS: list[Run] = [
    Run(hidden=16,  lr=0.10, optimizer="sgd",      iters=20000),
    Run(hidden=32,  lr=0.10, optimizer="sgd",      iters=20000),
    Run(hidden=64,  lr=0.10, optimizer="sgd",      iters=20000),
    Run(hidden=64,  lr=0.20, optimizer="sgd",      iters=20000),
    Run(hidden=64,  lr=0.05, optimizer="sgd",      iters=20000),
    Run(hidden=64,  lr=0.10, optimizer="momentum", iters=20000),
    Run(hidden=64,  lr=0.10, optimizer="adagrad",  iters=20000),
    Run(hidden=64,  lr=0.01, optimizer="adam",     iters=20000),
]
# ──────────────────────────────────────────────────────────────────────────────

# 전역 설정
SEED      = 0
N_SAMPLES = 200
NOISE     = 0.10
LOG_EVERY = 0       # 0 = 진행 로그 끔
OMEGA     = 6       # y = sin(OMEGA * π * x)


# ──────────────────── 모델 정의 ────────────────────
class TwoLayerNet(Model):
    def __init__(self, hidden_size: int, out_size: int = 1):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        return self.l2(F.sigmoid(self.l1(x)))


def make_optimizer(name: str, lr: float):
    name = name.lower()
    match name:
        case "sgd":      return optimizers.SGD(lr)
        case "momentum": return optimizers.MomentumSGD(lr)
        case "adagrad":  return optimizers.AdaGrad(lr)
        case "adam":     return optimizers.Adam(alpha=lr)
        case _:          raise ValueError(f"알 수 없는 옵티마이저: {name}")


# ──────────────────── 학습 ────────────────────
def train_one(run: Run, x: np.ndarray, y: np.ndarray) -> tuple[TwoLayerNet, float, float]:
    """주어진 Run 설정으로 학습 → (모델, train_mse, grid_mse) 반환."""
    model = TwoLayerNet(run.hidden)
    opt   = make_optimizer(run.optimizer, run.lr)
    opt.setup(model)

    for i in range(run.iters):
        y_pred = model(x)
        loss   = F.mean_squared_error(y, y_pred)
        model.cleargrads()
        loss.backward()
        opt.update()
        if LOG_EVERY > 0 and i % LOG_EVERY == 0:
            print(f"    [{i:6d}/{run.iters}] loss={float(loss.data):.6f}")

    y_pred    = model(x)
    train_mse = float(F.mean_squared_error(y, y_pred).data)

    t        = np.linspace(0, 1, 500)[:, np.newaxis]
    clean    = np.sin(OMEGA * np.pi * t)
    grid_mse = float(np.mean((model(t).data - clean) ** 2))

    return model, train_mse, grid_mse


# ──────────────────── 비교 실행 ────────────────────
def run_compare(runs: list[Run], seed: int, n_samples: int, noise: float, save: str) -> None:
    np.random.seed(seed)
    x  = np.random.rand(n_samples, 1)
    y  = np.sin(OMEGA * np.pi * x) + noise * np.random.rand(n_samples, 1)

    print(f"=== 비교 모드: {len(runs)}개 조합 · seed={seed} · n={n_samples} ===\n")

    results: list[tuple[Run, TwoLayerNet, float, float]] = []
    for idx, run in enumerate(runs):
        print(f"[{idx+1}/{len(runs)}] {run.label()}")
        model, tm, gm = train_one(run, x, y)
        results.append((run, model, tm, gm))
        print(f"         train MSE={tm:.6f}  grid MSE={gm:.6f}")

    # ── 요약 표 ──
    print("\n" + "─"*72)
    print(f"{'설정':<50} {'train MSE':>10} {'grid MSE':>10}")
    print("─"*72)
    for run, _, tm, gm in results:
        print(f"{run.label():<50} {tm:10.6f} {gm:10.6f}")
    print("─"*72)

    # ── 시각화 ──
    configure_korean_font()

    colors = _pick_colors(len(results))
    tt     = np.linspace(0, 1, 500)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(12, 6))

    # 연파랑 산점도 (학습 데이터)
    ax.scatter(x, y, s=14, alpha=0.30, c="lightsteelblue",
               edgecolors="none", label="train data (with noise)", zorder=1)

    # 검은 점선 (참값)
    ax.plot(tt, np.sin(OMEGA * np.pi * tt), "k--", lw=1.5, alpha=0.65,
            label=rf"참값 $y=\sin({OMEGA}\pi x)$", zorder=2)

    # 각 Run 예측 곡선
    for i, (run, model, tm, gm) in enumerate(results):
        yy    = model(tt).data
        label = (
            f"{run.label()}\n"
            f"  train MSE={tm:.4f}  grid MSE={gm:.4f}"
        )
        ax.plot(tt, yy, "-", lw=2.2, color=colors[i], label=label, zorder=3+i)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        rf"Q3 — 하이퍼파라미터 비교: $\sin({OMEGA}\pi x)$ 회귀" + "\n"
        f"동일 데이터 (seed={seed}, n={n_samples}, noise={noise})",
        fontsize=12,
    )
    ax.grid(True, alpha=0.25)

    # 우측 상단 범례 (각 색 = 어떤 하이퍼파라미터인지)
    ax.legend(
        loc="upper right",
        fontsize=8.5,
        framealpha=0.93,
        ncol=1,
        handlelength=2.0,
    )

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"\n그림 저장: {save}")
    plt.show()


def _pick_colors(n: int) -> list:
    if n <= 10:
        return [plt.cm.tab10(i) for i in range(n)]
    return [plt.cm.tab20(i % 20) for i in range(n)]


# ──────────────────── 단일 실행 (--single) ────────────────────
def run_single(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    x = np.random.rand(args.n_samples, 1)
    y = np.sin(OMEGA * np.pi * x) + args.noise * np.random.rand(args.n_samples, 1)

    run = Run(args.hidden, args.lr, args.optimizer, args.iters)
    model, tm, gm = train_one(run, x, y)
    print(f"train MSE={tm:.8f}  grid MSE={gm:.8f}")

    if args.no_plot:
        return

    configure_korean_font()
    tt = np.linspace(0, 1, 500)[:, np.newaxis]
    plt.figure(figsize=(9, 5))
    plt.scatter(x, y, s=12, alpha=0.5, label="train data (with noise)")
    plt.plot(tt, model(tt).data, "r-", lw=2, label="MLP prediction")
    plt.plot(tt, np.sin(OMEGA * np.pi * tt), "k--", lw=1.2, alpha=0.75,
             label=rf"$y=\sin({OMEGA}\pi x)$")
    plt.xlabel("x"); plt.ylabel("y")
    plt.title(rf"Q3 — $\sin({OMEGA}\pi x)$ 회귀")
    plt.legend(loc="upper right"); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
    plt.show()


# ──────────────────── 진입점 ────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=f"Q3: sin({OMEGA}πx) 하이퍼파라미터 비교")
    ap.add_argument("--seed",      type=int,   default=SEED)
    ap.add_argument("--n-samples", type=int,   default=N_SAMPLES)
    ap.add_argument("--noise",     type=float, default=NOISE)
    ap.add_argument("--save",      type=str,   default="", help="저장할 파일명 (예: q3.png)")
    ap.add_argument("--no-plot",   action="store_true")

    # --single 모드 전용 옵션
    ap.add_argument("--single",    action="store_true", help="단일 설정으로 실행")
    ap.add_argument("--hidden",    type=int,   default=64)
    ap.add_argument("--lr",        type=float, default=0.2)
    ap.add_argument("--iters",     type=int,   default=20000)
    ap.add_argument("--optimizer", default="sgd",
                    choices=["sgd", "momentum", "adagrad", "adam"])

    args = ap.parse_args()

    if args.single:
        run_single(args)
    else:
        run_compare(RUNS, args.seed, args.n_samples, args.noise, args.save)


if __name__ == "__main__":
    main()
