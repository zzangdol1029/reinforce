"""
PDF 16페이지 Quiz Q1 — Optimizer 비교
======================================
실습(dezero4.py)과 동일한 데이터·모델 구조로 다음 Optimizer 의 학습 곡선(loss)을 비교합니다.

    - MomentumSGD
    - AdaGrad
    - Adam

(비교 기준선으로 SGD 도 함께 그립니다.)

실행 예:
    python quiz_q1_optimizer_compare.py
    python quiz_q1_optimizer_compare.py --iters 8000 --no-plot   # 그래프 없이 수치만

주의:
    옵티마이저마다 안정적인 학습률 스케일이 다릅니다. 기본값은 경험적으로 맞춘 값이며,
    --lr-* 인자로 각각 조정할 수 있습니다.
"""
from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# week10/matplotlibrc 가 MacOSX 로 되어 있으면 Windows 에서 그래프 실패 → 강제 선택
if "--no-plot" in sys.argv:
    matplotlib.use("Agg", force=True)
elif sys.platform == "darwin":
    matplotlib.use("MacOSX", force=True)
else:
    matplotlib.use("TkAgg", force=True)

import matplotlib.pyplot as plt

import numpy as np

from mpl_korean_font import configure_korean_font

from dezero import Model
from dezero import optimizers
import dezero.layers as L
import dezero.functions as F

OUT_DIR = Path(__file__).resolve().parent / "results_quiz_q1_optimizer_compare"


@dataclass(frozen=True)
class ParamRow:
    """슬라이드·보고서용 변수 설명 (quiz_q1_optimizer_compare.md 와 동일)."""

    name: str
    meaning: str
    setting: str


# ★ 표 내용을 바꿀 때는 여기와 quiz_q1_optimizer_compare.md 를 함께 수정
PARAM_TABLE: tuple[ParamRow, ...] = (
    ParamRow(
        "optimizer",
        "가중치를 갱신하는 알고리즘",
        "4종 동시 비교: SGD(기준선), MomentumSGD, AdaGrad, Adam",
    ),
    ParamRow(
        "lr (학습률)",
        "1 step 당 파라미터 갱신 크기",
        "옵티마이저마다 별도 기본값 (--lr-sgd / --lr-momentum / --lr-adagrad / --lr-adam)",
    ),
    ParamRow(
        "iters",
        "학습 반복(iteration) 횟수",
        "모두 동일 (기본 10,000, --iters)",
    ),
    ParamRow(
        "hidden",
        "2층 MLP 은닉층 뉴런 수",
        "모두 동일 (기본 10, dezero4.py 와 동일, --hidden)",
    ),
    ParamRow(
        "seed",
        "난수 시드 (데이터·초기화 재현)",
        "모두 동일 (기본 0, --seed)",
    ),
    ParamRow(
        "초기 가중치",
        "학습 시작 시 W, b",
        "스냅샷 복원으로 4회 모두 동일 (공정 비교)",
    ),
    ParamRow(
        "데이터",
        "비선형 회귀 과제",
        "n=100, x~U(0,1), y=sin(2πx)+noise (dezero4.py 와 동일)",
    ),
    ParamRow(
        "log-every",
        "loss 기록 간격(step)",
        "모두 동일 (기본 10, --log-every)",
    ),
    ParamRow(
        "loss (지표)",
        "학습 성능 지표",
        "MSE; 그래프 y축 log scale",
    ),
)


def print_param_table() -> None:
    col_w = (14, 28, 52)
    header = ("변수", "의미", "비교 시 설정 (기본값)")
    sep = "─" * (sum(col_w) + 6)
    print(sep)
    print(f"{header[0]:<{col_w[0]}} | {header[1]:<{col_w[1]}} | {header[2]}")
    print(sep)
    for row in PARAM_TABLE:
        print(f"{row.name:<{col_w[0]}} | {row.meaning:<{col_w[1]}} | {row.setting}")
    print(sep)


def format_lr_row(label: str, lr: float) -> str:
    return f"  {label:<14} lr = {lr:g}"


def save_summary(
    args: argparse.Namespace,
    final_losses: dict[str, float],
) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "parameter_summary.txt"
    lines = [
        "Quiz Q1 — Optimizer 비교 (quiz_q1_optimizer_compare.py)",
        "",
        "[변수 의미·비교 설정]",
    ]
    for row in PARAM_TABLE:
        lines.append(f"{row.name}\t{row.meaning}\t{row.setting}")
    lines += [
        "",
        "[이번 실행 값]",
        f"seed: {args.seed}",
        f"iters: {args.iters}",
        f"hidden: {args.hidden}",
        f"log_every: {args.log_every}",
        format_lr_row("SGD", args.lr_sgd),
        format_lr_row("MomentumSGD", args.lr_momentum),
        format_lr_row("AdaGrad", args.lr_adagrad),
        format_lr_row("Adam", args.lr_adam),
        "",
        "[최종 MSE]",
    ]
    for label, mse in final_losses.items():
        lines.append(f"{label}: {mse:.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TwoLayerNet(Model):
    """dezero4.py 와 동일한 2층 MLP."""

    def __init__(self, hidden_size: int, out_size: int):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        return self.l2(y)


def snapshot_params(model: Model) -> list[np.ndarray]:
    """모든 학습 파라미터의 깊은 복사(동일 초기값 재현용).

    DeZero 의 Linear 등은 첫 forward 전까지 일부 Parameter 의 data 가 비어 있을 수 있어,
    스냅샷 전에 더미 입력으로 한 번 순전파를 돌립니다.
    """
    return [copy.deepcopy(np.asarray(p.data)) for p in model.params()]


def restore_params(model: Model, snaps: list[np.ndarray]) -> None:
    for p, arr in zip(model.params(), snaps):
        p.data[...] = arr


def build_optimizer(name: str, lr_sgd: float, lr_momentum: float, lr_adagrad: float, lr_adam: float):
    name = name.lower()
    if name == "sgd":
        return optimizers.SGD(lr_sgd)
    if name == "momentum":
        return optimizers.MomentumSGD(lr_momentum)
    if name == "adagrad":
        return optimizers.AdaGrad(lr_adagrad)
    if name == "adam":
        return optimizers.Adam(alpha=lr_adam)
    raise ValueError(name)


def train_one_optimizer(
    model: Model,
    optimizer,
    x: np.ndarray,
    y: np.ndarray,
    iters: int,
    log_every: int,
) -> list[float]:
    optimizer.setup(model)
    loss_hist: list[float] = []
    for i in range(iters):
        y_pred = model(x)
        loss = F.mean_squared_error(y, y_pred)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        if i % log_every == 0:
            loss_hist.append(float(loss.data))
    return loss_hist


def main() -> None:
    ap = argparse.ArgumentParser(description="Q1: Optimizer 비교 (PDF p.16)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--hidden", type=int, default=10, help="은닉층 크기 (dezero4 와 동일 10)")
    ap.add_argument("--log-every", type=int, default=10, help="몇 스텝마다 loss 기록할지")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--lr-sgd", type=float, default=0.2)
    ap.add_argument("--lr-momentum", type=float, default=0.2)
    ap.add_argument("--lr-adagrad", type=float, default=0.05)
    ap.add_argument("--lr-adam", type=float, default=0.02)
    args = ap.parse_args()

    print("=== Q1 Optimizer 비교 — 변수 표 ===\n")
    print_param_table()
    print("\n[이번 실행 학습률]")
    print(format_lr_row("SGD", args.lr_sgd))
    print(format_lr_row("MomentumSGD", args.lr_momentum))
    print(format_lr_row("AdaGrad", args.lr_adagrad))
    print(format_lr_row("Adam", args.lr_adam))
    print(f"  iters={args.iters}, hidden={args.hidden}, seed={args.seed}\n")

    np.random.seed(args.seed)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    # 동일 초기 가중치로 비교하기 위해 템플릿 모델에서 스냅샷
    template = TwoLayerNet(args.hidden, 1)
    template(np.zeros((1, 1), dtype=np.float64))  # 파라미터 할당·초기화
    init_snap = snapshot_params(template)

    runs = [
        ("SGD", "sgd"),
        ("MomentumSGD", "momentum"),
        ("AdaGrad", "adagrad"),
        ("Adam", "adam"),
    ]

    curves: dict[str, list[float]] = {}
    final_losses: dict[str, float] = {}

    for label, key in runs:
        model = TwoLayerNet(args.hidden, 1)
        model(np.zeros((1, 1), dtype=np.float64))
        restore_params(model, init_snap)
        opt = build_optimizer(
            key,
            lr_sgd=args.lr_sgd,
            lr_momentum=args.lr_momentum,
            lr_adagrad=args.lr_adagrad,
            lr_adam=args.lr_adam,
        )
        hist = train_one_optimizer(model, opt, x, y, args.iters, args.log_every)
        curves[label] = hist
        # 마지막 구간의 실제 loss 한 번 더 계산 (기록 간격과 무관하게 최종값)
        y_pred = model(x)
        final_losses[label] = float(F.mean_squared_error(y, y_pred).data)
        print(f"[{label:12s}] 최종 MSE = {final_losses[label]:.8f}")

    print("\n" + "─" * 40)
    print(f"{'Optimizer':<14} {'최종 MSE':>12}")
    print("─" * 40)
    for label, mse in final_losses.items():
        print(f"{label:<14} {mse:12.8f}")
    print("─" * 40)

    saved = save_summary(args, final_losses)
    print(f"\n표 요약 저장: {saved}")

    if args.no_plot:
        return

    configure_korean_font()

    steps = np.arange(0, args.iters, args.log_every)[: len(curves["SGD"])]
    plt.figure(figsize=(10, 5))
    for label, hist in curves.items():
        plt.semilogy(steps[: len(hist)], hist, label=label, lw=2)
    plt.xlabel("iteration")
    plt.ylabel("MSE loss (log scale)")
    plt.title("Q1 — Optimizer 비교 (동일 초기 가중치·동일 데이터)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
