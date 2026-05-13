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

import numpy as np
import matplotlib.pyplot as plt

from mpl_korean_font import configure_korean_font

from dezero import Model
from dezero import optimizers
import dezero.layers as L
import dezero.functions as F


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
