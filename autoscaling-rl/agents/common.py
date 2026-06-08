"""DeZero 공통 유틸.

교재(밑바닥부터 시작하는 딥러닝)의 DeZero 라이브러리를 사용한다.
DeZero는 구버전 numpy alias(np.int 등)를 사용하므로 최신 numpy에서
동작하도록 import 전에 호환 패치를 적용한다.
"""
from __future__ import annotations

import numpy as np

# ---- numpy 1.24+ 호환 패치 (dezero가 제거된 alias를 참조) ------------------
if not hasattr(np, "int"):
    np.int = int          # noqa: NPY001
if not hasattr(np, "float"):
    np.float = np.float64  # noqa: NPY001
if not hasattr(np, "bool"):
    np.bool = np.bool_     # noqa: NPY001

from dezero import Model            # noqa: E402
import dezero.layers as L           # noqa: E402
import dezero.functions as F        # noqa: E402


class MLP(Model):
    """2-hidden-layer MLP: in -> hidden -> hidden -> out."""

    def __init__(self, out_dim: int, hidden: int = 64, activation: str = "relu"):
        super().__init__()
        self.l1 = L.Linear(hidden)
        self.l2 = L.Linear(hidden)
        self.l3 = L.Linear(out_dim)
        self.act = F.relu if activation == "relu" else F.tanh

    def forward(self, x):
        h = self.act(self.l1(x))
        h = self.act(self.l2(h))
        return self.l3(h)


def clip_grads(models, max_norm: float):
    """global norm 기준 gradient clipping (DeZero에는 내장이 없어 직접 구현)."""
    params = [p for m in models for p in m.params() if p.grad is not None]
    total = float(np.sqrt(sum(float((p.grad.data ** 2).sum()) for p in params)))
    rate = max_norm / (total + 1e-6)
    if rate < 1.0:
        for p in params:
            p.grad.data *= rate
