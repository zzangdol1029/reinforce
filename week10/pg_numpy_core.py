"""
Policy / Value 순수 NumPy 구현 (CartPole 실습용)
- 교재 공식과 동일: grad(log π) softmax 경로에서는 ∇(-G log π) ∝ G·(π - one_hot)
- DeZero가 환경에서 동작하면 개별 스크립트에서 직교 구현 가능; 기본 실행은 여기 의존
"""
from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=-1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=-1, keepdims=True)


class SoftmaxPolicy:
    """2층 MLP 정책 (softmax)."""

    def __init__(
        self,
        state_dim: int,
        action_size: int,
        hidden: int = 128,
        lr: float = 2e-4,
    ):
        self.lr = lr
        self.action_size = action_size

        scale1 = np.sqrt(2.0 / state_dim)
        scale2 = np.sqrt(2.0 / hidden)
        rng = np.random.default_rng(seed=42)
        self.W1 = (rng.standard_normal((state_dim, hidden)) * scale1).astype(np.float32)
        self.b1 = np.zeros((1, hidden), dtype=np.float32)
        self.W2 = (rng.standard_normal((hidden, action_size)) * scale2).astype(np.float32)
        self.b2 = np.zeros((1, action_size), dtype=np.float32)

        self.zero_grad_buffers()

    def zero_grad_buffers(self) -> None:
        self._dW1 = np.zeros_like(self.W1)
        self._db1 = np.zeros_like(self.b1)
        self._dW2 = np.zeros_like(self.W2)
        self._db2 = np.zeros_like(self.b2)

    def predict(self, s: np.ndarray) -> tuple[np.ndarray, dict]:
        """s: (batch, state_dim). returns probs, cache."""
        z1 = s @ self.W1 + self.b1
        h = np.maximum(z1, 0.0)
        logits = h @ self.W2 + self.b2
        probs = softmax(logits)
        cache = {"s": s, "z1": z1, "h": h, "logits": logits, "probs": probs}
        return probs, cache

    def accumulate_pg_grad(self, cache: dict, action_idx: int, coeff: float) -> None:
        """
        단일 샘플 (batch=1): L += -coeff * log π(a|s)
        => ∂L/∂logits_j = coeff * (π_j - δ_{ja})
        """
        s = cache["s"]
        z1 = cache["z1"]
        h = cache["h"]
        probs = cache["probs"]

        oh = np.zeros_like(probs, dtype=np.float32)
        oh[0, action_idx] = 1.0
        grad_logits = (coeff * (probs - oh)).astype(np.float32)

        grad_h = grad_logits @ self.W2.T
        dz1 = np.where(z1 > 0, grad_h, 0.0).astype(np.float32)

        self._dW2 += h.T @ grad_logits
        self._db2 += grad_logits.sum(axis=0, keepdims=True)
        self._dW1 += s.T @ dz1
        self._db1 += dz1.sum(axis=0, keepdims=True)

    def step(self, grad_clip_norm: float | None = None) -> None:
        if grad_clip_norm is not None:
            total = np.sqrt(sum(np.sum(np.square(g)) for g in (self._dW1, self._db1, self._dW2, self._db2)))
            if total > grad_clip_norm and total > 0:
                factor = grad_clip_norm / total
                self._dW1 *= factor
                self._db1 *= factor
                self._dW2 *= factor
                self._db2 *= factor

        self.W1 -= self.lr * self._dW1
        self.b1 -= self.lr * self._db1
        self.W2 -= self.lr * self._dW2
        self.b2 -= self.lr * self._db2
        self.zero_grad_buffers()


class ValueNet:
    """V(s): 2층 MLP, 출력 1."""

    def __init__(
        self,
        state_dim: int,
        hidden: int = 128,
        lr: float = 5e-4,
    ):
        self.lr = lr
        rng = np.random.default_rng(seed=7)
        s1 = np.sqrt(2.0 / state_dim)
        s2 = np.sqrt(2.0 / hidden)
        self.W1 = (rng.standard_normal((state_dim, hidden)) * s1).astype(np.float32)
        self.b1 = np.zeros((1, hidden), dtype=np.float32)
        self.W2 = (rng.standard_normal((hidden, 1)) * s2).astype(np.float32)
        self.b2 = np.zeros((1, 1), dtype=np.float32)

        self.zero_grad_buffers()

    def zero_grad_buffers(self) -> None:
        self._dW1 = np.zeros_like(self.W1)
        self._db1 = np.zeros_like(self.b1)
        self._dW2 = np.zeros_like(self.W2)
        self._db2 = np.zeros_like(self.b2)

    def forward(self, s: np.ndarray) -> tuple[np.ndarray, dict]:
        z1 = s @ self.W1 + self.b1
        h = np.maximum(z1, 0.0)
        v = h @ self.W2 + self.b2  # (batch, 1)
        return v, {"s": s, "z1": z1, "h": h, "v": v}

    def accumulate_value_grad(self, cache: dict, dLdv: np.ndarray) -> None:
        """
        손실 L = ½(V - tgt)² (tgt는 역전파 막음) ⇒ ∂L/∂V = V − tgt 형태일 때,
        호출측에서 dLdv = (V − tgt) 로 넣으면 된다 (shape (batch, 1)).
        """
        s = cache["s"]
        z1 = cache["z1"]
        h = cache["h"]
        grad_out = dLdv.astype(np.float32)

        grad_h = grad_out @ self.W2.T
        dz1 = np.where(z1 > 0, grad_h, 0.0).astype(np.float32)

        self._dW2 += h.T @ grad_out
        self._db2 += grad_out.sum(axis=0, keepdims=True)
        self._dW1 += s.T @ dz1
        self._db1 += dz1.sum(axis=0, keepdims=True)

    def step(self) -> None:
        self.W1 -= self.lr * self._dW1
        self.b1 -= self.lr * self._db1
        self.W2 -= self.lr * self._dW2
        self.b2 -= self.lr * self._db2
        self.zero_grad_buffers()
