"""Double DQN (DeZero 직접 구현).

구성 요소
---------
- Q 네트워크 / Target 네트워크 : MLP (state_dim -> 64 -> 64 -> n_actions)
- Experience Replay            : 무작위 샘플링으로 시계열 상관 제거
- ε-greedy                     : 선형 감쇠 탐험
- Double DQN                   : 행동 선택은 online, 평가는 target 네트워크
                                 (Q값 과대추정 완화)
"""
from __future__ import annotations

import copy
import pickle
import random
from collections import deque

import numpy as np

from .common import MLP, F, clip_grads
from dezero import optimizers


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, *transition):
        self.buf.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return (s.astype(np.float32), a.astype(np.int64), r.astype(np.float32),
                s2.astype(np.float32), d.astype(np.float32))

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(self, state_dim: int, n_actions: int, total_steps: int, cfg: dict,
                 seed: int = 0):
        np.random.seed(seed)
        random.seed(seed)
        self.n_actions = n_actions
        self.cfg = cfg

        self.q = MLP(n_actions, cfg["hidden"], activation="relu")
        self.q(np.zeros((1, state_dim), dtype=np.float32))      # lazy init
        self.target = copy.deepcopy(self.q)
        self.opt = optimizers.Adam(cfg["lr"]).setup(self.q)
        self.buffer = ReplayBuffer(cfg["buffer_size"])

        self.step_count = 0
        self.eps_decay_steps = max(1, int(total_steps * cfg["eps_decay_frac"]))

    # ------------------------------------------------------------------
    @property
    def epsilon(self) -> float:
        c = self.cfg
        frac = min(1.0, self.step_count / self.eps_decay_steps)
        return c["eps_start"] + frac * (c["eps_end"] - c["eps_start"])

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        q = self.q(state[None].astype(np.float32)).data
        return int(q.argmax())

    def remember(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, float(done))

    # ------------------------------------------------------------------
    def learn(self) -> float | None:
        c = self.cfg
        self.step_count += 1
        if len(self.buffer) < c["learn_start"]:
            return None

        s, a, r, s2, d = self.buffer.sample(c["batch_size"])
        B = len(a)

        # Double DQN 타깃: online net이 고른 행동을 target net으로 평가
        q_next_online = self.q(s2).data
        a_next = q_next_online.argmax(axis=1)
        q_next_target = self.target(s2).data
        y = r + c["gamma"] * (1.0 - d) * q_next_target[np.arange(B), a_next]

        q_all = self.q(s)
        q_sa = F.get_item(q_all, (np.arange(B), a))
        loss = F.sum((q_sa - y.astype(np.float32)) ** 2) / B

        self.q.cleargrads()
        loss.backward()
        clip_grads([self.q], 10.0)
        self.opt.update()

        if self.step_count % c["target_update"] == 0:
            self.target = copy.deepcopy(self.q)
        return float(loss.data)

    # ------------------------------------------------------------------ I/O
    def save(self, path: str):
        self.q.save_weights(path)

    def load(self, path: str):
        self.q.load_weights(path)
        self.target = copy.deepcopy(self.q)

    def save_state(self, path: str):
        """분할 학습 재개용 전체 상태 저장 (버퍼 포함)."""
        with open(path, "wb") as f:
            pickle.dump({"buffer": self.buf_as_list(), "step_count": self.step_count}, f)

    def load_state(self, path: str):
        with open(path, "rb") as f:
            st = pickle.load(f)
        self.buffer.buf = deque(st["buffer"], maxlen=self.cfg["buffer_size"])
        self.step_count = st["step_count"]

    def buf_as_list(self):
        return list(self.buffer.buf)
