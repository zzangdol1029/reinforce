"""Stage 2 — Undertow worker thread pool 동적 튜닝 시뮬레이터 (Gymnasium 환경)

운영 환경(Undertow 기반 Spring Boot)에 맞춘 변형. 인스턴스 수 대신
JVM 내부 worker thread 수 c를 조절한다. 핵심 차이는 **비단조 용량 곡선**:

  capacity(c) = (c · μ_t) / (1 + α · max(0, c - c_knee)² / c_knee²)

- c ≤ c_knee : thread를 늘릴수록 용량 증가 (IO-bound라 병렬화 이득)
- c > c_knee : 컨텍스트 스위칭/CPU 경합으로 thread당 효율 하락
               → 어느 지점부터는 늘릴수록 오히려 용량이 줄어든다

즉 "많을수록 좋다"가 아니라 **트래픽에 따라 움직이는 최적점**이 존재하고,
에이전트는 그 최적점을 추적해야 한다. thread는 기동 지연이 없어
cold start는 없지만, 과도한 변경은 페널티(w_thrash)로 억제한다.

MDP 정의
--------
- State  (6,): [λ 정규화, λ 변화량, 이용률 ρ, 지연/SLA, c 정규화, backlog 정규화]
- Action (3) : 0=thread -step, 1=유지, 2=thread +step
- Reward     : -w_sla·SLA위반정도 - w_cost·(thread 비용) - w_thrash·(행동 비용)
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .traffic import make_traffic


class ThreadPoolEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        c_min: int = 4,
        c_max: int = 64,
        c_init: int = 16,
        c_step: int = 4,
        mu_thread: float = 25.0,
        c_knee: int = 32,
        alpha_cs: float = 2.0,
        lam_base: float = 250.0,
        lam_max: float = 900.0,
        sla: float = 0.2,
        episode_steps: int = 288,
        w_sla: float = 1.0,
        w_cost: float = 0.30,
        w_thrash: float = 0.02,
        c_base: int | None = None,   # 기본 thread 수: 초과분에만 비용 부과
        w_under: float = 0.0,
        seed: int | None = None,
    ):
        super().__init__()
        self.c_base, self.w_under = c_base, w_under
        self.c_min, self.c_max, self.c_init, self.c_step = c_min, c_max, c_init, c_step
        self.mu_thread, self.c_knee, self.alpha_cs = mu_thread, c_knee, alpha_cs
        self.lam_base, self.lam_max = lam_base, lam_max
        self.sla = sla
        self.episode_steps = episode_steps
        self.w_sla, self.w_cost, self.w_thrash = w_sla, w_cost, w_thrash

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ model
    def capacity(self, c: int) -> float:
        """thread 수 c에서의 시스템 처리 용량 (req/s). c_knee 초과 시 비선형 하락."""
        overhead = 1.0 + self.alpha_cs * max(0, c - self.c_knee) ** 2 / self.c_knee**2
        return c * self.mu_thread / overhead

    def _latency(self, lam: float) -> float:
        cap = self.capacity(self.c)
        rho = lam / cap
        w = 1.0 / self.mu_thread
        if rho < 1.0:
            w += rho / (cap * max(1.0 - rho, 0.02))
        else:
            w += 1.0 / (cap * 0.02)
        w += self.backlog / cap
        return w

    def _obs(self) -> np.ndarray:
        lam = self.traffic[self.t]
        lam_prev = self.traffic[self.t - 1] if self.t > 0 else lam
        rho = lam / self.capacity(self.c)
        return np.array(
            [
                lam / self.lam_max,
                (lam - lam_prev) / self.lam_max,
                min(rho, 2.0),
                min(self.latency / self.sla, 5.0),
                self.c / self.c_max,
                min(self.backlog / self.capacity(self.c_max), 5.0),
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ API
    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.traffic = make_traffic(self.episode_steps, self.lam_base, self._rng)
        self.t = 0
        self.c = self.c_init
        self.backlog = 0.0
        self.latency = 1.0 / self.mu_thread
        return self._obs(), {}

    def step(self, action: int):
        delta = (int(action) - 1) * self.c_step
        self.c = int(np.clip(self.c + delta, self.c_min, self.c_max))

        lam = self.traffic[self.t]
        cap = self.capacity(self.c)
        self.backlog = max(0.0, self.backlog + (lam - cap))
        self.latency = self._latency(lam)

        sla_excess = max(0.0, (self.latency - self.sla) / self.sla)
        if self.c_base is not None:
            cost = max(0, self.c - self.c_base) / max(self.c_max - self.c_base, 1)
            under = max(0, self.c_base - self.c) / self.c_base
        else:
            cost, under = self.c / self.c_max, 0.0
        r = (
            -self.w_sla * min(sla_excess, 5.0)
            - self.w_cost * cost
            - self.w_under * under
            - self.w_thrash * (abs(delta) / self.c_step)
        )

        self.t += 1
        truncated = self.t >= self.episode_steps
        info = dict(lam=lam, c=self.c, latency=self.latency,
                    sla_violated=self.latency > self.sla, cost=cost)
        return self._obs() if not truncated else self._obs_last(), r, False, truncated, info

    def _obs_last(self):
        self.t = self.episode_steps - 1
        o = self._obs()
        self.t = self.episode_steps
        return o
