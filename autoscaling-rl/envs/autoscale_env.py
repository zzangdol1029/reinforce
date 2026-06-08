"""Stage 1 — 컨테이너(서비스 인스턴스) 오토스케일링 시뮬레이터 (Gymnasium 환경)

MSA 환경에서 트래픽 변화에 따라 서비스 인스턴스(Undertow 기반 JVM 프로세스)
수를 조절하는 문제. 큐잉 이론(M/M/c 근사)으로 응답시간을 계산한다.

큐 모델
-------
활성 인스턴스 c개, 인스턴스당 처리율 μ (req/s)일 때 시스템 용량 = c·μ.
이용률 ρ = λ / (c·μ).

  대기시간 W ≈ 1/μ                    (순수 처리시간)
            + ρ / (c·μ·(1-ρ))         (M/M/c 정상상태 대기 근사)
            + backlog / (c·μ)         (과부하로 누적된 잔여 작업 소진 시간)

과부하(ρ>1)면 처리 못 한 요청이 backlog로 누적되어 다음 step의 지연을
키운다 — "한 번 밀리면 한동안 느려지는" 실제 서버 거동을 재현.

cold start: scale-out된 인스턴스는 `cold_start_steps` step 후에야 활성화
(JVM 기동 시간). 단, 비용은 기동 중에도 발생한다.

MDP 정의
--------
- State  (7,): [λ 정규화, λ 변화량, 이용률 ρ, 지연/SLA, c 정규화,
               기동중 인스턴스 정규화, backlog 정규화]
- Action (3) : 0=scale-in(-1), 1=유지, 2=scale-out(+1)
- Reward     : -w_sla·SLA위반정도 - w_cost·(인스턴스 비용) - w_thrash·(행동 비용)
- Episode    : 288 step (24h, 5분 간격)
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .traffic import make_traffic


class ContainerAutoscaleEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        c_min: int = 1,
        c_max: int = 20,
        c_init: int = 4,
        mu: float = 20.0,
        cold_start_steps: int = 2,
        lam_base: float = 80.0,
        lam_max: float = 400.0,
        sla: float = 0.2,
        episode_steps: int = 288,
        w_sla: float = 1.0,
        w_cost: float = 0.35,
        w_thrash: float = 0.02,
        c_base: int | None = None,   # 기본 확보 자원: 초과분에만 비용 부과
        w_under: float = 0.0,        # 기본값 미만 페널티
        seed: int | None = None,
    ):
        super().__init__()
        self.c_min, self.c_max, self.c_init = c_min, c_max, c_init
        self.c_base, self.w_under = c_base, w_under
        self.mu = mu
        self.cold_start_steps = cold_start_steps
        self.lam_base, self.lam_max = lam_base, lam_max
        self.sla = sla
        self.episode_steps = episode_steps
        self.w_sla, self.w_cost, self.w_thrash = w_sla, w_cost, w_thrash

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # -1 / 0 / +1
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ utils
    def _obs(self) -> np.ndarray:
        lam = self.traffic[self.t]
        lam_prev = self.traffic[self.t - 1] if self.t > 0 else lam
        rho = lam / (self.c * self.mu)
        return np.array(
            [
                lam / self.lam_max,
                (lam - lam_prev) / self.lam_max,
                min(rho, 2.0),
                min(self.latency / self.sla, 5.0),
                self.c / self.c_max,
                len(self.pending) / self.c_max,
                min(self.backlog / (self.c_max * self.mu), 5.0),
            ],
            dtype=np.float32,
        )

    def _latency(self, lam: float) -> float:
        """현재 활성 인스턴스 수와 backlog로 평균 응답시간(초)을 계산."""
        cap = self.c * self.mu
        rho = lam / cap
        w = 1.0 / self.mu                                   # 처리시간
        if rho < 1.0:
            w += rho / (cap * max(1.0 - rho, 0.02))         # 정상상태 대기
        else:
            w += 1.0 / (cap * 0.02)                          # 포화: 대기 폭증
        w += self.backlog / cap                              # 누적 잔여작업 소진
        return w

    # ------------------------------------------------------------------ API
    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.traffic = make_traffic(self.episode_steps, self.lam_base, self._rng)
        self.t = 0
        self.c = self.c_init          # 활성 인스턴스 수
        self.pending: list[int] = []  # 기동 중 인스턴스의 남은 step 수
        self.backlog = 0.0            # 처리 못 한 누적 요청 수
        self.latency = 1.0 / self.mu
        return self._obs(), {}

    def step(self, action: int):
        delta = int(action) - 1  # 0,1,2 -> -1,0,+1

        # --- 스케일링 적용 -------------------------------------------------
        total = self.c + len(self.pending)
        if delta == +1 and total < self.c_max:
            self.pending.append(self.cold_start_steps)      # cold start 시작
        elif delta == -1 and self.c > self.c_min:
            self.c -= 1                                     # 축소는 즉시

        # cold start 카운트다운 → 완료된 인스턴스 활성화
        self.pending = [p - 1 for p in self.pending]
        done_boot = sum(1 for p in self.pending if p <= 0)
        self.pending = [p for p in self.pending if p > 0]
        self.c = min(self.c + done_boot, self.c_max)

        # --- 큐 동역학 -----------------------------------------------------
        lam = self.traffic[self.t]
        cap = self.c * self.mu
        # 이번 step(단위시간) 동안 못 받아낸 만큼 backlog에 적립/소진
        self.backlog = max(0.0, self.backlog + (lam - cap))
        self.latency = self._latency(lam)

        # --- 보상 -----------------------------------------------------------
        # 비용: c_base(기본 확보분)까지는 0, 초과분에만 비례 부과.
        # 기본값보다 줄이면 w_under 페널티 (안전 여유 훼손 방지).
        sla_excess = max(0.0, (self.latency - self.sla) / self.sla)
        c_eff = self.c + len(self.pending)
        if self.c_base is not None:
            cost = max(0, c_eff - self.c_base) / max(self.c_max - self.c_base, 1)
            under = max(0, self.c_base - self.c) / self.c_base
        else:
            cost, under = c_eff / self.c_max, 0.0
        r = (
            -self.w_sla * min(sla_excess, 5.0)
            - self.w_cost * cost
            - self.w_under * under
            - self.w_thrash * abs(delta)
        )

        self.t += 1
        terminated = False
        truncated = self.t >= self.episode_steps
        info = dict(lam=lam, c=self.c, latency=self.latency,
                    sla_violated=self.latency > self.sla, cost=cost)
        return self._obs() if not truncated else self._obs_last(), r, terminated, truncated, info

    def _obs_last(self):
        self.t = self.episode_steps - 1
        o = self._obs()
        self.t = self.episode_steps
        return o
