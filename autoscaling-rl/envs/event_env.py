"""v2 — 이산사건 기반 오토스케일링 환경 (Gymnasium).

v1(공식 기반)과의 차이
----------------------
1. 요청 단위 시뮬레이션: 응답시간을 요청마다 정확히 계산 → p95 측정 가능
2. SLA 평가를 평균이 아닌 **p95(tail latency)** 로 수행 — 실무 SLA와 동일
3. 처리시간이 lognormal heavy-tail — '가끔 오는 무거운 요청'이 워커를
   오래 점유하는 현실 재현
4. 도메인 랜덤화: 에피소드마다 트래픽 패턴 + 처리시간 분포(sigma)가
   달라져 특정 환경 파라미터에 대한 과적합을 막는다

공통 MDP 구조
-------------
- 1 step = 결정 주기(윈도우) 10초. 에피소드 = 288 step
  (일일 트래픽 사이클을 압축 재생)
- State (8,): [λ 정규화, λ 변화량, 이용률, p95/SLA, 평균지연/SLA,
              c 정규화, 기동중 비율, backlog 정규화]
- Action (3): 자원 축소 / 유지 / 확대
- Reward: −w_sla·(p95 SLA 초과 비율) − w_cost·(자원 비용) − w_thrash·(행동)

두 가지 모드
------------
- EventContainerEnv : 인스턴스 수 조절 (인스턴스당 워커 슬롯 K개,
                      scale-out 시 cold start 지연)
- EventThreadPoolEnv: worker thread 수 조절 (thread 과다 시 컨텍스트
                      스위칭으로 처리시간이 길어지는 비단조 특성)
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .traffic import make_traffic
from .event_engine import FCFSQueueEngine, lognormal_services


class _EventEnvBase(gym.Env):
    """이산사건 환경의 공통 골격. 자원→워커 매핑만 서브클래스가 정의한다."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        c_min: int,
        c_max: int,
        c_init: int,
        window: float,           # 결정 주기 (초)
        episode_steps: int,
        lam_base: float,         # 기본 도착률 (req/s)
        lam_max: float,          # 정규화용 최대 도착률
        service_mean: float,     # 평균 처리시간 (초)
        sigma_range: tuple,      # lognormal sigma 범위 (도메인 랜덤화)
        sla: float,              # p95 SLA (초)
        w_sla: float,
        w_cost: float,
        w_thrash: float,
        c_base: int | None = None,   # 기본 확보 자원: 초과분에만 비용 부과
        w_under: float = 0.0,
        seed: int | None = None,
    ):
        super().__init__()
        self.c_min, self.c_max, self.c_init = c_min, c_max, c_init
        self.c_base, self.w_under = c_base, w_under
        self.window = window
        self.episode_steps = episode_steps
        self.lam_base, self.lam_max = lam_base, lam_max
        self.service_mean = service_mean
        self.sigma_range = sigma_range
        self.sla = sla
        self.w_sla, self.w_cost, self.w_thrash = w_sla, w_cost, w_thrash

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self._rng = np.random.default_rng(seed)

    # ---------------------------------------------------------- subclass API
    def _apply_action(self, delta: int):
        """delta(-1/0/+1)를 자원 변경으로 반영. 서브클래스 구현."""
        raise NotImplementedError

    def _workers(self) -> int:
        """현재 활성 워커 슬롯 수."""
        raise NotImplementedError

    def _pending_ratio(self) -> float:
        """기동 중(아직 비활성) 자원 비율. thread 모드는 0."""
        return 0.0

    def _service_scale(self) -> float:
        """처리시간 배율 (thread 모드의 컨텍스트 스위칭 오버헤드)."""
        return 1.0

    # ----------------------------------------------------------------- obs
    def _obs(self) -> np.ndarray:
        lam = self.traffic[self.t] if self.t < self.episode_steps else self.traffic[-1]
        lam_prev = self.traffic[self.t - 1] if self.t > 0 else lam
        cap = self._workers() / (self.service_mean * self._service_scale())
        util = min(lam / max(cap, 1e-6), 2.0)
        return np.array(
            [
                lam / self.lam_max,
                (lam - lam_prev) / self.lam_max,
                util,
                min(self.p95 / self.sla, 5.0),
                min(self.mean_lat / self.sla, 5.0),
                self.c / self.c_max,
                self._pending_ratio(),
                min(self.backlog / (self._workers() * self.window + 1e-6), 5.0),
            ],
            dtype=np.float32,
        )

    # ----------------------------------------------------------------- API
    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.traffic = make_traffic(self.episode_steps, self.lam_base, self._rng)
        # 도메인 랜덤화: 에피소드마다 처리시간 tail 두께가 달라짐
        self.sigma = float(self._rng.uniform(*self.sigma_range))
        self.t = 0
        self.now = 0.0
        self.c = self.c_init
        self._reset_resources()
        self.engine = FCFSQueueEngine(self._workers(), now=0.0)
        self.p95 = self.mean_lat = self.service_mean
        self.backlog = 0.0
        return self._obs(), {}

    def _reset_resources(self):
        pass

    def step(self, action: int):
        delta = int(action) - 1
        self._apply_action(delta)
        self.engine.resize(self._workers(), ready_time=self.now)

        # ---- 이번 윈도우의 요청 생성 (Poisson 도착 + lognormal 처리시간) ----
        lam = self.traffic[self.t]
        n = int(self._rng.poisson(lam * self.window))
        t0, t1 = self.now, self.now + self.window
        if n > 0:
            arrivals = np.sort(self._rng.uniform(t0, t1, size=n))
            services = lognormal_services(self._rng, n, self.service_mean, self.sigma)
            services *= self._service_scale()          # 비단조 오버헤드 (thread 모드)
            lat = self.engine.run_window(arrivals, services)
            self.p95 = float(np.percentile(lat, 95))
            self.mean_lat = float(lat.mean())
        else:
            self.p95 = self.mean_lat = self.service_mean
        self.now = t1
        self.backlog = self.engine.backlog_after(t1)

        # ---- 보상: p95 기준 SLA + 자원 비용 + 행동 비용 ----------------------
        sla_excess = max(0.0, (self.p95 - self.sla) / self.sla)
        cost = self._cost()
        under = (max(0, self.c_base - self.c) / self.c_base
                 if self.c_base is not None else 0.0)
        r = (
            -self.w_sla * min(sla_excess, 5.0)
            - self.w_cost * cost
            - self.w_under * under
            - self.w_thrash * abs(delta)
        )

        self.t += 1
        truncated = self.t >= self.episode_steps
        info = dict(lam=lam, c=self.c, latency=self.p95, mean_latency=self.mean_lat,
                    sla_violated=self.p95 > self.sla, cost=cost)
        return self._obs(), r, False, truncated, info

    def _cost(self) -> float:
        if self.c_base is not None:
            return max(0, self.c - self.c_base) / max(self.c_max - self.c_base, 1)
        return self.c / self.c_max


# ===================================================================== mode 1
class EventContainerEnv(_EventEnvBase):
    """인스턴스 수 조절 (v2). 인스턴스 1개 = 워커 슬롯 K개, cold start 존재."""

    def __init__(
        self,
        c_min: int = 1, c_max: int = 20, c_init: int = 4,
        workers_per_instance: int = 2,
        cold_start: float = 20.0,        # 초 (윈도우 2개 분량)
        window: float = 10.0,
        episode_steps: int = 288,
        lam_base: float = 30.0, lam_max: float = 160.0,
        service_mean: float = 0.1,
        sigma_range: tuple = (0.6, 1.0),
        sla: float = 0.4,                # p95 SLA (평균 SLA보다 여유)
        w_sla: float = 1.0, w_cost: float = 0.35, w_thrash: float = 0.02,
        c_base: int | None = None, w_under: float = 0.0,
        seed: int | None = None,
    ):
        super().__init__(c_min, c_max, c_init, window, episode_steps,
                         lam_base, lam_max, service_mean, sigma_range, sla,
                         w_sla, w_cost, w_thrash, c_base, w_under, seed)
        self.K = workers_per_instance
        self.cold_start = cold_start

    def _reset_resources(self):
        self.pending: list[float] = []   # 기동 완료 시각 목록

    def _apply_action(self, delta: int):
        # 기동 완료된 인스턴스를 활성화
        done = sum(1 for r in self.pending if r <= self.now)
        self.pending = [r for r in self.pending if r > self.now]
        self.c = min(self.c + done, self.c_max)

        total = self.c + len(self.pending)
        if delta == +1 and total < self.c_max:
            self.pending.append(self.now + self.cold_start)   # cold start 시작
        elif delta == -1 and self.c > self.c_min:
            self.c -= 1                                        # 축소는 즉시(drain)

    def _workers(self) -> int:
        return self.c * self.K

    def _pending_ratio(self) -> float:
        return len(self.pending) / self.c_max

    def _cost(self) -> float:
        # 기동 중 인스턴스도 비용 발생 (c_base 초과분 기준)
        c_eff = self.c + len(self.pending)
        if self.c_base is not None:
            return max(0, c_eff - self.c_base) / max(self.c_max - self.c_base, 1)
        return c_eff / self.c_max


# ===================================================================== mode 2
class EventThreadPoolEnv(_EventEnvBase):
    """worker thread 수 조절 (v2). thread 과다 시 처리시간이 길어지는 비단조."""

    def __init__(
        self,
        c_min: int = 4, c_max: int = 64, c_init: int = 16, c_step: int = 4,
        c_knee: int = 32, alpha_cs: float = 2.0,
        window: float = 5.0,
        episode_steps: int = 288,
        lam_base: float = 25.0, lam_max: float = 110.0,
        service_mean: float = 0.5,
        sigma_range: tuple = (0.6, 1.0),
        sla: float = 1.5,                # p95 SLA (처리시간 0.5s 기준 여유)
        w_sla: float = 1.0, w_cost: float = 0.30, w_thrash: float = 0.02,
        c_base: int | None = None, w_under: float = 0.0,
        seed: int | None = None,
    ):
        super().__init__(c_min, c_max, c_init, window, episode_steps,
                         lam_base, lam_max, service_mean, sigma_range, sla,
                         w_sla, w_cost, w_thrash, c_base, w_under, seed)
        self.c_step = c_step
        self.c_knee = c_knee
        self.alpha_cs = alpha_cs

    def _apply_action(self, delta: int):
        self.c = int(np.clip(self.c + delta * self.c_step, self.c_min, self.c_max))

    def _workers(self) -> int:
        return self.c

    def _service_scale(self) -> float:
        """컨텍스트 스위칭 오버헤드: knee 초과분에 비례해 처리시간 증가."""
        over = max(0, self.c - self.c_knee)
        return 1.0 + self.alpha_cs * over ** 2 / self.c_knee ** 2
