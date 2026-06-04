"""
Spring Boot MSA 게이트웨이 부하분산 시뮬레이터 (Gymnasium 환경)
================================================================

게이트웨이가 매 요청마다 N개의 백엔드 서비스 인스턴스 중 하나를 골라
요청을 라우팅한다. 인스턴스마다 처리 성능(service_rate)이 다르고,
요청 도착은 Poisson 과정을 따른다. 목표는 평균 응답시간과 SLA 위반을
최소화하도록 라우팅 정책을 학습하는 것.

큐 모델
-------
각 인스턴스를 단일 서버 FIFO 큐로 본다(M/G/1 유사). 인스턴스 i는
`free_at[i]` = 현재 큐에 쌓인 모든 작업을 끝내는 시각을 유지한다.
시각 t에 도착한 요청을 인스턴스 j로 라우팅하면:
    start  = max(free_at[j], t)
    finish = start + demand / service_rate[j]
    latency = finish - t            # 대기 + 처리 (정확값)
    free_at[j] = finish
이 방식으로 각 요청의 응답시간을 정확히 계산한다.

MDP 정의
--------
- State  : 인스턴스별 [정규화 backlog(예상 대기시간), 정규화 큐길이,
           최근 EWMA 지연, 정규화 처리율] + 전역 [정규화 총 backlog]
           -> shape (N*4 + 1,)
- Action : (이산) Discrete(N)        - 요청을 보낼 인스턴스 인덱스. DQN/PPO용
           (연속) Box(-1,1, (N,))    - 인스턴스별 가중치, argmax 라우팅. SAC용
- Reward : - latency
           - lambda_balance * (인스턴스 backlog 표준편차)
           - sla_penalty * (latency > SLA 임계값)

한 step = 하나의 요청 도착/라우팅 결정.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class LoadBalancerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_instances: int = 4,
        arrival_rate: float = 9.0,        # 평균 도착률 (req/s), Poisson
        service_rates=None,                # 인스턴스별 처리율 (work/s). None이면 이질적 기본값
        mean_demand: float = 1.0,          # 요청 1건 평균 작업량
        sla_threshold: float = 1.5,        # 응답시간 SLA 임계값 (s)
        episode_len: int = 500,            # 에피소드당 요청 수
        lambda_balance: float = 0.1,       # 부하균형 보상 가중치
        sla_penalty: float = 2.0,          # SLA 위반 페널티
        continuous: bool = False,          # True면 SAC용 Box 행동공간
        seed: int | None = None,
    ):
        super().__init__()
        self.n = n_instances
        self.arrival_rate = arrival_rate
        if service_rates is None:
            # 이질적 인스턴스: 느린 노드 ~ 빠른 노드. 정적 알고리즘이 약한 환경.
            service_rates = np.linspace(1.5, 3.5, self.n)
        self.service_rates = np.asarray(service_rates, dtype=np.float64)
        self.mean_demand = mean_demand
        self.sla_threshold = sla_threshold
        self.episode_len = episode_len
        self.lambda_balance = lambda_balance
        self.sla_penalty = sla_penalty
        self.continuous = continuous

        self._rng = np.random.default_rng(seed)

        if continuous:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(self.n)

        obs_dim = self.n * 4 + 1
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._max_rate = float(self.service_rates.max())

    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.free_at = np.zeros(self.n)          # 각 인스턴스가 현재 큐를 비우는 시각
        self.finish_times = [deque() for _ in range(self.n)]  # 진행중 요청들의 완료 시각
        self.ewma_latency = np.zeros(self.n)
        self.t = 0.0
        self.step_count = 0

        self.completed_latencies: list[float] = []
        self.sla_violations = 0
        self.total_completed = 0

        self._next_demand = self._sample_demand()
        return self._get_obs(), {}

    # ------------------------------------------------------------------ #
    def step(self, action):
        if self.continuous:
            j = int(np.argmax(np.asarray(action).ravel()))
        else:
            j = int(action)
        j = max(0, min(self.n - 1, j))

        # 1) 시간 진행: 다음 요청 도착까지의 간격
        dt = self._rng.exponential(1.0 / self.arrival_rate)
        self.t += dt
        self._collect_completed()  # 현재 시각 이전에 끝난 요청 정산

        demand = self._next_demand

        # 2) 라우팅된 요청의 정확한 응답시간 계산
        start = max(self.free_at[j], self.t)
        service_time = demand / self.service_rates[j]
        finish = start + service_time
        latency = finish - self.t            # 대기 + 처리

        self.free_at[j] = finish
        self.finish_times[j].append((finish, latency))
        self.ewma_latency[j] = 0.7 * self.ewma_latency[j] + 0.3 * latency

        # 3) 보상
        backlog = np.maximum(0.0, self.free_at - self.t)  # 인스턴스별 예상 대기시간
        balance_pen = float(np.std(backlog))
        sla_violate = latency > self.sla_threshold
        reward = -latency - self.lambda_balance * balance_pen
        if sla_violate:
            reward -= self.sla_penalty
            self.sla_violations += 1

        # 4) 다음 요청
        self._next_demand = self._sample_demand()
        self.step_count += 1
        truncated = self.step_count >= self.episode_len
        info = {"routed_to": j, "latency": latency, "sla_violate": sla_violate}
        return self._get_obs(), float(reward), False, truncated, info

    # ------------------------------------------------------------------ #
    def _collect_completed(self):
        """현재 시각 t 이전에 완료된 요청들을 통계로 옮긴다."""
        for i in range(self.n):
            dq = self.finish_times[i]
            while dq and dq[0][0] <= self.t:
                _, lat = dq.popleft()
                self.completed_latencies.append(lat)
                self.total_completed += 1

    def _queue_len(self) -> np.ndarray:
        return np.array([len(dq) for dq in self.finish_times], dtype=np.float64)

    # --- 베이스라인 정책이 참조하는 관측 헬퍼 ---
    def predicted_wait(self) -> np.ndarray:
        """인스턴스별 예상 대기시간(backlog). LeastWork가 사용."""
        return np.maximum(0.0, self.free_at - self.t)

    def inflight(self) -> np.ndarray:
        """현재 진행/대기 중인 요청 수. LeastConnection이 사용."""
        return np.array([sum(1 for f, _ in dq if f > self.t) for dq in self.finish_times],
                        dtype=np.float64)

    def _sample_demand(self) -> float:
        return float(self._rng.exponential(self.mean_demand))

    def _get_obs(self) -> np.ndarray:
        backlog = np.maximum(0.0, self.free_at - self.t)
        qlen = self._queue_len()
        max_b = max(1e-6, float(backlog.max()))
        max_q = max(1.0, float(qlen.max()))
        b = backlog / max_b
        q = qlen / max_q
        lat = np.tanh(self.ewma_latency / max(self.sla_threshold, 1e-6))
        rate = self.service_rates / self._max_rate
        glob = np.array([np.tanh(backlog.sum() / max(self.sla_threshold, 1e-6))])
        return np.concatenate([b, q, lat, rate, glob]).astype(np.float32)

    # ------------------------------------------------------------------ #
    def metrics(self) -> dict:
        """에피소드 종료 후 호출. 핵심 성능 지표 반환."""
        # 남은 진행중 요청도 정산해 공정 비교
        rem = [lat for dq in self.finish_times for (_, lat) in dq]
        lats = self.completed_latencies + rem
        lat = np.asarray(lats) if lats else np.array([0.0])
        backlog = np.maximum(0.0, self.free_at - self.t)
        return {
            "mean_latency": float(np.mean(lat)),
            "p95_latency": float(np.percentile(lat, 95)),
            "throughput": float(self.total_completed),
            "sla_violation_rate": float(self.sla_violations / max(1, self.step_count)),
            "load_imbalance": float(np.std(backlog)),
        }
