"""이산사건(event-driven) FCFS 큐 시뮬레이션 엔진.

v1(autoscale_env/threadpool_env)은 큐잉 공식으로 '평균' 지연을 근사하지만,
v2는 요청 하나하나를 워커에 배정해 응답시간을 정확히 계산한다.
이 덕분에 평균이 아닌 p95(tail latency) 기반 SLA 평가가 가능하다.

모델
----
- 워커 W개가 병렬로 요청을 처리하는 FCFS 시스템 (M/G/c)
- `free_at[i]` = 워커 i가 현재 잡힌 모든 작업을 끝내는 시각
- 시각 t에 도착한 요청 → 가장 먼저 비는 워커에 배정:
      start  = max(free_at[w], t)
      finish = start + service_time
      latency = finish - t          # 대기 + 처리 (정확값)
- 워커 수 변경:
    * 증가: 새 워커는 ready_time(cold start 반영) 이후부터 사용 가능
    * 감소: 가장 한가한(빨리 비는) 워커부터 제거 — 처리 중 요청은
      끊지 않고 새 요청만 안 받는 'drain' 방식과 동일한 효과

구현 노트
---------
- heapq 최소힙으로 '가장 먼저 비는 워커'를 O(log W)에 찾는다.
- 한 윈도우(결정 주기) 안의 요청 수가 수백 개 수준이 되도록
  시간 스케일을 잡으면 학습 전체가 CPU로 수 분 내에 끝난다.
"""
from __future__ import annotations

import heapq

import numpy as np


class FCFSQueueEngine:
    def __init__(self, n_workers: int, now: float = 0.0):
        # 힙 원소 = 워커가 비는 시각(free_at)
        self.free_at: list[float] = [now] * n_workers
        heapq.heapify(self.free_at)

    # ------------------------------------------------------------------
    @property
    def n_workers(self) -> int:
        return len(self.free_at)

    def backlog_after(self, t: float) -> float:
        """시각 t 이후에 남아 있는 총 잔여 작업 시간(초). 큐 적체의 지표."""
        return float(sum(max(0.0, f - t) for f in self.free_at))

    # ------------------------------------------------------------------
    def resize(self, target: int, ready_time: float):
        """워커 수를 target으로 조정.

        증가분은 ready_time부터 사용 가능(cold start), 감소는 한가한
        워커부터 제거(drain). 처리 중인 요청은 절대 중단되지 않는다.
        """
        cur = len(self.free_at)
        if target > cur:
            for _ in range(target - cur):
                heapq.heappush(self.free_at, ready_time)
        elif target < cur:
            # 빨리 비는(=한가한) 워커부터 제거
            self.free_at.sort()
            self.free_at = self.free_at[cur - target:]
            heapq.heapify(self.free_at)

    # ------------------------------------------------------------------
    def run_window(self, arrivals: np.ndarray, services: np.ndarray) -> np.ndarray:
        """한 윈도우 분량의 요청을 처리하고 요청별 응답시간 배열을 반환.

        arrivals : 도착 시각 (오름차순 정렬)
        services : 요청별 처리 시간 (arrivals와 같은 길이)
        """
        latencies = np.empty(len(arrivals), dtype=np.float64)
        free = self.free_at
        for i in range(len(arrivals)):
            at = arrivals[i]
            earliest = heapq.heappop(free)
            start = at if at > earliest else earliest
            finish = start + services[i]
            heapq.heappush(free, finish)
            latencies[i] = finish - at
        return latencies


def lognormal_services(rng: np.random.Generator, n: int,
                       mean: float, sigma: float) -> np.ndarray:
    """평균이 `mean`이 되도록 보정한 lognormal 처리시간 표본.

    sigma가 클수록 heavy-tail(가끔 매우 무거운 요청)이 강해진다.
    E[lognormal(μ,σ)] = exp(μ + σ²/2) 이므로 μ = ln(mean) − σ²/2 로 보정.
    """
    mu = np.log(mean) - 0.5 * sigma ** 2
    return rng.lognormal(mu, sigma, size=n)
