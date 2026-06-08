"""룰 기반 베이스라인.

1. ThresholdAutoscaler — Kubernetes HPA 방식의 임계값 정책:
   이용률 ρ > high 면 scale-out, ρ < low 면 scale-in, 사이면 유지.
   연속 스케일링 진동을 막기 위해 cooldown(연속 행동 금지 step)을 둔다.
   실제 운영에서 가장 널리 쓰이는 방식이라 RL의 직접 비교 대상.

2. StaticPolicy — 고정 c 정책. 평가 트래픽에 대해 가장 좋은 고정값을
   sweep으로 찾은 "오프라인 최적 고정 설정"과 비교하면
   동적 제어의 가치를 보여줄 수 있다.
"""
from __future__ import annotations

import numpy as np


class ThresholdAutoscaler:
    """HPA식 임계값 정책 (이용률 기반)."""

    def __init__(self, high: float = 0.7, low: float = 0.3, cooldown: int = 3):
        self.high, self.low, self.cooldown = high, low, cooldown
        self.cool = 0

    def reset(self):
        self.cool = 0

    def act(self, obs: np.ndarray, greedy: bool = True) -> int:
        rho = float(obs[2])  # 두 환경 모두 obs[2] = 이용률
        if self.cool > 0:
            self.cool -= 1
            return 1
        if rho > self.high:
            self.cool = self.cooldown
            return 2
        if rho < self.low:
            self.cool = self.cooldown
            return 0
        return 1


class StaticPolicy:
    """아무것도 하지 않는 고정 c 정책 (환경 reset 시 c_init으로 설정해 사용)."""

    def reset(self):
        pass

    def act(self, obs: np.ndarray, greedy: bool = True) -> int:
        return 1
