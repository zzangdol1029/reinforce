"""에피소드용 트래픽(요청 도착률) 생성기.

하루 24시간 패턴을 모사한다:
  λ(t) = base · (일일 사이클: 오전/저녁 피크) + 버스트 스파이크 + 노이즈

에피소드마다 위상·피크 크기·버스트 위치를 무작위로 바꿔서
에이전트가 특정 패턴을 외우지 못하고 "부하 변화에 반응"하도록 한다.
"""
from __future__ import annotations

import numpy as np


def make_traffic(n_steps: int, lam_base: float, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n_steps)

    # 일일 사이클: 오전 피크 + 저녁 피크 (에피소드마다 위상/크기 랜덤)
    phase1 = rng.uniform(-0.5, 0.5)
    phase2 = rng.uniform(-0.5, 0.5)
    amp1 = rng.uniform(0.6, 1.2)
    amp2 = rng.uniform(0.8, 1.6)
    daily = (
        1.0
        + amp1 * np.exp(-((t - (0.55 * np.pi + phase1)) ** 2) / 0.18)   # 오전 피크
        + amp2 * np.exp(-((t - (1.35 * np.pi + phase2)) ** 2) / 0.25)   # 저녁 피크
    )

    lam = lam_base * daily

    # 버스트: 1~3회, 각 5~15 step 동안 +50~150%
    for _ in range(rng.integers(1, 4)):
        start = rng.integers(0, n_steps - 16)
        width = rng.integers(5, 16)
        boost = rng.uniform(0.5, 1.5)
        lam[start : start + width] *= 1.0 + boost

    # 관측 노이즈
    lam *= rng.normal(1.0, 0.04, size=n_steps).clip(0.85, 1.15)
    return lam.clip(min=1.0)
