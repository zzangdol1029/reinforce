"""
통합 admission control 베이스라인 + 평가
========================================
모든 act(env, obs) -> 라우트별 ΔL index 벡터 (MultiDiscrete 행동).
독립 제어들은 공유 DB 결합을 무시 -> 피크/열화 때 집단 과부하.
통합 Oracle 은 DB 용량을 알고 우선순위로 예산 배분 -> 천장.
"""
from __future__ import annotations
import numpy as np
from envs.admission_env import L_DELTAS


def move_toward(curL, targetL):
    """라우트별로 target에 가장 가까워지는 ΔL index 벡터."""
    idx = []
    for c, t in zip(curL, targetL):
        idx.append(int(np.argmin([abs(c + d - t) for d in L_DELTAS])))
    return np.array(idx, dtype=int)


class Policy:
    name = "base"
    def reset(self, env): pass
    def act(self, env, obs): raise NotImplementedError


def _expected_lam(env):
    """오라클/그리디용 기대 도착률(노이즈 제외 결정 성분)."""
    phase = env._t / env.episode_steps
    wave = 0.5 * (1 + np.sin(2 * np.pi * phase - np.pi / 2))
    return np.array([r["base_rps"] + (r["peak_rps"] - r["base_rps"]) * wave
                     for r in env.routes])


class StaticLimit(Policy):
    """라우트별 고정 한계(정상 상태 기준 튜닝). 공유 DB 열화 때 과부하."""
    def __init__(self, factor, name):
        self.factor, self.name = factor, name
    def act(self, env, obs):
        target = np.array([max(env.min_L, r["was_cap"] * r["s"] * self.factor)
                           for r in env.routes])
        return move_toward(env.L, target)


class IndependentAIMD(Policy):
    """라우트마다 자기 지연만 보고 AIMD. 공유 DB 결합을 모름 -> 전환 지연·진동."""
    name = "Indep-AIMD"
    def act(self, env, obs):
        s = np.array([r["s"] for r in env.routes])
        sla = np.array([r["sla"] for r in env.routes])
        lat, L = env.last_latency, env.L
        target = L.copy()
        for i in range(env.N):
            if lat[i] > sla[i]:
                target[i] = max(env.min_L, L[i] * 0.6)
            elif lat[i] <= sla[i] * 0.9:
                target[i] = L[i] + 5
        return move_toward(L, target)


class GreedyIndep(Policy):
    """각 라우트가 자기 도착을 다 받으려 함(DB 무시) -> 집단 과부하."""
    name = "Greedy-Indep"
    def act(self, env, obs):
        lam = _expected_lam(env)
        s = np.array([r["s"] for r in env.routes])
        target = lam * s * 1.1
        return move_toward(env.L, target)


class Unlimited(Policy):
    name = "Unlimited"
    def act(self, env, obs):
        return move_toward(env.L, np.full(env.N, env.max_L))


class OracleJoint(Policy):
    """상한 참조: 공유 DB 용량 C_db(t)를 안다고 가정하고
    우선순위 높은 라우트부터 DB 예산을 배분 -> Σ served*cost ≈ C_db (db_util≈1).
    배포 불가(용량 비관측) -> RL이 노릴 천장."""
    name = "Oracle-Joint"
    def act(self, env, obs):
        lam = _expected_lam(env)
        cdb = env.current_db_cap()
        s = np.array([r["s"] for r in env.routes])
        was_cap = np.array([r["was_cap"] for r in env.routes])
        db_cost = np.array([r["db_cost"] for r in env.routes])
        prio = np.array([r["priority"] for r in env.routes])

        budget = cdb * 0.97          # 약간 보수적으로 (포화 회피)
        served = np.zeros(env.N)
        for i in np.argsort(-prio):  # 우선순위 내림차순
            cap_i = min(lam[i], was_cap[i])
            give = min(cap_i, budget / db_cost[i]) if budget > 0 else 0.0
            served[i] = max(0.0, give)
            budget -= served[i] * db_cost[i]
        target_L = np.maximum(env.min_L, served * s)
        return move_toward(env.L, target_L)


BASELINES = [
    StaticLimit(0.9, "Static-High"),
    StaticLimit(0.5, "Static-Mid"),
    GreedyIndep(),
    Unlimited(),
    IndependentAIMD(),
    OracleJoint(),
]


def evaluate_policy(policy, env, n_episodes=15, base_seed=2000):
    agg = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        policy.reset(env)
        done, R = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(policy.act(env, obs)); R += r
            done = term or trunc
        m = env.metrics(); m["reward"] = R; agg.append(m)
    return {k: float(np.mean([d[k] for d in agg])) for k in agg[0]}
