"""
베이스라인 라우팅 정책 + 공통 평가 함수
=======================================
정적/휴리스틱 부하분산 알고리즘들. RL 정책과 동일한 환경에서 비교한다.
"""
from __future__ import annotations

import numpy as np
from envs.load_balancer_env import LoadBalancerEnv


class BaselinePolicy:
    name = "base"

    def reset(self, n):
        pass

    def act(self, env: LoadBalancerEnv):
        raise NotImplementedError


class RoundRobin(BaselinePolicy):
    name = "RoundRobin"

    def reset(self, n):
        self.n = n
        self.i = 0

    def act(self, env):
        j = self.i % self.n
        self.i += 1
        return j


class WeightedRoundRobin(BaselinePolicy):
    """처리율(capacity)에 비례해 가중 라운드로빈."""
    name = "WeightedRR"

    def reset(self, n):
        self.n = n
        self.i = 0
        self.seq = None

    def act(self, env):
        if self.seq is None:
            rates = env.service_rates
            weights = np.maximum(1, np.round(rates / rates.min()).astype(int))
            self.seq = np.repeat(np.arange(self.n), weights)
        j = int(self.seq[self.i % len(self.seq)])
        self.i += 1
        return j


class LeastConnection(BaselinePolicy):
    """현재 큐 길이가 가장 짧은 인스턴스로."""
    name = "LeastConn"

    def reset(self, n):
        self.n = n

    def act(self, env):
        return int(np.argmin(env.inflight()))


class LeastWork(BaselinePolicy):
    """예상 대기시간(잔여작업/처리율)이 가장 작은 인스턴스로. (강력한 휴리스틱)"""
    name = "LeastWork"

    def reset(self, n):
        self.n = n

    def act(self, env):
        # 예상 대기시간 + 이 요청의 처리시간이 가장 작은 인스턴스
        wait = env.predicted_wait() + env._next_demand / env.service_rates
        return int(np.argmin(wait))


class RandomPolicy(BaselinePolicy):
    name = "Random"

    def reset(self, n):
        self.n = n
        self.rng = np.random.default_rng(0)

    def act(self, env):
        return int(self.rng.integers(self.n))


BASELINES = [RoundRobin(), WeightedRoundRobin(), LeastConnection(), LeastWork(), RandomPolicy()]


def evaluate_baseline(policy: BaselinePolicy, env_kwargs: dict, n_episodes: int = 20, base_seed: int = 1000):
    """베이스라인 정책을 n_episodes 동안 평가하고 평균 지표 반환."""
    agg = []
    for ep in range(n_episodes):
        env = LoadBalancerEnv(**env_kwargs)
        env.reset(seed=base_seed + ep)
        policy.reset(env.n)
        done = False
        while not done:
            a = policy.act(env)
            _, _, term, trunc, _ = env.step(a)
            done = term or trunc
        agg.append(env.metrics())
    return _mean_metrics(agg)


def _mean_metrics(agg: list[dict]) -> dict:
    keys = agg[0].keys()
    return {k: float(np.mean([m[k] for m in agg])) for k in keys}
