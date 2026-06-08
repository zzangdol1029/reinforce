"""
통합 적응형 동시성 제어 환경 (다중 라우트 + 공유 DB)
=====================================================
게이트웨이가 라우트별 동시 허용 수 L_i 를 동시에 조절한다.

2단 병목:
  (1) WAS 단계  : 라우트 i 는 자기 WAS 용량으로만 처리 (독립)
  (2) DB 단계   : 모든 라우트가 하나의 공유 DB 를 사용 -> 결합

핵심 물리:
  was_tp_i = min(arrival_i, L_i / s_i, was_cap_i)         # 라우트별 WAS 처리량 후보
  D        = Σ was_tp_i * db_cost_i                        # 공유 DB 총수요(cost 가중)
  db_util  = D / C_db(t)                                   # C_db 는 비노출·시변
  db_util>1 이면 DB 포화 -> 처리량 throttle + 모든 라우트 지연 폭증

상태(부분관측): 라우트별 6개 지표 × N. 공유 DB 용량 C_db 는 비노출.
행동: ΔL 벡터. action_mode = 'multi'(MultiDiscrete) | 'box'(연속) | 'flat'(Discrete 7^N)
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

L_DELTAS = np.array([-16, -4, -1, 0, 1, 4, 16])


class AdmissionEnv(gym.Env):
    def __init__(self, routes, db, s_db_base=0.03, episode_steps=300,
                 min_L=2, max_L=200, w_reject=0.5, w_sla=10.0, action_mode="multi"):
        super().__init__()
        self.routes = [dict(r) for r in routes]
        self.N = len(self.routes)
        self.db = dict(db)
        self.s_db_base = s_db_base
        self.episode_steps = episode_steps
        self.min_L, self.max_L = min_L, max_L
        self.w_reject, self.w_sla = w_reject, w_sla
        self.action_mode = action_mode

        self.K = len(L_DELTAS)
        # 관측: 라우트별 6개 [L정규화, 지연/sla, 지연추세, util, 거절율, 도착정규화]
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(self.N * 6,), dtype=np.float32)
        if action_mode == "multi":
            self.action_space = spaces.MultiDiscrete([self.K] * self.N)
        elif action_mode == "box":
            self.action_space = spaces.Box(-1.0, 1.0, shape=(self.N,), dtype=np.float32)
        elif action_mode == "flat":
            self.action_space = spaces.Discrete(self.K ** self.N)
        else:
            raise ValueError(action_mode)
        self._rng = np.random.default_rng()

    # ── 행동 디코딩 → 라우트별 ΔL ────────────────────────────────────────
    def _decode(self, action):
        if self.action_mode == "multi":
            idx = np.asarray(action, dtype=int).reshape(self.N)
            return L_DELTAS[idx]
        if self.action_mode == "box":
            a = np.clip(np.asarray(action, dtype=float).reshape(self.N), -1, 1)
            return np.round(a * 16).astype(int)
        # flat: base-K 디코딩
        a = int(action); out = []
        for _ in range(self.N):
            out.append(a % self.K); a //= self.K
        return L_DELTAS[np.array(out[::-1])]

    # ── 공유 DB 용량 C_db(t): 정상 ↔ 열화 마르코프 ──────────────────────
    def _db_capacity(self):
        return self._cdb

    def _step_db(self):
        if self._degraded:
            if self._rng.random() < self.db["recover_prob"]:
                self._degraded = False
                self._cdb = self.db["cap_healthy"]
        else:
            if self._rng.random() < self.db["degrade_prob"]:
                self._degraded = True
                self._cdb = self._rng.uniform(self.db["cap_deg_min"], self.db["cap_deg_max"])

    # ── 라우트별 도착률(일간 변동 + 노이즈) ─────────────────────────────
    def _arrivals(self):
        phase = self._t / self.episode_steps
        wave = 0.5 * (1 + np.sin(2 * np.pi * phase - np.pi / 2))  # 0→1→0
        out = np.zeros(self.N)
        for i, r in enumerate(self.routes):
            base, peak = r["base_rps"], r["peak_rps"]
            lam = base + (peak - base) * wave
            lam *= self._rng.uniform(0.85, 1.15)
            out[i] = max(1.0, lam)
        return out

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._degraded = False
        self._cdb = self.db["cap_healthy"]
        self.L = np.array([max(self.min_L, int(r["was_cap"] * r["s"] * 0.6))
                           for r in self.routes], dtype=float)
        self._prev_lat = np.array([r["s"] + self.s_db_base for r in self.routes])
        self.last_latency = self._prev_lat.copy()
        self._agg = []
        return self._obs(self._arrivals(), self.last_latency,
                         np.zeros(self.N), np.zeros(self.N)), {}

    # ── 한 스텝: ΔL 적용 → 2단 병목 계산 → 보상 ────────────────────────
    def step(self, action):
        dL = self._decode(action)
        self.L = np.clip(self.L + dL, self.min_L, self.max_L).astype(float)
        self._step_db()
        cdb = self._db_capacity()
        lam = self._arrivals()

        s = np.array([r["s"] for r in self.routes])
        was_cap = np.array([r["was_cap"] for r in self.routes])
        db_cost = np.array([r["db_cost"] for r in self.routes])
        sla = np.array([r["sla"] for r in self.routes])
        prio = np.array([r["priority"] for r in self.routes])

        # (1) WAS 처리량 후보: 동시성 한계 / 도착 / WAS 용량
        was_tp = np.minimum.reduce([lam, self.L / s, was_cap])

        # (2) 공유 DB 수요와 포화
        D = float(np.sum(was_tp * db_cost))
        db_util = D / max(cdb, 1e-6)

        if db_util <= 1.0:
            served = was_tp.copy()
            db_overflow = 0.0
        else:
            served = was_tp / db_util          # DB가 비례적으로 throttle
            db_overflow = db_util - 1.0

        # 지연: WAS 자체 큐잉(L 과다) + 공유 DB 포화(전 라우트 공통 인플레)
        was_util = (self.L / s) / was_cap
        lat = s + self.s_db_base
        lat = np.where(was_util > 1, lat * was_util, lat)        # 라우트별 WAS 큐잉
        if db_util > 1.0:
            lat = lat * (db_util ** 2)                           # 공유 DB 포화 → 큐잉 폭증(전 라우트 공통)

        rejected = np.maximum(0.0, lam - served)
        good = np.where(lat <= sla, served, served * 0.2)
        reward = float(np.sum(prio * good)
                       - self.w_reject * np.sum(rejected)
                       - self.w_sla * np.sum(np.maximum(0.0, lat - sla)))

        trend = lat - self._prev_lat
        self._prev_lat = lat.copy()
        self.last_latency = lat.copy()
        util = served / np.maximum(lam, 1e-6)
        rej_rate = rejected / np.maximum(lam, 1e-6)

        self._agg.append(dict(reward=reward,
                              sla_violation_rate=float(np.mean(lat > sla)),
                              mean_latency=float(np.mean(lat / sla)),
                              throughput=float(np.sum(served)),
                              rejected=float(np.sum(rejected)),
                              db_util=db_util,
                              prio_good=float(np.sum(prio * good))))
        self._t += 1
        term = self._t >= self.episode_steps
        info = dict(L=self.L.copy(), cdb=cdb, db_util=db_util,
                    latency=lat.copy(), served=served.copy(), lam=lam.copy())
        obs = self._obs(lam, lat, util, rej_rate)
        return obs, reward, term, False, info

    def _obs(self, lam, lat, util, rej_rate):
        s = np.array([r["s"] for r in self.routes])
        sla = np.array([r["sla"] for r in self.routes])
        peak = np.array([r["peak_rps"] for r in self.routes])
        trend = lat - self._prev_lat
        feats = []
        for i in range(self.N):
            feats += [self.L[i] / self.max_L,
                      min(lat[i] / sla[i], 4.0),
                      np.clip(trend[i] / sla[i], -2, 2),
                      util[i],
                      rej_rate[i],
                      lam[i] / peak[i]]
        return np.array(feats, dtype=np.float32)

    def current_db_cap(self):
        return self._cdb

    def metrics(self):
        ks = ["sla_violation_rate", "mean_latency", "throughput", "rejected", "db_util", "prio_good"]
        return {k: float(np.mean([m[k] for m in self._agg])) for k in ks}
