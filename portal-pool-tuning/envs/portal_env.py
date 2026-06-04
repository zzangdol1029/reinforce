"""
포털 WAS 자원(워커 스레드 + DB 커넥션 풀) 동시 튜닝 시뮬레이터
================================================================
Gymnasium 표준 인터페이스를 구현한 강화학습 환경.

■ 문제 배경
  실제 포털 WAS(Undertow)는 워커 스레드 수(W)와 HikariCP DB 커넥션풀 크기(D)를
  정적으로 설정한다. 부하와 요청 믹스가 변해도 두 값이 고정되어 있어 비최적 상태가
  지속된다. 이 환경은 그 두 값을 매 제어 스텝마다 동적으로 조절하는 에이전트를 학습
  시키기 위한 시뮬레이터다.

■ 결합 병목 모델 (핵심 개념)
  요청 한 건의 생애주기:
    [워커 스레드 점유] ──────────────────────────── 전체 체류시간(mean_s)
                       └── [DB 커넥션 점유] ──── DB 처리시간(mean_s × db_frac)

  처리능력 = min(CPU한계, 워커한계, DB한계)
    - CPU한계   = cores / cpu_처리시간
    - 워커한계  = W / 유효_체류시간
    - DB한계    = D / 유효_DB처리시간
  세 한계 중 가장 작은 쪽이 병목. 한쪽만 늘려도 다른 쪽이 막히면 효과 없음.

■ MDP 정의
  State  (10차원): 정규화된 운영 메트릭 (도착률, 추세, CPU/워커/DB 이용률 등)
  Action (이산25): ΔW 5단계 × ΔD 5단계  /  (연속2): SAC용 [-1,1]²
  Reward         : -SLA패널티 - 드롭패널티 - 자원비용 - 변동패널티
  Episode        : 200 제어 스텝 (부하·요청믹스 시계열 1회 순환)
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── 이산 행동 단계 정의 ────────────────────────────────────────────────────
# max_workers / max_db 대비 비율로 변화량을 결정하는 5단계 레벨
# 예) max_workers=2560 기준:
#   -0.08 → -205개 (큰 폭 감소)
#   -0.02 →  -51개 (소폭 감소)
#    0.00 →    0개 (유지)
#   +0.02 →  +51개 (소폭 증가)
#   +0.08 → +205개 (큰 폭 증가)
# 5×5 = 25가지 조합 → Discrete(25) 행동공간
W_LEVELS = (-0.08, -0.02, 0.0, 0.02, 0.08)   # 워커 변화 비율
D_LEVELS = (-0.08, -0.02, 0.0, 0.02, 0.08)   # DB풀 변화 비율


class PortalPoolEnv(gym.Env):
    """
    포털 WAS 워커·DB풀 동시 튜닝 환경.

    에이전트는 매 스텝(제어 주기)마다 (ΔW, ΔD) 행동을 선택해
    워커 수와 DB풀 크기를 조절한다. 환경은 부하와 요청 믹스가 변하는
    시계열을 시뮬레이션하며, 에이전트는 관측값만으로 최적 자원량을 추론해야 한다.

    관측 가능: 도착률, 이용률, 지연, 현재 자원 비율
    관측 불가: 실제 체류시간(mean_s), DB 비중(db_frac) → 부분관측(POMDP)
    """
    metadata = {"render_modes": []}

    def __init__(self, portal: dict, continuous: bool = False, seed: int | None = None):
        """
        Args:
            portal    : config.PORTALS 딕셔너리 중 하나 (포털별 사양)
            continuous: True  → SAC용 연속 행동공간 Box(-1,1,(2,))
                        False → DQN/PPO용 이산 행동공간 Discrete(25)
            seed      : 재현성 시드 (None이면 매 실행마다 다른 시나리오)
        """
        super().__init__()
        p = portal

        # ── 포털 하드웨어 사양 ────────────────────────────────────────────
        self.name = p["name"]
        self.cores = p["cores"]                              # 물리 CPU 코어 수
        self.max_w = p["max_workers"]                        # 워커 최대값
        self.min_w = p["min_workers"]                        # 워커 최솟값
        self.max_d = p["max_db"]                             # DB풀 최대값
        self.min_d = p["min_db"]                             # DB풀 최솟값
        self.current_w = p["current_workers"]                # 현재 운영 워커 수
        self.current_d = p["current_db"]                     # 현재 운영 DB풀 크기
        self.db_shared_max = p["db_shared_max"]              # 공유 DB 최대 커넥션

        # ── 부하 시뮬레이션 파라미터 ──────────────────────────────────────
        self.base_rps = p["base_rps"]                        # 기저(야간) RPS
        self.peak_rps = p["peak_rps"]                        # 피크 RPS
        self.ms_min = p["mean_s_min"]                        # 최소 체류시간(초)
        self.ms_max = p["mean_s_max"]                        # 최대 체류시간(초)
        self.dbf_min = p["dbfrac_min"]                       # DB 비중 최솟값
        self.dbf_max = p["dbfrac_max"]                       # DB 비중 최댓값

        # ── 보상 함수 파라미터 ────────────────────────────────────────────
        self.sla_threshold = p.get("sla_threshold", 0.5)    # SLA 지연 임계값(초)
        self.mem_cost = p["mem_cost"]                        # 워커 1개당 비용
        self.db_cost = p["db_cost"]                          # DB 커넥션 1개당 비용
        self.sla_weight = p.get("sla_weight", 4.0)          # SLA 위반 가중치
        self.drop_penalty = p.get("drop_penalty", 10.0)     # 드롭 패널티
        self.thrash = p.get("thrash", 0.01)                  # 자원 변동 패널티
        self.overhead = p.get("overhead", 0.0008)            # 컨텍스트 스위칭 계수
        self.db_contention = p.get("db_contention", 0.6)    # 공유 DB 경합 민감도

        # ── 에피소드·시계열 설정 ──────────────────────────────────────────
        self.episode_steps = p.get("episode_steps", 200)    # 에피소드 길이
        self.load_cycles = p.get("load_cycles", 4.0)        # 부하 주기 수
        self.ms_cycles = p.get("ms_cycles", 2.5)            # 체류시간 주기 수
        self.dbf_cycles = p.get("dbf_cycles", 3.0)          # DB 비중 주기 수

        self.continuous = continuous
        self._rng = np.random.default_rng(seed)

        # ── Gymnasium 행동/관측 공간 ──────────────────────────────────────
        if continuous:
            # SAC용: 2차원 연속 행동 [-1, 1] → (ΔW 비율, ΔD 비율)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        else:
            # DQN/PPO용: 0~24 정수 인덱스 → _decode()로 (ΔW, ΔD)로 변환
            self.action_space = spaces.Discrete(len(W_LEVELS) * len(D_LEVELS))

        # 관측 공간: 10차원 실수 벡터 (정규화된 메트릭)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

    # ── 에피소드 초기화 ────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        """
        에피소드 시작 시 호출. 부하 시계열을 새로 생성하고 자원을 현재 운영값으로 초기화.

        매 에피소드마다 랜덤 위상의 시계열이 생성되어 다양한 부하 패턴 학습 가능.
        자원(W, D)은 current_workers/current_db에서 시작해 에이전트가 조절.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # 에피소드 전체 부하 시계열 사전 생성 (랜덤 위상으로 매번 다름)
        self.lam_t, self.ms_t, self.dbf_t = self._build_traces()
        self.t = 0  # 현재 스텝 인덱스

        # 자원을 현재 운영 설정에서 시작 (에이전트가 여기서부터 최적화)
        self.W = self.current_w
        self.D = self.current_d

        # 추세 계산을 위한 이전 스텝값 초기화
        self.prev_lam = self.lam_t[0]
        self.last_latency = self.prev_latency = self.ms_min
        self.last_drop = 0.0
        self.last_cpu_util = self.last_worker_util = self.last_db_util = 0.0

        # 에피소드 통계 누적 변수 초기화 (metrics() 에서 사용)
        self.sum_lat = 0.0
        self.sum_drop = 0.0
        self.sum_sla = 0
        self.sum_W = 0.0
        self.sum_D = 0.0

        return self._get_obs(), {}

    # ── 핵심: 3중 병목 성능 모델 ──────────────────────────────────────────
    def perf(self, W, D, lam, mean_s, db_frac):
        """
        주어진 자원(W, D)과 부하 조건에서 응답 지연과 드롭율을 계산.

        ■ 공유 DB 과부하 모델
          DB풀(D)이 커질수록 공유 DB에 동시 접속이 늘어 경합 발생.
          db_t_eff = db_t × (1 + db_contention × D / db_shared_max)
          → D → db_shared_max 에 가까워질수록 DB 처리시간이 선형 증가

        ■ 3중 병목
          cpu_cap    = cores / cpu_t          (CPU가 처리할 수 있는 최대 RPS)
          worker_cap = W / mean_s_eff         (워커풀이 처리할 수 있는 최대 RPS)
          db_cap     = D / db_t_eff           (DB풀이 처리할 수 있는 최대 RPS)
          capacity   = min(세 한계)

        ■ 대기열 지연 (M/M/1 근사)
          rho = lam / capacity (이용률)
          rho < 1: lat = mean_s_eff / (1 - rho)  → 대기 포함 응답시간
          rho ≥ 1: 과부하 → 지연 폭발 + 드롭 발생

        Args:
            W       : 현재 워커 스레드 수
            D       : 현재 DB 커넥션풀 크기
            lam     : 초당 도착 요청 수 (RPS)
            mean_s  : 평균 요청 체류시간 (초)
            db_frac : 체류시간 중 DB 처리 비중 (0~1)

        Returns:
            (latency, drop_rate, capacity, mean_s_eff, db_t_eff)
        """
        cpu_t = (1 - db_frac) * mean_s                       # 순수 CPU 처리 시간
        db_t = db_frac * mean_s                               # 이상적 DB 처리 시간
        # 공유 DB 경합으로 인한 실제 DB 처리 시간 증가
        db_t_eff = db_t * (1 + self.db_contention * D / self.db_shared_max)
        mean_s_eff = cpu_t + db_t_eff                         # 실제 평균 체류시간

        cpu_cap = self.cores / max(cpu_t, 1e-6)              # CPU 처리 한계
        worker_cap = W / max(mean_s_eff, 1e-6)               # 워커풀 처리 한계
        db_cap = D / max(db_t_eff, 1e-6)                     # DB풀 처리 한계
        capacity = min(cpu_cap, worker_cap, db_cap)           # 실제 병목 처리량

        rho = lam / capacity                                   # 이용률

        # 코어 초과 워커로 인한 컨텍스트 스위칭 오버헤드 (초과분 제곱에 비례)
        overhead = self.overhead * (max(0, W - self.cores) / max(self.cores, 1)) ** 2

        if rho >= 1.0:
            # 과부하 상태: 처리 못한 요청은 드롭됨
            drop = (lam - capacity) / lam
            return self.sla_threshold * 3 + overhead, drop, capacity, mean_s_eff, db_t_eff

        # 정상 상태: M/M/1 대기열 공식으로 지연 계산
        latency = mean_s_eff / (1 - min(rho, 0.99)) + overhead
        return latency, 0.0, capacity, mean_s_eff, db_t_eff

    def step_reward(self, W, D, lam, mean_s, db_frac):
        """
        특정 (W, D) 설정에서의 즉각 보상 계산.
        Oracle 베이스라인의 격자 탐색에서 최적점을 찾기 위해 사용.
        (thrash 패널티 없음 — 단일 스텝 평가용)
        """
        lat, drop, _, _, _ = self.perf(W, D, lam, mean_s, db_frac)
        return (
            -self.sla_weight * max(0.0, lat - self.sla_threshold)
            - self.drop_penalty * drop
            - self.mem_cost * W
            - self.db_cost * D
        )

    def optimal_grid(self, lam, mean_s, db_frac, nW=24, nD=20):
        """
        (W, D) 격자 전체를 탐색해 현재 부하에서 보상이 최대인 자원량 반환.

        Oracle 베이스라인이 '미래를 알고 있을 때의 최상' 참조값을 구하기 위해 호출.
        NumPy 벡터화로 nW×nD = 480개 조합을 한 번에 계산해 효율적.

        Returns:
            (optimal_W, optimal_D) 정수 튜플
        """
        Ws = np.linspace(self.min_w, self.max_w, nW)
        Ds = np.linspace(self.min_d, self.max_d, nD)
        Wg, Dg = np.meshgrid(Ws, Ds, indexing="ij")          # (nW, nD) 격자

        cpu_t = (1 - db_frac) * mean_s
        db_t = db_frac * mean_s
        db_t_eff = db_t * (1 + self.db_contention * Dg / self.db_shared_max)
        mean_s_eff = cpu_t + db_t_eff
        cap = np.minimum(
            np.minimum(self.cores / max(cpu_t, 1e-6), Wg / mean_s_eff),
            Dg / db_t_eff
        )
        rho = lam / cap
        ov = self.overhead * (np.maximum(0, Wg - self.cores) / max(self.cores, 1)) ** 2
        lat = np.where(
            rho >= 1,
            self.sla_threshold * 3 + ov,
            mean_s_eff / (1 - np.minimum(rho, 0.99)) + ov
        )
        drop = np.where(rho >= 1, (lam - cap) / lam, 0.0)
        rew = (
            -self.sla_weight * np.maximum(0, lat - self.sla_threshold)
            - self.drop_penalty * drop
            - self.mem_cost * Wg
            - self.db_cost * Dg
        )
        # argmax로 최적 격자점 인덱스 찾기
        i = np.unravel_index(np.argmax(rew), rew.shape)
        return int(round(float(Wg[i]))), int(round(float(Dg[i])))

    # ── 행동 디코딩 ────────────────────────────────────────────────────────
    def _decode(self, action):
        """
        에이전트의 행동을 실제 (ΔW, ΔD) 변화량으로 변환.

        이산 모드 (DQN/PPO):
          action 0~24 → divmod(action, 5) → (wi, di)
          → W_LEVELS[wi] × max_w, D_LEVELS[di] × max_d
          예) action=13 → wi=2, di=3 → ΔW=0, ΔD=+102 (max_d=1280 기준)

        연속 모드 (SAC):
          action[0] ∈ [-1,1] → ΔW = action[0] × 0.08 × max_w
          action[1] ∈ [-1,1] → ΔD = action[1] × 0.08 × max_d
          → 이산 모드의 최대 변화량(±8%)과 동일한 범위
        """
        if self.continuous:
            a = np.asarray(action).ravel()
            dW = int(round(float(np.clip(a[0], -1, 1)) * 0.08 * self.max_w))
            dD = int(round(float(np.clip(a[1], -1, 1)) * 0.08 * self.max_d))
            return dW, dD
        idx = int(action)
        wi, di = divmod(idx, len(D_LEVELS))                   # 25 → (w인덱스, d인덱스)
        return int(round(W_LEVELS[wi] * self.max_w)), int(round(D_LEVELS[di] * self.max_d))

    # ── 환경 스텝 실행 ─────────────────────────────────────────────────────
    def step(self, action):
        """
        에이전트 행동 실행 → 자원 업데이트 → 성능 계산 → 보상 반환.

        ■ 보상 구성 (모두 음수 또는 0, 높을수록 좋음)
          R = -sla_weight × max(0, lat - sla_threshold)   SLA 위반 패널티
            - drop_penalty × drop_rate                    드롭 패널티
            - mem_cost × W                                워커 유지 비용
            - db_cost × D                                 DB풀 유지 비용
            - thrash × (|ΔW|/max_w + |ΔD|/max_d)         자원 변동 패널티

          SLA 패널티: 지연이 목표값 초과 시 초과분에 비례해 차감
          드롭 패널티: 처리 못한 요청 비율에 비례, 가중치가 높아 강력한 억제
          자원 비용: 불필요한 과다 할당 억제 (연속 학습 목적)
          변동 패널티: 매 스텝 큰 폭으로 바꾸는 것 억제 (운영 안정성)

        Returns:
            obs, reward, terminated(=False), truncated, info
            * terminated는 항상 False (자연 종료 없음)
            * truncated는 episode_steps 도달 시 True
        """
        dW, dD = self._decode(action)

        # 자원량 업데이트 (min~max 범위로 클리핑)
        newW = int(np.clip(self.W + dW, self.min_w, self.max_w))
        newD = int(np.clip(self.D + dD, self.min_d, self.max_d))
        appliedW = newW - self.W                               # 실제 적용된 변화량
        appliedD = newD - self.D
        self.W, self.D = newW, newD

        # 현재 스텝의 부하 파라미터
        lam = float(self.lam_t[self.t])                       # 초당 요청 수
        mean_s = float(self.ms_t[self.t])                     # 평균 체류시간
        dbf = float(self.dbf_t[self.t])                       # DB 처리 비중

        # 성능 계산 (3중 병목 모델)
        lat, drop, cap, ms_eff, db_t_eff = self.perf(self.W, self.D, lam, mean_s, dbf)
        served = cap if drop > 0 else lam                     # 실제 처리된 RPS

        # 자원 이용률 계산 (다음 관측값으로 사용)
        cpu_t = (1 - dbf) * mean_s
        self.last_cpu_util = min(1.0, served * cpu_t / self.cores)
        self.last_worker_util = min(1.0, served * ms_eff / max(self.W, 1))
        self.last_db_util = min(1.0, served * db_t_eff / max(self.D, 1))

        # 보상 계산
        reward = (
            -self.sla_weight * max(0.0, lat - self.sla_threshold)    # SLA 위반
            - self.drop_penalty * drop                                # 드롭
            - self.mem_cost * self.W                                  # 워커 비용
            - self.db_cost * self.D                                   # DB 비용
            - self.thrash * (abs(appliedW) / self.max_w              # 변동 패널티
                             + abs(appliedD) / self.max_d)
        )

        # 에피소드 통계 누적
        self.sum_lat += lat
        self.sum_drop += drop
        self.sum_W += self.W
        self.sum_D += self.D
        if lat > self.sla_threshold or drop > 0:
            self.sum_sla += 1                                  # SLA 위반 횟수

        # 추세 계산을 위한 이전값 보존
        self.prev_lam = lam
        self.prev_latency = self.last_latency
        self.last_latency = lat
        self.last_drop = drop

        self.t += 1
        truncated = (self.t >= self.episode_steps)             # 에피소드 종료 판정

        # 디버깅/평가용 상세 정보
        info = {
            "lam": lam, "mean_s": mean_s, "db_frac": dbf,
            "W": self.W, "D": self.D,
            "latency": lat, "drop": drop,
            "cpu_util": self.last_cpu_util,
            "worker_util": self.last_worker_util,
            "db_util": self.last_db_util,
            "capacity": cap,
        }
        return self._get_obs(), float(reward), False, truncated, info

    # ── 부하 시계열 생성 ───────────────────────────────────────────────────
    def _build_traces(self):
        """
        에피소드 전체 부하 시계열을 미리 생성.

        ■ 설계 의도
          - 부하(lam), 체류시간(ms), DB 비중(dbf) 세 변수가 독립적으로 변동
          - 랜덤 위상(ph1, ph2)으로 에피소드마다 다른 패턴 → 일반화 강제
          - 에이전트는 lam만 관측, ms·dbf는 숨겨진 상태 (부분관측 POMDP)
          - ms와 dbf의 변동이 '어느 자원이 병목인가'를 결정 → 에이전트는
            워커/DB 이용률 관측으로 간접 추론해야 함

        ■ 생성 방식
          lam : 사인파 일간 패턴 (base_rps ~ peak_rps) + 가우시안 노이즈 4%
          ms  : 독립 사인파 (ms_min ~ ms_max) + 노이즈
          dbf : 독립 사인파 (dbf_min ~ dbf_max) + 노이즈
          * 에피소드보다 14스텝 길게 생성 → future() 룩어헤드 버퍼

        Returns:
            (lam_trace, ms_trace, dbf_trace) 각각 길이 episode_steps+14 배열
        """
        T = self.episode_steps + 14                            # 룩어헤드 버퍼 포함
        t = np.arange(T)

        # 부하: 하루 주기 사인파 (새벽 낮음 → 낮 피크) + 노이즈
        daily = 0.5 * (1 + np.sin(
            2 * np.pi * t / (self.episode_steps / self.load_cycles) - np.pi / 2
        ))
        lam = (self.base_rps + (self.peak_rps - self.base_rps) * daily) \
              * (1 + self._rng.normal(0, 0.04, size=T))
        lam = np.maximum(self.base_rps * 0.4, lam)            # 최솟값 보장

        # 체류시간·DB 비중: 부하와 독립적인 사인파 (랜덤 시작 위상)
        ph1 = self._rng.uniform(0, 2 * np.pi)                 # 체류시간 시작 위상
        ph2 = self._rng.uniform(0, 2 * np.pi)                 # DB 비중 시작 위상

        ms = self.ms_min + (self.ms_max - self.ms_min) * 0.5 * (
            1 + np.sin(2 * np.pi * t / (self.episode_steps / self.ms_cycles) + ph1)
        )
        ms = np.clip(
            ms + self._rng.normal(0, 0.02 * (self.ms_max - self.ms_min), size=T),
            self.ms_min, self.ms_max
        )

        dbf = self.dbf_min + (self.dbf_max - self.dbf_min) * 0.5 * (
            1 + np.sin(2 * np.pi * t / (self.episode_steps / self.dbf_cycles) + ph2)
        )
        dbf = np.clip(
            dbf + self._rng.normal(0, 0.02, size=T),
            self.dbf_min, self.dbf_max
        )
        return lam, ms, dbf

    def future(self, h):
        """
        h 스텝 후의 부하 파라미터 반환.
        Oracle 베이스라인이 선행(anticipatory) 조절을 위해 호출.
        에피소드 끝을 넘으면 마지막 값으로 클램프.
        """
        i = min(self.t + h, len(self.lam_t) - 1)
        return float(self.lam_t[i]), float(self.ms_t[i]), float(self.dbf_t[i])

    # ── 관측값 생성 ────────────────────────────────────────────────────────
    def _get_obs(self):
        """
        10차원 관측 벡터 생성. 신경망 입력으로 사용되므로 모두 [-1~1] 근방으로 정규화.

        [0] lam / peak_rps           : 현재 부하 비율 (0=야간, 1=피크)
        [1] Δlam / peak_rps          : 부하 증가 추세 (양수=증가 중)
        [2] CPU 이용률                : 0~1  (1이면 CPU 병목)
        [3] 워커풀 이용률              : 0~1  (1이면 워커 병목 → 워커 늘려야)
        [4] DB풀 이용률                : 0~1  (1이면 DB 병목  → DB풀 늘려야)
        [5] tanh(lat/sla_threshold)   : 현재 지연 수준 (포화함수로 스케일)
        [6] 위 값의 변화량             : 지연 추세 (양수=악화 중)
        [7] W / max_workers           : 현재 워커 비율 (에이전트가 자기 상태 파악)
        [8] D / max_db                : 현재 DB풀 비율
        [9] 직전 드롭율               : 0~1 (0 초과 = 요청 유실 발생 중)

        * mean_s, db_frac는 노출하지 않음 → 에이전트가 이용률로 추론해야 함 (부분관측)
        """
        cur_lam = (self.lam_t[self.t] if self.t < len(self.lam_t) else self.lam_t[-1])
        return np.array([
            cur_lam / self.peak_rps,                                    # [0] 부하 비율
            (cur_lam - self.prev_lam) / self.peak_rps,                 # [1] 부하 추세
            self.last_cpu_util,                                         # [2] CPU 이용률
            self.last_worker_util,                                      # [3] 워커풀 이용률
            self.last_db_util,                                          # [4] DB풀 이용률
            np.tanh(self.last_latency / self.sla_threshold),           # [5] 지연 수준
            (np.tanh(self.last_latency / self.sla_threshold)           # [6] 지연 추세
             - np.tanh(self.prev_latency / self.sla_threshold)),
            self.W / self.max_w,                                        # [7] 워커 비율
            self.D / self.max_d,                                        # [8] DB풀 비율
            self.last_drop,                                             # [9] 드롭율
        ], dtype=np.float32)

    # ── 에피소드 요약 통계 ─────────────────────────────────────────────────
    def metrics(self):
        """
        에피소드 전체의 평균 성능 지표 반환.
        evaluate.py에서 정책 간 비교에 사용.

        Returns:
            dict with keys:
              sla_violation_rate : SLA 위반 스텝 비율 (낮을수록 좋음)
              mean_latency       : 평균 응답 지연(초) (낮을수록 좋음)
              mean_drop_rate     : 평균 드롭율 (낮을수록 좋음)
              mean_workers       : 평균 워커 수 (낮을수록 메모리 절약)
              mean_db_conns      : 평균 DB 커넥션 수 (낮을수록 공유 DB 부담 적음)
        """
        n = max(1, self.t)
        return {
            "sla_violation_rate": float(self.sum_sla / n),
            "mean_latency":       float(self.sum_lat / n),
            "mean_drop_rate":     float(self.sum_drop / n),
            "mean_workers":       float(self.sum_W / n),
            "mean_db_conns":      float(self.sum_D / n),
        }
