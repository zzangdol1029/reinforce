"""
비교 기준 정책(베이스라인) 모음
================================
RL 에이전트의 성능을 측정하기 위한 4가지 비교 기준 정책과 평가 함수.

■ 비교 정책 목적
  RL이 정말 의미 있는 개선을 했는지 판단하려면 단순한 기준들과 비교해야 한다.
  아무것도 안 하는 정책(Static-Current)보다 못하면 RL이 실패한 것.
  미래를 아는 Oracle에 가까울수록 RL이 성공한 것.

■ 정책 비교 구조
  Static-Low  <  Static-Current  <  HillClimb  <  RL  <  Oracle(상한)
  (최소 자원)    (현재 운영)       (반응형)       (목표)  (미래 알고 선행)

■ 행동 인코딩
  모든 이산 정책은 act() → Discrete(25) 인덱스 반환
  인덱스 = wi × 5 + di  (wi, di ∈ 0..4)
  → PortalPoolEnv._decode()에서 (ΔW, ΔD)로 변환
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from envs.portal_env import PortalPoolEnv, W_LEVELS, D_LEVELS


def to_action(env, target_w: int, target_d: int) -> int:
    """
    목표 자원량(target_w, target_d)으로 가장 가깝게 이동하는 행동 인덱스 반환.

    현재 (W, D)에서 목표까지의 차이를 계산하고,
    W_LEVELS / D_LEVELS 중 그 차이에 가장 가까운 단계를 선택.

    예) 현재 W=500, 목표 W=700, max_w=2560
        ΔW 필요 = +200
        W_LEVELS × max_w = [-205, -51, 0, +51, +205]
        가장 가까운 단계 = +205 (인덱스 4)
    """
    dw = target_w - env.W                                     # 필요한 워커 변화량
    dd = target_d - env.D                                     # 필요한 DB풀 변화량

    # 각 레벨별 실제 변화량과 필요 변화량의 차이가 최소인 인덱스 선택
    wi = int(np.argmin([abs(round(l * env.max_w) - dw) for l in W_LEVELS]))
    di = int(np.argmin([abs(round(l * env.max_d) - dd) for l in D_LEVELS]))
    return wi * len(D_LEVELS) + di                            # 2D 인덱스 → 1D 인덱스


# ── 기반 클래스 ────────────────────────────────────────────────────────────

class Policy(ABC):
    """모든 베이스라인 정책의 공통 인터페이스."""
    name = "base"

    def reset(self, env: PortalPoolEnv) -> None:  # noqa: ARG002
        """에피소드 시작 시 호출. 상태를 가진 정책(HillClimb 등)에서 초기화용."""

    @abstractmethod
    def act(self, env: PortalPoolEnv, obs) -> int:
        """관측값을 받아 행동 인덱스(0~24)를 반환."""


# ── 정책 1: StaticCurrent ─────────────────────────────────────────────────

class StaticCurrent(Policy):
    """
    현재 운영 중인 정적 설정을 그대로 유지하는 정책.

    연구포털: 워커 2048 / DB풀 150 고정
    사용자포털: 워커 256 / DB풀 100 고정
    관리포털: 워커 512 / DB풀 50 고정

    RL과 비교 시 '아무것도 안 한 것'보다 나아야 의미가 있다.
    이것이 이기지 못하면 RL 학습이 실패한 것.
    """
    name = "Static-Current"

    def act(self, env, obs) -> int:
        # 항상 current_w, current_d를 목표로 설정 → 현재 값 유지
        return to_action(env, env.current_w, env.current_d)


# ── 정책 2: StaticMax ─────────────────────────────────────────────────────

class StaticMax(Policy):
    """
    워커와 DB풀 모두 최대값으로 고정하는 정책.

    SLA 위반은 거의 없지만 메모리·DB 비용 낭비가 극심.
    '안전하지만 비효율적인' 극단적 사례로 참조.
    보상이 Static-Current보다 낮은 경우가 많음 (비용 패널티 때문).
    """
    name = "Static-Max"

    def act(self, env, obs) -> int:
        return to_action(env, env.max_w, env.max_d)


# ── 정책 3: StaticLow ─────────────────────────────────────────────────────

class StaticLow(Policy):
    """
    워커와 DB풀 모두 코어 수/최솟값 수준으로 낮게 고정하는 정책.

    자원 비용은 최소지만 피크 부하나 DB-heavy 구간에서 병목 발생.
    '최소 자원 설정'의 성능 하한선으로 참조.
    """
    name = "Static-Low"

    def act(self, env, obs) -> int:
        # 워커: max(min_w, 코어 수) — 코어 수 이하로는 줄이지 않음
        # DB풀: max(min_d, 10) — 최소 10개 유지
        return to_action(env, max(env.min_w, env.cores), max(env.min_d, 10))


# ── 정책 4: HillClimb ─────────────────────────────────────────────────────

class HillClimb(Policy):
    """
    관측 메트릭 기반의 반응형(reactive) 동시 조절 정책.

    규칙 기반으로 병목 자원을 파악하고 즉시 반응:
      - 지연 초과 + DB풀 포화 → DB 증설
      - 지연 초과 + 워커 포화 → 워커 증설
      - 여유 있음 → 양쪽 소폭 절감

    한계:
      - 부하 변화를 '보고 나서' 반응 → 전환 구간에서 항상 한 스텝 뒤처짐
      - 두 자원의 상호작용을 학습하지 못함 (단순 임계값 규칙)
      - RL은 이 패턴을 넘어 '미리 예측'하도록 학습해야 가치 있음
    """
    name = "HillClimb"

    def act(self, env, obs) -> int:
        lat = env.last_latency       # 직전 스텝 응답 지연
        wu = env.last_worker_util    # 워커풀 이용률 (0~1)
        du = env.last_db_util        # DB풀 이용률 (0~1)
        drop = env.last_drop         # 드롭율
        sla = env.sla_threshold      # SLA 임계값

        tw, td = env.W, env.D        # 목표 자원량 초기값 (현재와 동일)

        # 부하 상태 판단: 지연 초과 또는 드롭 발생 중
        busy = (lat > sla) or (drop > 0)

        if busy and du > 0.85:
            # DB풀이 85% 이상 포화 → DB가 병목 → DB 5% 증설
            td = env.D + max(1, round(0.05 * env.max_d))

        if busy and wu > 0.85:
            # 워커풀이 85% 이상 포화 → 워커가 병목 → 워커 5% 증설
            tw = env.W + max(1, round(0.05 * env.max_w))

        if not busy and lat < sla * 0.6:
            # 충분히 여유 있음 (지연이 SLA의 60% 미만)
            # → 두 자원을 소폭 절감해 비용 개선
            if wu < 0.5:
                tw = env.W - max(1, round(0.03 * env.max_w))   # 워커 3% 감소
            if du < 0.5:
                td = env.D - max(1, round(0.03 * env.max_d))   # DB풀 3% 감소

        return to_action(env, tw, td)


# ── 정책 5: Oracle ────────────────────────────────────────────────────────

class Oracle(Policy):
    """
    미래 부하를 미리 알고 선행(anticipatory) 조절하는 상한 참조 정책.

    실제 운영에서는 불가능한 '완벽한 예측자'로, RL의 성능 상한선 역할.
    Oracle 보상에 가까울수록 RL이 잘 학습된 것.

    ■ 동작 방식
      현재 스텝부터 horizon(=6) 스텝 앞까지의 부하 시계열을 미리 읽어
      각 시점에서의 최적 (W, D)를 격자 탐색으로 구한 뒤,
      그 중 최대값으로 미리 자원을 올려둠 (선행 프로비저닝).

    ■ 왜 6스텝인가?
      너무 짧으면 반응형과 차이 없고, 너무 길면 과프로비저닝.
      6스텝이 이 시뮬레이터의 변화 속도에서 경험적으로 좋은 값.
    """
    name = "Oracle(ref)"
    horizon = 6                                               # 룩어헤드 스텝 수

    def act(self, env, obs) -> int:
        tw = td = 0
        # 현재~horizon 스텝 범위에서 각 시점의 최적 자원량 계산
        for h in range(self.horizon + 1):
            lam, ms, dbf = env.future(h)                     # 미래 부하 파라미터
            w, d = env.optimal_grid(lam, ms, dbf)            # 그 시점 최적 (W, D)
            tw = max(tw, w)                                   # 가장 많이 필요한 값 선택
            td = max(td, d)
        return to_action(env, tw, td)


# ── 베이스라인 목록 ────────────────────────────────────────────────────────
# evaluate.py에서 이 리스트를 순회하며 모든 베이스라인을 평가
BASELINES = [StaticCurrent(), StaticMax(), StaticLow(), HillClimb(), Oracle()]


# ── 정책 평가 함수 ────────────────────────────────────────────────────────

def evaluate_policy(policy: Policy, portal: dict, n_episodes: int = 20,
                    base_seed: int = 4000) -> dict:
    """
    주어진 정책을 n_episodes 에피소드 실행하고 평균 성능 지표 반환.

    매 에피소드마다 다른 시드(base_seed + ep)를 사용해 다양한 부하 시나리오에서
    평가 → 특정 시나리오에 유리한 결과가 나오는 것 방지.

    Args:
        policy    : 평가할 Policy 인스턴스
        portal    : config.PORTALS 딕셔너리 항목
        n_episodes: 평가 에피소드 수 (많을수록 정확하지만 느림)
        base_seed : 첫 에피소드 시드 (ep마다 +1씩 증가)

    Returns:
        dict: reward, sla_violation_rate, mean_latency, mean_workers, mean_db_conns 평균값
    """
    agg = []
    for ep in range(n_episodes):
        env = PortalPoolEnv(portal, seed=base_seed + ep)
        obs, _ = env.reset(seed=base_seed + ep)
        policy.reset(env)                                     # 상태 초기화 (HillClimb 등)
        done = False
        R = 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(policy.act(env, obs))
            R += r
            done = term or trunc
        m = env.metrics()
        m["reward"] = R
        agg.append(m)

    # 전체 에피소드의 지표별 평균 반환
    return {k: float(np.mean([m[k] for m in agg])) for k in agg[0]}
