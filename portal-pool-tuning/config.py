"""
포털별 실제 사양 프리셋 (AP_DB자원설정 기반, 노드 1대 기준)
=============================================================

이 파일은 3개 포털의 하드웨어·소프트웨어 사양을 딕셔너리로 정의한다.
train.py / optimize.py / evaluate.py 모두 이 파일을 참조해 환경을 구성한다.

값 출처:
  - cores / RAM / Heap / Undertow worker / HikariCP pool : 제공된 자원설정 엑셀
  - 연구포털은 현재 WORKER_THREADS=2048, Hikari 150 을 '정적'으로 사용 → current 값
  - 사용자/관리는 Undertow 기본값(=io_threads×8=cores×16)을 current로 가정
  - 도착률(RPS)/요청믹스(mean_s, db_frac)는 동접·업무특성 기반 추정(가정) →
    실측(APM/부하테스트)으로 보정 권장. 노드 1대 기준값.

비용 계수 설계 원칙:
  mem_cost × 전형적W  ≈  db_cost × 전형적D  ≈  SLA 페널티 스케일
  → 보상 함수에서 워커 비용과 DB 비용이 서로 비슷한 크기로 작용해
    에이전트가 두 자원을 균형 있게 조절하도록 유도
"""

# ── 파라미터 설명 ──────────────────────────────────────────────────────────
#
# [하드웨어 사양]
#   cores          : 물리 CPU 코어 수. 처리능력의 상한선 역할
#   max_workers    : Undertow 워커 스레드 허용 최대값
#   min_workers    : 워커 스레드 허용 최솟값 (0으로 내려가지 않도록)
#   current_workers: 현재 운영 중인 워커 수 (RL이 개선해야 할 기준점)
#   max_db         : HikariCP 커넥션풀 허용 최대값
#   min_db         : 커넥션풀 허용 최솟값
#   current_db     : 현재 운영 중인 DB풀 크기
#   db_shared_max  : 공유 DB 서버가 감당 가능한 전체 커넥션 수
#                    → D가 이 값에 가까워지면 공유 DB 과부하로 실제 DB 처리시간 증가
#
# [부하 시뮬레이션]
#   base_rps       : 야간/비피크 초당 요청 수 (RPS)
#   peak_rps       : 피크 시간대 초당 요청 수
#   mean_s_min/max : 요청 평균 체류시간(초) 범위. 클수록 워커를 오래 점유
#   dbfrac_min/max : 요청 처리 중 DB 단계 비중 범위 (0=DB 없음, 1=전부 DB)
#                    클수록 DB 커넥션을 오래 점유
#
# [보상 함수 가중치]
#   sla_threshold  : SLA 응답시간 목표(초). 초과 시 패널티 발생
#   mem_cost       : 워커 스레드 1개당 유지 비용 (메모리 소비 반영)
#   db_cost        : DB 커넥션 1개당 유지 비용 (공유 DB 부하 반영)
#   sla_weight     : SLA 위반 패널티 가중치 (클수록 지연에 민감)
#   drop_penalty   : 처리 못한 요청(드롭) 패널티
#   thrash         : 자원을 자주 바꿀 때 부과하는 안정성 패널티
#   overhead       : 코어 수 초과 워커로 인한 컨텍스트 스위칭 오버헤드 계수
#   db_contention  : 공유 DB 경합 민감도. 클수록 D 증가 시 DB 처리시간 더 늘어남
#
# [에피소드 설정]
#   episode_steps  : 에피소드 길이 (제어 스텝 수)
#   load_cycles    : 에피소드 내 부하(RPS) 주기 반복 횟수
#   ms_cycles      : 에피소드 내 체류시간 변동 주기 반복 횟수
#   dbf_cycles     : 에피소드 내 DB 비중 변동 주기 반복 횟수
# ──────────────────────────────────────────────────────────────────────────

PORTALS = {

    # ── 연구포털 ────────────────────────────────────────────────────────────
    # 사양: 256코어/512GB × 4노드, Heap 256GB
    # 현재 설정: 워커 2048 / DB풀 150 (정적 고정)
    # 특성: 장시간 DB 요청이 많아 DB풀이 자주 병목 (dbfrac 최대 75%)
    # 문제: 워커 2048개는 대부분 과다 → 메모리 낭비
    #       DB-heavy 구간에서 DB풀 150이 부족 → 지연 급증
    # Oracle 최적: ~328W / ~143D (워커 84% 절감)
    "research": dict(
        name="research",
        cores=256,                      # CPU 코어 256개 (대형 서버)
        max_workers=2560,               # 워커 최대: 코어 수의 10배
        min_workers=16,                 # 워커 최소: 최소한의 서비스 유지
        current_workers=2048,           # 현재 운영 중 워커 수 (과다 상태)
        max_db=300,                     # DB풀 최대 (공유 DB 용량 내)
        min_db=5,                       # DB풀 최소
        current_db=150,                 # 현재 운영 중 DB풀 (DB-heavy 구간 부족)
        db_shared_max=300,              # 공유 DB 전체 허용 커넥션 수
        base_rps=80,                    # 야간 기저 부하 80 RPS
        peak_rps=500,                   # 피크 부하 500 RPS
        mean_s_min=0.08,                # 최소 체류시간 0.08초
        mean_s_max=0.45,                # 최대 체류시간 0.45초 (DB-heavy 요청)
        dbfrac_min=0.30,                # DB 처리 비중 최소 30%
        dbfrac_max=0.75,                # DB 처리 비중 최대 75% (장시간 DB 요청)
        sla_threshold=0.6,              # SLA 목표: 600ms 이내
        mem_cost=0.00004,               # 워커 1개당 비용 (대형 서버라 낮게 설정)
        db_cost=0.004,                  # DB 커넥션 1개당 비용
        sla_weight=4.0,                 # SLA 위반 패널티 가중치
        drop_penalty=10.0,              # 드롭 패널티
        thrash=0.02,                    # 자원 변동 패널티 계수
        overhead=0.0008,                # 컨텍스트 스위칭 오버헤드 계수
        db_contention=0.6,              # 공유 DB 경합 민감도
        episode_steps=200,              # 에피소드 200 스텝
        load_cycles=4.0,                # 에피소드 내 부하 4주기
        ms_cycles=2.5,                  # 체류시간 2.5주기
        dbf_cycles=3.0,                 # DB 비중 3주기
    ),

    # ── 사용자포털 ──────────────────────────────────────────────────────────
    # 사양: 16코어/64GB × 2노드, Heap 31GB
    # 현재 설정: 워커 256 / DB풀 100
    # 특성: 동접 5000명, 짧은 요청 (체류시간 0.03~0.12초)
    # 문제: 16코어가 CPU 병목 → 워커를 아무리 늘려도 CPU 한계에 막힘
    #       이 포털은 코어 증설이 근본 해결책 (RL의 한계 존재)
    # Oracle 최적: ~40W / ~20D (대폭 절감 가능)
    "user": dict(
        name="user",
        cores=16,                       # CPU 코어 16개 (소형 서버, CPU 병목 주요 원인)
        max_workers=512,                # 워커 최대: 코어의 32배
        min_workers=8,                  # 워커 최소
        current_workers=256,            # 현재 운영 중 워커 수
        max_db=150,                     # DB풀 최대
        min_db=5,                       # DB풀 최소
        current_db=100,                 # 현재 운영 중 DB풀
        db_shared_max=200,              # 공유 DB 전체 허용 커넥션 수
        base_rps=150,                   # 기저 부하 150 RPS (사용자 많음)
        peak_rps=600,                   # 피크 부하 600 RPS
        mean_s_min=0.03,                # 최소 체류시간 0.03초 (짧은 요청)
        mean_s_max=0.12,                # 최대 체류시간 0.12초
        dbfrac_min=0.30,                # DB 비중 최소 30%
        dbfrac_max=0.60,                # DB 비중 최대 60%
        sla_threshold=0.4,              # SLA 목표: 400ms (더 엄격)
        mem_cost=0.0003,                # 워커 비용 (소형 서버라 높게 설정)
        db_cost=0.004,
        sla_weight=4.0,
        drop_penalty=10.0,
        thrash=0.02,
        overhead=0.004,                 # 16코어라 오버헤드 계수 높음
        db_contention=0.6,
        episode_steps=200,
        load_cycles=4.0,
        ms_cycles=2.5,
        dbf_cycles=3.0,
    ),

    # ── 관리포털 ────────────────────────────────────────────────────────────
    # 사양: 32코어/128GB × 2노드, Heap 31GB
    # 현재 설정: 워커 512 / DB풀 50
    # 특성: 동접 200명, 부하 여유 (base 10 ~ peak 70 RPS)
    # 문제: 워커 512개가 훨씬 과다 → 메모리 낭비 (Oracle: ~37개면 충분)
    # Oracle 최적: ~37W / ~9D (가장 큰 개선 여지)
    "admin": dict(
        name="admin",
        cores=32,                       # CPU 코어 32개
        max_workers=512,                # 워커 최대
        min_workers=8,                  # 워커 최소
        current_workers=512,            # 현재 운영 중 워커 수 (최대값과 동일 = 완전 과다)
        max_db=80,                      # DB풀 최대
        min_db=5,                       # DB풀 최소
        current_db=50,                  # 현재 운영 중 DB풀
        db_shared_max=120,              # 공유 DB 전체 허용 커넥션 수
        base_rps=10,                    # 기저 부하 10 RPS (낮음)
        peak_rps=70,                    # 피크 부하 70 RPS
        mean_s_min=0.05,                # 최소 체류시간 0.05초
        mean_s_max=0.20,                # 최대 체류시간 0.20초
        dbfrac_min=0.30,                # DB 비중 최소 30%
        dbfrac_max=0.60,                # DB 비중 최대 60%
        sla_threshold=0.4,              # SLA 목표: 400ms
        mem_cost=0.0002,                # 워커 비용
        db_cost=0.004,
        sla_weight=4.0,
        drop_penalty=10.0,
        thrash=0.02,
        overhead=0.002,                 # 32코어, 중간 오버헤드
        db_contention=0.6,
        episode_steps=200,
        load_cycles=4.0,
        ms_cycles=2.5,
        dbf_cycles=3.0,
    ),
}

# 옵션 미지정 시 기본 포털
DEFAULT_PORTAL = "research"
