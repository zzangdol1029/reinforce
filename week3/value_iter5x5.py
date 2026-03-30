# 실습(Quiz p.42): 5x5 GridWorld — Value Iteration
#
# ================================
# Value Iteration(가치 반복)이란?
# ================================
# 목표: 최적 가치함수 V*(s)와 최적 정책 π*(s)를 구한다.
#
# 가치 반복은 다음을 반복합니다.
#   V_{k+1}(s) = max_a [ r(s,a,s') + γ V_k(s') ]
#
# 정책 반복과 비교하면,
# - 정책 반복: "평가(수렴)" ↔ "개선"을 번갈아 수행
# - 가치 반복: 최적 Bellman 백업을 계속 수행하여 V*로 직접 수렴
#
# V*를 얻은 뒤에는, 마지막에 greedy_policy(V*)로 π*를 구성합니다.

from collections import defaultdict

from gridworld5x5 import GridWorld5x5

import numpy as np
import matplotlib.pyplot as plt


def argmax(d):
    """
    딕셔너리 d에서 value가 최대인 key 반환.
    - 동률이면 마지막 key를 선택 (슬라이드/예제 코드 스타일).
    """
    max_value = max(d.values())
    max_key = -1
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key


def greedy_policy(V, env, gamma):
    """
    가치함수 V에 대한 탐욕적 정책(1-step lookahead).

    각 상태 s에서 다음을 최대화하는 행동 a를 고릅니다.
      a* = argmax_a [ r(s,a,s') + γ V(s') ]
    그리고 결정적 정책으로
      π(a*|s)=1, 나머지 0
    로 구성합니다.
    """
    pi = {}
    for state in env.states():
        # 벽/종료 상태는 정책이 의미 없으므로 확률 0으로 둠 (화살표도 그리지 않음)
        if state in env.wall_states or env.is_terminal(state):
            pi[state] = {0: 0, 1: 0, 2: 0, 3: 0}
            continue

        # 행동별 1-step 가치 계산
        action_values = {}
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            action_values[action] = r + gamma * V[next_state]

        # 최대 행동 선택 후 결정적 정책으로 구성
        max_action = argmax(action_values)
        action_probs = {0: 0, 1: 0, 2: 0, 3: 0}
        action_probs[max_action] = 1.0
        pi[state] = action_probs
    return pi


def value_iter_onestep(V, env, gamma):
    """
    가치 반복의 1-step 갱신.

    Bellman 최적 방정식:
      V(s) <- max_a [ r(s,a,s') + γ V(s') ]
    """
    for state in env.states():
        # (중요) 벽/종료 상태 처리
        # - 벽: 실제 위치가 아니므로 업데이트 불필요 → skip
        # - 종료 상태(goal/trap): V=0으로 고정(업데이트하지 않음)
        if state in env.wall_states:
            continue
        if env.is_terminal(state):
            V[state] = 0
            continue

        # Bellman optimal backup: 모든 행동을 평가해서 최대를 취함
        action_values = []
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            # 이 환경은 확률적 전이가 없으므로(결정적 전이),
            # action을 취했을 때의 1-step return을 그대로 사용
            action_values.append(r + gamma * V[next_state])

        # max_a (...) 로 업데이트 → '최적 행동을 한다'고 가정한 가치
        V[state] = max(action_values)

    return V


def _pick_snapshot_indices(n_snapshots, min_panels=5):
    """
    시각화할 스냅샷 인덱스 선택.
    - n_snapshots가 min_panels 이하면 전부 사용
    - 그보다 크면 0..n-1을 균등 간격으로 min_panels개 선택(시작·끝 포함)
    """
    if n_snapshots <= 0:
        return []
    if n_snapshots <= min_panels:
        return list(range(n_snapshots))
    return sorted(
        set(
            int(round(i * (n_snapshots - 1) / (min_panels - 1)))
            for i in range(min_panels)
        )
    )


def value_iter(V, env, gamma, threshold=0.001, is_render=True):
    """
    가치 반복 메인 루프.
    - 매 반복마다 V를 갱신하고, 최대 변화량(delta)로 수렴을 판단한다.
    """
    # 스냅샷을 모아서 한 창에 그리기 위한 리스트
    # - 모든 스텝을 저장한 뒤, 시각화 시 최소 5패널이 되도록 균등 간격으로 고릅니다.
    #
    # snapshots_all = [(iter_idx, V_copy), ...]
    # - iter_idx=0은 시작(V=0)
    snapshots_all = [(0, V.copy())]
    iter_idx = 0

    while True:

        old_V = V.copy()  # 수렴 판정용
        V = value_iter_onestep(V, env, gamma)  # 한 번 훑으며 최적 백업 적용
        iter_idx += 1

        # 모든 반복의 V를 저장 (5x5는 상태 수가 작아 메모리 부담이 거의 없음)
        snapshots_all.append((iter_idx, V.copy()))

        # delta = max_s |V_new(s) - V_old(s)| 계산
        delta = 0.0
        for state in env.states():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        # 임계값(threshold)보다 변화가 작으면 수렴으로 판단
        if delta < threshold:
            break

    # 최소 5개 패널이 나오도록 균등 간격으로 선택(스냅샷 개수가 5 미만이면 전부)
    n = len(snapshots_all)
    pick = _pick_snapshot_indices(n, min_panels=5)

    # 최종 반환용: [(name, V_snapshot), ...]
    snapshots = []
    for idx in pick:
        it, V_snap = snapshots_all[idx]
        if idx == 0:
            name = "start (V=0)"
        elif idx == n - 1:
            name = "final (V*)"
        else:
            name = f"mid (iter {it})"
        snapshots.append((name, V_snap))

    return V, snapshots, iter_idx


if __name__ == "__main__":
    # ============================================================
    # 하이퍼파라미터 비교 실행 (결과를 위·중·아래 세 줄로 표시)
    #
    # ┌────────────────────────┬────────┬─────────────┐
    # │                        │ gamma  │  threshold  │
    # ├────────────────────────┼────────┼─────────────┤
    # │ [Default]              │  0.9   │   0.001     │ ← 기본값
    # │ [gamma only]  (*)      │  0.99  │   0.001     │ ← gamma만 변경
    # │ [gamma+threshold] (**) │  0.99  │   1e-6      │ ← 둘 다 변경
    # └────────────────────────┴────────┴─────────────┘
    #
    # 1행 vs 2행 : gamma 변경 효과만 격리해서 확인
    # 2행 vs 3행 : threshold 변경 효과만 격리해서 확인
    #              (V 크기는 비슷, 수렴 정밀도·반복 횟수가 달라짐)
    #
    # gamma     : 할인율. 높을수록 먼 미래 보상을 더 중시 → V 크기 변화.
    # threshold : 수렴 임계값. 낮을수록 더 정밀하게 수렴 (반복 횟수↑, 속도↓).
    # ============================================================
    CONFIGS = [
        dict(
            label="[Default]            gamma=0.9,  threshold=0.001",
            gamma=0.9,           # 기본값
            threshold=0.001,     # 기본값
        ),
        dict(
            label="[gamma only] (*)     gamma=0.99, threshold=0.001",
            gamma=0.99,          # (*) 변경: 0.9 → 0.99  (먼 미래 보상 더 반영 → V 크기↑)
            threshold=0.001,     # 기본값 유지
        ),
        dict(
            label="[gamma+threshold](**) gamma=0.99, threshold=1e-6",
            gamma=0.99,          # (*) 동일
            threshold=1e-6,      # (**) 변경: 0.001 → 1e-6 (더 정밀한 수렴, 반복↑)
        ),
    ]

    all_rows = []
    for cfg in CONFIGS:
        env = GridWorld5x5()
        V = defaultdict(lambda: 0.0)
        V, snapshots, total_iters = value_iter(
            V, env, cfg["gamma"], threshold=cfg["threshold"]
        )
        pi = greedy_policy(V, env, cfg["gamma"])
        print(f"\n{cfg['label']}")
        print(f"  => 총 {total_iters}회 Bellman 백업 후 수렴")
        all_rows.append((cfg, snapshots, pi, env))

    # ==========================
    # 3행 x 5열 figure: 위=Default / 중=gamma만 변경 / 아래=둘 다 변경
    # ==========================
    N_ROWS = len(CONFIGS)
    N_COLS = 5
    fig, axes = plt.subplots(
        N_ROWS,
        N_COLS,
        figsize=(4.6 * N_COLS, 5.4 * N_ROWS),
        constrained_layout=True,
    )

    for row_i, (cfg, snapshots, pi, env) in enumerate(all_rows):
        row_axes = axes[row_i]
        n = len(snapshots)
        for col_i in range(N_COLS):
            ax = row_axes[col_i]
            if col_i < n:
                panel_name, V_snap = snapshots[col_i]
                is_last = col_i == n - 1
                policy = pi if is_last else None
                panel_title = "final (V*)+π*" if is_last else panel_name
                # 각 행의 첫 번째 패널 제목에 config 정보 표시
                if col_i == 0:
                    panel_title = f"{cfg['label']}\n{panel_title}"
                env.render_v(
                    V_snap,
                    policy,
                    use_matplotlib=True,
                    ax=ax,
                    show=False,
                    title=panel_title,
                    draw_colorbar=False,
                )
            else:
                ax.axis("off")

    plt.suptitle(
        "Value Iteration — Hyperparameter Comparison",
        fontsize=13,
        fontweight="bold",
    )
    plt.show()

