# 실습(Quiz p.42): 5x5 GridWorld — Policy Iteration
#
# ================================
# Policy Iteration(정책 반복)이란?
# ================================
# 목표: 5x5 GridWorld에서 최적 정책 μ*(s) 와 최적 가치함수 v*(s)를 구한다.
#
# 정책 반복은 크게 2단계를 번갈아 수행합니다.
#
# 1) Policy Evaluation (정책 평가)
#    - 현재 정책 π가 "고정"되어 있다고 가정하고, 그 정책의 가치함수 V^π를 계산합니다.
#    - Bellman 기대 방정식(Expectation Bellman equation)을 반복 적용하여 수렴시킵니다.
#
# 2) Policy Improvement (정책 개선)
#    - 계산된 V^π를 이용해, 1-step lookahead로 더 좋은 행동을 선택하는 탐욕적(greedy) 정책 π'를 만듭니다.
#    - π'가 π와 같아지면(더 이상 개선이 없으면) 최적 정책에 도달한 것으로 보고 종료합니다.
#
# 이 파일은 "5x5 환경"에 맞춘 구현이며,
# - 환경 정의: `gridworld5x5.py`
# - 시각화: `env.render_v(V, pi, use_matplotlib=True)` (matplotlib로 색/화살표 출력)
# 을 사용합니다.

from collections import defaultdict

from gridworld5x5 import GridWorld5x5

import numpy as np
import matplotlib.pyplot as plt


def argmax(d):
    """
    딕셔너리 d에서 value가 최대인 key 반환.
    - 동률(tie)이면 "마지막으로 등장한 key"를 선택합니다.
      (슬라이드 예제 코드의 동작과 동일)
    """
    max_value = max(d.values())
    max_key = -1
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key


def eval_onestep(pi, V, env, gamma=0.9):
    """
    정책 pi가 주어졌을 때, 모든 상태를 한 번 훑으며 V를 1회 갱신.

    Bellman 기대 방정식:
      V(s) <- Σ_a pi(a|s) [ r(s,a,s') + γ V(s') ]
    """
    for state in env.states():  # 모든 상태 순회 (벽/목표/함정 포함)
        # --- (중요) 업데이트에서 제외할 상태들 ---
        # 1) 벽: 실제로 에이전트가 "위치"할 수 없도록 설계(이동하면 제자리 유지)
        #        가치 업데이트 의미가 없으므로 건너뜁니다.
        if state in env.wall_states:
            continue
        # 2) terminal(목표/함정): 에피소드가 끝나는 상태.
        #    많은 교재/실습에서는 terminal의 V를 0으로 두고 업데이트하지 않습니다.
        if env.is_terminal(state):
            V[state] = 0
            continue

        # π(a|s): state에서 action을 선택할 확률 분포
        action_probs = pi[state]
        # 새로운 V(s)를 만들기 위해 누적할 변수 (기대값)
        new_V = 0.0

        # --- Bellman 기대 방정식의 Σ_a 부분 ---
        # 각 행동의 확률(action_prob)로, 다음 상태/보상을 가중합합니다.
        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            # (r + γ V(s')) 를 π(a|s)로 가중합
            new_V += action_prob * (r + gamma * V[next_state])

        V[state] = new_V

    return V


def policy_eval(pi, V, env, gamma, threshold=0.001):
    """
    정책 평가 반복:
      delta = max_s |V_new(s) - V_old(s)| 가 threshold 미만이면 종료.
    """
    # 반복적으로 1-step 업데이트를 수행하며 V^π로 수렴시킵니다.
    while True:
        old_V = V.copy()  # 수렴 판정을 위해 이전 V 저장
        V = eval_onestep(pi, V, env, gamma)  # 한 번 훑으며 업데이트

        # delta = max_s |V_new(s) - V_old(s)|  (최대 변화량)
        delta = 0.0
        for state in env.states():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        if delta < threshold:
            break

    return V


def greedy_policy(V, env, gamma):
    """
    현재 가치함수 V에 대해 1-step lookahead로 탐욕적 정책 생성.
      pi(s) = argmax_a [ r(s,a,s') + γ V(s') ]
    """
    # π'(s)를 만들 딕셔너리
    pi = {}
    for state in env.states():
        # 벽/종료 상태는 "행동을 선택할 필요가 없는 상태"로 취급합니다.
        # (화살표 표시에서도 제외되도록 확률 0으로 둠)
        if state in env.wall_states or env.is_terminal(state):
            pi[state] = {0: 0, 1: 0, 2: 0, 3: 0}
            continue

        # action_values[a] = Q(s,a) 비슷한 1-step lookahead 값
        action_values = {}
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            # 1-step lookahead: r + γ V(s')
            action_values[action] = r + gamma * V[next_state]

        # 가장 큰 값을 주는 행동 선택
        max_action = argmax(action_values)
        # 결정적(deterministic) 정책: 최적 행동은 확률 1, 나머지는 0
        action_probs = {0: 0, 1: 0, 2: 0, 3: 0}
        action_probs[max_action] = 1.0
        pi[state] = action_probs

    return pi


def _pick_snapshot_indices(n_snapshots, min_panels=5):
    """시각화할 스냅샷 인덱스: 최소 min_panels개(가능할 때) 균등 간격."""
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


def policy_iter(env, gamma, threshold=0.001, is_render=True):
    """
    정책 반복:
      (평가) V <- V^pi
      (개선) pi <- greedy(V)
      정책이 더 이상 변하지 않으면 종료.
    """
    # 초기 정책: 모든 상태에서 균등 무작위
    # (벽/종료 상태는 eval_onestep/greedy_policy에서 별도로 처리하므로 여기서는 동일하게 둬도 됩니다.)
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0.0)

    # 스냅샷(여러 개)을 한 창에 모아 그리기 위해 저장
    # snapshots = [(policy_name, V_snapshot, pi_snapshot), ...]
    snapshots = []

    # ==========================
    # (0) 시작: 무작위 정책 평가
    # ==========================
    V = policy_eval(pi, V, env, gamma, threshold)
    snapshots.append(("start (random π)", V.copy(), dict(pi)))

    k = 0
    while True:
        # ==========================
        # (개선) π <- greedy(V)
        # ==========================
        new_pi = greedy_policy(V, env, gamma)

        # ==========================
        # (평가) V <- V^{new_pi}
        # ==========================
        # 개선된 정책을 "고정"한다고 가정하고, 그 정책의 가치함수로 다시 수렴시킵니다.
        V = policy_eval(new_pi, V, env, gamma, threshold)

        k += 1
        snapshots.append((f"iter {k}", V.copy(), new_pi))

        # 정책이 더 이상 변하지 않으면 (정책 안정, policy stable) 종료
        if new_pi == pi:
            pi = new_pi
            break
        pi = new_pi

    # render는 여기서 하지 않고, 호출자가 한 창에 모아 그리도록 snapshots도 함께 반환
    return pi, V, snapshots


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
    #              (V 크기는 비슷, 수렴 정밀도·사이클 수가 달라짐)
    #
    # gamma     : 할인율. 높을수록 먼 미래 보상을 더 중시 → V 크기 변화.
    # threshold : 수렴 임계값. 낮을수록 더 정밀하게 수렴 (평가 반복↑, 속도↓).
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
        pi, V, snapshots = policy_iter(
            env, cfg["gamma"], threshold=cfg["threshold"]
        )
        pick = _pick_snapshot_indices(len(snapshots), min_panels=5)
        show_snaps = [snapshots[i] for i in pick]
        print(f"\n{cfg['label']}")
        print(f"  => 총 {len(snapshots)}개 평가·개선 사이클")
        all_rows.append((cfg, show_snaps, env))

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

    for row_i, (cfg, show_snaps, env) in enumerate(all_rows):
        row_axes = axes[row_i]
        n = len(show_snaps)
        for col_i in range(N_COLS):
            ax = row_axes[col_i]
            if col_i < n:
                name, V_snap, pi_snap = show_snaps[col_i]
                panel_title = name
                # 각 행의 첫 번째 패널 제목에 config 정보 표시
                if col_i == 0:
                    panel_title = f"{cfg['label']}\n{panel_title}"
                env.render_v(
                    V_snap,
                    pi_snap,
                    use_matplotlib=True,
                    ax=ax,
                    show=False,
                    title=panel_title,
                    draw_colorbar=False,
                )
            else:
                ax.axis("off")

    plt.suptitle(
        "Policy Iteration — Hyperparameter Comparison",
        fontsize=13,
        fontweight="bold",
    )
    plt.show()

