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


def value_iter(V, env, gamma, threshold=0.001, is_render=True):
    """
    가치 반복 메인 루프.
    - 매 반복마다 V를 갱신하고, 최대 변화량(delta)로 수렴을 판단한다.
    """
    # 스냅샷을 모아서 한 창에 그리기 위한 리스트
    # - value iteration은 반복 횟수가 많을 수 있으므로, 여기서는 "모든 반복"을 저장해두고
    #   나중에 균등 간격으로 4개를 뽑아(mid) 시각화합니다.
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

    # ====== start + 4 mid + final(총 6개) 선택 ======
    # snapshots_all 길이가 n이면, 인덱스 0..n-1 중에서
    # [0, 1/5, 2/5, 3/5, 4/5, 마지막] 지점을 반올림으로 선택합니다.
    n = len(snapshots_all)
    if n <= 1:
        pick = [0]
    else:
        pick = [
            0,
            int(round((n - 1) * 1 / 5)),
            int(round((n - 1) * 2 / 5)),
            int(round((n - 1) * 3 / 5)),
            int(round((n - 1) * 4 / 5)),
            n - 1,
        ]

    # 중복 제거 + 정렬(반올림 때문에 같은 인덱스가 나올 수 있음)
    pick = sorted(set(pick))

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

    return V, snapshots


if __name__ == "__main__":
    env = GridWorld5x5()
    gamma = 0.9  # 할인율
    V = defaultdict(lambda: 0.0)

    # 1) 최적 가치 함수 V* 찾기
    V, snapshots = value_iter(V, env, gamma, is_render=True)

    # 2) V*에 대한 최적 정책 pi* 찾기
    pi = greedy_policy(V, env, gamma)

    # ==========================
    # 한 창(figure) 안에 여러 결과를 서브플롯으로 출력
    # ==========================
    # start + 4 mid + final (총 6개)
    # 2줄로 보기 좋게: 2행 x 3열
    n = len(snapshots)
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.6 * nrows))
    axes = axes.flatten()

    for i, (name, V_snap) in enumerate(snapshots):
        ax = axes[i]
        # final만 greedy 정책까지 같이 표시하고, 나머지는 V만 표시
        policy = pi if i == n - 1 else None
        title = "final (V*) + greedy π*" if i == n - 1 else name

        env.render_v(
            V_snap,
            policy,
            use_matplotlib=True,
            ax=ax,
            show=False,
            title=title,
            draw_colorbar=False,
        )

    # 사용하지 않는 subplot 축은 숨김(예: n<6인 경우)
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

