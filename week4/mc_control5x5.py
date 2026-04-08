"""
Monte Carlo Control — 5x5 GridWorld  (Quiz p.22)
=================================================
Q함수를 ε-greedy 정책으로 학습하고, Q함수 및 최적 정책을 시각화합니다.

파라미터:
  ε (epsilon) : ε-greedy 탐험 확률.
                작을수록 현재 Q 최댓값 행동을 주로 선택 (exploitation).
                클수록 랜덤 탐험 비율 높음 (exploration).
  α (alpha)   : 학습률 (constant step-size).
                작을수록 과거 경험을 천천히 갱신 (안정적, 수렴 느림).
                클수록 최근 경험을 빠르게 반영 (빠름, 노이즈 큼).

환경:
  - 5x5 GridWorld (week3/gridworld5x5.py 재사용)
  - Goal (0,4): +1, Trap (0,3)·(3,3): -1, Wall (2,1)·(2,2)

알고리즘:
  On-policy First-Visit MC Control (상수 α 방식)
  1) ε-greedy 정책으로 에피소드 생성
  2) 역방향으로 Return G 계산
  3) First-visit (s,a)에 대해 Q(s,a) += α*(G - Q(s,a))
  4) 충분한 에피소드 후 greedy 정책 추출
"""

import os
import sys
import random
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ============================================================
# week3 경로 추가 (GridWorld5x5 재사용)
# ============================================================
_WEEK3 = os.path.join(os.path.dirname(__file__), "..", "week3")
sys.path.insert(0, _WEEK3)
from gridworld5x5 import GridWorld5x5  # noqa: E402

# ============================================================
# 폰트: 경로 하드코딩 없이 matplotlib 기본 메커니즘 사용
# - 앞에서부터 설치된 폰트를 쓰고, 없으면 DejaVu Sans(배포본 포함)로 폴백
# ============================================================
_FONT_CONFIGURED = False


def _setup_font():
    global _FONT_CONFIGURED
    if _FONT_CONFIGURED:
        return
    _FONT_CONFIGURED = True
    plt.rcParams["font.sans-serif"] = [
        "Malgun Gothic",
        "Apple SD Gothic Neo",
        "NanumGothic",
        "Nanum Gothic",
        "Noto Sans CJK KR",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# ε-greedy 정책으로 행동 선택
# ============================================================
def _epsilon_greedy(Q, state, actions, epsilon):
    """ε 확률로 랜덤 행동, (1-ε) 확률로 Q 최대 행동 선택."""
    if random.random() < epsilon:
        return random.choice(actions)
    q_vals = {a: Q[(state, a)] for a in actions}
    return max(q_vals, key=q_vals.get)


# ============================================================
# 에피소드 생성
# ============================================================
def _generate_episode(env, Q, epsilon, max_steps=500):
    """
    ε-greedy 정책으로 에피소드 1개를 생성.
    Returns:
        list of (state, action, reward)
    """
    state = env.reset()
    episode = []
    for _ in range(max_steps):
        if env.is_terminal(state):
            break
        action = _epsilon_greedy(Q, state, env.actions(), epsilon)
        next_state, reward, done = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


# ============================================================
# Greedy 정책 추출
# ============================================================
def extract_greedy_policy(Q, env):
    """Q 테이블에서 탐욕적(greedy) 정책 추출. dict/defaultdict 모두 지원."""
    pi = {}
    for state in env.states():
        if state in env.wall_states or env.is_terminal(state):
            pi[state] = {a: 0 for a in env.actions()}
            continue
        # defaultdict가 아닌 일반 dict도 지원하기 위해 .get 사용
        q_vals = {a: Q.get((state, a), 0.0) for a in env.actions()}
        best = max(q_vals, key=q_vals.get)
        pi[state] = {a: (1.0 if a == best else 0.0) for a in env.actions()}
    return pi


# ============================================================
# Monte Carlo Control (On-Policy, First-Visit, constant α)
# ============================================================
def mc_control(env, num_episodes=5000, epsilon=0.1, alpha=0.1, gamma=0.9, n_snaps=4):
    """
    On-policy First-Visit MC Control (상수 α 방식).

    Args:
        env          : GridWorld5x5 환경
        num_episodes : 총 에피소드 수
        epsilon      : ε-greedy 탐험 확률
        alpha        : 학습률 (상수 step-size)
        gamma        : 할인율
        n_snaps      : 스냅샷 개수 (시작 포함)

    Returns:
        Q        : 최종 Q 테이블  {(state, action): float}
        pi       : 최종 greedy 정책  {state: {action: prob}}
        q_snaps  : [(label, Q_dict), ...] 학습 과정 스냅샷
    """
    Q = defaultdict(float)

    # 스냅샷 저장 시점 (ep=0 포함, 균등 간격)
    snap_set = sorted(
        set(int(round(num_episodes * k / (n_snaps - 1))) for k in range(n_snaps))
    )

    q_snaps = []
    # ep=0 스냅샷 (학습 전 초기 Q=0)
    if 0 in snap_set:
        q_snaps.append(("ep 0 (init)", dict(Q)))

    for ep in range(1, num_episodes + 1):
        episode = _generate_episode(env, Q, epsilon)

        # Return G를 역방향으로 계산하고 First-Visit 업데이트
        G = 0.0
        visited = set()
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            if (state, action) not in visited:
                visited.add((state, action))
                # Q(s,a) ← Q(s,a) + α * (G - Q(s,a))
                Q[(state, action)] += alpha * (G - Q[(state, action)])

        if ep in snap_set and ep > 0:
            label = f"ep {ep}" if ep < num_episodes else f"ep {ep} (final)"
            q_snaps.append((label, dict(Q)))

    pi = extract_greedy_policy(Q, env)
    return Q, pi, q_snaps


# ============================================================
# Q 함수 시각화 — 삼각형 4분할
# ============================================================
_TRI_ANCHOR = {
    # action: (꼭짓점 상대좌표 오프셋 목록, 텍스트 무게중심 비율)
    # 셀 중심 (cx, cy), 반크기 0.5
    # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    0: ([(0, 0), (-0.5, -0.5), (0.5, -0.5)], (0, -0.32)),    # 위 삼각형
    1: ([(0, 0), (-0.5,  0.5), (0.5,  0.5)], (0,  0.32)),    # 아래 삼각형
    2: ([(0, 0), (-0.5, -0.5), (-0.5,  0.5)], (-0.32, 0)),   # 왼쪽 삼각형
    3: ([(0, 0), (0.5, -0.5), (0.5,  0.5)],  (0.32, 0)),     # 오른쪽 삼각형
}
_ARROW_DIR = {0: (0, -0.28), 1: (0, 0.28), 2: (-0.28, 0), 3: (0.28, 0)}


def render_q_triangles(Q, env, pi=None, ax=None, title=None, show=True,
                       vmax_global=None):
    """
    Q(s,a)를 각 셀을 4개의 삼각형(UP/DOWN/LEFT/RIGHT)으로 나눠 색상 표시.
      - 색: 빨강(음수) → 노랑(0) → 초록(양수)
      - 중앙 화살표: pi 또는 Q에서 argmax 행동
      - 벽: 회색, 목표/함정: 보상 색상에 레이블

    Args:
        Q            : Q 테이블 dict
        env          : GridWorld5x5 환경
        pi           : 표시할 정책 (None이면 Q에서 greedy 추출)
        ax           : matplotlib axes (None이면 새 figure)
        title        : 패널 제목
        show         : True면 plt.show() 호출
        vmax_global  : 색 스케일 최댓값 (None이면 자동)
    """
    _setup_font()
    H, W = env.height, env.width

    # 색 스케일
    all_q = [Q.get((s, a), 0.0)
             for s in env.states()
             for a in env.actions()
             if s not in env.wall_states and not env.is_terminal(s)]
    vmax = vmax_global if vmax_global else max(max((abs(v) for v in all_q), default=0.0), 0.01)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.cm.RdYlGn

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    if pi is None:
        pi = extract_greedy_policy(Q, env)

    for h in range(H):
        for w in range(W):
            state = (h, w)
            cx, cy = float(w), float(h)

            # ── 벽 ──
            if state in env.wall_states:
                ax.add_patch(plt.Polygon(
                    [(cx - 0.5, cy - 0.5), (cx + 0.5, cy - 0.5),
                     (cx + 0.5, cy + 0.5), (cx - 0.5, cy + 0.5)],
                    color="#555555", zorder=2,
                ))
                continue

            # ── 목표 / 함정 ──
            if env.is_terminal(state):
                r_val = env.reward_map[h, w]
                r_f = float(r_val) if r_val is not None else 0.0
                color = cmap(norm(r_f))
                ax.add_patch(plt.Polygon(
                    [(cx - 0.5, cy - 0.5), (cx + 0.5, cy - 0.5),
                     (cx + 0.5, cy + 0.5), (cx - 0.5, cy + 0.5)],
                    color=color, zorder=2,
                ))
                label = "G" if state == env.goal_state else "T"
                ax.text(cx, cy, label, ha="center", va="center",
                        fontsize=10, fontweight="bold", zorder=6)
                continue

            # ── 일반 상태: 4개 삼각형 ──
            for a in env.actions():
                q_val = Q.get((state, a), 0.0)
                color = cmap(norm(q_val))
                offsets, (tx_off, ty_off) = _TRI_ANCHOR[a]
                verts = [(cx + dx, cy + dy) for dx, dy in offsets]
                ax.add_patch(plt.Polygon(verts, color=color, zorder=2))
                # Q 값 숫자
                ax.text(cx + tx_off, cy + ty_off,
                        f"{q_val:.2f}",
                        ha="center", va="center",
                        fontsize=5.5, color="black", zorder=6)

            # 셀 테두리
            ax.add_patch(plt.Polygon(
                [(cx - 0.5, cy - 0.5), (cx + 0.5, cy - 0.5),
                 (cx + 0.5, cy + 0.5), (cx - 0.5, cy + 0.5)],
                fill=False, edgecolor="black", linewidth=0.8, zorder=3,
            ))
            # 대각선 (4분할 경계)
            ax.plot([cx - 0.5, cx + 0.5], [cy - 0.5, cy + 0.5],
                    color="black", linewidth=0.4, zorder=3)
            ax.plot([cx + 0.5, cx - 0.5], [cy - 0.5, cy + 0.5],
                    color="black", linewidth=0.4, zorder=3)

            # ── 최적 행동 화살표 ──
            best_a = max(pi[state], key=pi[state].get)
            if pi[state][best_a] > 0:
                dx, dy = _ARROW_DIR[best_a]
                ax.annotate(
                    "",
                    xy=(cx + dx, cy + dy),
                    xytext=(cx, cy),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="navy",
                        lw=1.5,
                        mutation_scale=10,
                    ),
                    zorder=7,
                )

    if title:
        ax.set_title(title, fontsize=8, pad=4)

    if standalone:
        plt.tight_layout()
    if show:
        plt.show()


# ============================================================
# 실험 실행 헬퍼
# ============================================================
def _run_comparison(configs, suptitle, num_episodes, gamma, n_snaps=4):
    """
    configs 목록으로 MC Control을 실행하고
    결과를 (행=config, 열=스냅샷) 형태의 figure에 표시.
    """
    N_ROWS = len(configs)
    N_COLS = n_snaps  # 스냅샷 개수와 열 수 맞춤

    fig, axes = plt.subplots(
        N_ROWS,
        N_COLS,
        figsize=(4.8 * N_COLS, 5.6 * N_ROWS),
        constrained_layout=True,
    )
    if N_ROWS == 1:
        axes = [axes]

    for row_i, cfg in enumerate(configs):
        env = GridWorld5x5()
        print(f"  {cfg['label']}  | episodes={num_episodes}")
        Q, pi, q_snaps = mc_control(
            env,
            num_episodes=num_episodes,
            epsilon=cfg["epsilon"],
            alpha=cfg["alpha"],
            gamma=gamma,
            n_snaps=n_snaps,
        )
        print(f"    => 스냅샷 {len(q_snaps)}개 저장")

        # 전체 Q 범위에서 공통 vmax 계산 (같은 행 안에서 색 스케일 통일)
        all_q_vals = [v for v in Q.values() if v != 0.0]
        vmax_row = max(max((abs(v) for v in all_q_vals), default=0.0), 0.01)

        row_axes = axes[row_i]
        for col_i in range(N_COLS):
            ax = row_axes[col_i] if N_COLS > 1 else row_axes
            if col_i < len(q_snaps):
                snap_label, Q_snap = q_snaps[col_i]
                is_final = col_i == len(q_snaps) - 1
                pi_snap = extract_greedy_policy(Q_snap, env) if is_final else None
                panel_title = snap_label
                if col_i == 0:
                    panel_title = f"{cfg['label']}\n{snap_label}"
                render_q_triangles(
                    Q_snap, env,
                    pi=pi_snap,
                    ax=ax,
                    title=panel_title,
                    show=False,
                    vmax_global=vmax_row,
                )
            else:
                ax.axis("off")

    plt.suptitle(suptitle, fontsize=13, fontweight="bold")
    plt.show()


# ============================================================
# __main__
# ============================================================
if __name__ == "__main__":
    # ============================================================
    # 공통 설정
    # ============================================================
    NUM_EPISODES = 5000   # 총 학습 에피소드 수
    GAMMA = 0.9           # 할인율
    N_SNAPS = 4           # 스냅샷 개수 (초기 포함)

    # ============================================================
    # (1) ε 비교 실험
    # ============================================================
    # ┌────────────────────────────┬──────┬──────┐
    # │                            │  ε   │  α   │
    # ├────────────────────────────┼──────┼──────┤
    # │ [Row 1] ε=0.1  (낮은 탐험) │ 0.1  │ 0.1  │ ← α 고정
    # │ [Row 2] ε=0.3  (중간 탐험) │ 0.3  │ 0.1  │
    # │ [Row 3] ε=0.7  (높은 탐험) │ 0.7  │ 0.1  │
    # └────────────────────────────┴──────┴──────┘
    #
    # ε 작음: exploitation 위주 → 빨리 수렴하지만 최적 외 경로를 잘 안 탐험
    # ε 큼  : exploration 위주 → 다양한 (s,a)를 방문하지만 수렴이 느릴 수 있음
    # ============================================================
    EPS_CONFIGS = [
        dict(
            label="[Row1] epsilon=0.1  (low  exploration) | alpha=0.1 (fixed)",
            epsilon=0.1,    # 탐험 10%
            alpha=0.1,      # 고정
        ),
        dict(
            label="[Row2] epsilon=0.3  (mid  exploration) | alpha=0.1 (fixed)",
            epsilon=0.3,    # 탐험 30%
            alpha=0.1,
        ),
        dict(
            label="[Row3] epsilon=0.7  (high exploration) | alpha=0.1 (fixed)",
            epsilon=0.7,    # 탐험 70%
            alpha=0.1,
        ),
    ]

    print("=" * 65)
    print("(1) ε 비교 실험  (alpha=0.1 고정, epsilon 변경)")
    print("=" * 65)
    _run_comparison(
        EPS_CONFIGS,
        suptitle=(
            f"Monte Carlo Control — epsilon Comparison\n"
            f"(gamma={GAMMA}, alpha=0.1 fixed, episodes={NUM_EPISODES})"
        ),
        num_episodes=NUM_EPISODES,
        gamma=GAMMA,
        n_snaps=N_SNAPS,
    )

    # ============================================================
    # (2) α 비교 실험
    # ============================================================
    # ┌──────────────────────────────┬──────┬──────┐
    # │                              │  ε   │  α   │
    # ├──────────────────────────────┼──────┼──────┤
    # │ [Row 1] α=0.01  (느린 학습)  │ 0.1  │ 0.01 │ ← ε 고정
    # │ [Row 2] α=0.1   (중간 학습)  │ 0.1  │ 0.1  │
    # │ [Row 3] α=0.5   (빠른 학습)  │ 0.1  │ 0.5  │
    # └──────────────────────────────┴──────┴──────┘
    #
    # α 작음: 과거 Q값 유지 비율 높음 → 안정적이지만 수렴 느림
    # α 큼  : 최근 Return G를 강하게 반영 → 빠르지만 노이즈에 민감
    # ============================================================
    ALPHA_CONFIGS = [
        dict(
            label="[Row1] alpha=0.01 (slow  learning) | epsilon=0.1 (fixed)",
            epsilon=0.1,    # 고정
            alpha=0.01,     # 느린 학습
        ),
        dict(
            label="[Row2] alpha=0.1  (mid   learning) | epsilon=0.1 (fixed)",
            epsilon=0.1,
            alpha=0.1,      # 중간 학습
        ),
        dict(
            label="[Row3] alpha=0.5  (fast  learning) | epsilon=0.1 (fixed)",
            epsilon=0.1,
            alpha=0.5,      # 빠른 학습
        ),
    ]

    print("\n" + "=" * 65)
    print("(2) α 비교 실험  (epsilon=0.1 고정, alpha 변경)")
    print("=" * 65)
    _run_comparison(
        ALPHA_CONFIGS,
        suptitle=(
            f"Monte Carlo Control — alpha Comparison\n"
            f"(gamma={GAMMA}, epsilon=0.1 fixed, episodes={NUM_EPISODES})"
        ),
        num_episodes=NUM_EPISODES,
        gamma=GAMMA,
        n_snaps=N_SNAPS,
    )
