"""
실습: 3x4 Grid World (03-Markov Decision Process.pdf 기반)
- Episodic task: terminal state에서 종료
- S: 3x4 그리드, A: 상/하/좌/우 (또는 상하좌우+대각 등)
- Reward: 목표 +1, 위험 -1, 빈칸 0 등
"""

import numpy as np
from typing import Tuple, List

# 그리드: 3행 4열. 좌표 (row, col), row=0이 위
# 일반적 설정: [0]=시작 구역, [3]=목표(+1), [7]=함정(-1) 등
ROWS, COLS = 3, 4
N_STATES = ROWS * COLS
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]


def state_to_idx(r: int, c: int) -> int:
    return r * COLS + c


def idx_to_state(idx: int) -> Tuple[int, int]:
    return idx // COLS, idx % COLS


def build_3x4_grid_mdp(
    goal_reward: float = 1.0,
    pit_reward: float = -1.0,
    step_reward: float = 0.0,
    slip_prob: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    3x4 Grid World (예시)
    - 상태 0: 시작 (1,1)
    - 상태 11 (3,4): 목표 +1
    - 상태 7 (2,2): 함정 -1 (있다면)
    - 나머지: step_reward

    Returns:
        P[s,a,s'], R[s,a,s'], terminal_states
    """
    n_states = N_STATES
    n_actions = 4
    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions, n_states))

    # 목표/터미널 (일반적으로 (0,3) 또는 (2,3)을 목표로 둠)
    goal_idx = state_to_idx(0, 3)   # (0,3) 목표
    pit_idx = state_to_idx(1, 1)   # (1,1) 함정으로 쓰거나 빈칸
    terminal = [goal_idx]

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # UP, DOWN, LEFT, RIGHT

    for s in range(n_states):
        if s in terminal:
            # terminal: 그대로 유지, reward 0 (이미 종료)
            for a in range(n_actions):
                P[s, a, s] = 1.0
                R[s, a, s] = 0.0
            continue

        r, c = idx_to_state(s)
        for a in range(n_actions):
            dr, dc = moves[a]
            rn, cn = r + dr, c + dc

            # 벽: 그리드 밖이면 제자리
            if rn < 0 or rn >= ROWS or cn < 0 or cn >= COLS:
                rn, cn = r, c

            s_next = state_to_idx(rn, cn)

            if slip_prob > 0:
                # 미끄러짐: 확률 slip_prob로 다른 방향으로 감
                P[s, a, s_next] += 1 - slip_prob
                for a2 in range(n_actions):
                    if a2 == a:
                        continue
                    dr2, dc2 = moves[a2]
                    rn2, cn2 = r + dr2, c + dc2
                    if rn2 < 0 or rn2 >= ROWS or cn2 < 0 or cn2 >= COLS:
                        rn2, cn2 = r, c
                    s_next2 = state_to_idx(rn2, cn2)
                    P[s, a, s_next2] += slip_prob / 3
            else:
                P[s, a, s_next] = 1.0

            if s_next == goal_idx:
                R[s, a, s_next] = goal_reward
            elif s_next == pit_idx and pit_reward != 0:
                R[s, a, s_next] = pit_reward
            else:
                R[s, a, s_next] = step_reward

    return P, R, terminal


def print_grid_info(P: np.ndarray, R: np.ndarray, terminal: List[int]):
    """그리드 구조 요약 출력"""
    print("3x4 Grid World (Episodic)")
    print("  상태 인덱스:")
    for r in range(ROWS):
        line = "    "
        for c in range(COLS):
            idx = state_to_idx(r, c)
            if idx in terminal:
                line += " [G] "
            else:
                line += " {:2d}  ".format(idx)
        print(line)
    print("  G: Goal (terminal), γ 사용해 Return 계산")


def run_random_policy_value(
    gamma: float = 0.9,
    goal_reward: float = 1.0,
    pit_reward: float = -1.0,
):
    """균일 랜덤 정책 π(a|s)=1/4에 대한 v_π 계산 (Policy Evaluation)"""
    P, R, terminal = build_3x4_grid_mdp(
        goal_reward=goal_reward, pit_reward=pit_reward
    )
    n_states = P.shape[0]
    n_actions = P.shape[1]

    # R(s,a) = Σ_{s'} P(s'|s,a) r(s,a,s')
    R_sa = np.einsum("sas,sas->sa", P, R)

    # Random policy: π(a|s) = 1/4
    policy = np.ones((n_states, n_actions)) / n_actions
    P_pi = np.einsum("sa,sas->ss", policy, P)
    R_pi = np.einsum("sa,sa->s", policy, R_sa)

    # Terminal은 0으로 유지
    v = np.zeros(n_states)
    for _ in range(500):
        v_new = R_pi + gamma * P_pi @ v
        v_new[terminal] = 0
        if np.max(np.abs(v_new - v)) < 1e-6:
            break
        v = v_new

    print_grid_info(P, R, terminal)
    print("\nRandom policy π(a|s)=1/4 에 대한 v_π(s) (γ={}):".format(gamma))
    for r in range(ROWS):
        line = "  "
        for c in range(COLS):
            idx = state_to_idx(r, c)
            if idx in terminal:
                line += "  G   "
            else:
                line += "{:5.2f} ".format(v[idx])
        print(line)
    return v, P, R, terminal


if __name__ == "__main__":
    print("=== 3x4 Grid World (Episodic Task) ===\n")
    run_random_policy_value(gamma=0.9, goal_reward=1.0, pit_reward=-1.0)
