"""
실습: 1x2 Grid World MDP (03-Markov Decision Process.pdf 기반)
- State: S = {L1, L2}
- Action: A(L1)=A(L2)= {right, left}
- Transition probability & Expected rewards (α, β 파라미터)
- 4가지 deterministic policy (μ1~μ4), Optimal policy = μ2
"""

import numpy as np
from typing import Dict, Tuple, List

# 상태/행동 인덱스
L1, L2 = 0, 1
LEFT, RIGHT = 0, 1
STATE_NAMES = ["L1", "L2"]
ACTION_NAMES = ["left", "right"]


def build_transition_and_reward(
    alpha: float, beta: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    PDF 전이 테이블:
    s   s'  a      p(s'|s,a)   r(s,a,s')
    L1  L1  left   1-α         -1
    L1  L2  right  α           +1
    L2  L1  left   β           0
    L2  L2  right  1-β         -1

    Returns:
        P[s,a,s'] = p(s'|s,a), R[s,a,s'] = r(s,a,s')
    """
    n_states, n_actions = 2, 2
    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions, n_states))

    # L1
    P[L1, LEFT, L1] = 1 - alpha
    R[L1, LEFT, L1] = -1
    P[L1, RIGHT, L2] = alpha
    R[L1, RIGHT, L2] = 1

    # L2
    P[L2, LEFT, L1] = beta
    R[L2, LEFT, L1] = 0
    P[L2, RIGHT, L2] = 1 - beta
    R[L2, RIGHT, L2] = -1

    return P, R


def get_expected_reward(P: np.ndarray, R: np.ndarray) -> np.ndarray:
    """R(s,a) = E[r|s,a] = Σ_{s'} P(s'|s,a) * r(s,a,s')"""
    return np.einsum("sas,sas->sa", P, R)


# 4가지 deterministic policy (PDF Q2)
# 행: L1, L2 / 열: left, right → 1인 열이 선택되는 행동
POLICY_MU1 = np.array([[0, 1], [0, 1]])   # L1→right, L2→right
POLICY_MU2 = np.array([[0, 1], [1, 0]])   # L1→right, L2→left  (Optimal)
POLICY_MU3 = np.array([[1, 0], [0, 1]])   # L1→left,  L2→right
POLICY_MU4 = np.array([[1, 0], [1, 0]])   # L1→left,  L2→left

POLICIES = {
    "μ1": POLICY_MU1,
    "μ2": POLICY_MU2,
    "μ3": POLICY_MU3,
    "μ4": POLICY_MU4,
}


def policy_evaluation(
    P: np.ndarray,
    R: np.ndarray,
    policy: np.ndarray,
    gamma: float = 0.9,
    tol: float = 1e-6,
    max_iter: int = 500,
) -> np.ndarray:
    """
    Bellman Expectation Equation으로 v_π(s) 반복 계산
    v_π(s) = Σ_a π(a|s) [ R(s,a) + γ Σ_{s'} P(s'|s,a) v_π(s') ]
    """
    n_states = P.shape[0]
    R_expected = get_expected_reward(P, R)  # [s,a]

    # π에 따른 전이: P_π(s'|s) = Σ_a π(a|s) P(s'|s,a), R_π(s) = Σ_a π(a|s) R(s,a)
    P_pi = np.einsum("sa,sas->ss", policy, P)
    R_pi = np.einsum("sa,sa->s", policy, R_expected)

    v = np.zeros(n_states)
    for _ in range(max_iter):
        v_new = R_pi + gamma * P_pi @ v
        if np.max(np.abs(v_new - v)) < tol:
            break
        v = v_new
    return v


def print_transition_table(alpha: float, beta: float):
    """PDF 형식의 전이 테이블 출력"""
    P, R = build_transition_and_reward(alpha, beta)
    print("전이 테이블: p(s'|s,a), r(s,a,s')")
    print("  s   s'   a      p(s'|s,a)   r(s,a,s')")
    print("  L1  L1  left   {:.4f}     {:.0f}".format(P[L1, LEFT, L1], R[L1, LEFT, L1]))
    print("  L1  L2  right  {:.4f}     {:.0f}".format(P[L1, RIGHT, L2], R[L1, RIGHT, L2]))
    print("  L2  L1  left   {:.4f}     {:.0f}".format(P[L2, LEFT, L1], R[L2, LEFT, L1]))
    print("  L2  L2  right {:.4f}     {:.0f}".format(P[L2, RIGHT, L2], R[L2, RIGHT, L2]))
    print()


def run_1x2_grid_world(alpha: float = 0.5, beta: float = 0.5, gamma: float = 0.9):
    """1x2 Grid World 실습: 4 policies의 v_π 비교, optimal = μ2"""
    P, R = build_transition_and_reward(alpha, beta)

    print("=== 1x2 Grid World MDP ===\n")
    print("S = {L1, L2}, A = {left, right}, γ = {}\n".format(gamma))
    print_transition_table(alpha, beta)

    print("4가지 정책 (deterministic):")
    print("       L1    L2")
    print("       left  right")
    for name, pi in POLICIES.items():
        a_l1 = "left " if pi[L1, LEFT] == 1 else "right"
        a_l2 = "left " if pi[L2, LEFT] == 1 else "right"
        print("  {}   {}   {}".format(name, a_l1, a_l2))
    print()

    print("Policy Evaluation (Bellman Expectation), γ = {}:".format(gamma))
    results = {}
    for name, policy in POLICIES.items():
        v = policy_evaluation(P, R, policy, gamma=gamma)
        results[name] = v
        print("  v_π({})  L1={:.4f}, L2={:.4f}".format(name, v[L1], v[L2]))

    # Optimal: μ2가 가장 높은 value를 주는지 확인
    print("\n→ Optimal policy = μ2 (L1→right, L2→left)")
    v_mu2 = results["μ2"]
    for name, v in results.items():
        if name == "μ2":
            continue
        if np.all(v <= v_mu2 + 1e-9) and np.any(v < v_mu2 - 1e-9):
            print("  μ2 dominates {} (모든 상태에서 μ2의 value가 크거나 같음)".format(name))
    return results, P, R


if __name__ == "__main__":
    run_1x2_grid_world(alpha=0.5, beta=0.5, gamma=0.9)
