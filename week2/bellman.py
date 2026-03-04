"""
실습: Bellman Equation (4.pdf 기반)
- Bellman Expectation Equation: v_π(s), q_π(s,a)
- Bellman Optimality Equation: v*(s), q*(s,a)
- Policy Evaluation (반복) 및 Optimal Value 계산
"""

import sys
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from grid_world_1x2 import (
    build_transition_and_reward,
    get_expected_reward,
    POLICIES,
    L1,
    L2,
    LEFT,
    RIGHT,
)


def bellman_expectation_v(
    P: np.ndarray,
    R: np.ndarray,
    policy: np.ndarray,
    v: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    v_π(s) = Σ_a π(a|s) [ R(s,a) + γ Σ_{s'} P(s'|s,a) v_π(s') ]
    한 스텝 backup.
    """
    R_sa = get_expected_reward(P, R)
    n_states = P.shape[0]
    v_new = np.zeros(n_states)
    for s in range(n_states):
        for a in range(policy.shape[1]):
            v_new[s] += policy[s, a] * (
                R_sa[s, a] + gamma * np.dot(P[s, a, :], v)
            )
    return v_new


def bellman_optimality_v(
    P: np.ndarray,
    R: np.ndarray,
    v: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    v*(s) = max_a q*(s,a)
    q*(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) v*(s')
    한 스텝 backup.
    """
    R_sa = get_expected_reward(P, R)
    n_states, n_actions = P.shape[0], P.shape[1]
    v_new = np.zeros(n_states)
    for s in range(n_states):
        q_s = np.array([
            R_sa[s, a] + gamma * np.dot(P[s, a, :], v)
            for a in range(n_actions)
        ])
        v_new[s] = np.max(q_s)
    return v_new


def policy_evaluation_iterative(
    P: np.ndarray,
    R: np.ndarray,
    policy: np.ndarray,
    gamma: float = 0.9,
    tol: float = 1e-6,
    max_iter: int = 500,
) -> np.ndarray:
    """Bellman Expectation 반복으로 v_π 수렴"""
    n_states = P.shape[0]
    v = np.zeros(n_states)
    for _ in range(max_iter):
        v_new = bellman_expectation_v(P, R, policy, v, gamma)
        if np.max(np.abs(v_new - v)) < tol:
            break
        v = v_new
    return v


def value_iteration(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float = 0.9,
    tol: float = 1e-6,
    max_iter: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bellman Optimality 반복으로 v* 수렴.
    Returns: v*, optimal policy (deterministic) π[s,a]
    """
    n_states, n_actions = P.shape[0], P.shape[1]
    R_sa = get_expected_reward(P, R)
    v = np.zeros(n_states)
    for _ in range(max_iter):
        v_new = bellman_optimality_v(P, R, v, gamma)
        if np.max(np.abs(v_new - v)) < tol:
            break
        v = v_new

    # v*에서 greedy policy 추출: π*(s) = argmax_a q*(s,a)
    policy = np.zeros((n_states, n_actions))
    for s in range(n_states):
        q_s = np.array([
            R_sa[s, a] + gamma * np.dot(P[s, a, :], v)
            for a in range(n_actions)
        ])
        best_a = np.argmax(q_s)
        policy[s, best_a] = 1.0
    return v, policy


def run_bellman_demo(alpha: float = 0.5, beta: float = 0.5, gamma: float = 0.9):
    """1x2 Grid World에서 Bellman Expectation vs Optimality 시연"""
    P, R = build_transition_and_reward(alpha, beta)

    print("=== Bellman Equation 실습 (4.pdf) ===\n")
    print("1) Bellman Expectation Equation: v_π(s) = Σ_a π(a|s)[ R(s,a) + γ Σ_{s'} P(s'|s,a) v_π(s') ]")
    print("   → 주어진 policy π에 대해 각 state value 계산\n")

    for name, policy in POLICIES.items():
        v = policy_evaluation_iterative(P, R, policy, gamma=gamma)
        print("   Policy {}: v_π(L1)={:.4f}, v_π(L2)={:.4f}".format(name, v[L1], v[L2]))

    print("\n2) Bellman Optimality Equation: v*(s) = max_a [ R(s,a) + γ Σ_{s'} P(s'|s,a) v*(s') ]")
    print("   → Value Iteration으로 v* 계산\n")

    v_star, pi_star = value_iteration(P, R, gamma=gamma)
    print("   v*(L1)={:.4f}, v*(L2)={:.4f}".format(v_star[L1], v_star[L2]))
    print("   Optimal policy (deterministic):")
    print("     L1 → {},  L2 → {}".format(
        "left" if pi_star[L1, LEFT] == 1 else "right",
        "left" if pi_star[L2, LEFT] == 1 else "right",
    ))
    print("   → PDF와 동일: Optimal policy = μ2 (L1→right, L2→left)")
    return v_star, pi_star


if __name__ == "__main__":
    run_bellman_demo(alpha=0.5, beta=0.5, gamma=0.9)
