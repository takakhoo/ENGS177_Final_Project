"""Finite-state infinite-horizon discounted MDP solvers used by QMDP.

Implements:
    - value_iteration(P, R, lam, eps): Lec-7 Bellman iteration with
      stopping criterion `eps*(1-lam)/(2*lam)`.
    - policy_iteration(P, R, lam): exact PI via linear solve.
    - extract_greedy_policy(P, R, V, lam): argmax operator.

Shapes:
    P: (A, S, S) — transition tensor, P[a, s, s'] = p(s' | s, a)
                   (NOTE: in our problem regimes are exogenous to action, so
                    P[a, s, s'] = T_HMM[s, s'] for every action a.)
    R: (S, A)    — expected reward per (state, action).
    V: (S,)      — value function.
"""
from __future__ import annotations

import numpy as np


def value_iteration(
    P: np.ndarray, R: np.ndarray, lam: float = 0.95, eps: float = 1e-4, max_iter: int = 10_000
) -> tuple[np.ndarray, np.ndarray, int]:
    """Quiz-3 formula-sheet VI.

    Returns
    -------
    V : (S,) optimal value
    pi : (S,) optimal action index per state
    n : number of iterations
    """
    S = R.shape[0]
    V = np.zeros(S)
    tol = eps * (1 - lam) / (2 * lam)  # Quiz 3 stopping criterion
    for n in range(1, max_iter + 1):
        # Q[s, a] = R[s, a] + lam * sum_{s'} P[a, s, s'] V[s']
        Q = R + lam * np.einsum("ass,s->as", P, V).T  # (S, A)
        V_next = Q.max(axis=1)
        if np.max(np.abs(V_next - V)) < tol:
            V = V_next
            break
        V = V_next
    pi = (R + lam * np.einsum("ass,s->as", P, V).T).argmax(axis=1)
    return V, pi, n


def policy_evaluation_exact(
    P: np.ndarray, R: np.ndarray, pi: np.ndarray, lam: float
) -> np.ndarray:
    """Closed-form policy eval: v = (I - lam * P_pi)^-1 r_pi."""
    S = R.shape[0]
    P_pi = np.array([P[pi[s], s, :] for s in range(S)])   # (S, S)
    r_pi = np.array([R[s, pi[s]] for s in range(S)])      # (S,)
    return np.linalg.solve(np.eye(S) - lam * P_pi, r_pi)


def policy_iteration(
    P: np.ndarray, R: np.ndarray, lam: float = 0.95, max_iter: int = 1_000
) -> tuple[np.ndarray, np.ndarray, int]:
    """Exact policy iteration with greedy improvement."""
    S, A = R.shape
    pi = np.zeros(S, dtype=int)
    for n in range(1, max_iter + 1):
        V = policy_evaluation_exact(P, R, pi, lam)
        Q = R + lam * np.einsum("ass,s->as", P, V).T  # (S, A)
        pi_new = Q.argmax(axis=1)
        if np.array_equal(pi_new, pi):
            return V, pi, n
        pi = pi_new
    return V, pi, n


def q_function(P: np.ndarray, R: np.ndarray, V: np.ndarray, lam: float) -> np.ndarray:
    """Q*(s, a) = R(s, a) + lam * sum_{s'} P(s' | s, a) V(s'). Shape (S, A)."""
    return R + lam * np.einsum("ass,s->as", P, V).T
