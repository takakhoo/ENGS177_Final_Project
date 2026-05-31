"""QMDP approximation for POMDPs (Littman, Cassandra, Kaelbling 1995).

Given a fully-observable optimal Q-function Q*(s, a) and a belief b over states,
    pi_QMDP(b) = argmax_a sum_s b(s) Q*(s, a).

We also implement the standard Bayesian belief filter:
    b_t(s') ∝ O(o_t | s') * sum_s T(s' | s) b_{t-1}(s).
"""
from __future__ import annotations

import numpy as np


def qmdp_action(belief: np.ndarray, Q_star: np.ndarray) -> int:
    """QMDP greedy action given current belief.

    belief: (S,) summing to 1.
    Q_star: (S, A).
    Returns argmax_a sum_s belief[s] Q_star[s, a].
    """
    return int((belief @ Q_star).argmax())


def update_belief(
    belief: np.ndarray,
    T: np.ndarray,
    obs_likelihood: np.ndarray,
) -> np.ndarray:
    """One step of Bayesian belief filter.

    belief: (S,) prior over states at t-1, summing to 1.
    T: (S, S) transition matrix T[s, s'] = p(s' | s). Regime is exogenous to action.
    obs_likelihood: (S,) p(o_t | s) under each state's emission model.

    Returns: (S,) posterior belief at t.
    """
    predictive = belief @ T                # (S,), predictive prior at t
    unnorm = obs_likelihood * predictive    # (S,), joint with observation
    Z = unnorm.sum()
    if Z <= 0 or not np.isfinite(Z):
        # Numerical underflow: return uniform as fallback.
        return np.full_like(belief, 1.0 / belief.size)
    return unnorm / Z


def stationary_distribution(T: np.ndarray) -> np.ndarray:
    """Left eigenvector of T for eigenvalue 1, normalized to a probability vector."""
    eigvals, eigvecs = np.linalg.eig(T.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    v = np.real(eigvecs[:, idx])
    v = np.maximum(v, 0)
    return v / v.sum()
