"""Utility functions for the reward model.

Log utility:   U(W) = log(W)
CRRA utility:  U(W) = (W^{1-gamma} - 1) / (1 - gamma) for gamma != 1, else log(W).
"""
from __future__ import annotations

import numpy as np


def crra(wealth: np.ndarray | float, gamma: float = 2.0) -> np.ndarray | float:
    """Constant-relative-risk-aversion utility. wealth > 0."""
    wealth = np.maximum(wealth, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return np.log(wealth)
    return (np.power(wealth, 1 - gamma) - 1.0) / (1 - gamma)


def log_utility(wealth: np.ndarray | float) -> np.ndarray | float:
    return crra(wealth, gamma=1.0)


def expected_utility_per_regime(
    regime_returns: dict[int, np.ndarray],  # regime_id -> (n_sim, n_assets) sample returns
    actions: np.ndarray,                    # (n_actions, n_assets) discrete portfolio grid
    gamma: float = 2.0,
) -> np.ndarray:
    """Build R(s, a) by Monte Carlo over each regime's return distribution.

    Returns (n_regimes, n_actions) reward array.
    """
    n_regimes = len(regime_returns)
    n_actions = actions.shape[0]
    R = np.zeros((n_regimes, n_actions))
    for s in range(n_regimes):
        rets = regime_returns[s]  # (n_sim, n_assets)
        for a_idx in range(n_actions):
            port_ret = rets @ actions[a_idx]   # (n_sim,)
            wealth = 1.0 + port_ret             # next-step wealth from $1
            R[s, a_idx] = crra(wealth, gamma).mean()
    return R
