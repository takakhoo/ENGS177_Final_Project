"""Step 4 of pipeline: build R(s, a), solve the underlying MDP via VI and PI (cross-check),
emit policy maps and value functions to results/."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.models.mdp import value_iteration, policy_iteration, q_function  # noqa: E402
from src.utils.utility import expected_utility_per_regime                  # noqa: E402

DATA = REPO / "data" / "processed"
RES = REPO / "results"
RES.mkdir(exist_ok=True)

N_STATES = 2
LAMBDA = 0.95
EPS = 1e-4
N_SIM_PER_REGIME = 5000
GAMMA_CRRA = 2.0

# Discrete portfolio actions: (stock weight, bond weight)
ACTIONS = np.array([
    [0.0, 1.0],
    [0.2, 0.8],
    [0.4, 0.6],
    [0.6, 0.4],
    [0.8, 0.2],
    [1.0, 0.0],
])


def simulate_regime_returns(model, n_sim: int, rng: np.random.Generator) -> dict[int, np.ndarray]:
    """Sample per-regime *asset returns* by drawing observation samples from each HMM emission
    and mapping through the asset-return distribution.

    For the baseline we use a simple proxy: each regime has its own (mean, std) for stock
    and bond returns. In the full implementation those come from regime-conditional sample
    moments in `regime_returns.csv`. Here we recover them from the data on the fly.
    """
    df = pd.read_csv(DATA / "monthly.csv")
    dc = "DATE" if "DATE" in df.columns else df.columns[0]
    df[dc] = pd.to_datetime(df[dc]); df = df.set_index(dc)
    obs_cols = [c for c in ["vix", "term_spread", "hy_oas"] if c in df.columns]
    states = model.predict(df[obs_cols].values)
    out: dict[int, np.ndarray] = {}
    for k in range(model.n_components):
        mask = states == k
        if mask.sum() < 2:
            mu = df[["spy_ret", "agg_ret"]].mean().values
            cov = df[["spy_ret", "agg_ret"]].cov().values
        else:
            mu = df.loc[mask, ["spy_ret", "agg_ret"]].mean().values
            cov = df.loc[mask, ["spy_ret", "agg_ret"]].cov().values
        out[k] = rng.multivariate_normal(mu, cov, size=n_sim)
    return out


def main() -> None:
    with open(DATA / f"hmm_{N_STATES}state.pkl", "rb") as f:
        model = pickle.load(f)

    rng = np.random.default_rng(0)
    regime_returns = simulate_regime_returns(model, N_SIM_PER_REGIME, rng)

    # R(s, a) by Monte-Carlo expectation of CRRA utility of the portfolio's gross return.
    R = expected_utility_per_regime(regime_returns, ACTIONS, gamma=GAMMA_CRRA)

    # Build transition tensor: regime dynamics are exogenous to action.
    T_hmm = model.transmat_                                  # (S, S)
    A = ACTIONS.shape[0]
    P = np.repeat(T_hmm[np.newaxis, :, :], A, axis=0)        # (A, S, S)

    V_vi, pi_vi, n_iter_vi = value_iteration(P, R, lam=LAMBDA, eps=EPS)
    V_pi, pi_pi, n_iter_pi = policy_iteration(P, R, lam=LAMBDA)

    Q_star = q_function(P, R, V_vi, lam=LAMBDA)

    # Cross-check
    assert np.array_equal(pi_vi, pi_pi), "VI and PI gave different greedy policies!"
    print(f"VI converged in {n_iter_vi} iters; PI converged in {n_iter_pi}.")
    print(f"V*  : {np.round(V_vi, 4)}")
    print(f"pi* : {pi_vi}  → actions {[ACTIONS[i] for i in pi_vi]}")

    pd.DataFrame({
        "regime": np.arange(N_STATES),
        "V_star": V_vi,
        "best_action_idx": pi_vi,
        "stock_weight": [ACTIONS[i, 0] for i in pi_vi],
        "bond_weight":  [ACTIONS[i, 1] for i in pi_vi],
    }).to_csv(RES / "mdp_policy.csv", index=False)

    np.save(RES / "Q_star.npy", Q_star)
    print(f"Wrote {RES / 'mdp_policy.csv'} and {RES / 'Q_star.npy'}")


if __name__ == "__main__":
    main()
