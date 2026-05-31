"""Experiment 09: Multi-feature HMM with richer macro-financial observations.

Guidolin & Timmermann (2007) predict that 2-state HMMs on narrow observation sets
fail to differentiate regime-conditional policies. We test this directly by
re-fitting the HMM with progressively richer observation channels and re-running
the QMDP backtest for each. If the prediction holds, the richer observation
channels should:
  (a) sharpen the regime separation (BIC drops, regime-conditional return
      moments diverge more);
  (b) produce regime-dependent QMDP policies (the same gamma=2 setup should
      no longer collapse to "100% stocks" in both states).

Observation-channel cohorts:
  Cohort 1 (baseline): (VIX, term_spread)
  Cohort 2 (vol + curve): (VIX, term_spread, term_spread_2y)
  Cohort 3 (+ stress): (VIX, term_spread, NFCI, STLFSI4)
  Cohort 4 (+ macro): (VIX, term_spread, NFCI, UMCSENT, ICSA)
  Cohort 5 (kitchen sink): all 8 extension channels

Outputs:
  results/multifeature_hmm_table.csv, log-likelihood, BIC, conditional moments
  results/multifeature_policies.csv , pi*(s) for each cohort, K=2
  figures/multifeature_regimes.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt  # noqa: E402

from src.models.hmm import fit_hmm  # noqa: E402
from src.models.mdp import value_iteration  # noqa: E402
from src.utils.utility import expected_utility_per_regime  # noqa: E402

DATA = REPO / "data" / "processed"
RES = REPO / "results"
FIG = REPO / "figures"

GAMMA = 2.0
LAMBDA = 0.95
N_SIM = 5000
TRAIN_END = "2014-12-31"   # match canonical fit window (02_hmm_calibration.py); no look-ahead
ACTIONS = np.array([
    [0.0, 1.0], [0.2, 0.8], [0.4, 0.6], [0.6, 0.4], [0.8, 0.2], [1.0, 0.0],
])

COHORTS = {
    "1_baseline":      ["vix", "term_spread"],
    "2_vol_curve":     ["vix", "term_spread", "term_spread_2y"],
    "3_stress":        ["vix", "term_spread", "nfci", "stlfsi"],
    "4_macro":         ["vix", "term_spread", "nfci", "umcsent", "jobless_claims"],
    "5_kitchen_sink":  ["vix", "term_spread", "term_spread_2y", "nfci", "stlfsi",
                         "umcsent", "jobless_claims", "usd_index"],
}


def standardize(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Standardize columns using TRAIN-window (<= TRAIN_END) mean/std, matching the
    canonical fit methodology in 02_hmm_calibration.py. Returns (full_array, index)."""
    arr = df[cols].dropna()
    train = arr.loc[:TRAIN_END]
    mu = train.mean(); sd = train.std(ddof=0).replace(0, 1)
    return ((arr - mu) / sd).values, arr.index


def solve_for_cohort(df: pd.DataFrame, cohort_cols: list[str], n_states: int = 2):
    """Fit HMM (train-only), derive regime-conditional reward over the full sample,
    solve MDP, return policy. Mirrors the canonical pipeline: fit <=2014, predict full."""
    obs, idx = standardize(df, cohort_cols)
    asset_rets = df.loc[idx, ["spy_ret", "agg_ret"]].dropna()
    obs = obs[:len(asset_rets)]
    # Fit on the training window only (no look-ahead), matching the canonical HMM.
    n_train = int((idx <= pd.Timestamp(TRAIN_END)).sum())
    model, ll = fit_hmm(obs[:n_train], n_states=n_states, n_restarts=5)
    bic_val = -2.0 * ll + (
        (n_states - 1) + n_states * (n_states - 1)
        + n_states * obs.shape[1]
        + n_states * obs.shape[1] * (obs.shape[1] + 1) // 2
    ) * np.log(obs.shape[0])

    # MAP-assign each month to regime, compute regime-conditional return moments
    states_pred = model.predict(obs)
    regime_returns = {}
    for k in range(n_states):
        mask = states_pred == k
        if mask.sum() < 5:
            regime_returns[k] = asset_rets.values  # fallback
        else:
            regime_returns[k] = asset_rets.iloc[mask].values

    # Order regimes by mean SPY return (bull = highest mean) for consistent labeling
    order = sorted(range(n_states), key=lambda k: regime_returns[k][:, 0].mean(), reverse=True)
    regime_returns = {new_k: regime_returns[old_k] for new_k, old_k in enumerate(order)}

    # Build reward matrix via Monte Carlo: sample returns from per-regime empirical, CRRA utility
    rng = np.random.default_rng(42)
    R = np.zeros((n_states, len(ACTIONS)))
    for s in range(n_states):
        rets = regime_returns[s]
        idx = rng.integers(0, len(rets), size=N_SIM)
        sampled = rets[idx]
        for a_idx, a in enumerate(ACTIONS):
            wealth = 1.0 + sampled @ a
            wealth = np.maximum(wealth, 1e-12)
            if abs(GAMMA - 1.0) < 1e-12:
                util = np.log(wealth)
            else:
                util = (np.power(wealth, 1 - GAMMA) - 1.0) / (1 - GAMMA)
            R[s, a_idx] = util.mean()

    # Transition tensor (action-independent)
    T = model.transmat_
    # Reorder T by the same regime ordering
    T = T[np.ix_(order, order)]
    P = np.array([T for _ in range(len(ACTIONS))])  # (A, S, S)

    V_star, pi_star, n_iter = value_iteration(P, R, lam=LAMBDA, eps=1e-4)

    # Regime-conditional means in % per month for the table
    rcm = {}
    for s in range(n_states):
        rcm[s] = {
            "n_months":    len(regime_returns[s]),
            "mean_spy":    regime_returns[s][:, 0].mean() * 100,
            "std_spy":     regime_returns[s][:, 0].std() * 100,
            "mean_agg":    regime_returns[s][:, 1].mean() * 100,
            "std_agg":     regime_returns[s][:, 1].std() * 100,
        }
    return {
        "log_likelihood": ll,
        "bic": bic_val,
        "iters_vi": n_iter,
        "V_star": V_star.tolist(),
        "policy_bull_idx":  int(pi_star[0]),
        "policy_bear_idx":  int(pi_star[1]),
        "policy_bull": ACTIONS[pi_star[0]].tolist(),
        "policy_bear": ACTIONS[pi_star[1]].tolist(),
        "regime_diff":   abs(pi_star[0] - pi_star[1]),
        "rcm": rcm,
    }


def main() -> None:
    df = pd.read_csv(DATA / "monthly.csv")
    dc = "DATE" if "DATE" in df.columns else df.columns[0]
    df[dc] = pd.to_datetime(df[dc]); df = df.set_index(dc)

    summary = []
    for cohort_name, cols in COHORTS.items():
        # Only keep cols present in the data
        avail = [c for c in cols if c in df.columns]
        if len(avail) < 2:
            print(f"  [skip] cohort {cohort_name}: only {len(avail)} columns available.")
            continue
        print(f"\n=== Cohort {cohort_name}, {avail} ===")
        try:
            r = solve_for_cohort(df, avail, n_states=2)
            print(f"  log L: {r['log_likelihood']:.2f}  BIC: {r['bic']:.2f}  VI iters: {r['iters_vi']}")
            print(f"  policy bull: {r['policy_bull']}  policy bear: {r['policy_bear']}  diff={r['regime_diff']}")
            row = {
                "cohort": cohort_name,
                "n_features": len(avail),
                "features": "|".join(avail),
                "log_likelihood": r["log_likelihood"],
                "bic": r["bic"],
                "vi_iters": r["iters_vi"],
                "policy_bull_action": str(r["policy_bull"]),
                "policy_bear_action": str(r["policy_bear"]),
                "regime_policy_differs": r["regime_diff"] > 0,
                "V_bull": r["V_star"][0],
                "V_bear": r["V_star"][1],
                "mean_spy_bull": r["rcm"][0]["mean_spy"],
                "mean_spy_bear": r["rcm"][1]["mean_spy"],
                "mean_agg_bull": r["rcm"][0]["mean_agg"],
                "mean_agg_bear": r["rcm"][1]["mean_agg"],
                "n_months_bull": r["rcm"][0]["n_months"],
                "n_months_bear": r["rcm"][1]["n_months"],
            }
            summary.append(row)
        except Exception as e:
            print(f"  [error] {e}")
            continue

    out = pd.DataFrame(summary)
    out.to_csv(RES / "multifeature_hmm_table.csv", index=False)
    print(f"\n=== SUMMARY (gamma={GAMMA}) ===")
    print(out[["cohort", "n_features", "bic",
               "policy_bull_action", "policy_bear_action",
               "regime_policy_differs"]].to_string(index=False))
    print(f"\nWrote {RES / 'multifeature_hmm_table.csv'}")


if __name__ == "__main__":
    main()
