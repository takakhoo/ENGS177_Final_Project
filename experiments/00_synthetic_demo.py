"""Synthetic-data end-to-end demo (no network required).

Generates a 25-year monthly stream from a *known* 2-regime HMM, runs the full
pipeline (HMM fit → MDP value iteration → QMDP belief filtering → backtest)
and writes a figure + metrics table.

The point: verify the full machinery on data where we control ground truth.
This is a standalone validation; the real backtest is in 05_backtest_compare.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt  # noqa: E402
from src.models.mdp import value_iteration, policy_iteration, q_function  # noqa: E402
from src.models.qmdp import update_belief, stationary_distribution  # noqa: E402
from src.utils.metrics import summarize  # noqa: E402
from src.utils.utility import expected_utility_per_regime  # noqa: E402

FIG = REPO / "figures"
RES = REPO / "results"
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)

# ----- ground truth ----------------------------------------------------------
RNG = np.random.default_rng(42)
N_MONTHS = 25 * 12   # 25 years
T_TRUE = np.array([           # regime persistence: bull sticky, bear sticky-ish
    [0.97, 0.03],             # bull -> bull, bull -> bear
    [0.10, 0.90],             # bear -> bull, bear -> bear
])
# Per-regime asset return moments (monthly): (mean_stock, mean_bond)
RET_MU = {
    0: np.array([0.010, 0.003]),    # bull: stocks +1%/mo, bonds +0.3%/mo
    1: np.array([-0.015, 0.005]),   # bear: stocks -1.5%/mo, bonds +0.5%/mo (flight-to-quality)
}
RET_COV = {
    0: np.array([[0.0016, 0.00002], [0.00002, 0.0001]]),   # bull: low vol
    1: np.array([[0.0064, -0.00005], [-0.00005, 0.0002]]),  # bear: high vol
}
# Observation emissions (proxy for VIX, term spread, HY OAS), 3-D
OBS_MU = {
    0: np.array([15.0,  1.0,  3.5]),
    1: np.array([35.0, -0.5,  8.0]),
}
OBS_COV = {
    0: np.diag([25.0, 0.5, 1.5]),
    1: np.diag([100.0, 0.9, 6.0]),
}

ACTIONS = np.array([[0.0, 1.0], [0.2, 0.8], [0.4, 0.6], [0.6, 0.4], [0.8, 0.2], [1.0, 0.0]])
LAMBDA = 0.95
TXCOST_BPS = 5
GAMMA_CRRA = 2.0


def simulate_path():
    states = np.zeros(N_MONTHS, dtype=int)
    obs = np.zeros((N_MONTHS, 3))
    rets = np.zeros((N_MONTHS, 2))
    s = 0
    for t in range(N_MONTHS):
        states[t] = s
        obs[t] = RNG.multivariate_normal(OBS_MU[s], OBS_COV[s])
        rets[t] = RNG.multivariate_normal(RET_MU[s], RET_COV[s])
        s = int(RNG.choice(2, p=T_TRUE[s]))
    return states, obs, rets


def build_reward_table(regime_return_samples: dict[int, np.ndarray]) -> np.ndarray:
    return expected_utility_per_regime(regime_return_samples, ACTIONS, gamma=GAMMA_CRRA)


def gauss_pdf(x, mu, cov):
    d = x.shape[0]
    diff = x - mu
    inv = np.linalg.inv(cov + 1e-9 * np.eye(d))
    norm = 1.0 / np.sqrt((2 * np.pi) ** d * max(np.linalg.det(cov), 1e-30))
    return float(norm * np.exp(-0.5 * diff @ inv @ diff))


def run_policy(obs, rets, T, means, covs, Q_star, policy_name):
    b = stationary_distribution(T)
    a_prev = np.array([0.6, 0.4])
    out_r, out_w = [], []
    for t in range(len(obs)):
        like = np.array([gauss_pdf(obs[t], means[s], covs[s]) for s in range(2)])
        b = update_belief(b, T, like)
        if policy_name == "static":
            a = np.array([0.6, 0.4])
        elif policy_name == "qmdp":
            a = ACTIONS[int((b @ Q_star).argmax())]
        elif policy_name == "myopic":
            # one-step-lookahead with assumed regime moments (use the known means here)
            mu_mat = np.vstack([RET_MU[0], RET_MU[1]])  # (S, 2)
            scores = b @ (mu_mat @ ACTIONS.T)
            a = ACTIONS[int(scores.argmax())]
        else:
            raise ValueError(policy_name)
        cost = TXCOST_BPS / 1e4 * np.abs(a - a_prev).sum()
        out_r.append(float(rets[t] @ a) - cost)
        out_w.append(a)
        a_prev = a
    idx = pd.date_range("2000-01-31", periods=len(obs), freq="ME")
    return pd.Series(out_r, index=idx, name=policy_name), pd.DataFrame(out_w, index=idx, columns=["stock", "bond"])


def main():
    print("Simulating synthetic 25-year path from known 2-regime HMM…")
    states, obs, rets = simulate_path()
    print(f"  regime 0 (bull) fraction = {(states == 0).mean():.2f}")
    print(f"  regime 1 (bear) fraction = {(states == 1).mean():.2f}")

    # For the reward MC, use samples drawn from the ground-truth return distributions.
    regime_return_samples = {
        0: RNG.multivariate_normal(RET_MU[0], RET_COV[0], size=5000),
        1: RNG.multivariate_normal(RET_MU[1], RET_COV[1], size=5000),
    }
    R = build_reward_table(regime_return_samples)
    print(f"\nReward R(s, a):\n{R.round(4)}")

    A = ACTIONS.shape[0]
    P = np.repeat(T_TRUE[np.newaxis, :, :], A, axis=0)
    V_vi, pi_vi, n_vi = value_iteration(P, R, lam=LAMBDA)
    V_pi, pi_pi, n_pi = policy_iteration(P, R, lam=LAMBDA)
    assert np.array_equal(pi_vi, pi_pi), "VI and PI disagree!"
    print(f"\nMDP solved: VI {n_vi} iters, PI {n_pi} iters.")
    print(f"  V* = {V_vi.round(4)},  pi* indices = {pi_vi},  → weights = {ACTIONS[pi_vi].tolist()}")
    Q_star = q_function(P, R, V_vi, lam=LAMBDA)

    # Use the ground-truth emission for the belief filter (an "oracle" filter scenario).
    means = [OBS_MU[0], OBS_MU[1]]
    covs = [OBS_COV[0], OBS_COV[1]]

    results, weights_all = {}, {}
    for name in ("static", "qmdp", "myopic"):
        r, w = run_policy(obs, rets, T_TRUE, means, covs, Q_star, name)
        results[name] = r
        weights_all[name] = w

    rows = [summarize(r, weights_all[k], name=k) for k, r in results.items()]
    metrics = pd.DataFrame(rows).round(4)
    print("\n--- Backtest metrics (synthetic, oracle filter) ---")
    print(metrics.to_string(index=False))
    metrics.to_csv(RES / "synthetic_metrics.csv", index=False)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for name, r in results.items():
        ax1.plot(r.index, (1 + r).cumprod(), label=name, lw=1.4)
    ax1.set_yscale("log"); ax1.legend(); ax1.set_ylabel("Cumulative wealth (log)")
    ax1.set_title("Synthetic 25-year backtest — QMDP vs Myopic vs Static 60/40")

    ax2.fill_between(pd.date_range("2000-01-31", periods=len(obs), freq="ME"),
                     0, (states == 1).astype(int), step="post", color="0.85", label="bear regime")
    ax2.set_ylim(-0.05, 1.05); ax2.set_ylabel("bear regime"); ax2.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(FIG / "synthetic_equity_curve.pdf")
    fig.savefig(FIG / "synthetic_equity_curve.png", dpi=160)
    print(f"Wrote {FIG / 'synthetic_equity_curve.pdf'}")


if __name__ == "__main__":
    main()
