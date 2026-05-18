"""Step 7 of pipeline: sweep CRRA risk aversion gamma and document at what point
the QMDP policy stops being 'all stocks always' and starts to defend in bears.

Outputs:
  results/gamma_sensitivity.csv         — metrics per (gamma, policy)
  results/gamma_policy_table.csv        — pi*(s) per (gamma, regime)
  figures/gamma_sensitivity_sharpe.pdf  — Sharpe vs gamma per policy
  figures/gamma_equity_curves.pdf       — equity curves at gamma=2, 5, 10
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt  # noqa: E402

from src.models.mdp import value_iteration, policy_iteration, q_function  # noqa: E402
from src.models.qmdp import update_belief, stationary_distribution         # noqa: E402
from src.utils.metrics import summarize                                    # noqa: E402
from src.utils.utility import expected_utility_per_regime                  # noqa: E402

DATA = REPO / "data" / "processed"
RES = REPO / "results"
FIG = REPO / "figures"

LAMBDA = 0.95
TXCOST_BPS = 5
N_SIM_PER_REGIME = 5000
K = 2

GAMMAS = (1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0)
ACTIONS = np.array([
    [0.0, 1.0], [0.2, 0.8], [0.4, 0.6], [0.6, 0.4], [0.8, 0.2], [1.0, 0.0],
])


def _gauss_pdf(x, mu, cov):
    d = x.shape[0]
    diff = x - mu
    inv = np.linalg.inv(cov + 1e-9 * np.eye(d))
    norm = 1.0 / np.sqrt((2 * np.pi) ** d * max(np.linalg.det(cov), 1e-30))
    return float(norm * np.exp(-0.5 * diff @ inv @ diff))


def load_data():
    df = pd.read_csv(DATA / "monthly.csv")
    dc = "DATE" if "DATE" in df.columns else df.columns[0]
    df[dc] = pd.to_datetime(df[dc]); df = df.set_index(dc)
    obs_cols = [c for c in ["vix", "term_spread", "hy_oas"] if c in df.columns]
    return df, obs_cols


def regime_samples(df, model, rng):
    obs_cols = [c for c in ["vix", "term_spread", "hy_oas"] if c in df.columns]
    states = model.predict(df[obs_cols].values)
    out = {}
    for k in range(model.n_components):
        mask = states == k
        if mask.sum() < 2:
            mu = df[["spy_ret", "agg_ret"]].mean().values
            cov = df[["spy_ret", "agg_ret"]].cov().values
        else:
            mu = df.loc[mask, ["spy_ret", "agg_ret"]].mean().values
            cov = df.loc[mask, ["spy_ret", "agg_ret"]].cov().values
        out[k] = rng.multivariate_normal(mu, cov, size=N_SIM_PER_REGIME)
    return out


def run_policy(obs, asset_rets, T, means, covs, Q_star, name):
    n, K = obs.shape[0], T.shape[0]
    b = stationary_distribution(T)
    a_prev = np.array([0.6, 0.4])
    rets, weights = [], []
    for t in range(n):
        like = np.array([_gauss_pdf(obs[t], means[s], covs[s]) for s in range(K)])
        b = update_belief(b, T, like)
        if name == "static":
            a = np.array([0.6, 0.4])
        elif name == "qmdp":
            a = ACTIONS[int((b @ Q_star).argmax())]
        elif name == "myopic":
            mu = asset_rets.iloc[max(0, t - 12):t + 1].mean().values
            a = ACTIONS[int((ACTIONS @ mu).argmax())]
        else:
            raise ValueError(name)
        cost = TXCOST_BPS / 1e4 * np.abs(a - a_prev).sum()
        rets.append(float(asset_rets.iloc[t].values @ a) - cost)
        weights.append(a)
        a_prev = a
    idx = asset_rets.index
    return pd.Series(rets, index=idx, name=name), pd.DataFrame(weights, index=idx, columns=["stock", "bond"])


def main():
    df, obs_cols = load_data()
    with open(DATA / f"hmm_{K}state.pkl", "rb") as f:
        model = pickle.load(f)
    rng = np.random.default_rng(0)
    rrs = regime_samples(df, model, rng)
    T_hmm = model.transmat_
    means = list(model.means_); covs = list(model.covars_)
    obs = df[obs_cols].values
    asset_rets = df[["spy_ret", "agg_ret"]]
    A = ACTIONS.shape[0]
    P = np.repeat(T_hmm[np.newaxis, :, :], A, axis=0)

    metric_rows, policy_rows = [], []
    eq_curves = {}  # gamma -> {policy: cum_return_series}

    for gamma in GAMMAS:
        R = expected_utility_per_regime(rrs, ACTIONS, gamma=gamma)
        V_vi, pi_vi, _ = value_iteration(P, R, lam=LAMBDA)
        V_pi, pi_pi, _ = policy_iteration(P, R, lam=LAMBDA)
        assert np.array_equal(pi_vi, pi_pi)
        Q_star = q_function(P, R, V_vi, lam=LAMBDA)

        for s in range(K):
            policy_rows.append({
                "gamma": gamma, "regime": s,
                "action_idx": int(pi_vi[s]),
                "stock_weight": float(ACTIONS[pi_vi[s], 0]),
                "bond_weight":  float(ACTIONS[pi_vi[s], 1]),
                "V_star": float(V_vi[s]),
            })

        eq_curves[gamma] = {}
        for name in ("static", "qmdp", "myopic"):
            r, w = run_policy(obs, asset_rets, T_hmm, means, covs, Q_star, name)
            row = summarize(r, w, name=name); row["gamma"] = gamma
            metric_rows.append(row)
            eq_curves[gamma][name] = (1 + r).cumprod()

        # Pretty print
        print(f"gamma={gamma:5.2f}  pi*={pi_vi.tolist()}  "
              f"QMDP Sharpe {metric_rows[-2]['sharpe']:.2f}  "
              f"static Sharpe {metric_rows[-3]['sharpe']:.2f}")

    pd.DataFrame(metric_rows).to_csv(RES / "gamma_sensitivity.csv", index=False)
    pd.DataFrame(policy_rows).to_csv(RES / "gamma_policy_table.csv", index=False)
    print(f"\nWrote {RES / 'gamma_sensitivity.csv'} and {RES / 'gamma_policy_table.csv'}")

    # Figure: Sharpe vs gamma
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    df_m = pd.DataFrame(metric_rows)
    for name, c in zip(("static", "qmdp", "myopic"), ("0.4", "C0", "C2")):
        sub = df_m[df_m["name"] == name].sort_values("gamma")
        ax.plot(sub["gamma"], sub["sharpe"], marker="o", color=c, label=name, lw=1.8)
    ax.set_xlabel(r"CRRA risk aversion $\gamma$")
    ax.set_ylabel("Annualized Sharpe ratio")
    ax.set_title("QMDP policy unlocks at higher risk aversion")
    ax.axhline(0, color="0", lw=0.4); ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "gamma_sensitivity_sharpe.pdf")
    fig.savefig(FIG / "gamma_sensitivity_sharpe.png", dpi=150)
    plt.close(fig)

    # Figure: equity curves at gamma = 2, 5, 10, 20
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), sharex=True, sharey=True)
    for ax, gamma in zip(axes.flat, (2.0, 5.0, 10.0, 20.0)):
        for name, c in zip(("static", "qmdp", "myopic"), ("0.4", "C0", "C2")):
            cum = eq_curves[gamma][name]
            ax.plot(cum.index, cum.values, color=c, lw=1.4, label=name)
        ax.set_yscale("log")
        ax.set_title(f"$\\gamma = {gamma}$")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("Backtest equity curves under varying CRRA risk aversion", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "gamma_equity_curves.pdf")
    fig.savefig(FIG / "gamma_equity_curves.png", dpi=150)
    plt.close(fig)

    print(f"Wrote {FIG / 'gamma_sensitivity_sharpe.pdf'}")
    print(f"Wrote {FIG / 'gamma_equity_curves.pdf'}")


if __name__ == "__main__":
    main()
