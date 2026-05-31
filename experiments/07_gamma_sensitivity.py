"""Step 7 of pipeline: sweep CRRA risk aversion gamma and document at what point
the QMDP policy stops being 'all stocks always' and starts to defend in bears.

Outputs:
  results/gamma_sensitivity.csv        , metrics per (gamma, policy)
  results/gamma_policy_table.csv       , pi*(s) per (gamma, regime)
  figures/gamma_sensitivity_sharpe.pdf , Sharpe vs gamma per policy
  figures/gamma_equity_curves.pdf      , equity curves at gamma=2, 5, 10
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

from src.utils.plotting import apply_style, PALETTE, annotate_event  # noqa: E402
apply_style()

from src.models.mdp import value_iteration, policy_iteration, q_function  # noqa: E402
from src.models.qmdp import update_belief, stationary_distribution         # noqa: E402
from src.models.hmm import standardized_obs                                # noqa: E402
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
    states = model.predict(standardized_obs(df, obs_cols))
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
    obs = standardized_obs(df, obs_cols)   # belief filter must match fit-time space
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

    label_map = {
        "static": "Static 60/40 (benchmark)",
        "qmdp":   "QMDP (regime-aware)",
        "myopic": "Myopic 12-mo trend",
    }
    for name in ("static", "qmdp", "myopic"):
        sub = df_m[df_m["name"] == name].sort_values("gamma")
        ax.plot(sub["gamma"], sub["sharpe"], marker="o", markersize=6,
                color=PALETTE[name], label=label_map[name], lw=2.0)

    # Reference lines
    static_sharpe = df_m[df_m["name"] == "static"]["sharpe"].iloc[0]
    ax.axhline(static_sharpe, color=PALETTE["static"], lw=0.7, ls=":", alpha=0.6)

    # Find crossover: smallest gamma where QMDP Sharpe >= static Sharpe
    qmdp_df = df_m[df_m["name"] == "qmdp"].sort_values("gamma")
    cross = qmdp_df[qmdp_df["sharpe"] >= static_sharpe]["gamma"]
    if len(cross):
        gx = float(cross.iloc[0])
        sx = float(qmdp_df[qmdp_df["gamma"] == gx]["sharpe"].iloc[0])
        ax.axvline(gx, color=PALETTE["highlight"], lw=0.8, ls="--", alpha=0.7)
        ax.annotate(
            f"QMDP unlocks at $\\gamma={gx:.0f}$\n(Sharpe {sx:.2f} > {static_sharpe:.2f})",
            xy=(gx, sx), xytext=(40, 30),
            textcoords="offset points", fontsize=10, color=PALETTE["highlight"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["highlight"], lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PALETTE["highlight"], lw=0.6),
        )

    ax.set_xlabel(r"CRRA risk aversion $\gamma$  (higher = more conservative)")
    ax.set_ylabel("Annualised Sharpe ratio (2003–2026)")
    ax.set_title(
        "QMDP Sharpe vs CRRA risk aversion\n"
        "Crossover at $\\gamma\\approx 8$, but $\\pi^\\ast(\\mathrm{bull})=\\pi^\\ast(\\mathrm{bear})$ at every $\\gamma$"
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(0.3, 1.4)
    fig.tight_layout()
    fig.savefig(FIG / "gamma_sensitivity_sharpe.pdf")
    fig.savefig(FIG / "gamma_sensitivity_sharpe.png")
    plt.close(fig)

    # Figure: equity curves at gamma = 2, 5, 10, 20
    label_map = {
        "static": "Static 60/40",
        "qmdp":   "QMDP",
        "myopic": "Myopic 12-mo",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharex=True, sharey=True)
    for ax, gamma in zip(axes.flat, (2.0, 5.0, 10.0, 20.0)):
        for name in ("static", "qmdp", "myopic"):
            cum = eq_curves[gamma][name]
            ax.plot(cum.index, cum.values, color=PALETTE[name], lw=1.7,
                    label=label_map[name])
            # Terminal value annotation on the right
            ax.annotate(
                f"${cum.iloc[-1]:.1f}$", xy=(cum.index[-1], cum.iloc[-1]),
                xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=8.5, color=PALETTE[name], weight="bold",
            )
        # Identify the policy for QMDP at this gamma to label panel
        policy_row = pd.DataFrame(policy_rows)
        policy_row = policy_row[policy_row["gamma"] == gamma].iloc[0]
        sw = int(round(policy_row["stock_weight"] * 100))
        bw = int(round(policy_row["bond_weight"] * 100))
        ax.set_title(f"$\\gamma={gamma:.0f}$  →  QMDP policy: {sw}/{bw} stocks/bonds")
        ax.set_yscale("log")
        ax.legend(loc="upper left", fontsize=9)
    for ax in axes[1, :]:
        ax.set_xlabel("Date")
    for ax in axes[:, 0]:
        ax.set_ylabel("Cumulative wealth (log scale)")
    fig.suptitle("Equity curves under varying CRRA risk aversion $\\gamma$ "
                 ",  higher $\\gamma$ shifts QMDP toward bonds across-the-board",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(FIG / "gamma_equity_curves.pdf")
    fig.savefig(FIG / "gamma_equity_curves.png")
    plt.close(fig)

    print(f"Wrote {FIG / 'gamma_sensitivity_sharpe.pdf'}")
    print(f"Wrote {FIG / 'gamma_equity_curves.pdf'}")


if __name__ == "__main__":
    main()
