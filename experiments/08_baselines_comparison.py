"""Experiment 08: All-baselines horse race on the headline 2003-2026 panel.

Runs every strategy in src/models/baselines.py PLUS QMDP from the existing
pipeline against SPY/AGG monthly log returns, applying the same 5 bps
transaction cost. Produces a comparison table and an equity-curve figure
suitable for the extended report.

Outputs:
  results/baselines_metrics.csv     — full metrics table (Sharpe, Sortino, Omega, etc.)
  results/baselines_weights.csv     — month-end weights per strategy (stacked)
  figures/baselines_equity.{pdf,png}  — log-scale equity curves
  figures/baselines_drawdown.{pdf,png} — drawdown over time
  figures/baselines_sharpe_bar.{pdf,png} — bar chart of headline metrics

Note: HMM-conditional MV and Black-Litterman with HMM views require the
fitted HMM and per-date belief. We compute belief on the fly using the
already-saved 2-state HMM in data/processed/.
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

from src.models.baselines import (  # noqa: E402
    static_60_40, equal_weight, inverse_volatility, risk_parity_2asset,
    mean_variance_lw, faber_10mo_sma, ts_momentum_12mo, vol_target_60_40,
    myopic_12mo, hmm_conditional_mv, black_litterman_hmm_views,
    backtest_from_weights,
)
from src.models.qmdp import update_belief, stationary_distribution  # noqa: E402
from src.utils.metrics import summarize, max_drawdown  # noqa: E402

DATA = REPO / "data" / "processed"
RES = REPO / "results"
FIG = REPO / "figures"
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

N_STATES = 2
LAMBDA = 0.95
TXCOST_BPS = 5.0
GAMMA = 2.0


def _gauss_pdf(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    d = x.shape[0]
    diff = x - mu
    inv = np.linalg.inv(cov + 1e-9 * np.eye(d))
    norm = 1.0 / np.sqrt((2 * np.pi) ** d * max(np.linalg.det(cov), 1e-30))
    return float(norm * np.exp(-0.5 * diff @ inv @ diff))


def compute_belief_series(obs: np.ndarray, model) -> pd.DataFrame:
    """Forward-pass Bayesian filter producing per-date belief vector."""
    T_hmm = model.transmat_
    means = list(model.means_)
    covs = list(model.covars_)
    belief = stationary_distribution(T_hmm)
    out = []
    for t in range(obs.shape[0]):
        likelihood = np.array([_gauss_pdf(obs[t], means[s], covs[s]) for s in range(N_STATES)])
        belief = update_belief(belief, T_hmm, likelihood)
        out.append(belief.copy())
    return pd.DataFrame(out)


def qmdp_weights(asset_rets: pd.DataFrame, belief: pd.DataFrame,
                 Q_star: np.ndarray, action_grid: np.ndarray) -> pd.DataFrame:
    """QMDP weights from belief and stored Q*. Drop-in matches the baselines API."""
    weights = []
    cols = asset_rets.columns
    for i in range(len(asset_rets)):
        b = belief.iloc[i].values
        scores = b @ Q_star
        a = action_grid[int(scores.argmax())]
        weights.append(a)
    return pd.DataFrame(weights, index=asset_rets.index, columns=cols)


def main() -> None:
    # ---- Load data + HMM + Q* ----
    df = pd.read_csv(DATA / "monthly.csv")
    dc = "DATE" if "DATE" in df.columns else df.columns[0]
    df[dc] = pd.to_datetime(df[dc]); df = df.set_index(dc)
    with open(DATA / f"hmm_{N_STATES}state.pkl", "rb") as f:
        model = pickle.load(f)

    asset_rets = df[["spy_ret", "agg_ret"]].copy()

    # Belief series for HMM-aware baselines
    obs_cols = ["vix", "term_spread"]
    obs = df[obs_cols].values
    belief = compute_belief_series(obs, model)
    belief.index = df.index

    regime_means_obs = list(model.means_)  # (K,) of shape (d_obs,) — emissions, not returns
    regime_covs_obs = list(model.covars_)

    # Build regime-conditional asset return moments
    # Hard-assign each month to MAP regime, then compute conditional mean+cov of (spy_ret, agg_ret)
    map_state = belief.values.argmax(axis=1)
    regime_means_rets, regime_covs_rets = [], []
    for k in range(N_STATES):
        mask = map_state == k
        if mask.sum() < 5:
            regime_means_rets.append(asset_rets.mean().values)
            regime_covs_rets.append(np.cov(asset_rets.values.T))
        else:
            regime_means_rets.append(asset_rets.iloc[mask].mean().values)
            regime_covs_rets.append(np.cov(asset_rets.iloc[mask].values.T))

    # Action grid for QMDP and myopic (same as 05_backtest)
    action_grid = np.array([
        [0.0, 1.0], [0.2, 0.8], [0.4, 0.6],
        [0.6, 0.4], [0.8, 0.2], [1.0, 0.0],
    ])
    Q_star = np.load(RES / "Q_star.npy")

    # ---- Compute weights per strategy ----
    strategies = {
        "static_60_40":      static_60_40(asset_rets),
        "equal_weight":      equal_weight(asset_rets),
        "inverse_vol":       inverse_volatility(asset_rets, lookback=36),
        "risk_parity":       risk_parity_2asset(asset_rets, lookback=36),
        "mean_variance_lw":  mean_variance_lw(asset_rets, lookback=60, risk_aversion=2.0),
        "faber_10mo_sma":    faber_10mo_sma(asset_rets, lookback=10),
        "ts_momentum_12mo":  ts_momentum_12mo(asset_rets, lookback=12),
        "vol_target_60_40":  vol_target_60_40(asset_rets, target_vol=0.10, lookback=36),
        "myopic_12mo":       myopic_12mo(asset_rets, action_grid=action_grid, lookback=12),
        "hmm_conditional_mv": hmm_conditional_mv(
            asset_rets, belief, regime_means_rets, regime_covs_rets,
            risk_aversion=GAMMA, long_only=True,
        ),
        "black_litterman_hmm": black_litterman_hmm_views(
            asset_rets, belief, regime_means_rets,
            market_weights=np.array([0.6, 0.4]),
            risk_aversion=2.5, tau=0.05, view_confidence=0.3, lookback=60,
        ),
        "qmdp":              qmdp_weights(asset_rets, belief, Q_star, action_grid),
    }

    # ---- Backtest each ----
    bench = backtest_from_weights(strategies["static_60_40"], asset_rets, TXCOST_BPS)
    rets = {}
    weights_combined = []
    for name, w in strategies.items():
        r = backtest_from_weights(w, asset_rets, TXCOST_BPS)
        r.name = name
        rets[name] = r
        weights_combined.append(w.assign(strategy=name))

    # Combine weights into long-form for CSV
    weights_long = pd.concat([w.reset_index().assign(strategy=name)
                              for name, w in strategies.items()],
                             ignore_index=True)
    weights_long.to_csv(RES / "baselines_weights.csv", index=False)

    # ---- Metrics ----
    rows = []
    for name, r in rets.items():
        m = summarize(r, weights=strategies[name], benchmark=bench, name=name)
        rows.append(m)
    metrics = pd.DataFrame(rows).set_index("name").round(4)
    metrics.to_csv(RES / "baselines_metrics.csv")
    print("\n=== ALL-BASELINES BACKTEST (2003-2026, 5 bps cost, monthly) ===")
    print(metrics[["cagr", "vol", "sharpe", "sortino", "max_drawdown", "calmar", "turnover_avg"]])

    # Save concatenated return series for figures
    returns_df = pd.concat(rets, axis=1)
    returns_df.to_csv(RES / "baselines_returns.csv")

    # ---- Equity-curve figure ----
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for name, r in rets.items():
        cum = (1 + r).cumprod()
        ax.plot(cum.index, cum.values, lw=1.4, label=name)
    ax.set_yscale("log")
    ax.set_ylabel("Cumulative wealth (log scale), $1 → $X")
    ax.set_title("All-baselines backtest 2003–2026 — SPY/AGG monthly, 5 bps cost")
    ax.legend(loc="best", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "baselines_equity.pdf")
    fig.savefig(FIG / "baselines_equity.png", dpi=150)
    plt.close(fig)

    # ---- Drawdown figure ----
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for name, r in rets.items():
        cum = (1 + r).cumprod()
        rm = cum.cummax()
        dd = cum / rm - 1.0
        ax.plot(dd.index, dd.values * 100, lw=1.2, label=name)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown timeseries — all baselines")
    ax.legend(loc="best", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "baselines_drawdown.pdf")
    fig.savefig(FIG / "baselines_drawdown.png", dpi=150)
    plt.close(fig)

    # ---- Sharpe bar chart ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    sorted_metrics = metrics.sort_values("sharpe", ascending=False)
    colors = ["#1f77b4" if n == "qmdp" else "#777" for n in sorted_metrics.index]
    sorted_metrics["sharpe"].plot.bar(ax=axes[0, 0], color=colors)
    axes[0, 0].set_title("Sharpe ratio (annualised)"); axes[0, 0].axhline(0, color="k", lw=0.5)
    sorted_metrics.sort_values("sortino", ascending=False)["sortino"].plot.bar(
        ax=axes[0, 1], color="#888"); axes[0, 1].set_title("Sortino ratio")
    sorted_metrics.sort_values("calmar", ascending=False)["calmar"].plot.bar(
        ax=axes[1, 0], color="#888"); axes[1, 0].set_title("Calmar ratio")
    sorted_metrics.sort_values("max_drawdown", ascending=True)["max_drawdown"].plot.bar(
        ax=axes[1, 1], color="#a33"); axes[1, 1].set_title("Max drawdown (more-negative is worse)")
    for ax in axes.ravel():
        ax.tick_params(axis="x", labelrotation=45)
    fig.suptitle("Risk-adjusted performance — all baselines", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "baselines_sharpe_bar.pdf")
    fig.savefig(FIG / "baselines_sharpe_bar.png", dpi=150)
    plt.close(fig)

    print(f"\nWrote:\n  {RES / 'baselines_metrics.csv'}\n  {RES / 'baselines_returns.csv'}\n"
          f"  {FIG / 'baselines_equity.pdf'}\n  {FIG / 'baselines_drawdown.pdf'}\n"
          f"  {FIG / 'baselines_sharpe_bar.pdf'}")


if __name__ == "__main__":
    main()
