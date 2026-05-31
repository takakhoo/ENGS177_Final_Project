"""Experiment 12: Subperiod-robustness check.

Split the 2003-2026 sample into three sub-periods and re-run the headline
horse race on each. This tests whether the rankings hold across regimes of
the broader macro environment:

  Period A: 2003-10 to 2010-12  (pre-Fed-QE era; includes GFC)
  Period B: 2011-01 to 2019-12  (post-GFC + ZIRP era)
  Period C: 2020-01 to 2026-05  (COVID + post-pandemic inflation era)

Outputs:
  results/subperiod_metrics.csv
  figures/subperiod_sharpe_heatmap.{pdf,png}
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.models.baselines import (  # noqa: E402
    static_60_40, equal_weight, inverse_volatility, risk_parity_2asset,
    mean_variance_lw, faber_10mo_sma, ts_momentum_12mo, vol_target_60_40,
    myopic_12mo, hmm_conditional_mv, black_litterman_hmm_views,
    backtest_from_weights,
)
from src.models.qmdp import update_belief, stationary_distribution  # noqa: E402
from src.models.hmm import standardized_obs  # noqa: E402
from src.utils.metrics import summarize  # noqa: E402

DATA = REPO / "data" / "processed"
RES = REPO / "results"
FIG = REPO / "figures"

PERIODS = {
    "A_2003_2010": ("2003-10-31", "2010-12-31"),
    "B_2011_2019": ("2011-01-31", "2019-12-31"),
    "C_2020_2026": ("2020-01-31", "2026-12-31"),
}

ACTIONS = np.array([
    [0.0, 1.0], [0.2, 0.8], [0.4, 0.6], [0.6, 0.4], [0.8, 0.2], [1.0, 0.0],
])
N_STATES = 2
TXCOST_BPS = 5.0


def _gauss_pdf(x, mu, cov):
    d = x.shape[0]; diff = x - mu
    inv = np.linalg.inv(cov + 1e-9 * np.eye(d))
    norm = 1.0 / np.sqrt((2 * np.pi) ** d * max(np.linalg.det(cov), 1e-30))
    return float(norm * np.exp(-0.5 * diff @ inv @ diff))


def compute_belief(obs, model):
    T_hmm = model.transmat_; means = list(model.means_); covs = list(model.covars_)
    b = stationary_distribution(T_hmm); out = []
    for t in range(obs.shape[0]):
        L = np.array([_gauss_pdf(obs[t], means[s], covs[s]) for s in range(N_STATES)])
        b = update_belief(b, T_hmm, L); out.append(b.copy())
    return pd.DataFrame(out)


def main():
    df = pd.read_csv(DATA / "monthly.csv")
    dc = "DATE" if "DATE" in df.columns else df.columns[0]
    df[dc] = pd.to_datetime(df[dc]); df = df.set_index(dc)
    with open(DATA / f"hmm_{N_STATES}state.pkl", "rb") as f:
        model = pickle.load(f)

    asset_rets_full = df[["spy_ret", "agg_ret"]]
    obs_cols = ["vix", "term_spread"]
    obs_full = standardized_obs(df, obs_cols)  # match fit-time z-scored space
    belief_full = compute_belief(obs_full, model); belief_full.index = df.index

    map_state = belief_full.values.argmax(axis=1)
    regime_means_rets, regime_covs_rets = [], []
    for k in range(N_STATES):
        mask = map_state == k
        if mask.sum() < 5:
            regime_means_rets.append(asset_rets_full.mean().values)
            regime_covs_rets.append(np.cov(asset_rets_full.values.T))
        else:
            regime_means_rets.append(asset_rets_full.iloc[mask].mean().values)
            regime_covs_rets.append(np.cov(asset_rets_full.iloc[mask].values.T))

    Q_star = np.load(RES / "Q_star.npy")

    all_rows = []
    for period_name, (start, end) in PERIODS.items():
        sub = df.loc[start:end]
        if len(sub) < 24:
            print(f"  [skip] {period_name}: only {len(sub)} months")
            continue
        a_rets = sub[["spy_ret", "agg_ret"]]
        b_sub = belief_full.loc[sub.index]

        strategies = {
            "static_60_40":      static_60_40(a_rets),
            "equal_weight":      equal_weight(a_rets),
            "inverse_vol":       inverse_volatility(a_rets, lookback=36),
            "risk_parity":       risk_parity_2asset(a_rets, lookback=36),
            "mean_variance_lw":  mean_variance_lw(a_rets, lookback=60),
            "faber_10mo_sma":    faber_10mo_sma(a_rets, lookback=10),
            "ts_momentum_12mo":  ts_momentum_12mo(a_rets, lookback=12),
            "vol_target_60_40":  vol_target_60_40(a_rets, target_vol=0.10, lookback=36),
            "myopic_12mo":       myopic_12mo(a_rets, action_grid=ACTIONS, lookback=12),
            "hmm_conditional_mv": hmm_conditional_mv(
                a_rets, b_sub, regime_means_rets, regime_covs_rets,
                risk_aversion=2.0, long_only=True,
            ),
            "black_litterman_hmm": black_litterman_hmm_views(
                a_rets, b_sub, regime_means_rets,
                market_weights=np.array([0.6, 0.4]),
                risk_aversion=2.5, tau=0.05, view_confidence=0.3, lookback=60,
            ),
            "qmdp": pd.DataFrame(
                [ACTIONS[int((b @ Q_star).argmax())] for b in b_sub.values],
                index=a_rets.index, columns=a_rets.columns,
            ),
        }

        for name, w in strategies.items():
            r = backtest_from_weights(w, a_rets, TXCOST_BPS)
            m = summarize(r, weights=w, name=name)
            m["period"] = period_name; m["n_months"] = len(a_rets)
            all_rows.append(m)

    out = pd.DataFrame(all_rows)
    out.to_csv(RES / "subperiod_metrics.csv", index=False)

    # Pivot for heatmap: rows = strategy, columns = period, values = Sharpe
    sharpe_pivot = out.pivot(index="name", columns="period", values="sharpe")
    # Sort by full-sample Sharpe for visual ordering, use period A as proxy if absent
    sharpe_pivot = sharpe_pivot.sort_values(sharpe_pivot.columns[-1], ascending=False)

    print("\n=== SUBPERIOD SHARPE TABLE ===")
    print(sharpe_pivot.round(2))

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(sharpe_pivot.values, cmap="RdYlGn", vmin=-0.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(len(sharpe_pivot.columns))); ax.set_xticklabels(sharpe_pivot.columns)
    ax.set_yticks(range(len(sharpe_pivot.index))); ax.set_yticklabels(sharpe_pivot.index)
    for i in range(sharpe_pivot.shape[0]):
        for j in range(sharpe_pivot.shape[1]):
            ax.text(j, i, f"{sharpe_pivot.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=10,
                    color="white" if abs(sharpe_pivot.values[i, j]) > 1.5 else "black")
    fig.colorbar(im, ax=ax, label="Sharpe ratio (annualised)")
    ax.set_title("Subperiod Sharpe heatmap, robustness across 2003-2010, 2011-2019, 2020-2026")
    fig.tight_layout()
    fig.savefig(FIG / "subperiod_sharpe_heatmap.pdf")
    fig.savefig(FIG / "subperiod_sharpe_heatmap.png", dpi=150)
    print(f"\nWrote {FIG / 'subperiod_sharpe_heatmap.pdf'}")


if __name__ == "__main__":
    main()
