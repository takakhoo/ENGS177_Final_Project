"""Experiment 11: Visualize regime timelines under the feature cohorts from Exp 09.

Confirms the multi-feature finding visually: cohort 3 (VIX, T10Y3M, NFCI, STLFSI4)
should produce sharper bear-regime probability spikes at 2008, 2020, 2022 than
cohort 1 (VIX, T10Y3M only). The cohort 3 policy is the one that flipped from
"100/0 in both regimes" to "100/0 bull, 0/100 bear" at gamma=2.

Outputs:
  figures/cohort_regime_timelines.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.models.hmm import fit_hmm  # noqa: E402

DATA = REPO / "data" / "processed"
FIG = REPO / "figures"
TRAIN_END = "2014-12-31"   # match canonical fit window (02 / 09); no look-ahead

COHORTS = {
    "1: VIX + 10Y3M (baseline)":         ["vix", "term_spread"],
    "3: + NFCI + STLFSI4 (+ stress)":    ["vix", "term_spread", "nfci", "stlfsi"],
    "5: kitchen sink (8 channels)":      ["vix", "term_spread", "term_spread_2y", "nfci", "stlfsi",
                                          "umcsent", "jobless_claims", "usd_index"],
}


def main() -> None:
    df = pd.read_csv(DATA / "monthly.csv")
    dc = "DATE" if "DATE" in df.columns else df.columns[0]
    df[dc] = pd.to_datetime(df[dc]); df = df.set_index(dc)

    fig, axes = plt.subplots(len(COHORTS), 1, figsize=(13, 3.0 * len(COHORTS)), sharex=True)

    # NBER recession bands
    nber = df["nber_recession"].fillna(0).astype(float)
    in_rec = nber.diff().fillna(0)
    starts = nber.index[in_rec > 0]
    ends   = nber.index[in_rec < 0]
    if len(starts) > len(ends):
        ends = ends.append(pd.DatetimeIndex([df.index[-1]]))

    for ax_idx, (cohort_name, cols) in enumerate(COHORTS.items()):
        ax = axes[ax_idx]
        avail = [c for c in cols if c in df.columns]
        sub = df[avail].dropna()
        arr = sub.values
        # Standardize with TRAIN-window stats and fit on the train window only
        # (no look-ahead), matching the canonical methodology in 02 / 09.
        n_train = int((sub.index <= pd.Timestamp(TRAIN_END)).sum())
        mu = arr[:n_train].mean(axis=0); sd = arr[:n_train].std(axis=0); sd[sd == 0] = 1
        std = (arr - mu) / sd
        # Fit 2-state on the training window, smooth over the full sample
        model, _ = fit_hmm(std[:n_train], n_states=2, n_restarts=5, seed=ax_idx + 1)
        # Posterior (smoothed) probabilities
        gamma_post = model.predict_proba(std)
        # Reorder so state 0 = bull = highest mean SPY return
        asset_rets = df.loc[sub.index, "spy_ret"]
        state_means = []
        for k in range(2):
            mask = model.predict(std) == k
            state_means.append(asset_rets.iloc[mask].mean() if mask.sum() else 0.0)
        order = sorted(range(2), key=lambda k: state_means[k], reverse=True)
        gamma_post = gamma_post[:, order]

        # Plot bear-regime probability as filled red region
        bear_prob = gamma_post[:, 1]
        ax.fill_between(sub.index, 0, bear_prob, color="#d62728", alpha=0.45,
                        label="P(bear regime)")
        ax.plot(sub.index, bear_prob, color="#a31818", lw=0.8)

        # NBER bands
        for s, e in zip(starts, ends):
            ax.axvspan(s, e, color="gray", alpha=0.25)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("P(bear)")
        ax.set_title(f"Cohort {cohort_name}", loc="left", fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Smoothed P(bear regime) under three observation cohorts\n"
                 "(NBER recessions shaded gray)", y=1.00)
    fig.tight_layout()
    fig.savefig(FIG / "cohort_regime_timelines.pdf")
    fig.savefig(FIG / "cohort_regime_timelines.png", dpi=150)
    print(f"Wrote {FIG / 'cohort_regime_timelines.pdf'}")


if __name__ == "__main__":
    main()
