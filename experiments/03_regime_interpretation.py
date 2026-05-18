"""Step 3 of pipeline: interpret HMM regimes, plot timeline vs NBER, save tables."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt  # noqa: E402
from src.models.hmm import standardize_with_train_stats  # noqa: E402

DATA = REPO / "data" / "processed"
FIG = REPO / "figures"
RES = REPO / "results"
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)

OBS_CANDIDATES = ["vix", "term_spread", "hy_oas"]
N_STATES = 2  # default to two-state baseline; change to 3 for sensitivity run
TRAIN_END = "2014-12-31"


def main() -> None:
    df = pd.read_csv(DATA / "monthly.csv")
    date_col = "DATE" if "DATE" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    OBS_COLS = [c for c in OBS_CANDIDATES if c in df.columns]
    train = df.loc[:TRAIN_END]
    _, obs = standardize_with_train_stats(train, df, OBS_COLS)
    with open(DATA / f"hmm_{N_STATES}state.pkl", "rb") as f:
        model = pickle.load(f)

    # Smoothed posterior P(s_t | o_{1:T})
    gamma = model.predict_proba(obs)  # (T, K)
    states = model.predict(obs)
    df = df.assign(state=states, **{f"p_state{k}": gamma[:, k] for k in range(N_STATES)})

    # Identify which state is "bear" by lowest mean SPY return in-sample
    mean_spy_by_state = df.loc[:TRAIN_END].groupby("state")["spy_ret"].mean()
    bear_state = int(mean_spy_by_state.idxmin())
    print(f"Bear state inferred = {bear_state} (mean SPY ret = {mean_spy_by_state[bear_state]:.4f})")

    # Save regime returns table
    summary = (
        df.loc[:TRAIN_END]
        .groupby("state")
        .agg(
            n_months=("spy_ret", "size"),
            mean_spy=("spy_ret", "mean"),
            std_spy=("spy_ret", "std"),
            mean_agg=("agg_ret", "mean"),
            std_agg=("agg_ret", "std"),
            mean_vix=("vix", "mean"),
        )
        .round(4)
    )
    summary.to_csv(RES / "regime_returns.csv")
    print(summary)

    # Plot regime probability vs NBER
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(df.index, 0, df["nber_recession"], color="0.85", step="post",
                    label="NBER recession")
    ax.plot(df.index, df[f"p_state{bear_state}"], lw=1.2, color="C3", label=f"P(bear={bear_state})")
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("Probability")
    ax.set_title(f"Filtered probability of bear regime vs NBER recession ({N_STATES}-state HMM)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "regime_timeline.pdf")
    print(f"Wrote {FIG / 'regime_timeline.pdf'}")


if __name__ == "__main__":
    main()
