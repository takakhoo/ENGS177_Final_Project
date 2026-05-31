"""Step 5 of pipeline: the recursively-filtered belief over the latent regime.

This is the artifact that demonstrates the belief state is genuine: starting from the
stationary prior, we run the two-step Bayesian filter (predict then update) forward over
all 271 months and plot P(bear regime)_t. It also prints one concrete month-to-month
update with the predict/update intermediates, used as the worked example in the report.

Outputs:
  figures/belief_trajectory.{pdf,png}
  results/belief_series.csv
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

from src.models.qmdp import update_belief, stationary_distribution  # noqa: E402
from src.models.hmm import standardized_obs  # noqa: E402

DATA = REPO / "data" / "processed"
RES = REPO / "results"
FIG = REPO / "figures"
N_STATES = 2
OBS_COLS = ["vix", "term_spread"]


def gauss_pdf(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    d = x.shape[0]
    diff = x - mu
    inv = np.linalg.inv(cov + 1e-9 * np.eye(d))
    norm = 1.0 / np.sqrt((2 * np.pi) ** d * max(np.linalg.det(cov), 1e-30))
    return float(norm * np.exp(-0.5 * diff @ inv @ diff))


def main() -> None:
    df = pd.read_csv(DATA / "monthly.csv")
    dc = "DATE" if "DATE" in df.columns else df.columns[0]
    df[dc] = pd.to_datetime(df[dc]); df = df.set_index(dc)
    with open(DATA / f"hmm_{N_STATES}state.pkl", "rb") as f:
        model = pickle.load(f)

    obs = standardized_obs(df, OBS_COLS)                  # z-scored to match fit-time space
    T = model.transmat_
    means = list(model.means_); covs = list(model.covars_)

    # Identify which state is "bear" = lower mean SPY return under the MAP assignment.
    states = model.predict(obs)
    bear = int(np.argmin([df.loc[states == s, "spy_ret"].mean() for s in range(N_STATES)]))

    # Recursive Bayesian filter, forward over all months.
    b = stationary_distribution(T)
    beliefs = []
    example_rows = []
    for t in range(len(obs)):
        predictive = b @ T                                # predict step
        like = np.array([gauss_pdf(obs[t], means[s], covs[s]) for s in range(N_STATES)])
        post = like * predictive
        post = post / post.sum()                          # update step
        beliefs.append(post.copy())
        # Capture the Aug-2015 (China/oil shock) update as the worked example: a clean
        # one-step flip from near-certain bull to near-certain bear on a VIX spike.
        if df.index[t].strftime("%Y-%m") == "2015-08":
            example_rows.append((t, df.index[t], b.copy(), predictive.copy(),
                                 like.copy(), post.copy(), df["vix"].iloc[t]))
        b = post

    B = np.array(beliefs)
    pbear = B[:, bear]
    out = pd.DataFrame({"P_bear": pbear}, index=df.index)
    out.to_csv(RES / "belief_series.csv")

    # ---- Worked example print (real numbers for the report) ----
    if example_rows:
        t, dt, prior, pred, like, post, vix = example_rows[0]
        print("=== Worked belief update (real numbers) ===")
        print(f"Month: {dt.date()}  (VIX = {vix:.1f})")
        print(f"  prior      b_(t-1) = (bull {prior[1-bear]:.3f}, bear {prior[bear]:.3f})")
        print(f"  predict    b_hat   = (bull {pred[1-bear]:.3f}, bear {pred[bear]:.3f})")
        print(f"  likelihood ratio   = O(o|bear)/O(o|bull) = {like[bear]/max(like[1-bear],1e-30):.2f}")
        print(f"  posterior  b_t      = (bull {post[1-bear]:.3f}, bear {post[bear]:.3f})")

    # ---- Figure ----
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.fill_between(df.index, 0, df["nber_recession"], step="post",
                    color="0.80", alpha=0.9, label="NBER recession", zorder=1)
    ax.fill_between(df.index, 0, pbear, color="#d62728", alpha=0.40,
                    label=r"$b_t(\mathrm{bear})$, recursive Bayes filter", zorder=3)
    ax.plot(df.index, pbear, color="#a31818", lw=1.3, zorder=4)
    for date_str, lab in [("2008-10-31", "GFC"), ("2015-09-30", "China/oil"),
                          ("2020-04-30", "COVID"), ("2022-09-30", "Inflation")]:
        t = pd.Timestamp(date_str)
        if df.index[0] <= t <= df.index[-1]:
            ax.axvline(t, color="0.3", lw=0.6, ls=":", alpha=0.6, zorder=2)
            ax.text(t, 1.04, lab, rotation=0, fontsize=8, ha="center", color="0.3")
    ax.set_ylim(-0.02, 1.08); ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel(r"$P(\mathrm{bear}\mid \mathbf{o}_{1:t})$")
    ax.set_xlabel("Date")
    ax.set_title("Recursively-filtered belief over the latent regime (canonical 2-state HMM, "
                 "VIX + term spread)", fontsize=10)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG / "belief_trajectory.pdf")
    fig.savefig(FIG / "belief_trajectory.png", dpi=150)
    plt.close(fig)
    print(f"\nWrote {FIG / 'belief_trajectory.pdf'} and {RES / 'belief_series.csv'}")
    print(f"P(bear): mean={pbear.mean():.3f}, frac>0.5={np.mean(pbear>0.5):.2f}, "
          f"max={pbear.max():.3f}")


if __name__ == "__main__":
    main()
