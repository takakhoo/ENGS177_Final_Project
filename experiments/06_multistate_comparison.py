"""Step 6 of pipeline: side-by-side comparison of K=2, K=3, K=4 regime models.

For each K we:
  1. Load the pre-fitted HMM (from experiments/02_hmm_calibration.py).
  2. Build R(s, a) by Monte-Carlo over regime-conditional asset returns.
  3. Solve underlying MDP via VI + PI (cross-checked).
  4. Run QMDP backtest against static 60/40 and myopic baselines.
  5. Aggregate metrics + plot regime timelines and equity curves.

Outputs:
  results/multistate_metrics.csv    , one row per (K, policy)
  results/multistate_policies.csv   , per-regime greedy actions for each K
  figures/multistate_regime_timeline.pdf, 3-panel plot
  figures/multistate_equity_curves.pdf  , overlay of all backtests
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
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)

LAMBDA = 0.95
TXCOST_BPS = 5
GAMMA_CRRA = 2.0
N_SIM_PER_REGIME = 5000

ACTIONS = np.array([
    [0.0, 1.0], [0.2, 0.8], [0.4, 0.6], [0.6, 0.4], [0.8, 0.2], [1.0, 0.0],
])


def load_data():
    df = pd.read_csv(DATA / "monthly.csv")
    dc = "DATE" if "DATE" in df.columns else df.columns[0]
    df[dc] = pd.to_datetime(df[dc]); df = df.set_index(dc)
    obs_cols = [c for c in ["vix", "term_spread", "hy_oas"] if c in df.columns]
    return df, obs_cols


def regime_return_samples(df: pd.DataFrame, model, rng: np.random.Generator) -> dict[int, np.ndarray]:
    obs_cols = [c for c in ["vix", "term_spread", "hy_oas"] if c in df.columns]
    states = model.predict(standardized_obs(df, obs_cols))
    out: dict[int, np.ndarray] = {}
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


def _gauss_pdf(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    d = x.shape[0]
    diff = x - mu
    inv = np.linalg.inv(cov + 1e-9 * np.eye(d))
    norm = 1.0 / np.sqrt((2 * np.pi) ** d * max(np.linalg.det(cov), 1e-30))
    return float(norm * np.exp(-0.5 * diff @ inv @ diff))


def run_policy(obs, asset_rets, T, means, covs, Q_star, policy_name):
    n = obs.shape[0]
    b = stationary_distribution(T)
    a_prev = np.array([0.6, 0.4])
    rets, weights = [], []
    K = T.shape[0]
    for t in range(n):
        like = np.array([_gauss_pdf(obs[t], means[s], covs[s]) for s in range(K)])
        b = update_belief(b, T, like)
        if policy_name == "static":
            a = np.array([0.6, 0.4])
        elif policy_name == "qmdp":
            a = ACTIONS[int((b @ Q_star).argmax())]
        elif policy_name == "myopic":
            mu_assets = asset_rets.iloc[max(0, t - 12):t + 1].mean().values
            scores = ACTIONS @ mu_assets
            a = ACTIONS[int(scores.argmax())]
        else:
            raise ValueError(policy_name)
        cost = TXCOST_BPS / 1e4 * np.abs(a - a_prev).sum()
        rets.append(float(asset_rets.iloc[t].values @ a) - cost)
        weights.append(a)
        a_prev = a
    idx = asset_rets.index
    return pd.Series(rets, index=idx, name=policy_name), pd.DataFrame(
        weights, index=idx, columns=["stock", "bond"]
    )


def run_for_K(df: pd.DataFrame, K: int) -> dict:
    """Full pipeline for one K. Returns dict with policies, metrics, etc."""
    obs_cols = [c for c in ["vix", "term_spread", "hy_oas"] if c in df.columns]
    obs = standardized_obs(df, obs_cols)   # belief filter must match fit-time space
    asset_rets = df[["spy_ret", "agg_ret"]]

    with open(DATA / f"hmm_{K}state.pkl", "rb") as f:
        model = pickle.load(f)

    rng = np.random.default_rng(0)
    rrs = regime_return_samples(df, model, rng)
    R = expected_utility_per_regime(rrs, ACTIONS, gamma=GAMMA_CRRA)

    T_hmm = model.transmat_
    A = ACTIONS.shape[0]
    P = np.repeat(T_hmm[np.newaxis, :, :], A, axis=0)

    V_vi, pi_vi, n_vi = value_iteration(P, R, lam=LAMBDA)
    V_pi, pi_pi, n_pi = policy_iteration(P, R, lam=LAMBDA)
    assert np.array_equal(pi_vi, pi_pi)
    Q_star = q_function(P, R, V_vi, lam=LAMBDA)

    means = list(model.means_); covs = list(model.covars_)

    rets, weights = {}, {}
    for name in ("static", "qmdp", "myopic"):
        r, w = run_policy(obs, asset_rets, T_hmm, means, covs, Q_star, name)
        rets[name] = r
        weights[name] = w

    metrics_rows = [summarize(r, weights[k], name=k) for k, r in rets.items()]
    for row in metrics_rows:
        row["K"] = K

    # Per-regime characterization (in-sample)
    states = model.predict(obs)
    regime_stats = []
    for s in range(K):
        mask = states == s
        if mask.sum() == 0:
            continue
        regime_stats.append({
            "K": K, "regime": s, "n_months": int(mask.sum()),
            "mean_spy": float(df.loc[mask, "spy_ret"].mean()),
            "std_spy":  float(df.loc[mask, "spy_ret"].std()),
            "mean_agg": float(df.loc[mask, "agg_ret"].mean()),
            "std_agg":  float(df.loc[mask, "agg_ret"].std()),
            "mean_vix": float(df.loc[mask, "vix"].mean()),
            "best_action_idx": int(pi_vi[s]),
            "best_action_stock": float(ACTIONS[pi_vi[s], 0]),
            "best_action_bond": float(ACTIONS[pi_vi[s], 1]),
            "V_star": float(V_vi[s]),
        })

    return {
        "K": K,
        "model": model,
        "states": states,
        "rets": rets,
        "weights": weights,
        "metrics": metrics_rows,
        "regime_stats": regime_stats,
        "n_iter_vi": n_vi,
        "n_iter_pi": n_pi,
    }


def main() -> None:
    df, obs_cols = load_data()
    print(f"Data: {len(df)} months, observation cols = {obs_cols}")
    print(f"Period: {df.index.min().date()} → {df.index.max().date()}\n")

    all_metrics = []
    all_regime_stats = []
    runs: dict[int, dict] = {}

    for K in (2, 3, 4):
        print(f"--- K = {K} regimes ---")
        run = run_for_K(df, K)
        runs[K] = run
        all_metrics.extend(run["metrics"])
        all_regime_stats.extend(run["regime_stats"])
        print(f"  VI iters: {run['n_iter_vi']}, PI iters: {run['n_iter_pi']}")
        for row in run["metrics"]:
            print(f"  {row['name']:>7s}  CAGR {row['cagr']:7.2%}  Sharpe {row['sharpe']:6.2f}  "
                  f"MDD {row['max_drawdown']:7.2%}  Calmar {row['calmar']:6.2f}")
        print()

    pd.DataFrame(all_metrics).to_csv(RES / "multistate_metrics.csv", index=False)
    pd.DataFrame(all_regime_stats).to_csv(RES / "multistate_policies.csv", index=False)
    print(f"Wrote {RES / 'multistate_metrics.csv'} ({len(all_metrics)} rows)")
    print(f"Wrote {RES / 'multistate_policies.csv'} ({len(all_regime_stats)} rows)")

    # ---- Figure 1: side-by-side regime timelines (3 panels) ----
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.0), sharex=True)
    stress_events = [
        ("2008-10-31", "GFC"),
        ("2015-09-30", "China"),
        ("2020-04-30", "COVID"),
        ("2022-09-30", "Inflation"),
    ]
    for ax, K in zip(axes, (2, 3, 4)):
        run = runs[K]
        proba = run["model"].predict_proba(standardized_obs(df, obs_cols))
        bear = int(np.argmin([np.mean(df.loc[run["states"] == s, "spy_ret"]) for s in range(K)]))
        bear_prob = proba[:, bear]
        # NBER recession (background)
        ax.fill_between(df.index, 0, df["nber_recession"], color=PALETTE["nber"],
                        step="post", alpha=0.9, label="NBER recession", zorder=1)
        # Bear probability as filled area, this is the headline
        ax.fill_between(df.index, 0, bear_prob, color=PALETTE["bear"], alpha=0.45,
                        zorder=3, label=f"P(bear), state {bear}")
        ax.plot(df.index, bear_prob, color=PALETTE["bear"], lw=1.6, zorder=4)
        # Other states as thin lines
        for s in range(K):
            if s == bear:
                continue
            color = PALETTE["bull"] if s == 0 else (PALETTE["neutral"] if s == 1 else "#9467bd")
            ax.plot(df.index, proba[:, s], lw=1.0, color=color, alpha=0.7,
                    label=f"P(state {s})", zorder=2)
        # Annotate stress events on top panel only
        if K == 2:
            for date_str, label in stress_events:
                t = pd.Timestamp(date_str)
                if df.index[0] <= t <= df.index[-1]:
                    ax.axvline(t, color="0.2", lw=0.5, ls=":", alpha=0.55, zorder=0)
                    ax.text(t, 1.08, label, rotation=0, fontsize=9, ha="center",
                            color="0.25", va="bottom", weight="bold")
        ax.set_ylim(-0.02, 1.05)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_ylabel(f"$K={K}$\nP(state)", fontsize=11)
        ax.legend(loc="center left", fontsize=8.5, bbox_to_anchor=(1.005, 0.5))
    axes[0].set_title(
        "Filtered regime probabilities under $K\\in\\{2,3,4\\}$-state HMMs\n"
        "Bear regime (worst mean SPY) shown as filled red region.  "
        "NBER recessions shaded gray.",
        pad=24,
    )
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(FIG / "multistate_regime_timeline.pdf")
    fig.savefig(FIG / "multistate_regime_timeline.png")
    plt.close(fig)

    # ---- Figure 2: equity curves overlay ----
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    styles = {2: "-", 3: "--", 4: ":"}
    static_curve = (1 + runs[2]["rets"]["static"]).cumprod()
    ax.plot(static_curve.index, static_curve.values, lw=2.4, color=PALETTE["static"],
            label="Static 60/40 (benchmark)", zorder=5)
    for K in (2, 3, 4):
        for name in ("qmdp", "myopic"):
            r = runs[K]["rets"][name]
            cum = (1 + r).cumprod()
            ax.plot(cum.index, cum.values, lw=1.6, ls=styles[K], color=PALETTE[name],
                    alpha=0.85, label=f"{name.upper()}  ($K={K}$)")
    # Annotate final wealth values for static and best of each
    end_x = static_curve.index[-1]
    end_static = static_curve.iloc[-1]
    end_qmdp_2 = (1 + runs[2]["rets"]["qmdp"]).cumprod().iloc[-1]
    end_myo_2 = (1 + runs[2]["rets"]["myopic"]).cumprod().iloc[-1]
    ax.annotate(f"${end_myo_2:.2f}$", xy=(end_x, end_myo_2), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=9,
                color=PALETTE["myopic"], weight="bold")
    ax.annotate(f"${end_qmdp_2:.2f}$", xy=(end_x, end_qmdp_2), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=9,
                color=PALETTE["qmdp"], weight="bold")
    ax.annotate(f"${end_static:.2f}$", xy=(end_x, end_static), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=9,
                color=PALETTE["static"], weight="bold")
    ax.set_yscale("log")
    ax.set_ylabel("Cumulative wealth (log scale), \\$1 at start of backtest")
    ax.set_xlabel("Date")
    ax.set_title(
        "Out-of-sample backtest 2003–2026, equity curves\n"
        "Solid / dashed / dotted: $K=2,3,4$ regimes.  Right labels: terminal wealth."
    )
    ax.legend(loc="upper left", ncol=2, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "multistate_equity_curves.pdf")
    fig.savefig(FIG / "multistate_equity_curves.png")
    plt.close(fig)

    # ---- Figure 3: regime-conditional mean returns bar chart ----
    fig, ax = plt.subplots(figsize=(9, 5.0))
    width = 0.22
    colors_state = [PALETTE["bull"], PALETTE["bear"], PALETTE["neutral"], "#9467bd"]
    for ki, K in enumerate((2, 3, 4)):
        means_per_regime = [s["mean_spy"] for s in runs[K]["regime_stats"]]
        means_per_regime = means_per_regime + [np.nan] * (4 - len(means_per_regime))
        for i, m in enumerate(means_per_regime):
            if not np.isnan(m):
                bar = ax.bar(ki + (i - 1.5) * width, m * 100, width=width,
                             color=colors_state[i], edgecolor="0.3", lw=0.5,
                             label=f"state {i}" if ki == 0 else None)
                ax.text(ki + (i - 1.5) * width, m * 100 + (0.06 if m >= 0 else -0.18),
                        f"{m*100:.2f}", ha="center", fontsize=8, color="0.2")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([f"$K={K}$" for K in (2, 3, 4)])
    ax.set_ylabel("Mean monthly SPY return per regime (\\%)")
    ax.set_xlabel("HMM regime count")
    ax.axhline(0, color="0.4", lw=0.7)
    ax.set_title(
        "In-sample regime-conditional mean SPY return\n"
        "$K=2$ produces a clear bull/bear split; $K=3,4$ collapse extra states"
    )
    ax.legend(loc="best", fontsize=10, title="Regime", title_fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "multistate_returns_bar.pdf")
    fig.savefig(FIG / "multistate_returns_bar.png")
    plt.close(fig)

    print("\nWrote 3 figures:")
    print("  figures/multistate_regime_timeline.pdf")
    print("  figures/multistate_equity_curves.pdf")
    print("  figures/multistate_returns_bar.pdf")


if __name__ == "__main__":
    main()
