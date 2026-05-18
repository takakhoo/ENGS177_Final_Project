"""Step 5 of pipeline: walk-forward backtest of QMDP vs myopic vs static 60/40.

For simplicity, this script uses the (already-fitted) global HMM rather than refitting
annually. A walk-forward refit is the next iteration; this run produces the headline
numbers we'll show in the report's first results figure.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt  # noqa: E402
from src.models.qmdp import update_belief, stationary_distribution  # noqa: E402
from src.utils.metrics import summarize  # noqa: E402

DATA = REPO / "data" / "processed"
RES = REPO / "results"
FIG = REPO / "figures"

N_STATES = 2
LAMBDA = 0.95
TXCOST_BPS = 5
ACTIONS = np.array([
    [0.0, 1.0],
    [0.2, 0.8],
    [0.4, 0.6],
    [0.6, 0.4],
    [0.8, 0.2],
    [1.0, 0.0],
])


def backtest_policy(
    obs: np.ndarray,
    asset_rets: pd.DataFrame,
    T_hmm: np.ndarray,
    means: list[np.ndarray],         # per-state emission mean
    covs: list[np.ndarray],          # per-state emission cov
    Q_star: np.ndarray | None,
    policy_name: str,
) -> tuple[pd.Series, pd.DataFrame]:
    """Run a single policy through the full time series. Returns (monthly_returns, weights)."""
    n_steps = obs.shape[0]
    belief = stationary_distribution(T_hmm)
    prev_a = np.array([0.6, 0.4])  # start at 60/40 weights

    rets, weights = [], []
    for t in range(n_steps):
        # Belief update from observation o_t (Lec 2 Bayes filter)
        likelihood = np.array(
            [_gauss_pdf(obs[t], means[s], covs[s]) for s in range(N_STATES)]
        )
        belief = update_belief(belief, T_hmm, likelihood)

        # Policy
        if policy_name == "static":
            a = np.array([0.6, 0.4])
        elif policy_name == "qmdp":
            a_idx = int((belief @ Q_star).argmax())
            a = ACTIONS[a_idx]
        elif policy_name == "myopic":
            # one-step-lookahead: argmax_a E[U | b] under each regime's MEAN return only
            mean_ret = np.array([means[s][:2] if len(means[s]) >= 4 else None for s in range(N_STATES)])
            # fallback: weighted expected portfolio return (linear utility)
            mu_assets = np.array(
                [asset_rets.iloc[max(0, t - 12):t + 1].mean().values for _ in range(N_STATES)]
            ) if t == 0 else np.tile(asset_rets.iloc[max(0, t - 12):t + 1].mean().values, (N_STATES, 1))
            scores = belief @ (mu_assets @ ACTIONS.T)
            a = ACTIONS[int(scores.argmax())]
        else:
            raise ValueError(policy_name)

        # Apply transaction cost
        turnover = np.abs(a - prev_a).sum()
        cost = TXCOST_BPS / 1e4 * turnover

        # Realize portfolio return for this period
        port_ret = float(asset_rets.iloc[t].values @ a) - cost
        rets.append(port_ret)
        weights.append(a)
        prev_a = a

    idx = asset_rets.index
    return pd.Series(rets, index=idx, name=policy_name), pd.DataFrame(
        weights, index=idx, columns=["stock", "bond"]
    )


def _gauss_pdf(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
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

    obs_cols = [c for c in ["vix", "term_spread", "hy_oas"] if c in df.columns]
    obs = df[obs_cols].values
    asset_rets = df[["spy_ret", "agg_ret"]]

    means = list(model.means_)
    covs = list(model.covars_)
    T_hmm = model.transmat_
    Q_star = np.load(RES / "Q_star.npy")

    results = {}
    weights_all = {}
    for policy_name in ("static", "qmdp", "myopic"):
        rets, weights = backtest_policy(obs, asset_rets, T_hmm, means, covs, Q_star, policy_name)
        results[policy_name] = rets
        weights_all[policy_name] = weights

    # Metrics
    rows = [summarize(rets, weights_all[k], name=k) for k, rets in results.items()]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RES / "metrics.csv", index=False)
    print(metrics.round(4))

    # Equity curve plot
    fig, ax = plt.subplots(figsize=(11, 5))
    for k, rets in results.items():
        cum = (1 + rets).cumprod()
        ax.plot(cum.index, cum.values, label=k, lw=1.5)
    ax.set_yscale("log")
    ax.set_ylabel("Cumulative wealth (log scale)")
    ax.set_title(f"Backtest 2000–2024 — QMDP vs Myopic vs 60/40 ({N_STATES}-state HMM)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "equity_curve.pdf")
    print(f"Wrote {FIG / 'equity_curve.pdf'} and {RES / 'metrics.csv'}")


if __name__ == "__main__":
    main()
