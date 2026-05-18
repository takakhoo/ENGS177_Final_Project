# 04 · Implementation Plan

End-to-end plan from raw data to backtest result. Each numbered step has a one-file experiment script in `experiments/` and supporting modules in `src/`.

---

## Pipeline

```
┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ 01_fetch_data    │ ─► │ 02_hmm_calib    │ ─► │ 03_regime_interp │ ─► │ 04_qmdp_solve    │ ─► │ 05_backtest      │
│ FRED + Yahoo     │    │ Gaussian HMM    │    │ vs NBER          │    │ VI + QMDP        │    │ walk-forward     │
│ → data/raw       │    │ → hmm.pkl       │    │ → figures/       │    │ → results/       │    │ → results/       │
└──────────────────┘    └─────────────────┘    └──────────────────┘    └──────────────────┘    └──────────────────┘
```

## 1. Data fetching (`experiments/01_fetch_data.py`)

**Inputs:** None (downloads from network).

**Outputs:**
- `data/raw/vix.csv` (FRED VIXCLS)
- `data/raw/term_spread.csv` (FRED T10Y3M)
- `data/raw/hy_oas.csv` (FRED BAMLH0A0HYM2)
- `data/raw/spy.csv` (Yahoo, total return adjusted close)
- `data/raw/agg.csv` (Yahoo, AGG ETF)
- `data/raw/nber.csv` (FRED USREC)
- `data/processed/monthly.csv` — single aligned monthly frame:
  `date, vix, term_spread, hy_oas, spy_ret, agg_ret, nber_recession`

**Tools:** `pandas_datareader` (FRED), `yfinance` (Yahoo).

**Time series cleaning:**
- Resample daily → monthly (end-of-month last observation).
- Standardize each observation series to zero mean, unit variance (rolling 60-month window to avoid look-ahead).
- Drop rows with any NaN.

## 2. HMM calibration (`experiments/02_hmm_calibration.py`)

**Inputs:** `data/processed/monthly.csv`.

**Outputs:**
- `data/processed/hmm_2state.pkl`, `hmm_3state.pkl` (pickled `hmmlearn.GaussianHMM` objects).
- `results/hmm_selection.csv` — table of BIC and held-out log-likelihood per state count.

**Method:**
1. Split: 1990–2014 train, 2015–present test.
2. Standardize observations using train-set mean/std.
3. Fit Gaussian HMM with `n_components ∈ {2, 3, 4}`, full covariance.
4. EM (Baum–Welch): 5 restarts per `n_components`, take the highest train log-likelihood.
5. BIC: $-2 \log L + k \log n$ where $k$ is number of free parameters.
6. Pick **best on BIC** (parsimony) but report all three.

**Validation:**
- Posterior `predict_proba(test)` should spike during 2008 and 2020 recessions.
- One of the HMM states should have lower mean SPY return and higher VIX in its emission distribution → "bear" state.

## 3. Regime interpretation (`experiments/03_regime_interpretation.py`)

**Inputs:** `hmm_*.pkl`, `data/processed/monthly.csv`, `data/raw/nber.csv`.

**Outputs:**
- `figures/regime_timeline.pdf` — filtered regime probabilities (Viterbi or smoothed) overlaid with NBER recession shading.
- `figures/regime_emissions.pdf` — emission means and 1σ ellipses per state in (VIX, spread) space.
- `results/regime_durations.csv` — expected dwell time per regime from $\pi P^n$.
- `results/regime_returns.csv` — average SPY and AGG return per regime (in-sample).

**Sanity checks before moving on:**
- Bear regime mean SPY return < 0 (or much lower than bull regime).
- Bear regime mean VIX > bull regime mean VIX (large effect).
- Regime probabilities spike on NBER dates (visual).

## 4. QMDP solver (`experiments/04_qmdp_solve.py`)

**Inputs:** `hmm_*.pkl`, `data/processed/monthly.csv`.

**Outputs:**
- `results/mdp_value_function.csv` — $V^*(s)$ per regime.
- `results/mdp_policy.csv` — $\pi^*_{\text{MDP}}(s)$ greedy policy on regime grid.
- `results/qmdp_policy.csv` — $\pi_{\text{QMDP}}(b)$ on a discretized belief simplex.
- `figures/policy_map.pdf` — policy as a function of belief over "bear" regime (1-D slice for 2-state model).

**Method:**

**Step A — Build reward $R(s, a)$.**
For each regime $s$, expected utility of action $a$ is computed by Monte Carlo from the HMM emission of the asset return distribution. We use log utility $U(W) = \log(1 + a^\top R)$ baseline + CRRA $\gamma \in \{1, 2, 5\}$ sensitivity.

Transaction cost subtracted: $-c \|a - a_{\text{prev}}\|_1$ with $c = 5$ bps. For step A we use no-cost baseline; cost added at policy-evaluation time.

**Step B — Value iteration.**
$$V^{n+1}(s) = \max_{a \in A}\left\{ R(s, a) + \lambda \sum_{s' \in S} T(s' | s)\, V^n(s') \right\}$$
Stop when $\|V^{n+1} - V^n\|_\infty < \varepsilon(1-\lambda)/(2\lambda)$. Default $\lambda = 0.95$, $\varepsilon = 1e-4$.

**Step C — Cross-check with policy iteration.**
Solve $(I - \lambda P_\pi) v = r_\pi$ and improve. Verify identical greedy policy.

**Step D — QMDP.**
At each backtest decision point with belief $b_t$:
$$\pi_{\text{QMDP}}(b_t) = \argmax_a \sum_s b_t(s) Q^*_{\text{MDP}}(s, a).$$

## 5. Walk-forward backtest (`experiments/05_backtest_compare.py`)

**Inputs:** `data/processed/monthly.csv`, `hmm_*.pkl`, `results/qmdp_policy.csv`.

**Outputs:**
- `results/backtest_equity.csv` — cumulative wealth per policy per month.
- `results/metrics.csv` — table of CAGR, vol, Sharpe, MDD, Calmar, turnover.
- `figures/equity_curve.pdf` — log-scale cumulative return for the three policies.
- `figures/drawdown.pdf` — drawdown series.
- `figures/turnover.pdf` — turnover over time, per policy.

**Method:**

```
For each year t ∈ [2000, 2024]:
    train_window = data[1990 : t-1]
    test_window  = data[t : t+1]   # 12 months
    refit HMM on train_window
    rebuild reward R(s, a) and MDP value function
    for month m in test_window:
        observe o_m
        update belief: b_m = filter(b_{m-1}, a_{m-1}, o_m)
        choose a_m = π_QMDP(b_m)   (also myopic and static for comparators)
        apply a_m, observe realized portfolio return
    record monthly returns
Aggregate all years into one OOS equity curve per policy.
```

**Initial belief at t=1990 reset:** stationary distribution of fitted HMM transition matrix.

## 6. Sensitivity analysis (`experiments/06_sensitivity.py`)

A 4-way grid over:
- `n_regimes ∈ {2, 3}`
- `gamma ∈ {1, 2, 5}` (CRRA risk aversion; gamma=1 ⇒ log utility)
- `rebalance_freq ∈ {monthly, quarterly}`
- `cost_bps ∈ {0, 5, 20}`

24 cells. Each cell yields a full backtest. We report a heatmap of Sharpe by (n_regimes, gamma) at the headline cost level.

## 7. Monte Carlo validation (stretch — `experiments/07_montecarlo_validation.py`)

Simulate 1000 paths of length 25 years from the fitted HMM. Run QMDP policy on each. Compare distribution of Sharpe to live backtest. Confirms live backtest is not an artifact of one realized path.

---

## Compute / wall-clock estimate

| Step | Wall-clock |
|---|---|
| Data fetch | < 1 min |
| HMM fit (single config) | < 10 sec |
| Full HMM model selection | < 1 min |
| Value iteration on a 2-state, 6-action MDP | < 1 sec |
| Single-year backtest | < 5 sec |
| Full walk-forward (2000–2024, 25 years × 12 months) | < 2 min |
| Full sensitivity grid (24 cells) | < 30 min |
| MC validation (1000 paths) | < 5 min |

Everything runs on a laptop. No GPU.

## Dependencies

See `requirements.txt`. Core:
- `numpy`, `pandas`, `scipy`
- `hmmlearn`
- `yfinance`, `pandas_datareader`
- `matplotlib`, `seaborn`
- `joblib` (for caching)
