# 04 · Experiments Guide (What Each Script Does, In Order)

> Every script in `experiments/` is independently runnable. They are numbered so the directory listing matches the pipeline order. This file maps each script to its inputs, outputs, and the question it answers.

---

## Pipeline at a glance

```
┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ 01_fetch_data    │ ─► │ 02_hmm_calib    │ ─► │ 03_regime_interp │ ─► │ 04_qmdp_solve    │
│ FRED + Yahoo     │    │ Gaussian HMM    │    │ vs NBER          │    │ VI + PI + Q*     │
│ → data/raw       │    │ → hmm_*.pkl     │    │ → figures/       │    │ → Q_star.npy     │
└──────────────────┘    └─────────────────┘    └──────────────────┘    └────────┬─────────┘
                                                                                │
        ┌───────────────────────────────────────────────────────────────────────┘
        ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ 06_multistate    │    │ 07_gamma_sens    │    │ 08_baselines     │
│ K=2,3,4 compare  │    │ γ sweep + plots  │    │ 12-way horse race│
│ → figures        │    │ → figures        │    │ → metrics tables │
└──────────────────┘    └──────────────────┘    └────────┬─────────┘
                                                          │
        ┌─────────────────────────────────────────────────┘
        ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ 09_richer_obs    │    │ 10_walk_forward  │    │ 12_subperiod     │
│ feature cohort   │    │ refit cadences   │    │ A/B/C heatmap    │
│ → cohort table   │    │ → equity curves  │    │ → Sharpe heatmap │
└────────┬─────────┘    └──────────────────┘    └──────────────────┘
         │
         ▼
┌──────────────────┐
│ 11_cohort_viz    │
│ P(bear) timeline │
│ → figures        │
└──────────────────┘
```

Plus `00_synthetic_demo.py`, a no-network smoke test against a known generative model. Use this to verify the install is working before fetching real data.

---

## Each script

### `00_synthetic_demo.py`, No-network smoke test
**Question:** does the install work?
**Inputs:** none (generates 25 years of synthetic monthly data from a known 2-regime HMM).
**Outputs:** `figures/synthetic_equity_curve.pdf`, `results/synthetic_metrics.csv`.
**Why it exists:** lets a teammate verify the full HMM → MDP → QMDP → backtest pipeline works end-to-end before touching FRED/Yahoo.

### `01_fetch_data.py`, Download FRED + Yahoo data
**Question:** what's the raw observation panel?
**Inputs:** none (downloads from FRED public CSV endpoint + Yahoo via `yfinance`).
**Outputs:** `data/raw/{vix, term_spread, hy_oas, nber, term_spread_2y, nfci, stlfsi, umcsent, jobless_claims, usd_index, wti_oil, fed_funds, spy, agg}.csv`, plus aligned `data/processed/monthly.csv`.
**Note:** all data is already committed to the repo, so this script is only needed to *refresh* to the latest available month.

### `02_hmm_calibration.py`, Fit the Gaussian HMM
**Question:** what are the regime parameters?
**Inputs:** `data/processed/monthly.csv`.
**Outputs:** `data/processed/hmm_{2,3,4}state.pkl`, `results/hmm_selection.csv` (BIC + log-likelihood per K).
**Method:** Baum-Welch (EM) via `hmmlearn.GaussianHMM`, 5 random restarts.

### `03_regime_interpretation.py`, Plot regime timeline vs NBER
**Question:** does the HMM recover known crises?
**Inputs:** monthly panel + fitted HMM.
**Outputs:** `figures/regime_timeline.pdf` (note: this was replaced in the cleanup; current canonical is `multistate_regime_timeline.pdf` from script 06).

### `04_qmdp_solve.py`, Solve the underlying MDP
**Question:** what's the optimal MDP policy and Q*?
**Inputs:** monthly panel + fitted HMM.
**Outputs:** `results/Q_star.npy`, `results/mdp_policy.csv`, `results/regime_returns.csv`.
**Method:** value iteration + policy iteration (cross-checked), CRRA reward via Monte Carlo over per-regime asset returns.

### `06_multistate_comparison.py`, K∈{2,3,4} HMM comparison
**Question:** does adding states help?
**Inputs:** monthly panel.
**Outputs:** `figures/multistate_regime_timeline.{pdf,png}`, `multistate_equity_curves.{pdf,png}`, `multistate_returns_bar.{pdf,png}`; `results/multistate_metrics.csv`, `multistate_policies.csv`.
**Finding:** K=2 produces the cleanest regime separation; K=3 and K=4 collapse extra states (271 monthly observations is too small to identify richer structure from 2 observation channels).

### `07_gamma_sensitivity.py`, CRRA γ sweep
**Question:** at what risk aversion does QMDP start beating static?
**Inputs:** monthly panel + fitted HMM.
**Outputs:** `figures/gamma_sensitivity_sharpe.{pdf,png}`, `gamma_equity_curves.{pdf,png}`; `results/gamma_sensitivity.csv`, `gamma_policy_table.csv`.
**Finding:** QMDP Sharpe crosses static at γ=8, saturates near 0.89 at γ≥15. But the optimal MDP policy is the same in bull and bear at *every* γ tested, the Sharpe gain is a bondward shift, not regime tilting.

### `08_baselines_comparison.py`, Twelve-strategy horse race (the headline)
**Question:** how does QMDP compare to the academic + practitioner consensus?
**Inputs:** monthly panel + fitted HMM + Q*.
**Outputs:** `figures/baselines_equity.{pdf,png}`, `baselines_drawdown.{pdf,png}`, `baselines_sharpe_bar.{pdf,png}`; `results/baselines_metrics.csv`, `baselines_returns.csv`, `baselines_weights.csv`.
**Strategies:** static 60/40, equal-weight, inverse-vol, risk parity (ERC), Markowitz + Ledoit-Wolf, Faber 10-mo SMA, TS momentum 12-mo, vol-target 60/40, myopic 12-mo, HMM-conditional MV, Black-Litterman + HMM views, QMDP.
**Finding:** Faber wins (Sharpe 1.68); QMDP at γ=2 finishes last (0.73).

### `09_richer_observations.py`, Multi-feature HMM cohort study (Unlock 1)
**Question:** is the QMDP collapse fundamental or an observation-channel problem?
**Inputs:** monthly panel.
**Outputs:** `results/multifeature_hmm_table.csv`, for each cohort, the optimal MDP policy at γ=2.
**Finding:** adding NFCI + STLFSI4 produces a fully regime-differentiated policy (100/0 bull, 0/100 bear) at the same γ=2. **The headline collapse is an observation-channel problem.**

### `10_walk_forward_refit.py`, Walk-forward refit cadences (Unlock 2)
**Question:** does refit cadence change the headline verdict?
**Inputs:** monthly panel.
**Outputs:** `figures/walk_forward_equity.{pdf,png}`; `results/walk_forward_metrics.csv`.
**Finding:** expanding-window walk-forward refit lifts QMDP Sharpe 0.81 → 1.08 (+33%), max DD −34% → −16.5% (more than halves).

### `11_cohort_regime_visualization.py`, Visual confirmation of Unlock 1
**Question:** can we *see* the multi-feature unlock?
**Inputs:** monthly panel.
**Outputs:** `figures/cohort_regime_timelines.{pdf,png}`, P(bear) timelines under three cohorts side-by-side.
**Finding:** cohort 3 (+NFCI +STLFSI4) produces sharp high-confidence bear spikes that cleanly bracket every NBER recession + 2015-16 oil/China + 2022 inflation. Cohort 1 baseline is noisy. Cohort 5 (kitchen sink) over-fits.

### `12_subperiod_robustness.py`, Period A/B/C robustness check
**Question:** does the ranking hold across economically distinct subperiods?
**Inputs:** monthly panel + Q*.
**Outputs:** `figures/subperiod_sharpe_heatmap.{pdf,png}`; `results/subperiod_metrics.csv`.
**Finding:** Faber dominates every column (Sharpe 1.41, 2.01, 1.45). QMDP's worst column is Period A (GFC era, Sharpe 0.33), exactly where a regime-aware model should help most. HMM-aware non-QMDP baselines (HMM-MV, BL+HMM) get Sharpe 1.03–1.25 in Period A vs QMDP's 0.33: signal was there, QMDP's γ=2 projection destroyed it.

---

## How to reproduce the headline results from a fresh clone

```bash
git clone git@github.com:takakhoo/ENGS177_Final_Project.git
cd ENGS177_Final_Project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Data + fitted HMMs are already committed, so you can skip 01 and 02.
python experiments/04_qmdp_solve.py
python experiments/08_baselines_comparison.py   # headline horse race
python experiments/09_richer_observations.py    # Unlock 1
python experiments/10_walk_forward_refit.py     # Unlock 2
python experiments/12_subperiod_robustness.py   # robustness
```

To refresh to the latest available data (network required):
```bash
python experiments/01_fetch_data.py
python experiments/02_hmm_calibration.py
# (then run downstream experiments as above)
```

Total wall time, end-to-end with all 12 experiments: under 5 minutes on a 2019 MacBook Pro.
