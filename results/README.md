# `results/` — Numerical Outputs of Every Experiment

> What each CSV is, who produces it, and which finding it supports.

---

## Map: file → producer → finding

| File | Produced by | Headline finding it supports |
|---|---|---|
| [`baselines_metrics.csv`](baselines_metrics.csv) | `experiments/08_baselines_comparison.py` | Twelve-strategy horse race — Faber Sharpe 1.68 dominates; QMDP last at 0.73 |
| [`baselines_returns.csv`](baselines_returns.csv) | `experiments/08_baselines_comparison.py` | Per-month return series for every strategy |
| [`baselines_weights.csv`](baselines_weights.csv) | `experiments/08_baselines_comparison.py` | Per-month weights chosen by every strategy (long-form) |
| [`multifeature_hmm_table.csv`](multifeature_hmm_table.csv) | `experiments/09_richer_observations.py` | Unlock 1 — adding NFCI + STLFSI4 differentiates the policy at γ=2 |
| [`walk_forward_metrics.csv`](walk_forward_metrics.csv) | `experiments/10_walk_forward_refit.py` | Unlock 2 — expanding-window refit lifts QMDP Sharpe to 1.08 |
| [`subperiod_metrics.csv`](subperiod_metrics.csv) | `experiments/12_subperiod_robustness.py` | Robustness — Faber dominates Period A/B/C; QMDP worst in GFC era |
| [`gamma_sensitivity.csv`](gamma_sensitivity.csv) | `experiments/07_gamma_sensitivity.py` | γ-sweep — QMDP Sharpe crosses static at γ=8 |
| [`gamma_policy_table.csv`](gamma_policy_table.csv) | `experiments/07_gamma_sensitivity.py` | π*(s) vs γ — same action in bull/bear at every γ |
| [`multistate_metrics.csv`](multistate_metrics.csv) | `experiments/06_multistate_comparison.py` | K=2,3,4 backtest comparison |
| [`multistate_policies.csv`](multistate_policies.csv) | `experiments/06_multistate_comparison.py` | K=2,3,4 optimal policies |
| [`hmm_selection.csv`](hmm_selection.csv) | `experiments/02_hmm_calibration.py` | BIC + log-likelihood for K ∈ {2,3,4} |
| [`mdp_policy.csv`](mdp_policy.csv) | `experiments/04_qmdp_solve.py` | Optimal MDP policy at K=2, γ=2 |
| [`regime_returns.csv`](regime_returns.csv) | `experiments/04_qmdp_solve.py` | Per-regime asset return moments used to build the reward |
| [`Q_star.npy`](Q_star.npy) | `experiments/04_qmdp_solve.py` | Action-value table consumed by QMDP at decision time |
| [`synthetic_metrics.csv`](synthetic_metrics.csv) | `experiments/00_synthetic_demo.py` | No-network smoke-test metrics |

---

## The three headline tables, inline

### Twelve-strategy horse race (`baselines_metrics.csv`)

Out-of-sample 2003-10-31 → 2026-04-30, monthly rebalance, 5 bps tx cost, CRRA γ=2 for QMDP. Sorted by Sharpe.

| Strategy | CAGR | Sharpe | Sortino | Max DD | Calmar |
|---|---:|---:|---:|---:|---:|
| Faber 10-mo SMA | 17.24% | **1.68** | 3.57 | −13.5% | **1.28** |
| Myopic 12-mo trend | 15.34% | 1.41 | 2.70 | −15.1% | 1.01 |
| TS momentum 12-mo | 9.06% | 1.10 | 1.74 | −22.0% | 0.41 |
| HMM-conditional MV | 6.38% | 1.08 | 1.78 | −18.4% | 0.35 |
| BL + HMM views | 6.17% | 0.94 | 1.45 | −18.0% | 0.34 |
| Inverse-vol | 4.78% | 0.92 | 1.42 | −16.8% | 0.28 |
| Risk parity (ERC) | 4.87% | 0.91 | 1.41 | −16.8% | 0.29 |
| Equal weight 50/50 | 6.69% | 0.84 | 1.26 | −28.6% | 0.23 |
| MV + Ledoit-Wolf | 6.58% | 0.82 | 1.32 | −22.5% | 0.29 |
| Static 60/40 (bench) | 7.40% | 0.81 | 1.21 | −34.2% | 0.22 |
| Vol-target 60/40 | 7.50% | 0.77 | 1.12 | −39.1% | 0.19 |
| **QMDP (γ=2)** | 10.01% | **0.73** | 1.07 | **−53.0%** | 0.19 |

### Unlock 1: multi-feature HMM (`multifeature_hmm_table.csv`)

Same γ=2 as the headline; just changing the observation channels fed to the HMM.

| Cohort | Features | π*(bull) | π*(bear) | Differs? |
|---|---|---|---|---|
| 1. Baseline | VIX, T10Y3M | 100/0 | 100/0 | No |
| 2. +Yield curve | +T10Y2Y | 100/0 | 100/0 | No |
| **3. +Stress** | **+NFCI, +STLFSI4** | **100/0** | **0/100** | **Full flip** |
| 4. +Macro | +UMCSENT, +ICSA | 100/0 | 0/100 | Yes |
| 5. Kitchen sink | 8 channels | 100/0 | 60/40 | Yes |

### Unlock 2: walk-forward refit (`walk_forward_metrics.csv`)

| Variant | Sharpe | Max DD | Calmar |
|---|---:|---:|---:|
| Static 60/40 (bench) | 0.81 | −34.2% | 0.22 |
| QMDP fixed (report baseline) | 0.96 | −34.2% | 0.29 |
| QMDP annual refit (5y window) | 0.85 | −25.1% | 0.36 |
| QMDP quarterly refit | 0.83 | −23.4% | 0.37 |
| **QMDP expanding window** | **1.08** | **−16.5%** | **0.60** |

---

## Synthesis

The headline finding "QMDP at γ=2 underperforms 60/40" holds only in the joint configuration of (a) narrow observations, (b) fixed HMM, (c) low CRRA. Each component is empirically reversible. See [`../docs/01_project_story.md`](../docs/01_project_story.md) for the full narrative.
