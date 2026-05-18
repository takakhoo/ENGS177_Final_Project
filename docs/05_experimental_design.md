# 05 · Experimental Design

This file documents *what we're testing*, *what we expect*, and *how we'll know we got there*. Every experiment is a hypothesis with a falsifiable prediction.

---

## Headline experiment

> **H0**, A QMDP policy operating on a 2-state Gaussian HMM regime model does not beat a static 60/40 stock/bond portfolio out-of-sample, after 5 bps transaction costs, over the period 2000–2024.

We reject H0 if **annualized Sharpe ratio of QMDP > Sharpe of 60/40** with a > 50% improvement (sensitivity in `06_sensitivity.py`).

## Detailed experiments

### Experiment 1, HMM model selection

| Variable | Levels |
|---|---|
| Number of regimes $K$ | 2, 3, 4 |
| Covariance type | full |
| Observation set | {VIX}, {VIX, term-spread}, {VIX, term-spread, HY-OAS} |

**Metric:** held-out log-likelihood, BIC.

**Prediction:** 3-state full-covariance with all three observations wins on log-likelihood but 2-state wins on BIC due to parsimony.

**Falsification:** if BIC is monotone-decreasing in $K$ over our range, we keep adding states until BIC tops out (data-driven).

### Experiment 2, Regime interpretability

| Quantity | Expected |
|---|---|
| Bear-regime mean VIX | > bull-regime mean VIX, by > 5 vol points |
| Bear-regime mean SPY return | < 0 monthly, or much lower than bull |
| Bear-regime probability spikes during 2008-09 | True (visual) |
| Bear-regime probability spikes during 2020-03 | True (visual) |

**Falsification:** if regime labels are not economically interpretable (e.g., the "bear" regime has high SPY returns), the HMM is overfitting or the observation set is wrong; revisit step 1.

### Experiment 3, MDP solver agreement

| Method | Stopping criterion | Result |
|---|---|---|
| Value iteration | $\|v^{n+1} - v^n\|_\infty < \varepsilon(1-\lambda)/(2\lambda)$ with $\varepsilon = 10^{-4}$, $\lambda = 0.95$ | $V^*_\text{VI}(s)$ |
| Policy iteration | $\pi_{n+1} = \pi_n$ | $V^*_\text{PI}(s)$ |

**Prediction:** the two methods yield identical greedy policies and $\|V^*_\text{VI} - V^*_\text{PI}\|_\infty < \varepsilon$.

**Falsification:** if they differ, we have an off-by-one or sign error somewhere.

### Experiment 4, QMDP vs. baselines (the headline)

| Policy | Description |
|---|---|
| **Static 60/40** | $\pi(b) = (0.6, 0.4)$ always |
| **Myopic** | $\pi(b) = \argmax_a \sum_s b(s) R(s, a)$ |
| **QMDP** (ours) | $\pi(b) = \argmax_a \sum_s b(s) Q^*_\text{MDP}(s, a)$ |

Backtest: 2000–2024 walk-forward with annual HMM refit, 5 bps cost.

**Metrics reported:**
- CAGR, annualized volatility, Sharpe ratio
- Max drawdown (MDD), Calmar (= CAGR / |MDD|)
- Average annual turnover
- Worst-year return; best-year return
- Returns during NBER recessions vs. expansions

**Predictions:**
1. **QMDP Sharpe > Myopic Sharpe > Static Sharpe.** Order from theory.
2. **QMDP MDD better than Static.** The whole point.
3. **QMDP turnover higher than Static, lower than Myopic.** Myopic has no incentive to smooth.
4. **QMDP outperformance concentrated in recessions.** Expansion-period returns roughly similar across policies (because in expansion you want to be long anyway).

### Experiment 5, Sensitivity grid

$K \in \{2, 3\} \times \gamma \in \{1, 2, 5\} \times \text{freq} \in \{\text{M}, \text{Q}\} \times \text{cost} \in \{0, 5, 20\}$ bps = 36 cells.

**Outputs:**
- `figures/sensitivity_sharpe.pdf`, Sharpe heatmap by $(K, \gamma)$ at headline cost.
- `figures/sensitivity_cost.pdf`, line plot of Sharpe vs. cost level for each policy.

**Prediction:** QMDP beats 60/40 across all reasonable cells. Edge shrinks but does not vanish at 20 bps.

**Falsification:** if the QMDP edge requires zero transaction cost, the result is not actionable.

### Experiment 6, Statistical significance

The headline Sharpe difference is one number on one realized history. To estimate uncertainty:

1. **Block bootstrap** the monthly returns of QMDP and 60/40 with block size 24 months. 10,000 resamples. Report 95% CI on Sharpe difference.
2. **Deflated Sharpe ratio** (Bailey et al. 2014) accounting for the multiple-testing implied by our sensitivity grid.

**Prediction:** 95% CI on Sharpe difference does not include 0, and the deflated Sharpe remains positive.

### Experiment 7 (Stretch), Monte Carlo validation

Simulate from the fitted HMM. Run policies on simulated paths.

**Why:** to disentangle "QMDP is genuinely better" from "QMDP got lucky on the realized path."

**Output:** histograms of Sharpe over 1000 sim paths, side by side for each policy. The medians and CIs should align with the single-history backtest.

### Experiment 8 (Stretch), Belief-space discretization

QMDP uses the underlying-MDP $Q$-function. A more sophisticated POMDP solver (PBVI) maintains alpha-vectors over the belief simplex.

**Test:** does PBVI's policy materially differ from QMDP's? On a 2-state regime, the belief simplex is just $[0, 1]$, so this is cheap.

**Prediction:** PBVI ≈ QMDP for 2-state; PBVI > QMDP for 3-state where information value matters more.

---

## Figures inventory (everything we plan to commit to `figures/`)

| Filename | Source experiment | Caption |
|---|---|---|
| `regime_timeline.pdf` | Exp 2 | Filtered regime probabilities overlaid with NBER recession shading. |
| `regime_emissions.pdf` | Exp 2 | Per-regime emission means and covariances in (VIX, spread) space. |
| `policy_map.pdf` | Exp 3 | QMDP policy as a function of belief over "bear" (1-D for 2-state). |
| `equity_curve.pdf` | Exp 4 | Cumulative log-return for QMDP, myopic, 60/40. |
| `drawdown.pdf` | Exp 4 | Drawdown series for each policy. |
| `turnover.pdf` | Exp 4 | Monthly turnover series. |
| `sensitivity_sharpe.pdf` | Exp 5 | Sharpe heatmap, $(K, \gamma)$ at headline cost. |
| `sensitivity_cost.pdf` | Exp 5 | Sharpe vs. cost level. |
| `bootstrap_sharpe.pdf` | Exp 6 | Bootstrap distribution of Sharpe difference (QMDP - static). |
| `mc_sharpe_hist.pdf` | Exp 7 | MC distribution of Sharpe per policy. |
| `pbvi_vs_qmdp.pdf` | Exp 8 | Value difference of PBVI vs. QMDP across belief simplex. |

## Tables inventory (`results/*.csv`)

| Filename | Columns |
|---|---|
| `hmm_selection.csv` | n_states, train_ll, test_ll, bic |
| `mdp_policy.csv` | regime, action_stock, action_bond |
| `qmdp_policy.csv` | belief_bear, action_stock, action_bond |
| `metrics.csv` | policy, cagr, vol, sharpe, mdd, calmar, turnover_avg |
| `bootstrap_ci.csv` | policy_pair, sharpe_diff, ci_lo, ci_hi, p_value |
| `sensitivity.csv` | n_states, gamma, freq, cost_bps, policy, sharpe, mdd |
