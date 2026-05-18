# Initial Findings — Real Backtest 2003–2026

First end-to-end real-data backtest is in. Worth recording the surprise *before* it shapes the rest of the work.

## Setup
- HMM: 2-state Gaussian on (VIX, term spread), 135-month train window (2003–2014).
- Bull (state 0): mean SPY +1.16%/mo, VIX 15.2. (87 months in training set.)
- Bear (state 1): mean SPY −0.14%/mo, VIX 27.7. (48 months in training set.)
- Reward: CRRA utility, γ=2.
- Discount λ=0.95. Cost 5 bps/side.
- Actions: 6-point stock/bond grid {0/100, 20/80, …, 100/0}.

## What VI converged to
$V^* = (0.1054, 0.1003)$, $\pi^* = (\text{100 stocks}, \text{100 stocks})$.

That is: **even in the bear regime, the MDP solver prefers 100% stocks**. Under γ=2 CRRA, bear-regime mean stock return is barely negative but bond return is also small; over a discount horizon, the optimizer still prefers risk because the *discounted continuation value* dominates the single-period penalty.

Consequence: QMDP, which is $\argmax_a \sum_s b(s) Q^*(s,a)$, picks "all stocks" for *every* belief. So QMDP $=$ a static all-stock allocation.

## Backtest results

| Policy | CAGR | Vol | Sharpe | MDD | Calmar | Turnover |
|---|---|---|---|---|---|---|
| Static 60/40 | 7.35% | 9.4% | 0.81 | −34.2% | 0.22 | 0.00 |
| QMDP (all-stocks) | 9.90% | 14.7% | 0.72 | −53.0% | 0.19 | 0.00 |
| Myopic | 13.99% | 11.3% | **1.22** | −21.1% | 0.66 | 0.24 |

**QMDP underperforms static 60/40 in risk-adjusted terms** here — it's just a high-leverage equity portfolio.

**Myopic** (which uses 12-month rolling mean returns and linear scoring) does much better: it has real signal because it captures *trend*, not just regime label.

## Why this is actually useful for the project

This is *the* sensitivity analysis result. Three things to do next:

### 1. Higher risk aversion γ
CRRA with γ ∈ {3, 5, 10} should make bear-regime utility more strongly negative for stock-heavy actions, pushing the optimal bear-regime action toward bonds.

### 2. Better regime separation
Add HY-OAS back (we need a FRED API key to bypass the public CSV truncation) or substitute another stress indicator (TED spread, financial conditions index). With a stronger bear emission, the HMM will produce more discriminating beliefs.

### 3. Different reward structure
Penalize drawdown explicitly, or use a Sharpe-style reward, or add a leverage constraint that caps stock weight at 100% (already done). Could add a CVaR penalty.

## Why the myopic policy wins here
Myopic's score function is

$$\text{score}(a) = \sum_s b(s)\, \mathbb{E}[r_{\text{stock}} | s] w_{\text{stock}} + \mathbb{E}[r_{\text{bond}} | s] w_{\text{bond}}$$

with empirical means estimated on the trailing 12 months. So it captures *momentum* — a recent crash drops $\mathbb{E}[r_{\text{stock}}]$ below $\mathbb{E}[r_{\text{bond}}]$ and shifts the policy to bonds. The QMDP, by contrast, uses *long-run regime-conditional* means which are positive enough for stocks even in the bear state.

This is a great natural result to discuss in the report. The QMDP underperformance is itself a meaningful finding: **regime-aware allocation only adds value when the regime signal differentiates expected returns strongly enough relative to the discount factor, and log/γ=2 utility is too risk-tolerant to act on the signal we have**.

## Next planned experiment

Re-run with γ ∈ {2, 5, 10}. Prediction: QMDP starts beating static at γ ≥ 5. Output: `results/sensitivity_gamma.csv` + `figures/sensitivity_gamma.pdf`.
