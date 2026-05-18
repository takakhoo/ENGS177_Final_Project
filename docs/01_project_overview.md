# 01 · Project Overview (Canonical Brief)

**If you are an LLM agent or new collaborator: start here.** This file is the single source of truth for what we are building, why, and how to plug in.

---

## 1. Problem

A long-only investor holds a portfolio of two assets, equities (SPY) and bonds (AGG), and must choose, at each monthly rebalance date, the weights $a_t = (w_t^{\text{stock}}, w_t^{\text{bond}})$ with $w^{\text{stock}}_t + w^{\text{bond}}_t = 1$, $w^{\text{stock}}_t, w^{\text{bond}}_t \geq 0$.

Asset returns are *non-stationary*: they are drawn from a different distribution depending on the latent macro-financial regime (broadly: an *expansionary* regime with low volatility and positive drift, vs. a *stress* regime with elevated volatility and compressed term spreads). The investor cannot observe the regime directly; they observe noisy macro-financial proxies (VIX, term spread, high-yield OAS).

**Goal:** maximize expected discounted utility of long-run wealth, net of transaction costs, using only the publicly observable information stream.

## 2. Why POMDP

Three observations together force the POMDP formulation:

1. The **regime is latent**, we never see it directly.
2. The **observation stream is noisy and partial**, the same VIX value can occur in either regime.
3. The **decision is sequential**, current allocation affects future wealth, and current information updates beliefs about future regimes.

A POMDP $(\mathcal{T}, S, A, T, \Omega, O, R, \lambda)$ formalizes exactly this: hidden state, observation channel, belief updates, sequential reward.

## 3. POMDP tuple (verbatim from HW2)

- **States $S$** (latent regimes): two-state baseline $\{s_\text{expansion}, s_\text{stress}\}$, three-state variant $\{\text{bull}, \text{neutral}, \text{bear}\}$.
- **Actions $A$**: discrete portfolio grid, e.g. $\{(0.0, 1.0), (0.2, 0.8), (0.4, 0.6), (0.6, 0.4), (0.8, 0.2), (1.0, 0.0)\}$.
- **Transition $T(s' | s)$**: HMM transition matrix estimated by Baum–Welch on monthly observation series. *No action effect*, the market is exogenous to our portfolio choice.
- **Observation space $\Omega$**: continuous vector $o_t = (\text{VIX}_t, \text{term-spread}_t, \text{HY-OAS}_t)$, standardized.
- **Observation function $O(o | s)$**: per-state Gaussian emission learned by HMM, $o_t | s_t = k \sim \mathcal{N}(\mu_k, \Sigma_k)$.
- **Belief state $b_t \in \Delta^{|S|-1}$**: posterior over regimes given the observation history.
- **Reward $R(s, a)$**: expected utility of single-period portfolio return minus turnover cost,
  $$R(s, a) = \mathbb{E}[U(a^\top R^{\text{asset}}) \mid s] - c \|a_t - a_{t-1}\|_1,$$
  with $U$ log or CRRA utility and $c = 5$ bps per side per unit weight changed.
- **Discount $\lambda \in (0, 1)$**: default $\lambda = 0.95$ at monthly frequency (≈ 5-year half-life).

## 4. Solution approach

**Step 1, HMM calibration.** Fit Gaussian HMM on $(\text{VIX}, \text{spread}, \text{HY-OAS})$ via Baum–Welch (EM). Model selection by BIC and out-of-sample log-likelihood. Initialize from k-means with multiple restarts.

**Step 2, Underlying MDP solution.** Solve the *fully-observable* infinite-horizon discounted MDP
$$V^*(s) = \max_{a \in A}\left\{ R(s, a) + \lambda \sum_{s' \in S} T(s' | s)\, V^*(s') \right\}$$
by **value iteration** (Lec 7) on the discrete regime × action grid. Output: $Q^*_{\text{MDP}}(s, a)$.

**Step 3, QMDP approximation.** At decision time $t$ with belief $b_t$,
$$\pi_{\text{QMDP}}(b_t) = \argmax_{a \in A} \sum_{s \in S} b_t(s)\, Q^*_{\text{MDP}}(s, a).$$
This is exact when $b_t$ is a delta (fully-observable) and a sensible heuristic when uncertainty is moderate.

**Step 4, Belief update (Bayesian filter, Lec 2).** Given prior $b_{t-1}$, action $a_{t-1}$, observation $o_t$,
$$b_t(s') \propto O(o_t | s') \sum_s T(s' | s)\, b_{t-1}(s).$$

**Step 5, Backtest.** Walk-forward evaluation with **annual HMM refitting** to avoid look-ahead bias. Compare three policies: QMDP, myopic one-step-lookahead, static 60/40.

## 5. Comparator policies

- **Static 60/40:** $\pi_{\text{static}}(b) = (0.6, 0.4)$ for every $b$. Benchmark.
- **Myopic:** $\pi_{\text{myopic}}(b) = \argmax_a \sum_s b(s) R(s, a)$. Greedy on next-period expected utility; ignores future evolution of beliefs.
- **QMDP (ours):** as above.

## 6. Evaluation metrics

- **CAGR** (compound annual growth rate)
- **Annualized volatility**
- **Sharpe ratio** (annualized)
- **Max drawdown** and **Calmar ratio**
- **Turnover** (per-period $\ell_1$ change in weights)
- **Hit rate of regime calls** vs. NBER recession dates (qualitative)

## 7. Inputs (data sources)

| Series | Source | Frequency | Identifier |
|---|---|---|---|
| VIX (CBOE Volatility Index) | FRED | Daily → resampled monthly | `VIXCLS` |
| 10Y–3M Treasury spread | FRED | Daily → monthly | `T10Y3M` |
| ICE BofA HY OAS | FRED | Daily → monthly | `BAMLH0A0HYM2` |
| S&P 500 total return proxy | Yahoo Finance | Daily → monthly | `SPY` |
| Aggregate bond total return | Yahoo Finance | Daily → monthly | `AGG` |
| NBER recession dates | FRED | Monthly | `USREC` |

All time series aligned at monthly frequency, end-of-month. Sample: 1990–present (≈ 35 years; AGG limits to 2003+ for live data, earlier history reconstructed via Treasury index).

## 8. Project flow (week by week)

- **Wk 3–4:** Proposal, problem definition, data assembly, done.
- **Wk 5–6:** HMM calibration, regime interpretation, HW2 / HW3.
- **Wk 7–8:** QMDP solver + initial backtest, HW3, now.
- **Wk 9:** Sensitivity analysis, write-up, slides.
- **Wk 10:** Final presentation, report, peer review.

## 9. Team responsibilities (preliminary)

| Member | Primary | Secondary |
|---|---|---|
| Dario | HMM calibration | Regime interpretation |
| Even | QMDP solver | MDP value iteration |
| Kyle (KDLL) | Backtest engine | Performance metrics |
| Taka | POMDP formulation + utility/reward + report writing | Plots, all integrations |

(Update in `docs/06_deliverables.md` as we go; final contribution table goes in the report.)

## 10. Open questions for the team

1. CRRA risk-aversion parameter $\gamma$, fix at 2 or sweep $\{1, 2, 5\}$?
2. Two-state vs. three-state HMM, pick one for headline result, show the other in sensitivity?
3. Should we include cash (T-bills) as a third action dimension, or stay with stock/bond only?
4. Transaction cost, flat 5 bps or scaled by turnover regime?
5. What's our "stretch" experiment if QMDP works?, point-based VI (PBVI)? Online belief-space tree search?
