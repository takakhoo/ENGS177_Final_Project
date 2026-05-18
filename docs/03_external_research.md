# 03 · External Research & Literature Map

This file collects the external references we draw from, beyond the ENGS 177 lectures. Cite this file in the report's "Background" and "Methods" sections.

---

## Tier 1, Foundational regime-switching papers

### Hamilton (1989), "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle"
*Econometrica*, 57(2), 357–384.

- First to formalize a two-state Markov-switching model for GDP growth.
- Provides the parametric framework we use, but on macro returns rather than GDP.
- **What we take:** the idea that economic time series live in a small number of discrete regimes that switch according to a Markov chain.

### Ang & Bekaert (2002), "International Asset Allocation with Regime Shifts"
*Review of Financial Studies*, 15(4), 1137–1187.

- Demonstrates that ignoring regimes produces substantial welfare losses in international allocation.
- Uses a regime-switching VAR for joint returns.
- **What we take:** evidence base that regime-aware allocation pays off out-of-sample → motivates our project goal.

### Guidolin & Timmermann (2007), "Asset Allocation Under Multivariate Regime Switching"
*Journal of Economic Dynamics and Control*, 31(11), 3503–3544.

- Formalizes multi-asset allocation under regime switching with dynamic optimization.
- Shows utility gains across CRRA risk aversions $\gamma \in \{2, 5, 10\}$.
- **What we take:** the CRRA utility sensitivity analysis is a direct template for our $\gamma$ sensitivity plot.

### Nystrup, Madsen & Lindström (2018), "Dynamic Portfolio Optimization Across Hidden Market Regimes"
*Quantitative Finance*, 18(1), 83–95.

- Practical estimation challenges in HMM-based allocation. Discusses overfitting, lookback selection, and walk-forward validation.
- **What we take:** the *annual refit* protocol for HMMs to avoid look-ahead bias.

## Tier 2, POMDP foundations

### Kaelbling, Littman & Cassandra (1998), "Planning and Acting in Partially Observable Stochastic Domains"
*Artificial Intelligence*, 101(1–2), 99–134.

- The canonical POMDP reference. Defines POMDP tuple, belief MDP, alpha-vector representation of value functions.
- **What we take:** the POMDP-as-belief-MDP formulation, the proof that the value function is piecewise linear and convex in belief.

### Littman, Cassandra & Kaelbling (1995), "Learning Policies for Partially Observable Environments: Scaling Up"
*ICML 1995*.

- Introduces the **QMDP** approximation:
$$Q_{\text{QMDP}}(b, a) = \sum_s b(s)\, Q^*_{\text{MDP}}(s, a).$$
- Shows QMDP is exact on fully observable problems and a reasonable heuristic on POMDPs with moderate uncertainty.
- **What we take:** QMDP is the core solver we implement. We cite this paper in the report's Methods section.

### Pineau, Gordon & Thrun (2003), "Point-Based Value Iteration: An Anytime Algorithm for POMDPs"
*IJCAI 2003*.

- PBVI maintains a finite set of belief points and prunes alpha-vectors.
- **What we take:** *extension*. If QMDP underperforms, PBVI is the next thing to try (mentioned in `docs/05_experimental_design.md` as a stretch experiment).

## Tier 3, Practical HMM tooling

### Rabiner (1989), "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition"
*Proceedings of the IEEE*, 77(2), 257–286.

- The HMM tutorial. Defines forward-backward, Baum–Welch (EM), Viterbi.
- **What we take:** the entire HMM training procedure. `hmmlearn` (Python) implements all three.

### `hmmlearn` library (Python)
- `GaussianHMM` for continuous emissions.
- Supports `n_components` (number of states), `covariance_type` ('full', 'diag', 'tied'), and `n_iter` for EM.
- Used in `src/models/hmm.py`.

## Tier 4, Backtesting / finance practice

### Lopez de Prado (2018), *Advances in Financial Machine Learning*
- Walk-forward / combinatorial purged cross-validation.
- Embargo period to prevent leakage.
- **What we take:** the protocol for walk-forward backtesting with refits.

### Bailey, Borwein, Lopez de Prado, Zhu (2014), "Pseudo-Mathematics and Financial Charlatanism"
- The Sharpe ratio inflation from backtest overfitting.
- **What we take:** the *deflated Sharpe ratio* as a sanity check on our reported Sharpe.

## Tier 5, Data references

### NBER Business Cycle Dating Committee
- https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
- We use NBER recession dummies (FRED `USREC`) for qualitative regime validation: "do our model's 'bear' regime probabilities spike when NBER calls a recession?"

### FRED (Federal Reserve Economic Data)
- VIX: `VIXCLS` (Dec 1989 – present).
- 10Y–3M Treasury spread: `T10Y3M`.
- HY OAS: `BAMLH0A0HYM2` (Dec 1996 – present).

### Yahoo Finance via `yfinance` Python package
- `SPY` (S&P 500 ETF, Jan 1993 – present), `AGG` (US Aggregate Bond ETF, Sep 2003 – present).

---

## Concept compatibility map: their paper → our code

| External concept | Lives in our repo as |
|---|---|
| HMM Baum-Welch (Rabiner) | `src/models/hmm.py::fit_hmm` |
| POMDP belief update (Kaelbling) | `src/models/belief_filter.py::update_belief` |
| QMDP approximation (Littman) | `src/models/qmdp.py::qmdp_policy` |
| Walk-forward refit (Nystrup, Lopez de Prado) | `src/backtest/backtest.py::walk_forward` |
| CRRA utility (Guidolin-Timmermann) | `src/utils/utility.py::crra` |
| Deflated Sharpe (Bailey) | `src/utils/metrics.py::deflated_sharpe` |
