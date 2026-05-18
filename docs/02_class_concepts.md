# 02 · Mathematical Foundations

Every component of this project is built on a standard set of mathematical tools from sequential decision-making under uncertainty. This file is the bridge between those tools and the code in `src/`.

---

## Bayesian inference and the belief filter

The belief update $b_t \to b_{t+1}$ given observation $o_{t+1}$ is a recursive application of Bayes' rule:

$$b_{t+1}(s') = \frac{O(o_{t+1} \mid s')\sum_s T(s' \mid s)\,b_t(s)}{\sum_{s''}O(o_{t+1} \mid s'')\sum_s T(s' \mid s)\,b_t(s)}.$$

Prior to predictive to posterior. Implemented in `src/models/qmdp.py::update_belief`.

## Markov chain over regimes

The latent state sequence $\{s_t\}$ is a Markov chain on $S = \{s_1, \ldots, s_K\}$ with transition matrix $T \in \mathbb{R}^{K \times K}$. We use the stationary distribution of this chain as the initial regime prior $b_0$. Code: `src/models/hmm.py` (initial-state probabilities) and `src/models/qmdp.py::stationary_distribution`.

We also use Markov-chain results on hitting times and mean recurrence times to interpret stationary regime durations (e.g., "expected length of a bear regime").

## Why not a decision tree or influence diagram

A decision tree at horizon $H$ has $|A|^H \cdot |O|^H$ leaves, intractable beyond a few periods. An influence diagram exhibits the same exponential blow-up when copied across decision epochs. Neither shares state structure across rebalances, which is exactly what an MDP/POMDP formulation gives us for free.

## MDP framework

The underlying fully-observable MDP is the tuple $(\mathcal{T}, S, A, T, R, \lambda)$. We use this vocabulary throughout the codebase: states, actions, transitions, rewards, discount factor.

## Backward induction (finite-horizon)

As a sanity check, for a small horizon $T = 12$ months we run

$$V_t^*(s) = \max_a \{R(s,a) + \lambda \sum_{s'} T(s'|s)\, V_{t+1}^*(s')\}$$

backward in $t$ to verify our infinite-horizon value iteration converges to the right neighbourhood. This gives a finite-horizon vs. infinite-horizon comparison plot.

## Infinite-horizon dynamic programming

The core MDP solver:

- **Value iteration:** $v^{n+1}(s) = \max_a \{R(s,a) + \lambda \sum_{s'} T(s'|s)\, v^n(s')\}$, stopping when $\|v^{n+1} - v^n\|_\infty < \varepsilon(1-\lambda)/(2\lambda)$.
- **Policy iteration:** solve $(I - \lambda P_\pi)v = r_\pi$ (closed-form policy evaluation) then greedy improvement.

Both implemented in `src/models/mdp.py` and demonstrated to agree exactly in `experiments/04_qmdp_solve.py`.

## Multi-armed bandits (discussion)

The exploration-exploitation trade-off appears when we relax to model-free reinforcement learning (Q-learning, UCB-style state-action exploration). We frame our POMDP setting as "bandit with side information and known dynamics" and discuss in the report's conclusions why model-based methods dominate in this domain.

## Monte Carlo validation

We Monte-Carlo simulate the HMM-implied return process and run the policy on simulated paths, then compare to the live backtest. Bias and confidence bands come from MC. Planned code: `experiments/06_montecarlo_validation.py` (stretch).

## Temporal difference / Q-learning (extension)

Discussed in conclusions as an alternative model-free baseline. We do not run TD in core experiments because the data is short (271 months) and the model-based pipeline already dominates with known structure.

---

## Formula-to-code map

| Mathematical object | Implemented in |
|---|---|
| Closed-form policy evaluation $u^\pi_\lambda = (I - \lambda P_\pi)^{-1} r_\pi$ | `src/models/mdp.py::policy_evaluation_exact` |
| Bellman operator $Lv = \max_\pi\{r_\pi + \lambda P_\pi v\}$ | `src/models/mdp.py::value_iteration` (inner loop) |
| Value-iteration stopping criterion $\varepsilon(1-\lambda)/(2\lambda)$ | `src/models/mdp.py::value_iteration` |
| Optimality equation $v^*(s) = \max_a \{r + \lambda \sum_j p(j|s,a) v^*(j)\}$ | core of `value_iteration` |
| Optimal policy rule $\pi^*(s) \in \argmax_a\{...\}$ | `src/models/mdp.py` (return of VI/PI) |
| Bayesian belief filter | `src/models/qmdp.py::update_belief` |

---

## Extensions not in the syllabus, used here

- **QMDP approximation** (Littman, Cassandra, Kaelbling, 1995): natural extension of infinite-horizon MDPs to partial observability.
- **Baum-Welch EM for HMM** (Rabiner, 1989): the standard fitting procedure for Gaussian HMMs.
- **Walk-forward backtest with annual refitting:** standard finance practice for evaluating dynamic strategies without look-ahead bias.

Each is cited explicitly in the report and the external-research note.
