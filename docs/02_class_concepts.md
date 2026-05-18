# 02 · Class Concepts Used (ENGS 177 Lectures Mapped to Project)

Every component of this project is built directly on material covered in ENGS 177 lectures. This file is the bridge between course notes and code. Cite this file in the report's "Methods" section.

---

## Lecture 2 — Bayesian Networks & Bayesian Inference
**Where used:** Belief update $b_t \to b_{t+1}$ given observation $o_{t+1}$.
$$b_{t+1}(s') = \frac{O(o_{t+1} \mid s')\, \sum_s T(s' \mid s)\, b_t(s)}{\sum_{s''} O(o_{t+1} \mid s'')\, \sum_s T(s'' \mid s)\, b_t(s)}.$$

This is Bayes' rule applied recursively: prior $\to$ predictive $\to$ posterior. Implemented in `src/models/belief_filter.py`.

## Lecture 3 — Markov Chains
**Where used:** Regime dynamics. The latent state $\{s_t\}$ is a Markov chain on $S = \{s_1, \ldots, s_K\}$ with transition matrix $T \in \mathbb{R}^{K \times K}$. We use the **stationary distribution** of this chain as the initial regime prior $b_0$. Code: `src/models/hmm.py` (initial-state probabilities).

We also reference Lecture-3 results on **hitting times** and **mean recurrence times** to interpret the stationary regime durations (e.g., "expected length of a bear regime").

## Lecture 4 — Decision Trees & Influence Diagrams
**Where used (negatively):** In the proposal and report we argue *why* a decision tree / influence diagram is the wrong tool here (exponential blow-up over time, no shared state). We include a small influence diagram in the report illustrating one rebalance decision to show what would happen at the wrong horizon.

## Lecture 5 — MDP Formulation
**Where used:** The *underlying* fully-observable MDP $(\mathcal{T}, S, A, T, R, \lambda)$ is exactly the Lec 5 tuple. We use Lec 5 vocabulary verbatim (states, actions, rewards, transitions, discount).

## Lecture 6 — Finite-Horizon DP / Backward Induction
**Where used:** As a *sanity check*. For a small horizon $T = 12$ months, we run **backward induction** to verify our infinite-horizon value iteration converges to the right neighborhood. Provides a finite-horizon vs. infinite-horizon comparison plot.

## Lecture 7 — Infinite-Horizon DP (Value & Policy Iteration)
**Where used:** The core MDP solver.
- **Value iteration** on the regime grid: $v^{n+1}(s) = \max_a \{R(s,a) + \lambda \sum_{s'} T(s'|s) v^n(s')\}$.
- Stopping criterion $\|v^{n+1} - v^n\|_\infty < \varepsilon(1-\lambda)/(2\lambda)$.
- **Policy iteration** for cross-check: solve $(I - \lambda P_\pi)v = r_\pi$ then greedy improvement.
- Both implemented in `src/models/mdp.py` and demonstrated as agreeing.

## Lecture 8 — Multi-Armed Bandits (NOT directly used, but referenced)
**Where used:** In **discussion / extensions**. The exploration-exploitation trade-off appears if we relax to model-free RL (Q-learning, UCB-style state-action exploration). We frame our POMDP setting as "MAB with side information and known dynamics" and discuss in the report why model-based methods dominate here.

## Lecture 9 — Monte Carlo Methods (course week)
**Where used:** Validation. We Monte-Carlo simulate the HMM-implied return process and run the policy on simulated paths, then compare to the live backtest. Bias and confidence bands from MC. Code: `experiments/06_montecarlo_validation.py` (planned).

## Lecture 10 — Temporal Difference / Q-learning (not on Quiz 3, but covered)
**Where used:** Extension only. Discussed in conclusions as alternative model-free baseline. We do not run TD in core experiments because (a) the data is too short, (b) the model-based pipeline dominates with known structure.

---

## Specific formulas from the Quiz 3 formula sheet that appear in code

| Sheet item | Implemented where |
|---|---|
| Closed-form policy eval $\vect{u}^\pi_\lambda = (I - \lambda P_\pi)^{-1} r_\pi$ | `src/models/mdp.py::policy_eval_exact` |
| Bellman operator $Lv = \max_\pi\{r_\pi + \lambda P_\pi v\}$ | `src/models/mdp.py::bellman_step` |
| VI stopping criterion $\varepsilon(1-\lambda)/(2\lambda)$ | `src/models/mdp.py::value_iteration` |
| Optimality eq $v^*(s) = \max_a \{r + \lambda \sum_j p(j|s,a) v^*(j)\}$ | core of `value_iteration` |
| Optimal policy rule $\pi^*(s) \in \argmax_a\{...\}$ | `extract_greedy_policy` |
| Bayes filter (Lec 2) | `src/models/belief_filter.py` |

---

## Concepts NOT in class but used (cited as extensions)

- **QMDP approximation** (Littman, Cassandra, Kaelbling 1995): outside the syllabus but a natural extension of Lec 7 MDPs to partial observability.
- **Baum–Welch EM for HMM** (Rabiner 1989): outside Lec 2's exact-inference scope; covered as part of HMM tooling.
- **Walk-forward backtest with annual refitting**: standard finance practice, not class material.

We are explicit about this in the report's "Methods" section: every non-class concept gets a one-sentence justification + reference.
