# 09 · ENGS 177 Supplementary Material: Theoretical Anchors for the QMDP Project

> Reading of the eleven supplementary papers in the ENGS 177 folder. Anchors every theoretical claim in our extended QMDP paper to a published result.

---

## 1. Per-Paper Summaries

### Auer, Cesa-Bianchi & Fischer (2002), *Finite-time Analysis of the Multiarmed Bandit Problem*
Closes the gap between Lai-Robbins (1985) asymptotic logarithmic regret and what one can actually compute in finite time. Four theorems for index policies on K-armed bandit with bounded rewards in [0,1]:
- **UCB1 (Thm 1):** expected regret ≤ 8 Σ (ln n)/Δᵢ + (1 + π²/3) Σ Δⱼ uniformly in n.
- **UCB2 (Thm 2):** tightens leading constant arbitrarily close to 1/(2Δᵢ²).
- **ε_n-GREEDY (Thm 3):** ε_n = c/n achieves logarithmic per-step regret if c is large enough.
- **UCB1-NORMAL (Thm 4):** logarithmic regret for Gaussian arms with unknown mean and variance.

Canonical reference for "UCB has O(log T) regret." Used in our paper for the no-info bandit baseline.

### Dayan (1992), *The Convergence of TD(λ) for General λ*
Extends Sutton 1988 from TD(0) to TD(λ) for all λ ∈ [0,1] via Watkins' λ-weighted n-step backup recast. Three contributions: in-expectation theorem for general λ; convergence with linearly dependent state representations (to LS projection, not LMS); **with-probability-one convergence** of a modified TD prediction algorithm via Watkins' Action Replay Process. Legitimizes the entire TD(λ) family on absorbing Markov chains, what Lecture 10 silently relies on.

### Jaakkola, Jordan & Singh (1994), *Convergence of Stochastic Iterative DP Algorithms*
The **unifying stochastic-approximation theorem** subsuming Watkins/Dayan + Dayan/Sutton. Central theorem: Δ_{n+1}(x) = (1 − αₙ) Δₙ + βₙ Fₙ converges to 0 w.p.1 if (i) Robbins-Monro conditions hold; (ii) the expected update operator is a contraction in some weighted max-norm with rate γ<1; (iii) variance bounded by C(1+‖Δₙ‖)². Q-learning and TD(λ) convergence are one-line corollaries. **One-stop citation** for "standard stochastic-approximation arguments."

### Precup, Sutton & Singh (2000), *Eligibility Traces for Off-Policy Policy Evaluation*
Introduces and analyzes five eligibility-trace algorithms for off-policy evaluation in model-free tabular setting. Data from behavior policy b; evaluate Q^π for target policy π where b is soft. Key methods:
- **First-visit IS estimator** Q^IS = (1/M) Σ R_m w_m, consistent, unbiased, high-variance.
- **Weighted IS estimator** Q^ISW, biased, consistent, lower-variance.
- **Per-decision IS (Thm 1)**, breaks trajectory weight into per-step factors; consistent and unbiased; reduces variance further.
- **Per-decision weighted IS**, biased, consistent.
- **Tree-backup**, no IS correction needed (averages over all target-policy actions).

Needed the moment we evaluate a fixed candidate QMDP policy on data from a different exploratory policy.

### Singh & Sutton (1996), *RL with Replacing Eligibility Traces*
Distinction between **accumulating** and **replacing** traces in TD(λ).
- **Thm 1 (equivalence of batch replace-TD(1) and first-visit MC):** With repeated presentations, batch replace-trace TD(1) computes exactly the first-visit MC estimate.
- **Thm 3 (equivalence of batch accumulate-TD(1) and every-visit MC):** Accumulating-trace computes every-visit MC (the LMS estimate).
- **Thm 4 (w.p.1 convergence of replace TD(λ))** under Jaakkola conditions.

Theoretical anchor for "first-visit MC is unbiased", sentence Lecture 9 uses without citation.

### Singh, Jaakkola, Littman & Szepesvári (2000), *Convergence Results for Single-Step On-Policy RL*
**First published proof that SARSA(0) converges to Q*** under reasonable exploration.
- **Thm 1 (GLIE):** If learning policy is Greedy in Limit with Infinite Exploration, then SARSA(0) Q_t → Q* and π_t → π* w.p.1. Extends Jaakkola contraction so the contraction property need only hold asymptotically. Appendix B shows ε-greedy with ε_t(s) = c/n_t(s) is GLIE.
- **Thm 2:** SARSA(0) under fixed rank-based exploration converges to Q̄ that solves a modified Bellman equation.

The "GLIE convergence" result Lecture 9 alludes to.

### Van Hasselt (2010), *Double Q-learning*
Diagnoses Q-learning's **maximization bias**: same samples used to select and evaluate the max action, so max_a Q̂(s',a) is positively biased for max_a E[Q̂(s',a)] (Jensen's inequality on convex max function).
- **Double estimator (Lemma 1):** Use independent estimator sets; one selects the maximizer, the other evaluates. Shown to have *negative* bias.
- **Double Q-learning:** Maintain Q^A and Q^B; coin-flip per step, update one using the other for evaluation. Convergence to Q* w.p.1 in finite MDPs under Robbins-Monro + infinite visitation.

Empirical sections: massive gains on stochastic gridworlds.

### Watkins & Dayan (1992), *Q-Learning*
Formal convergence proof:
> Given bounded rewards |rₙ| ≤ R, learning rates 0 ≤ αₙ < 1 with Σ α_{nᵢ(x,a)} = ∞ and Σ α²_{nᵢ(x,a)} < ∞ for every (x,a), and every (x,a) visited infinitely often, Qₙ(x,a) → Q*(x,a) w.p.1.

Proof via **Action Replay Process (ARP)**: constructed auxiliary Markov process where each "card" corresponds to one observed tuple. Two lemmas: (A) Q-values are optimal action-values of ARP at level n; (B) ARP transition probabilities and rewards converge w.p.1 to true MDP.

### Powell (2019), *A Unified Framework for Stochastic Optimization*
27-page EJOR review proposing single notational/conceptual umbrella for 15 overlapping communities.

**Universal model:** max_π E[Σ_t C_t(S_t, X^π_t(S_t), W_{t+1}) | S_0] with transition S_{t+1} = S^M(S_t, x_t, W_{t+1}) and exogenous info (W_1,...,W_T). Find best *policy*, not best decision.

**Four meta-classes of policies:**
1. **Policy Function Approximations (PFAs).** Direct parametric maps X^π(S | θ).
2. **Cost Function Approximations (CFAs).** X^π(S | θ) = argmax C̄^π(S, x | θ). **UCB is a CFA.**
3. **Value Function Approximations (VFAs).** X^π(S) = argmax (C(S,x) + E[V̄_{t+1}(S_{t+1})]). Classical ADP/RL.
4. **Direct Lookahead Approximations (DLAs).** Solve simplified lookahead model online: deterministic, rollout, MCTS, two-stage SP.

All four classes should be considered for any problem. PFA+CFA = "policy search"; VFA+DLA = "lookahead approximations". For POMDPs (Section 2.15), reformulate as belief MDP whose state is belief b^n; notoriously hard, motivating point-based solvers and approximations.

### Shapiro (2021), *Tutorial on Risk-Neutral, Distributionally Robust, and Risk-Averse Multistage Stochastic Programming*
Pedagogical review of MSP as **distinct paradigm** from MDP/DP. Nested-expectation formulation: min_π E[Σ c_t(x_t, ξ_t)] s.t. dynamics. Covers scenario-tree discretization (exponential N^T blow-up); DP reformulation (cost-to-go V_t); risk-averse formulations via nested coherent risk measures (CVaR, MV) and time-consistency issues; distributionally robust formulations via Wasserstein or moment-based ambiguity sets.

For us: alternative paradigm to discuss in related work, instead of approximating Q*, write the problem as a multistage SP.

---

## 2. Mapping to ENGS 177 Lectures

| Paper | Lecture(s) | What lecture uses |
|---|---|---|
| Auer 2002 | L8 (MAB) | UCB1 regret theorem; ε_n-GREEDY analysis |
| Dayan 1992 | L10 (TD) | Lecture 10 §2.3 cites Thm 1 verbatim |
| Jaakkola 1994 | L9, L10 | Stochastic-approximation skeleton |
| Precup 2000 | L9 §6 (Off-Policy MC) | IS formulation, per-decision algorithm |
| Singh 1996 | L9 §4.3 | "First-visit MC is unbiased" |
| Singh 2000 | L9, L10 | GLIE convergence; SARSA convergence |
| Van Hasselt 2010 | L10 §7 | Cited by name; Jensen bias reproduced |
| Watkins 1992 | L10 §5 | Convergence of Q-learning |
| Powell 2019 | (meta) | Positioning in stochastic-opt literature |
| Shapiro 2021 | (meta) | Alternative paradigm for related work |

---

## 3. Relevance to Our QMDP Project

**Singh 1996 / Singh 2000, MC convergence backbone.** Our MC reward estimation under QMDP-induced policy: Singh 1996 gives unbiasedness of first-visit / bias of every-visit; Singh 2000 provides asymptotic argument under GLIE. ε-greedy with ε_t = c/n_t is the exact recipe.

**Dayan 1992, TD(0) convergence to back any TD extension.** If we add bootstrapped value-function estimation over belief states (or underlying state with QMDP control), Dayan 1992 is the convergence reference.

**Watkins & Dayan 1992, Q-learning convergence if we add model-free baseline.** Natural baseline: "what if we did not assume the model and learned Q* by Q-learning instead?"

**Van Hasselt 2010, Double Q-learning to avoid maximization bias.** Should use over vanilla Q-learning whenever reward distribution is stochastic (typical POMDP regime).

**Precup, Sutton & Singh 2000, Off-policy MC evaluation via IS.** Citation if we evaluate candidate QMDP policy from data generated under different policy. Per-decision IS (Thm 1) is lowest-variance unbiased option.

**Auer 2002, UCB bandit baseline.** "Single-state" baseline that ignores belief, treats each action as a bandit arm. Even ignoring partial observability, best stateless policy incurs logarithmic regret.

**Powell 2019, Taxonomy positioning.** **The most important meta-citation.** QMDP fits naturally as a **direct lookahead approximation (DLA)**: X^QMDP_t(b_t) = argmax_a Σ_s b_t(s) Q*_MDP(s,a). Equivalently a **VFA** in which belief-state value V(b) is approximated by linear lower envelope Σ_s b(s) max_a Q*(s,a). In Powell's language: **using MDP value function as VFA on belief MDP and selecting actions via one-step DLA**. Explicit framing motivates alternative design choices (PFA: parametric belief→action; CFA: exploration-bonus QMDP; full DLA: POMCP-style online belief search).

**Jaakkola 1994, Unified convergence umbrella.** Appendix proofs: cite as "by standard stochastic-approximation arguments."

**Shapiro 2021, Alternative paradigm.** Stochastic programming approaches problem differently. Pointer for "another way to handle multistage decisions under uncertainty is the stochastic-programming approach, which scales differently with horizon (exponential in scenarios) versus our approximate-DP (polynomial in belief dimension)."

---

## 4. Synthesis: Most Theoretically Grounded Extension

**Paragraph 1, Anchoring current implementation.** Our QMDP pipeline does two things: (1) solves fully observable MDP for Q*(s,a); (2) generates trajectories under QMDP policy and uses MC to estimate realized expected return. Anchors: for (1), L7's VI analysis (contraction in max-norm with rate λ) suffices. For (2), Singh & Sutton (1996, Thm 3) for first-visit MC unbiasedness; Singh et al. (2000, Thm 1) for asymptotic convergence under GLIE if any randomization is used. State Robbins-Monro conditions explicitly, cite Jaakkola/Jordan/Singh (1994, Thm 1) as general SA result. **Crucial reframing using Powell (2019)'s taxonomy in introduction:** "QMDP is a VFA policy in which the belief-MDP value function is approximated by the lower envelope V_QMDP(b) = max_a Σ_s b(s) Q*_MDP(s,a), with a one-step DLA at decision time. We compare this against alternative policy classes within Powell's (2019) four-class taxonomy." This sentence gives the project theoretical structure.

**Paragraph 2, Strongest extension: off-policy evaluation + model-free baseline.** **Off-policy evaluation of QMDP using per-decision IS (Precup 2000) on data generated by an exploratory behavior policy**, paired with **a Double Q-learning baseline (Van Hasselt 2010) that does not assume model knowledge**. Reasoning: current MC estimator is on-policy, forecloses ability to evaluate competing candidate policies without expensive re-simulation. Precup's per-decision IS lets one set of trajectories evaluate every candidate in parallel. Adding Double Q-learning gives: (a) model-free comparison point that does not assume QMDP transition matrix is correct; (b) addresses maximization bias that vanilla Q-learning suffers in stochastic POMDPs. Plus one-paragraph "alternative paradigm" citing Shapiro (2021), VFA/DLA vs multistage SP. Plus one-paragraph "no-information baseline" citing Auer (2002) UCB1, agent ignores belief, treats actions as bandit arms with O(log T) regret. **All positioned within Powell (2019)'s four-class taxonomy.**

---

## 5. BibTeX Block (paste into references.bib)

```bibtex
@article{Auer2002,
  author  = {Auer, Peter and Cesa-Bianchi, Nicol\`o and Fischer, Paul},
  title   = {Finite-time Analysis of the Multiarmed Bandit Problem},
  journal = {Machine Learning},
  volume  = {47}, number = {2--3}, pages = {235--256}, year = {2002}
}
@article{Dayan1992,
  author  = {Dayan, Peter},
  title   = {The Convergence of {TD}($\lambda$) for General $\lambda$},
  journal = {Machine Learning}, volume = {8}, number = {3--4},
  pages   = {341--362}, year = {1992}
}
@inproceedings{Jaakkola1994,
  author    = {Jaakkola, Tommi and Jordan, Michael I. and Singh, Satinder P.},
  title     = {Convergence of Stochastic Iterative Dynamic Programming Algorithms},
  booktitle = {NIPS 6}, pages = {703--710}, year = {1994}
}
@inproceedings{Precup2000,
  author    = {Precup, Doina and Sutton, Richard S. and Singh, Satinder},
  title     = {Eligibility Traces for Off-Policy Policy Evaluation},
  booktitle = {ICML 2000}, pages = {759--766}, year = {2000}
}
@article{Singh1996,
  author  = {Singh, Satinder P. and Sutton, Richard S.},
  title   = {RL with Replacing Eligibility Traces},
  journal = {Machine Learning}, volume = {22}, pages = {123--158}, year = {1996}
}
@article{Singh2000,
  author  = {Singh, Satinder and Jaakkola, Tommi and Littman, Michael L. and Szepesv\'ari, Csaba},
  title   = {Convergence Results for Single-Step On-Policy RL Algorithms},
  journal = {Machine Learning}, volume = {39}, pages = {287--308}, year = {2000}
}
@inproceedings{VanHasselt2010,
  author    = {van Hasselt, Hado},
  title     = {Double {Q}-learning},
  booktitle = {NIPS 23}, pages = {2613--2621}, year = {2010}
}
@article{Watkins1992,
  author  = {Watkins, Christopher J. C. H. and Dayan, Peter},
  title   = {{Q}-learning},
  journal = {Machine Learning}, volume = {8}, pages = {279--292}, year = {1992}
}
@article{Powell2019,
  author  = {Powell, Warren B.},
  title   = {A Unified Framework for Stochastic Optimization},
  journal = {European Journal of Operational Research},
  volume  = {275}, number = {3}, pages = {795--821}, year = {2019},
  doi     = {10.1016/j.ejor.2018.07.014}
}
@article{Shapiro2021,
  author  = {Shapiro, Alexander},
  title   = {Tutorial on Risk Neutral, Distributionally Robust and Risk Averse Multistage Stochastic Programming},
  journal = {European Journal of Operational Research},
  volume  = {288}, number = {1}, pages = {1--13}, year = {2021},
  doi     = {10.1016/j.ejor.2020.03.065}
}
```
