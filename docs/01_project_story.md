# 01 · The Project Story

> One narrative, beginning to end. If you only read one file in this folder, read this one.

---

## Where it started

In 2022, a textbook 60/40 stock-bond portfolio lost roughly **17%**. Both stocks AND bonds fell together under persistent inflation, exactly the failure mode the 60/40 mix is sold as protecting against. We asked: *can a decision-theoretic regime overlay, built end-to-end on public macroeconomic data, beat the static benchmark out-of-sample after realistic transaction costs?*

By "decision-theoretic" we meant three things:
1. The allocation rule is the **solution of an explicit optimisation**, not a discretionary heuristic.
2. The latent macro regime is treated as a **hidden state** and inferred by a **Bayesian filter**, not labelled by hand.
3. The policy is the **QMDP approximation of an underlying POMDP**, making it falsifiable, comparable, and explainable.

## The framework: POMDP

Three observations together force the POMDP formulation:
- The **regime is latent**, we never observe it directly.
- The **observations are noisy**, VIX and term spread are imperfect proxies.
- The **decisions are sequential**, current allocation affects future wealth and future beliefs.

Decision trees explode with horizon. Influence diagrams blow up the same way for repeated rebalances. Fully-observable MDPs assume what we can't observe. POMDP is the minimum tool.

In Powell's (2019) four-class taxonomy, our QMDP solution is a **value function approximation** (the underlying MDP's $Q^\ast$ used as a piecewise-linear lower envelope over the belief simplex) combined with a **one-step direct lookahead** at decision time.

## Headline finding (in the original config)

Out-of-sample backtest, 2003–2026, SPY + AGG, monthly rebalance, 5 bps tx cost. We benchmarked QMDP against a static 60/40 and ten other strategies spanning the academic and practitioner consensus.

> **QMDP at CRRA γ=2 finishes LAST of twelve on Sharpe (0.73 vs static's 0.81).** Faber's 10-month moving-average rule wins (Sharpe 1.68). The underlying MDP collapses to "100% stocks" in BOTH regimes, so QMDP is effectively a leveraged equity strategy with the deepest drawdown in the comparison (−53% in 2008).

If we'd stopped there, the story would be: "regime-aware allocation is a bad idea, simple trend-following wins." That's how every regime-switching paper *that publishes a negative result* concludes.

## But the negative finding has a structure

We ran three diagnostic experiments. Each one reverses the headline in isolation.

### Diagnostic 1: γ-sweep
At γ ≥ 8, the underlying MDP shifts to 40/60 across both regimes. QMDP Sharpe crosses the static benchmark. **But** π*(bull) = π*(bear) at *every* γ we tested. The Sharpe gain is a second-moment effect (less equity → less vol → higher Sharpe), not regime-aware tilting.

### Diagnostic 2: multi-feature observation cohort
Guidolin–Timmermann (2007) predict that narrow observation sets fail to differentiate regime-conditional policies. We tested directly: re-fit the 2-state HMM with five different observation cohorts at the same γ=2:

| Cohort | Features | Bull | Bear | Differs? |
|---|---|---|---|---|
| 1. Baseline | VIX, T10Y3M | 100/0 | 100/0 | No |
| 2. +Yield curve | +T10Y2Y | 100/0 | 100/0 | No |
| **3. +Stress** | **+NFCI, +STLFSI4** | **100/0** | **0/100** | **Yes, full flip** |
| 4. +Macro | +NFCI, +UMCSENT, +ICSA | 100/0 | 0/100 | Yes |
| 5. Kitchen sink | 8 channels | 100/0 | 60/40 | Yes |

**Adding the Chicago Fed NFCI and St. Louis Fed STLFSI4 fixes the policy collapse at the same γ=2.** The headline collapse was an observation-channel problem, not a fundamental QMDP problem.

### Diagnostic 3: walk-forward refit
The original backtest fit the HMM ONCE on 2003–2014 and held it fixed. Nystrup et al. (2018) recommend refitting. We tested four cadences:

| Variant | Sharpe | Max DD | Calmar |
|---|---:|---:|---:|
| Static 60/40 (bench) | 0.81 | −34.2% | 0.22 |
| QMDP fixed (report baseline) | 0.96 | −34.2% | 0.29 |
| QMDP annual refit (5y window) | 0.85 | −25.1% | 0.36 |
| QMDP quarterly refit | 0.83 | −23.4% | 0.37 |
| **QMDP expanding window** | **1.08** | **−16.5%** | **0.60** |

**Expanding-window refit lifts QMDP Sharpe from 0.81 → 1.08 (+33%), max drawdown −34% → −16% (more than halves), Calmar 0.22 → 0.60 (nearly triples).**

## The synthesis

Our headline "QMDP loses" holds **only** in the joint configuration of:
- (a) narrow VIX+spread observations,
- (b) fixed in-sample HMM,
- (c) low CRRA risk aversion (γ=2).

Each component is empirically reversible. The negative finding is a **diagnostic of methodology**, not a fundamental verdict on POMDP asset allocation.

This aligns tightly with the literature: Ang–Bekaert (2002) predict it (no risk-free leg, so regime label has nowhere to act); Guidolin–Timmermann (2007) predict it (needs 4+ states or richer observations); Nystrup et al. (2018) predict it (regime models need walk-forward refit); Tu (2010) predicts it (low γ leaves no room for regime signal to dominate). What is unique about our contribution: we publish the **joint diagnostic decomposition**, exactly which methodological component drives the headline verdict, with all three failure modes traced to the relevant prior work.

## The honest verdict

We do **not** claim a Sharpe win for POMDP asset allocation. We **do** claim:
1. The POMDP framework is the right *minimum* tool for the decision problem.
2. The headline negative finding is a diagnostic of methodology, not fundamentals, reversed in isolation by richer observations, walk-forward refit, or higher CRRA.
3. The HMM signal itself is informative, HMM-MV and BL+HMM both beat QMDP, indicating QMDP's projection at γ=2 destroys it.
4. Trend-following is the practitioner-consensus bar, we did not clear it.

## Three extensions in priority order

1. **CVaR-penalised reward**: replace CRRA with $R(s,a)=\mathbb{E}[a^\top r\mid s]-\kappa \mathrm{CVaR}_\alpha[a^\top r\mid s]$ to explicitly penalise bear-regime tail risk.
2. **Point-based value iteration** (Pineau et al. 2003): tighter $\alpha$-vector bound on the 1-D belief simplex; cheap.
3. **Off-policy MC + Double Q-learning** (Precup 2000 + Van Hasselt 2010): model-free baseline that doesn't assume the QMDP transition matrix is correct.

---

## Where to go next in this folder

- **[`02_class_concepts.md`](02_class_concepts.md)**, math machinery (Bayes filter, Markov chains, VI, PI, QMDP) mapped to code.
- **[`03_intuition_primer.md`](03_intuition_primer.md)**, ten analogies (doctor-and-symptoms, casino dealer, surfing waves) that explain every moving part without math.
- **[`04_experiments_guide.md`](04_experiments_guide.md)**, what each `experiments/0X_*.py` script does, in order.
- **[`05_practitioner_baselines_survey.md`](05_practitioner_baselines_survey.md)**, 3,700-word survey of what real institutional allocators actually do.
- **[`06_academic_literature_survey.md`](06_academic_literature_survey.md)**, 3,500-word survey of regime-switching asset allocation literature.
- **[`07_supplementary_papers_synthesis.md`](07_supplementary_papers_synthesis.md)**, 3,400-word synthesis of the 11 ENGS 177 supplementary papers; introduces the Powell (2019) taxonomy reframing.

For full math derivations, code listings, glossary, and per-experiment detail, see [`../report/extended_report.pdf`](../report/extended_report.pdf) (38 pages).
