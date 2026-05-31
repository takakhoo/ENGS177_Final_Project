# 03 · Intuition Primer (Ten Analogies)

> If you've read the [`01_project_story.md`](01_project_story.md), this file tells you *why* each piece works, in plain English. If the math in the extended report or `02_class_concepts.md` is overwhelming, read this first.

---

## 1. The whole project, in one analogy

**Imagine you are a doctor.** Your patient is the stock-bond market. At every check-up (monthly rebalance) you must prescribe a treatment (portfolio weights). The patient is either *healthy* (bull regime) or *sick* (bear regime), but you cannot examine their internal state directly, you only see symptoms (VIX level, yield-curve slope). Different treatments work better in different states. Your goal: read the symptoms, infer the latent state, prescribe accordingly.

This is exactly a POMDP. Latent state $s_t$ = regime; observation $\mathbf{o}_t$ = symptom vector; action $\mathbf{a}_t$ = portfolio weight; reward $R(s,\mathbf{a})$ = utility of realised return; transition $T(s'\mid s)$ = how diseases progress.

---

## 2. HMM as "a casino dealer secretly switching decks"

**Imagine a poker dealer who, between rounds, secretly swaps between two decks.** Deck A has lots of high cards (the bull regime: positive expected returns). Deck B has lots of low cards (the bear regime: negative). You never see which deck is in play. You only see the cards.

- **Transition matrix** $T$ = the dealer's probability of switching decks. We estimate $\hat T \approx [[0.96, 0.04], [0.06, 0.94]]$, sticky.
- **Emission** $O(\mathbf{o}\mid s) = \mathcal{N}(\boldsymbol\mu_s, \boldsymbol\Sigma_s)$ = the card distribution under each deck.
- **Baum–Welch** (EM) is how we figure out, from a long stream of observed cards, what each deck's distribution must be and how often the dealer switches.

---

## 3. The Bayes filter as "updating your guess about which deck is in play"

After every card the dealer reveals, you update your guess about which deck is active.

- **Predict step:** "Even before seeing this card, the dealer might have switched." $\hat b_t(s') = \sum_s T(s'\mid s)\,b_{t-1}(s)$. If you were 70% sure it was Deck A last round and switch probability is 4%, you become 69% sure even without seeing anything.
- **Update step:** "A high card just came out, more likely Deck A." $b_t(s') \propto O(\mathbf{o}_t \mid s') \cdot \hat b_t(s')$. Multiply by likelihood, renormalise.

One observation of VIX=35 (a "low card" under the bull dealer, high under the bear) flips belief from 70/30 bull to 93/7 bear in one step. That responsiveness is the whole point of the filter.

---

## 4. The MDP as "the dealer dealt face-up"

**Suppose the dealer turned the decks face-up**, you can see at every moment which deck is in use. What's the best action in each deck? That's the underlying MDP. Solve the Bellman equation by value iteration or policy iteration. The answer is a table: "in Deck A, do X; in Deck B, do Y."

This is the easier sub-problem. Once we have $Q^\ast(s,a)$ (value of taking action $a$ when we *know* we're in state $s$), the POMDP question becomes: given that I don't know which deck I'm in but I have a belief, what should I do?

---

## 5. QMDP as "acting as if the dealer secretly told you"

The QMDP rule: act as if the dealer privately whispered "you're in Deck $s$ with probability $b(s)$," then take the action that maximises expected $Q^\ast$ under that whisper:

$$\pi_{\rm QMDP}(\mathbf{b}) = \arg\max_a \sum_s b(s) Q^\ast(s, a).$$

**Concrete walkthrough.** Belief after VIX=35 observation: $\mathbf{b} = (0.07, 0.93)$. At γ=8 the $Q^\ast$ table is regime-differentiated:

| Action | Q*(bull) | Q*(bear) | b · Q*(·, a) |
|---|---:|---:|---:|
| 100/0 (all stocks) | 0.32 | 0.10 | 0.115 |
| 60/40 | 0.24 | 0.20 | 0.203 |
| **40/60** | 0.20 | **0.22** | **0.219** |
| 0/100 (all bonds) | 0.05 | 0.18 | 0.171 |

QMDP picks 40/60. Bear-heavy belief steers toward a bond-heavy action. **At γ=2 (our headline), the 100/0 column dominates both regimes**, so QMDP always picks 100/0 regardless of belief. The collapse.

---

## 6. Why QMDP collapsed at γ=2: "a low-risk-aversion gambler bets the same regardless of suspicion"

A gambler with low risk aversion has so much faith in expected value over downside that even 93% suspicious it's the "bad deck," the long-run discounted continuation value of the high-EV action still wins. Only a sufficiently risk-averse gambler (γ ≥ 8) lets the suspicion shift the policy.

---

## 7. Why richer observations fix it: "more thermometers, better diagnosis"

Two thermometers (VIX, term spread) measure roughly the same axis: how spooked is the equity market. They rise together during stress; they don't add independent information.

**Adding NFCI and STLFSI4** is like giving the doctor a blood test and an X-ray. NFCI integrates equity, bond, money-market, and shadow-banking signals. STLFSI4 aggregates 18 different stress indicators. They move *differently* from VIX in subtle ways, during the 2022 inflation shock, VIX was moderate but NFCI/STLFSI spiked sharply. With these channels the HMM can identify a richer bear regime where conditional bond return meaningfully beats conditional stock return, sharp enough that even a low-γ MDP shifts toward bonds.

Our cohort study confirms: VIX+spread only → regime label non-actionable. Add NFCI+STLFSI4 → same γ=2 MDP produces fully regime-differentiated policy (100/0 bull, 0/100 bear).

---

## 8. Why walk-forward refit fixes it: "re-reading the casino's recent hands"

A fixed HMM is like a player who learned the casino's house edges 20 years ago and never updated. The casino has changed since then, new dealers, new decks, different switching probabilities. Strategies based on stale parameters mis-time their bets.

**Walk-forward expanding-window refit** re-estimates the HMM every year using all data observed up to that point. The model tracks slow drift in regime moments (e.g., post-2008 the bear-state mean became less negative as central banks intervened earlier). With proper refit, QMDP's Sharpe lifts 0.81 → 1.08, a 33% improvement just from re-reading the recent hands.

---

## 9. Why Faber wins: "surf the wave instead of predicting which wave is coming"

**Time-series momentum / Faber 10-month moving-average:** hold an asset if its price is above its 10-month average, else move to cash. No regime model, no Bayesian filter, no MDP. Just: "is the trend my friend right now?"

Why it works: trend-following exploits *persistence in returns* (momentum), which is a stylised fact across centuries (Hurst–Ooi–Pedersen 2017). It is the empirical sweet spot of robustness vs simplicity. Our 12-strategy horse race confirms: Faber dominates Sharpe 1.68 vs static 0.81 vs QMDP 0.73. The simpler model wins because it doesn't try to predict the wave, it just rides it.

**The lesson:** regime-aware allocation is theoretically appealing but empirically fragile. Even with all our diagnostic unlocks (richer observations + walk-forward refit + higher γ), our best QMDP variant achieves Sharpe 1.08, competitive with trend-following but not dominant.

---

## 10. The honest verdict (claims vs non-claims)

| We claim | We do NOT claim |
|---|---|
| POMDP is the right *minimum* tool for this decision problem | POMDP asset allocation beats simple trend-following |
| The headline negative finding is a diagnostic of methodology, not fundamentals | The original QMDP at γ=2 is a useful production policy |
| The HMM signal is informative (other HMM-aware baselines beat QMDP) | Our walk-forward QMDP variant is the new state-of-the-art |
| Three components separately reverse the negative finding (observations, refit, γ) | Our Sharpe gains net of transaction costs are large enough to matter to a real allocator |

---

## End of primer

The remaining docs and the [`extended_report.pdf`](../report/extended_report.pdf) provide the formal treatment, referencing these analogies by name.
