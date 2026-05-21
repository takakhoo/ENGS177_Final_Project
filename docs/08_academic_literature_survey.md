# 08 · Academic Literature Survey: Regime-Switching Asset Allocation

> Survey of ~25 papers on regime-switching asset allocation, POMDP/QMDP in finance, and tactical-allocation baselines. Built to position our negative finding within the literature.

---

## 1. Foundational Papers

### 1.1 Ang & Bekaert (2002) — "International Asset Allocation with Regime Shifts"
*Review of Financial Studies*, 15(4), 1137–1187.

The canonical reference for the welfare-economics framing of regime-switching portfolio choice. Two-regime VAR on US, UK, German equity returns with "normal" vs "bear" states; CRRA dynamic-programming investor; constant-correlation single-regime benchmark.

**Welfare-gain claim and the catch:** *the costs of ignoring regimes are small for all-equity portfolios but become economically meaningful only when the investor can hold a conditionally risk-free asset*. Regime information is mostly valuable as a switch into cash during high-vol regimes; in an all-stock universe, regimes barely change the optimal weights. **Directly analogous to our negative finding**: with risky-only assets (SPY/AGG), the regime label doesn't move the policy much. Risk aversion γ ∈ {5, 10, 20}.

### 1.2 Ang & Bekaert (2004) — "How Regimes Affect Asset Allocation"
*Financial Analysts Journal*, 60(2), 86–99 (NBER WP 10080).

Practitioner-facing companion. Switching between *cash, bonds, and equities* is where the regime model adds OOS value: the model recommends shifting heavily to cash during persistent high-vol regimes. Essentially **a dynamic vol-targeting story dressed up as a regime story** — and again why our 2-asset SPY/AGG universe at γ=2 finds nothing.

### 1.3 Guidolin & Timmermann (2007) — "Asset Allocation Under Multivariate Regime Switching"
*Journal of Economic Dynamics and Control*, 31(11), 3503–3544.

Gold-standard multi-asset regime-switching allocation paper. **4-state model** (crash, slow growth, bull, recovery) for joint US stock-bond returns, monthly 1954–1999, CRRA with learning over state probabilities. Optimal allocations vary considerably across states; in the crash state, long-horizon investor allocates *more* to stocks (mean-reversion). Certainty-equivalent gain ~1–2%/yr over static.

**Relevance:** Regime-conditioned allocation matters *only with 4+ states and long horizon*. Two states + monthly horizon (our setup) is exactly where they would predict no gain.

### 1.4 Guidolin & Timmermann (2008) — "International Asset Allocation under Regime Switching, Skew, and Kurtosis Preferences"
*Review of Financial Studies*, 21(2), 889–935.

Extends to international ICAPM with 4-moment preferences (mean, variance, skew, kurtosis). Regime switching **increases** optimal US-stock holding once skew/kurtosis aversion is in the objective. **Predicts our CRRA(2) finding**: regime models look weak in pure mean-variance / log / CRRA-without-higher-moments setups.

### 1.5 Tu (2010) — "Is Regime Switching in Stock Returns Important in Portfolio Decisions?"
*Management Science*, 56(7), 1198–1215.

The skeptic's paper and the most important for framing our null. Bayesian framework jointly accounting for **model, parameter, and regime uncertainty**. Headline: certainty-equivalent loss from ignoring regimes is ~2%/yr, up to 10% at high γ.

**Caveat:** Tu's framework is single-equity-vs-cash, not two risky assets, and his "regime matters" is largely a vol-timing/cash-switch. He does *not* find regime label moves the equity-vs-bond mix at moderate γ in CRRA — **the closest published prior to our null finding**.

### 1.6 Bulla et al. (2011) — "Markov-Switching Asset Allocation: Do Profitable Strategies Exist?"
*Journal of Asset Management*, 12(5), 310–321.

Two-state HMM on daily returns of US/German/Japanese equity indices, 40 years. Simple rule: hold the index in the low-vol state, hold cash in the high-vol state. Annualised excess returns 18.5–201.6 bp over buy-and-hold — *but the entire strategy is a vol-timing/exposure rule, not an alpha rule*. Confirms Ang-Bekaert: binding margin is the cash/equity switch.

### 1.7 Nystrup, Madsen & Lindström (2018) — "Dynamic Portfolio Optimization Across Hidden Market Regimes"
*Quantitative Finance*, 18(1), 83–95.

The canonical **walk-forward refit** paper. 2-state HMM with online (recursive) re-estimation; Markowitz mean-variance subject to 1-day-delayed regime signal. Outperforms static on Sharpe and max drawdown. **Regime models work only when regime persistence exceeds inference latency.**

**Relevance:** Their walk-forward setup is the rigorous baseline we should be benchmarking against. Sharpe lifts ~0.1–0.3 over static and large drawdown reductions.

### 1.8 Kritzman, Page & Turkington (2012) — "Regime Shifts: Implications for Dynamic Strategies"
*Financial Analysts Journal*, 68(3), 22–39.

State Street's institutional take. 2-state Markov switching on **market turbulence, inflation, and economic growth** (not returns themselves). Their explicit use of macro/financial state variables (close in spirit to our VIX + term spread) is notable.

### 1.9 Bae, Kim & Mulvey (2014); Costa & Kwon (2019); ML/DRL Work (2023–2026)
Bae/Kim/Mulvey use multivariate HMM feeding a stochastic-programming scenario-tree optimizer. Costa & Kwon derive robust regime-dependent MV. Recent ML papers (Cohen et al. 2025; Wasserstein-HMM 2026) report Sharpe up to 2.18 — caveats: short backtests, single feature sets, possible look-ahead.

---

## 2. POMDP / QMDP in Finance

### 2.1 Hauskrecht (2000) — "Value-Function Approximations for POMDPs"
*JAIR*, 13, 33–94.

Reference for QMDP, FIB, and upper-bound approximations. **Crucial for our framing**: QMDP cannot motivate information-gathering actions, but since there is **no action that improves observability of the regime** in our setup (we observe (VIX, spread) regardless of allocation), QMDP is asymptotically near-optimal in our problem class.

### 2.2 Littman, Cassandra & Kaelbling (1995); Pineau, Gordon & Thrun (2003)
LCK 1995 is the original QMDP. PGT 2003 introduces PBVI — tighter alternative we should acknowledge. On 2 latent states, 1-D belief simplex, no info-gathering actions, PBVI is overkill — exactly what we should argue in the paper.

### 2.3 POMDP in finance is **sparse**

Continuous-time relatives exist (Dai/Zhang/Zhu 2010 "Trend Following under Regime Switching"; Sass & Haussmann 2004 *Finance and Stochastics*) but use HJB. **Discrete-time POMDP + QMDP framing in finance is genuinely uncommon — methodological selling point.**

---

## 3. Tactical Baselines

### 3.1 Faber (2007) — "A Quantitative Approach to TAA"
*Journal of Wealth Management*, 9(4), 69–79. 10-month SMA rule across 5 asset classes since 1900. Equity-like returns with bond-like vol. **Must benchmark against this.**

### 3.2 Hurst, Ooi & Pedersen / AQR (2017) — "A Century of Evidence on Trend-Following"
*JPM*, 44(1), 15–29. TS-momentum across 67 markets, 1880–2016. Positive every decade since 1880, positive in 8 of 10 worst 60/40 drawdowns. **The "what regime models try to recover."**

### 3.3 Asness/Frazzini/Pedersen (2012) — "Leverage Aversion and Risk Parity"
*FAJ*, 68(1), 47–59. Theoretical underpinning of RP. **Include risk-parity baseline alongside 60/40.**

### 3.4 Ledoit & Wolf (2004); Black-Litterman variants
Ledoit-Wolf shrinkage for covariance; Idzorek 2005 for tactical Black-Litterman.

---

## 4. Falsifiable Claim and Reality

**Implicit claim:** A Markov-state model of joint returns, fed to a CRRA optimizer, produces OOS certainty-equivalent gains over a constant-mix benchmark, driven by the **regime label** changing the optimal portfolio mix (not just by exposure scaling).

**Reality:**
- **Almost always holds** if cash/bonds-as-risk-free in universe and γ ≥ 5–10. Mechanism: high-vol regime → shift to risk-free. (Ang-Bekaert; Bulla; Nystrup; Kritzman.)
- **Often** with 4+ states or skew/kurtosis prefs. (Guidolin-Timmermann 2007, 2008.)
- **Rarely** for low γ (≤2), 2-state HMM, two risky assets. **This is where our null result lives.** Closest published acknowledgment: Tu (2010); Ang-Bekaert explicit "cost of ignoring regimes small for all-equity portfolios."

**Bottom line:** Our null is consistent with — and actually predicted by — the most rigorous papers, but almost never stated as a headline. Our contribution: make the null explicit and identify the *parameter region* in which regime models fail to add policy value.

---

## 5. Ranked Relevance to Our Project

| Rank | Paper | Why |
|---|---|---|
| 1 | Tu 2010 MS 56(7) | Closest acknowledgment of small-gain region at low γ |
| 2 | Ang-Bekaert 2002 RFS 15(4) | "Cost of ignoring regimes small for all-equity" |
| 3 | Guidolin-Timmermann 2007 JEDC | Predicts our 2-state result needs 4+ states |
| 4 | Nystrup et al. 2018 QF 18(1) | Walk-forward refit baseline we should implement |
| 5 | Bulla et al. 2011 JAM 12(5) | Confirms regime label moves policy only via cash-switching |
| 6 | Guidolin-Timmermann 2008 RFS 21(2) | Predicts CRRA(2) finding: needs skew/kurtosis prefs |
| 7 | Faber 2007 JWM 9(4) | Mandatory tactical baseline |
| 8 | Hauskrecht 2000 JAIR 13 | QMDP justification |
| 9 | Pineau et al. 2003 IJCAI | Must acknowledge as tighter alternative |
| 10 | Kritzman et al. 2012 FAJ 68(3) | Closest precedent for macro-state features |
| 11 | Hurst-Ooi-Pedersen 2017 JPM 44(1) | Trend-following benchmark |
| 12 | Costa-Kwon 2019 J. Risk | Robust MV with regime |
| 13 | Bae-Kim-Mulvey 2014 EJOR 234(2) | Stochastic programming alternative |
| 14 | Asness et al. 2012 FAJ 68(1) | Risk-parity baseline |
| 15 | Ledoit-Wolf 2004 JPM 30(4) | Shrinkage covariance |

---

## 6. What Makes Our Project Unique

1. **Discrete POMDP + QMDP framing in finance is genuinely rare.** Most regime papers solve an MDP and bolt on a filter; we solve the POMDP correctly via QMDP.
2. **Explicit feature set (VIX, 10Y-3M)** rather than fitting on returns. Kritzman uses turbulence/inflation/growth; we use the two cleanest macro-financial regime signals.
3. **Null result with diagnostic decomposition.** Almost nobody publishes nulls. Our "regime label never moves policy at γ=2; only shifts when γ≥8" is genuinely useful negative knowledge — we cite Ang-Bekaert and Tu as supporting why this is *expected*.
4. **VI vs PI cross-check** — methodological rigor most empirical-finance papers skip.
5. **QMDP vs optimal MDP vs 60/40 simultaneous comparison** — most papers compare only to 60/40.

## 7. Where Our Project Is NOT Unique

| Element | Canonical reference |
|---|---|
| 2-state HMM on macro/financial signals | Ang-Bekaert 2002; Bulla 2011 |
| Baum-Welch fitting for finance | Hamilton 1989; Hardy 2001 |
| CRRA dynamic portfolio with regimes | Ang-Bekaert 2002; Guidolin-Timmermann 2007 |
| Walk-forward HMM + portfolio refit | **Nystrup et al. 2018 — must implement** |
| QMDP algorithm | Littman/Cassandra/Kaelbling 1995; Hauskrecht 2000 |
| Tactical SPY/cash with simple signal | Faber 2007; Bulla 2011 |
| γ sweep | Standard in Ang-Bekaert and Guidolin-Timmermann |
| 60/40 benchmark | Universal |

## 8. Concrete Recommendations

1. **Add three baselines:** (i) Faber 10-mo SMA; (ii) Nystrup walk-forward HMM+MV; (iii) risk-parity ERC.
2. **Re-frame null** in related-work: "consistent with Ang-Bekaert (2002) and Tu (2010) which document regime gains vanish for low γ in risky-only universes; we provide the first explicit decomposition showing the regime label is non-binding at γ=2 and becomes binding only at γ≥8."
3. **Sensitivity table** over (K ∈ {2,3,4}, γ ∈ {1,2,5,8,15}, feature set ∈ {(VIX), (spread), (both), (both+credit)}). Directly tests Guidolin-Timmermann prediction.
4. **Cite Hauskrecht and PGT** explicitly when justifying QMDP over PBVI.
5. **Nystrup walk-forward methodology** is the obvious benchmark for our work.
