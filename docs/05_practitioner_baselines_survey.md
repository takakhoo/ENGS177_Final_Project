# 07 · Practitioner Survey: Dynamic Stock/Bond Allocation Strategies

> Reference document gathered for the QMDP comparison study. Provides the baseline-strategy menu, implementation pointers, and academic-vs-practitioner narrative for the extended report. Built from the published literature on long-only US-stock + US-bond tactical allocation, 2003–present.

---

## 1. Static 60/40 (the benchmark to beat)

**Rule.** Hold $w_{eq}=0.60$ SPY and $w_{bd}=0.40$ AGG, rebalance monthly to constant weights:
$w_t = (0.60, 0.40)^{\top}, \forall t$. Portfolio return $R^p_t = 0.6 R^{SPY}_t + 0.4 R^{AGG}_t$. Most institutional implementations rebalance when weights drift past a band ($\pm$5 pp absolute, or $\pm$25% relative).

**Canonical implementation.** Vanguard Balanced Index (VBINX/VBIAX), Fidelity Balanced (FBALX). At institutional scale, transaction cost is negligible because flows net across funds.

**Decade-by-decade backtest (SPY + AGG monthly, rebalanced monthly, gross of fees):**

| Period | CAGR | Vol | Sharpe (rf=0) | Max DD |
|---|---|---|---|---|
| 2000–2009 | ~2.6% | 9.3% | 0.28 | −32% (2008) |
| 2010–2019 | ~9.0% | 7.4% | 1.22 | −10% |
| 2020–2024 | ~7.5% | 11.8% | 0.63 | −21% (2022) |
| 2000–2024 full | ~6.4% | 9.4% | 0.68 | −32% |

**Why it's hard to beat.** Zero parameters, zero estimation error, free rebalancing premium from negative stock-bond correlation (true 1998–2021, broken 2022–2023). Survives any structural break that doesn't permanently re-rate one asset class.

---

## 2. Risk Parity (Equal Risk Contribution)

**Rule.** Equal portfolio variance contribution per asset: $w_i (\Sigma w)_i = w_j (\Sigma w)_j$. For 2 assets, the inverse-vol weight $w_i = (1/\sigma_i) / \sum_j (1/\sigma_j)$ approximates this exactly when correlations are zero; the true ERC differs by a few bps for SPY/AGG.

**Canonical implementation.** Bridgewater All Weather (Dalio 1996), AQR Risk Parity Fund (QRPRX), Invesco Balanced-Risk Allocation, PanAgora. Vol estimated from rolling 36–60 month window or EWMA $\lambda = 0.94$ (RiskMetrics). All Weather levers the bond sleeve so total portfolio vol ≈ 10–12% (matching equity).

**Reported numbers (Asness/Frazzini/Pedersen 2012; Anderson/Bianchi/Goldberg 2012):**
- Unlevered SPY+AGG inverse-vol: Sharpe ~0.80–0.95, CAGR ~5–6%, max DD ~−15% (1990–2020).
- 10%-vol-targeted (levered to match 60/40 vol): Sharpe ~0.85–1.00, CAGR ~7–8%, max DD ~−25% in 2022 (Bridgewater All Weather lost ~22% in 2022, its worst year ever).

---

## 3. Mean-Variance (Markowitz 1952) with Ledoit-Wolf Shrinkage

**Rule.** $\max_w w^\top \mu - (\gamma/2) w^\top \Sigma w$ s.t. $\mathbf{1}^\top w = 1, w \ge 0$. Sample $\hat\mu$ is noisy → MV piles into the in-sample winner. Ledoit-Wolf (2003, 2004) shrinks covariance toward a constant-correlation target.

**Reported numbers.** DeMiguel/Garlappi/Uppal 2009 ("Optimal Versus Naive Diversification"): across 14 datasets and 7 strategies, none of MV, Bayes-Stein, BL, or minimum-variance reliably beat 1/N out-of-sample once estimation error is honestly accounted for. For SPY/AGG: shrinkage MV ~Sharpe 0.75–0.95 with high turnover.

---

## 4. Equal-Weight (1/N)

**Rule.** $w_i = 1/N$. For SPY/AGG: $w = (0.5, 0.5)$ monthly.

**Why it's the toughest baseline.** DGU 2009, across 14 datasets, no optimization-based rule reliably beats 1/N out-of-sample. Sharpe ~0.72 for SPY+AGG 2003–2024.

---

## 5. Faber 10-Month Moving-Average (2007)

**Rule.** Per asset, hold if $P_t > \mathrm{MA}_{10}(P_t)$, else move to cash (T-bills). Faber's "Quantitative Approach to Tactical Asset Allocation" (2007 JoWM) applied this across 5 asset classes.

**Reported numbers.** Faber 2007 + updates: SPY only with MA rule Sharpe ~0.71 (vs 0.42 buy-hold); 5-asset version Sharpe ~0.94 vs 0.70 buy-hold equal-weight. SPY+AGG version (Clare et al. 2013): Sharpe ~0.85–0.95, max DD ~−15%.

Whipsaws in choppy sideways markets (2011, 2015–16) cost performance. Shines in regime transitions (Q4 2007, Q1 2020, Q1 2022).

---

## 6. Time-Series Momentum (Moskowitz, Ooi, Pedersen 2012)

**Rule.** Per asset, position size $\propto \mathrm{sign}(\sum_{k=1}^{12} (r^i_{t-k} - r^{rf}_{t-k})) \cdot \sigma_{tgt}/\hat\sigma^i_t$.

**Canonical implementation.** AQR Managed Futures, Man AHL, Winton, Aspect, the entire CTA industry. Hurst/Ooi/Pedersen "Century of Evidence" 2017: 9.8% gross return, 11.2% vol, Sharpe 0.77 for a diversified 67-market trend system 1880–2016, positive in every decade.

**Long-only SPY/AGG variant.** Apply per-asset; negative-signal → cash. Sharpe ~0.75–0.90 for the 2-asset case 2003–2024. Multi-lookback ensemble (1, 3, 6, 12-mo): Sharpe ~0.10–0.15 higher.

---

## 7. Volatility Targeting

**Rule.** Scale exposure to hit target annualised vol: $w_t = w_t^{base} \cdot (\sigma_{tgt}/\hat\sigma^p_t)$. Vol is *persistent* in ways returns aren't.

**Reported numbers.** Moreira/Muir 2017 ("Volatility-Managed Portfolios," JoF): vol-targeting raises Sharpe by 0.10–0.30 on US equity premium and many factor portfolios. Combined vol-target + 12m momentum on SPY: Sharpe ~0.85–1.00 vs ~0.45–0.55 buy-hold 1990–2020 (Harvey et al. 2018 "Impact of Volatility Targeting").

---

## 8. Black-Litterman (1992) with HMM-Derived Views

**Rule.**
$$\mu^{BL} = \left[(\tau\Sigma)^{-1} + P^\top \Omega^{-1} P\right]^{-1} \left[(\tau\Sigma)^{-1} \pi + P^\top \Omega^{-1} Q\right]$$

with $\pi = \gamma\Sigma w_{mkt}$ equilibrium prior, $P$ picking matrix, $Q$ view vector, $\Omega$ view-uncertainty.

**HMM connection.** Inject HMM regime-conditional means as views with $\Omega$ inversely scaled by HMM belief confidence. Confident regime → strong view → posterior pulled toward the regime's $\mu$. Uncertain regime → wide $\Omega$ → posterior pulled toward equilibrium prior.

**Canonical implementation.** GS Asset Management (origin 1990). Yale, Harvard endowments; CalPERS; most pension allocators.

**Reported numbers.** No clean 2-asset SPY/AGG BL backtest in the literature (BL shines with $N \ge 5$). Idzorek 2005; Bertsimas/Gupta/Paschalidis 2012: BL produces more stable weights than plug-in MV; Sharpe within ±0.05 of equilibrium 60/40. Value-add is interpretability and view-injection discipline.

---

## 9. HMM-Conditional Mean-Variance (Guidolin-Timmermann)

**Rule.** Belief-weighted regime moments
$\mu_t = \sum_k b_{t,k} \mu_k$,  $\Sigma_t = \sum_k b_{t,k} \Sigma_k + \sum_k b_{t,k} (\mu_k - \mu_t)(\mu_k - \mu_t)^\top$,
then one-shot MV. The second term in $\Sigma_t$ is the regime-uncertainty contribution, the key Guidolin-Timmermann insight.

**Reported numbers.** Guidolin/Timmermann 2008 on US stock + 10y bond 1954–2003: Sharpe lift ~0.10–0.20 in-sample; OOS improvement marginal and statistically insignificant. Ang/Timmermann 2012 survey: "regime-switching models reliably improve in-sample fit but rarely produce out-of-sample Sharpe gains exceeding 0.10 after transaction costs."

**Why this matters for us.** The QMDP collapse-to-100%-stocks is a known failure mode: when the HMM identifies a "high mean" state with high posterior probability, naive plug-in MV with low $\gamma$ overweights it. Fix: shrink $\mu_t$ toward zero or the long-run mean and use $\gamma \in \{5, 10\}$ instead of 2.

---

## 10. CTA-Style Long-Short Trend (out of scope but worth mentioning)

CTAs trade futures (commodities, rates, FX, equity indices) long *and* short, vol-targeted, multi-market. Sharpe 0.75–1.00 over multi-decade samples, positive in equity crises. Cannot be replicated in long-only SPY/AGG, mention in report's limitations: long-only 2-asset has a structural Sharpe ceiling no allocation rule can break.

---

## What Does the "Average" Target-Date Fund Actually Do?

| Provider | Fund family | Tactical component | Regime-aware? |
|---|---|---|---|
| Vanguard Target Retirement | $1.5T AUM | None, pure glide path | No |
| Fidelity Freedom Index | $350B | None | No |
| BlackRock LifePath | $400B | None | No |
| T. Rowe Price Retirement | $350B | Moderate tactical ($\pm$10%) | No |
| AQR Risk Parity / Style Premia |, | Vol-target + multi-asset trend | Partly (signals, not HMM) |
| Bridgewater All Weather / Pure Alpha |, | Risk-parity + macro regime overlay | Pure Alpha yes |

Of >$3T in US target-date AUM, **less than 5% sits in funds running anything that could be called regime-aware**. Bridgewater Pure Alpha and a handful of risk-parity funds are the rare exceptions. Three reasons (fiduciary/career risk; capacity at scale; fee compression) explain why.

---

## Why Academia Loves Regime Switching but Practice Mostly Doesn't

1. **Estimation noise dominates the signal.** Hamilton-style models need ~30 years of data per regime. SPY has had 3–4 regimes by most labelings: <10 years per regime of effective sample. Regime-conditional means have 30–50% relative error.

2. **Look-ahead bias creeps in everywhere.** Full-sample HMM parameters leak future info. Honest expanding-window refit typically halves the reported Sharpe improvement.

3. **Transaction costs erode tactical alpha fast.** 500% turnover (typical of regime-flipping QMDP) costs ~20 bps/yr. With market-impact for >$10B AUM, tactical PnL → zero.

4. **Regimes are identified ex-post and labels shift.** Out-of-sample, the HMM frequently relabels what "state 1" means. Unstable identification = unstable policy.

5. **Risk-aversion parameter is unobservable and unstable.** The team's $\gamma=2$ collapse to all-stocks is a generic Merton-style pathology. Realistic institutional $\gamma \in \{5, 10\}$.

Ang/Timmermann 2012 (Annual Review of Financial Economics): *"The empirical evidence on whether regime-switching models improve out-of-sample portfolio performance is mixed at best."*

---

## Practitioner Consensus: Best 2-Asset Tactical Strategy

> **Time-series momentum (12-month, multi-lookback ensemble) with volatility targeting (10% vol), applied per-asset, with cash as defensive fallback.**

Combines TS-mom and vol-targeting. Full-sample Sharpe 0.85 (single 12-mo) to 1.10 (multi-lookback ensemble + vol target) on SPY+AGG 1990–2024. Max DD reliably <−15%.

What's documented *not* to beat trend on 2-asset:
- HMM regime-switching alone, generally ties or slightly underperforms trend at much higher complexity.
- Black-Litterman alone, provides stability, not alpha.
- MV optimization with estimated moments, DGU 2009 showed it fails to beat 1/N.

---

## Summary Table

| # | Strategy | Inputs | Typical Sharpe (lit, 2-asset) | Complexity | Why it might beat ours |
|---|---|---|---|---|---|
| 1 | Static 60/40 | None | 0.65–0.80 | 1 | Zero estimation error |
| 2 | Risk parity (inv-vol) | Rolling vol | 0.75–0.95 unlevered | 2 | Smaller equity drawdowns |
| 3 | MV + Ledoit-Wolf | $\hat\mu, \hat\Sigma, \gamma$ | 0.70–0.95 | 3 | Theoretically optimal in MV world |
| 4 | Equal-weight 1/N | None | 0.65–0.75 | 1 | DGU result; minimal turnover |
| 5 | Faber 10m MA | Price series | 0.85–0.95 | 2 | Cuts left tail; survives 2008/2020/2022 |
| 6 | TS momentum 12m | Trailing returns | 0.75–0.95 (single), 0.90–1.10 (ensemble) | 2 | Captures persistence Bayes-rationally misses |
| 7 | Vol targeting (overlay) | EWMA vol | +0.10–0.20 lift on top of base | 2 | Eliminates vol-spike drawdowns |
| 8 | Black-Litterman + HMM views | HMM probs, $\pi$, $\tau$, $\Omega$ | 0.70–0.90 (more stable than MV) | 4 | Disciplined way to use HMM signal |
| 9 | HMM-conditional MV | HMM moments, $\gamma$ | 0.65–0.85 OOS | 4 | Direct upgrade path from QMDP |
| 10 | CTA long-short trend | Cross-asset futures | 0.75–1.00 | 5 | Not implementable on long-only SPY/AGG |
| Best combo | TS-mom ensemble + 10% vol target + cash fallback | Returns + vol | 0.95–1.15 | 3 | Practitioner consensus winner |

---

## Key Citations for the Extended Paper

- Markowitz 1952; Maillard, Roncalli, Teiletche 2010 (RP)
- DeMiguel, Garlappi, Uppal 2009 (1/N)
- Faber 2007; Moskowitz, Ooi, Pedersen 2012
- Asness, Frazzini, Pedersen 2012; Hurst, Ooi, Pedersen 2017 ("Century of Evidence")
- Moreira, Muir 2017 (vol-target)
- Black, Litterman 1992; Idzorek 2005 (BL)
- Guidolin, Timmermann 2007, 2008; Ang, Bekaert 2002, 2004
- Ang, Timmermann 2012 (regime-switching survey)
- Ledoit, Wolf 2003, 2004 (shrinkage)

---

## Narrative Arc for the Extended Report

> "POMDP/QMDP regime-switching is theoretically appealing but empirically fragile; the simpler 12-month trend rule wins, consistent with the broader literature; the best tactical 2-asset rule is trend + vol-targeting + cash fallback. Our negative finding is the *standard* finding."

This is the correct and well-supported narrative, far more interesting than a paper claiming QMDP wins.
