# Regime-Switching Asset Allocation via a POMDP

**ENGS 177, Decision Making Under Uncertainty, Spring 2026** · Type: *New Application*
Dario Blanco Morales · Even Hogberget · Kyle David Ledda-Lewaren · Taka Khoo
Instructor: Prof. Wesley Marrero, Thayer School of Engineering, Dartmouth College

> A long-only investor must split capital between U.S. equities (SPY) and aggregate bonds (AGG) every month. We treat the prevailing macro-financial regime as a hidden state, infer it from VIX and the 10Y–3M Treasury term spread via a Gaussian Hidden Markov Model, and choose portfolio weights using a QMDP approximation of the resulting POMDP. We benchmark against a static 60/40 plus ten other strategies spanning the academic and practitioner consensus, then trace exactly which methodological choices drive the headline verdict.

---

## TL;DR

The 60/40 portfolio lost 17% in 2022 when stocks and bonds fell together. We asked: **can a fully decision-theoretic regime overlay—POMDP + HMM + QMDP, end-to-end on public data—beat 60/40 after realistic frictions?**

In the original configuration (VIX+spread observations, fixed HMM, CRRA γ=2), the answer is **no**: QMDP finishes last of twelve strategies on Sharpe (0.73 vs static's 0.81 and Faber's 1.68). But three diagnostic experiments reverse this in isolation:
- Adding the Chicago Fed **NFCI + STLFSI4** as observation channels → the same γ=2 policy goes from "100% stocks in both regimes" to "100% stocks in bull, 100% bonds in bear."
- Switching to **expanding-window walk-forward refit** → QMDP Sharpe lifts 0.81 → 1.08 and max drawdown cuts from −34% to −16.5%.
- Raising CRRA to γ ≥ 8 → optimal MDP shifts to 40/60 across regimes and crosses static on Sharpe.

**The negative headline is a diagnostic of methodology, not a fundamental verdict on POMDP asset allocation.** We do not claim a Sharpe win — Faber's trend rule wins, consistent with the practitioner-consensus literature. We do claim that the POMDP framework is the right *minimum* tool for the decision problem, that the HMM signal itself is informative, and that each headline-driving methodological choice can be separately reversed.

---

## Deliverables

| Item | Pages | Audience | Link |
|---|---|---|---|
| Class report (Canvas submission, 8–10 body + refs spec) | 11 | Instructor + peer reviewer | [`report/report.pdf`](report/report.pdf) |
| Presentation slides (17 frames for the 8–10 min talk) | 17 | Class presentation | [`presentation/slides.pdf`](presentation/slides.pdf) |
| Extended technical report (intuition primer + full math + glossary + code) | 38 | Teammates + depth seekers | [`report/extended_report.pdf`](report/extended_report.pdf) |
| Original proposal | 2 | — | [`proposal/ENGS177_Term_Project_Proposal.docx`](proposal/ENGS177_Term_Project_Proposal.docx) |

For the *story* behind everything, start with [`docs/01_project_story.md`](docs/01_project_story.md). For *intuition without math*, read [`docs/03_intuition_primer.md`](docs/03_intuition_primer.md). For the *experimental order*, see [`docs/04_experiments_guide.md`](docs/04_experiments_guide.md).

---

## The research question

The textbook 60/40 is the default recommendation in every personal-finance and pension textbook. It assumes returns are roughly stationary. Reality violates this in two specific ways: returns are conditionally heteroskedastic (calm decades broken by short crises), and stress periods exhibit higher cross-asset correlations so bond diversification erodes when it is most needed. In 2022 a 60/40 portfolio lost about 17% as both legs sold off under persistent inflation.

Whether to *think* about regimes is not new — Hamilton (1989), Ang & Bekaert (2002), Guidolin & Timmermann (2007), Nystrup et al. (2018). What is less settled is the **methods question** at the heart of this project:

> **Can a fully decision-theoretic regime overlay — POMDP solved end-to-end on public macroeconomic data — beat the static 60/40 benchmark out-of-sample after realistic transaction costs?**

By "decision-theoretic" we mean three things:
1. The allocation rule is the **solution of an explicit optimisation**, not a discretionary heuristic.
2. The latent regime is treated as a **hidden state** and inferred by a **Bayesian filter**, not labelled by hand.
3. The policy is the **QMDP approximation of an underlying POMDP** — falsifiable, comparable, explainable.

### Why POMDP, not a tree, not an MDP

| Naive alternative | Why it fails |
|---|---|
| Decision tree | Tree explodes combinatorially with horizon; no shared state |
| Influence diagram | Same blow-up; not natural for repeated monthly rebalance |
| Fully observable MDP | Requires knowing the regime, but it is *latent* |
| **POMDP (ours)** | Hidden state + noisy observation + sequential decision — exactly our setting |

In Powell (2019)'s four-class taxonomy of sequential-decision policies, our QMDP solution is a **value function approximation** (the underlying MDP's $Q^\ast$ as a piecewise-linear lower envelope over the belief simplex) combined with a **one-step direct lookahead** at decision time. This is the project's theoretical anchor.

---

## What we actually built

End-to-end pipeline (each box maps to one or more scripts in [`experiments/`](experiments/)):

```
┌────────────┐  ┌───────────┐  ┌───────────┐  ┌─────────────┐  ┌────────────┐
│ FRED+Yahoo │→ │ Monthly   │→ │ Gaussian  │→ │ Underlying  │→ │ Q*(s,a)    │
│ CSVs       │  │ panel     │  │ HMM       │  │ MDP (VI+PI) │  │ table      │
└────────────┘  └─────┬─────┘  └─────┬─────┘  └─────────────┘  └──────┬─────┘
                      │              │                                 │
                      ▼              ▼                                 ▼
              ┌───────────────────────────────┐               ┌──────────────┐
              │  Bayesian filter → belief b_t │──────────────►│ QMDP rule    │
              └───────────────────────────────┘               └──────┬───────┘
                                                                     │
                                                              ┌──────▼──────┐
                                                              │ Backtest +  │
                                                              │ 12 baselines│
                                                              └─────────────┘
```

### Components

| File / module | What it does |
|---|---|
| [`src/data/fetch_data.py`](src/data/fetch_data.py) | Downloads 14 series from FRED + Yahoo, resamples to month-end, aligns into panel |
| [`src/models/hmm.py`](src/models/hmm.py) | Fits Gaussian HMM via Baum-Welch (EM) with 5 random restarts; BIC model selection |
| [`src/models/mdp.py`](src/models/mdp.py) | Value iteration + policy iteration; cross-checked agreement to 2.6×10⁻⁶ |
| [`src/models/qmdp.py`](src/models/qmdp.py) | Bayesian belief filter + QMDP rule; stationary-distribution initialiser |
| [`src/models/baselines.py`](src/models/baselines.py) | 11 alternative strategies (60/40, equal-weight, inverse-vol, risk parity, MV+LW, Faber, TS-mom, vol-target, myopic, HMM-MV, BL+HMM) |
| [`src/utils/metrics.py`](src/utils/metrics.py) | 12 performance metrics (CAGR, Sharpe, Sortino, Omega, max DD, Calmar, Ulcer, tail ratio, hit rate, turnover, info ratio) |
| [`src/utils/utility.py`](src/utils/utility.py) | CRRA / log utility + Monte Carlo reward builder |

### Experiments (numbered to match the pipeline order)

Full guide: [`docs/04_experiments_guide.md`](docs/04_experiments_guide.md). Each script is independently runnable.

```
00_synthetic_demo.py            — no-network smoke test
01_fetch_data.py                — FRED + Yahoo download (data already committed)
02_hmm_calibration.py           — Baum-Welch fit, BIC model selection
03_regime_interpretation.py     — plot regimes vs NBER
04_qmdp_solve.py                — VI + PI cross-check, build Q*
06_multistate_comparison.py     — K ∈ {2, 3, 4} comparison
07_gamma_sensitivity.py         — CRRA γ sweep
08_baselines_comparison.py      — 12-strategy horse race (the headline)
09_richer_observations.py       — observation-cohort study (Unlock 1)
10_walk_forward_refit.py        — Nystrup-style refit protocol (Unlock 2)
11_cohort_regime_visualization.py — visual confirmation of Unlock 1
12_subperiod_robustness.py      — Period A/B/C heatmap
```

---

## Data: real, public, all committed to this repo

Every series has a clickable source link AND a clickable local CSV. No synthetic data is used in any reported finding — the file `00_synthetic_demo.py` exists only as a no-network smoke test.

### Headline observations (used in original report)

| Series | Source | Local | Role |
|---|---|---|---|
| **VIX** | [FRED VIXCLS](https://fred.stlouisfed.org/series/VIXCLS) | [`data/raw/vix.csv`](data/raw/vix.csv) | HMM observation #1 — equity-vol signal |
| **10Y–3M Treasury spread** | [FRED T10Y3M](https://fred.stlouisfed.org/series/T10Y3M) | [`data/raw/term_spread.csv`](data/raw/term_spread.csv) | HMM observation #2 — yield-curve signal |
| **HY OAS** | [FRED BAMLH0A0HYM2](https://fred.stlouisfed.org/series/BAMLH0A0HYM2) | [`data/raw/hy_oas.csv`](data/raw/hy_oas.csv) | Proposed third channel; FRED CSV truncates to ~3y → **dropped from headline** |
| **NBER recessions** | [NBER cycle dates](https://www.nber.org/research/business-cycle-dating) · [FRED USREC](https://fred.stlouisfed.org/series/USREC) | [`data/raw/nber.csv`](data/raw/nber.csv) | Qualitative validation only — **never inside the model** |
| **SPY** | [Yahoo SPY](https://finance.yahoo.com/quote/SPY/) | [`data/raw/spy.csv`](data/raw/spy.csv) | Equity asset return |
| **AGG** | [Yahoo AGG](https://finance.yahoo.com/quote/AGG/) | [`data/raw/agg.csv`](data/raw/agg.csv) | Bond asset return (Sep-2003 inception sets panel start) |

### Extension channels (used in [`09_richer_observations.py`](experiments/09_richer_observations.py))

| Series | Source | Local | Role |
|---|---|---|---|
| 10Y–2Y spread | [FRED T10Y2Y](https://fred.stlouisfed.org/series/T10Y2Y) | [`data/raw/term_spread_2y.csv`](data/raw/term_spread_2y.csv) | Alt yield-curve metric |
| **NFCI** | [FRED NFCI](https://fred.stlouisfed.org/series/NFCI) | [`data/raw/nfci.csv`](data/raw/nfci.csv) | **Financial-conditions index — the channel that fixes the policy collapse** |
| **STLFSI4** | [FRED STLFSI4](https://fred.stlouisfed.org/series/STLFSI4) | [`data/raw/stlfsi.csv`](data/raw/stlfsi.csv) | Cross-validation stress signal |
| Consumer sentiment | [FRED UMCSENT](https://fred.stlouisfed.org/series/UMCSENT) | [`data/raw/umcsent.csv`](data/raw/umcsent.csv) | Household-driven leading indicator |
| Initial jobless claims | [FRED ICSA](https://fred.stlouisfed.org/series/ICSA) | [`data/raw/jobless_claims.csv`](data/raw/jobless_claims.csv) | High-frequency labour-market signal |
| Trade-weighted USD | [FRED DTWEXBGS](https://fred.stlouisfed.org/series/DTWEXBGS) | [`data/raw/usd_index.csv`](data/raw/usd_index.csv) | Global stress (flight to USD) |
| WTI crude oil | [FRED DCOILWTICO](https://fred.stlouisfed.org/series/DCOILWTICO) | [`data/raw/wti_oil.csv`](data/raw/wti_oil.csv) | Supply-driven inflation episodes |
| Fed funds rate | [FRED DFF](https://fred.stlouisfed.org/series/DFF) | [`data/raw/fed_funds.csv`](data/raw/fed_funds.csv) | Monetary-policy stance |

### Derived files

| File | Description |
|---|---|
| [`data/processed/monthly.csv`](data/processed/monthly.csv) | Aligned monthly panel: 272 rows × 13 columns (10 obs + NBER + 2 returns) |
| [`data/processed/hmm_{2,3,4}state.pkl`](data/processed/) | Pickled `hmmlearn.GaussianHMM` instances |

See [`data/README.md`](data/README.md) for the full schema and refresh instructions.

---

## Headline findings (in one place)

### 1. Twelve-strategy horse race ([`results/baselines_metrics.csv`](results/baselines_metrics.csv))

Out-of-sample 2003-10-31 → 2026-04-30, monthly rebalance, 5 bps tx cost. Sorted by Sharpe.

| Strategy | CAGR | Vol | Sharpe | Sortino | Max DD | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| **Faber 10-mo SMA** | **17.24%** | 9.8% | **1.68** | **3.57** | −13.5% | **1.28** |
| Myopic 12-mo trend | 15.34% | 10.6% | 1.41 | 2.70 | −15.1% | 1.01 |
| TS momentum 12-mo | 9.06% | 8.3% | 1.10 | 1.74 | −22.0% | 0.41 |
| HMM-conditional MV | 6.38% | 5.9% | 1.08 | 1.78 | −18.4% | 0.35 |
| Black-Litterman + HMM views | 6.17% | 6.7% | 0.94 | 1.45 | −18.0% | 0.34 |
| Inverse-volatility | 4.78% | 5.2% | 0.92 | 1.42 | −16.8% | 0.28 |
| Risk parity (ERC) | 4.87% | 5.4% | 0.91 | 1.41 | −16.8% | 0.29 |
| Equal-weight 50/50 | 6.69% | 8.1% | 0.84 | 1.26 | −28.6% | 0.23 |
| Mean-variance + LW | 6.58% | 8.2% | 0.82 | 1.32 | −22.5% | 0.29 |
| Static 60/40 (bench) | 7.40% | 9.4% | 0.81 | 1.21 | −34.2% | 0.22 |
| Vol-target 60/40 | 7.50% | 10.1% | 0.77 | 1.12 | −39.1% | 0.19 |
| **QMDP (CRRA γ=2)** | 10.01% | 14.6% | **0.73** | 1.07 | **−53.0%** | 0.19 |

Faber's simple trend rule wins. QMDP comes last on Sharpe with the deepest drawdown. The two HMM-aware non-QMDP baselines (HMM-MV and BL+HMM) beat QMDP, indicating the HMM signal *is* informative — QMDP's projection at γ=2 destroys it.

### 2. Unlock 1: richer observations fix the policy collapse ([`results/multifeature_hmm_table.csv`](results/multifeature_hmm_table.csv))

Same γ=2 as headline; we just change which observation channels feed the HMM:

| Cohort | Features | Bull policy | Bear policy | Regime-differentiated? |
|---|---|---|---|---|
| 1. Baseline | VIX, T10Y3M | 100/0 | 100/0 | No (the collapse) |
| 2. +Yield curve | +T10Y2Y | 100/0 | 100/0 | No |
| **3. +Stress** | **+NFCI, +STLFSI4** | **100/0** | **0/100** | **Yes — full flip** |
| 4. +Macro | +NFCI, +UMCSENT, +ICSA | 100/0 | 0/100 | Yes |
| 5. Kitchen sink (8 channels) | all extension | 100/0 | 60/40 | Yes |

**The headline collapse is an observation-channel problem, not a fundamental QMDP problem.** Confirms Guidolin–Timmermann (2007)'s prediction.

### 3. Unlock 2: walk-forward refit lifts QMDP Sharpe to 1.08 ([`results/walk_forward_metrics.csv`](results/walk_forward_metrics.csv))

| Variant | Sharpe | Max DD | Calmar |
|---|---:|---:|---:|
| Static 60/40 (bench) | 0.81 | −34.2% | 0.22 |
| QMDP fixed (report baseline) | 0.96 | −34.2% | 0.29 |
| QMDP annual refit (rolling 5y) | 0.85 | −25.1% | 0.36 |
| QMDP quarterly refit (rolling 5y) | 0.83 | −23.4% | 0.37 |
| **QMDP expanding window** | **1.08** | **−16.5%** | **0.60** |

QMDP Sharpe lifts 0.81 → **1.08** (+33%), max drawdown −34% → **−16.5%** (more than halves), Calmar nearly triples.

### Synthesis

Our headline negative finding holds *only* in the joint configuration of (a) narrow observations, (b) fixed HMM, (c) low CRRA. **Each component is empirically reversible.** This aligns with the literature: Ang–Bekaert (2002) predict it (no risk-free leg); Guidolin–Timmermann (2007) predict it (richer obs needed); Nystrup et al. (2018) predict it (walk-forward refit needed); Tu (2010) predicts it (low γ leaves no room).

---

## Figures

### Twelve-strategy horse race
![Equity curves](figures/baselines_equity.png)
![Drawdown timeseries](figures/baselines_drawdown.png)
![Sharpe / Sortino / Calmar / MaxDD bars](figures/baselines_sharpe_bar.png)

### Unlock 1 — observations matter (cohort 3 cleanly separates regimes)
![Cohort regime timelines](figures/cohort_regime_timelines.png)

### Unlock 2 — walk-forward refit changes the QMDP verdict
![Walk-forward equity curves](figures/walk_forward_equity.png)

### Subperiod robustness — Faber wins everywhere
![Subperiod Sharpe heatmap](figures/subperiod_sharpe_heatmap.png)

### Regime detection — HMM cleanly recovers known crises
![Multistate regime timeline](figures/multistate_regime_timeline.png)

---

## Quick start

```bash
git clone git@github.com:takakhoo/ENGS177_Final_Project.git
cd ENGS177_Final_Project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# All raw data + fitted HMMs are committed, so you can skip 01 and 02:
python experiments/04_qmdp_solve.py             # build Q*
python experiments/08_baselines_comparison.py   # 12-strategy horse race
python experiments/09_richer_observations.py    # Unlock 1
python experiments/10_walk_forward_refit.py     # Unlock 2
python experiments/12_subperiod_robustness.py   # robustness heatmap

# To refresh data to the latest available month (network required):
python experiments/01_fetch_data.py
python experiments/02_hmm_calibration.py
```

Total wall time end-to-end with all 12 experiments: under 5 minutes on a 2019 MacBook Pro.

### Loading the data and a fitted HMM

```python
import pickle, pandas as pd

df = pd.read_csv("data/processed/monthly.csv")
df["observation_date"] = pd.to_datetime(df["observation_date"])
df = df.set_index("observation_date")

with open("data/processed/hmm_2state.pkl", "rb") as f:
    hmm = pickle.load(f)

print(df.tail())
print("Transition matrix:", hmm.transmat_)
print("Emission means (VIX, term spread):", hmm.means_)
```

---

## Repo layout

```
ENGS177_Final_Project/
├── README.md                       ← you are here
├── docs/
│   ├── 01_project_story.md         ← single narrative beginning-to-end (start here)
│   ├── 02_class_concepts.md        ← math machinery mapped to code
│   ├── 03_intuition_primer.md      ← 10 analogies (no math required)
│   ├── 04_experiments_guide.md     ← what each experiment script does, in order
│   ├── 05_practitioner_baselines_survey.md  ← 3.7k-word survey of real allocators
│   ├── 06_academic_literature_survey.md     ← 3.5k-word survey of regime-switching lit
│   └── 07_supplementary_papers_synthesis.md ← 3.4k-word synthesis of ENGS 177 supplementary pdfs
├── report/
│   ├── report.tex / .pdf           ← 11-page Canvas submission
│   └── extended_report.tex / .pdf  ← 38-page technical deep-dive
├── presentation/
│   └── slides.tex / .pdf           ← 17-frame beamer deck for the 8–10 min talk
├── proposal/                       ← original Canvas proposal
├── homework/                       ← per-author HW1-3
├── data/
│   ├── README.md                   ← per-series schema + sources
│   ├── raw/                        ← 14 raw CSVs (committed)
│   └── processed/                  ← aligned panel + pickled HMMs (committed)
├── src/
│   ├── data/fetch_data.py
│   ├── models/{hmm, mdp, qmdp, baselines}.py
│   └── utils/{metrics, utility, plotting}.py
├── experiments/                    ← 12 numbered runnable scripts (00 - 12, no 05)
├── figures/                        ← all PDF + PNG outputs
└── results/
    ├── README.md                   ← what each CSV is + headline tables inline
    └── *.csv  *.npy                ← every numerical output
```

---

## References (selected; full list in [`report/extended_report.pdf`](report/extended_report.pdf) §References)

- Hamilton (1989) — regime-switching for GDP. *Econometrica.*
- Ang & Bekaert (2002) — regime gains small for all-equity. *RFS.*
- Guidolin & Timmermann (2007) — 4-state MS for multi-asset allocation. *JEDC.*
- Tu (2010) — regime gains depend on uncertainty handling. *Management Science.*
- Nystrup et al. (2018) — walk-forward refit matters. *Quantitative Finance.*
- Kaelbling, Littman, Cassandra (1998) — POMDP framework. *Artificial Intelligence.*
- Littman, Cassandra, Kaelbling (1995) — QMDP algorithm. *ICML.*
- Pineau, Gordon, Thrun (2003) — point-based VI. *IJCAI.*
- Powell (2019) — unified framework for stochastic optimization. *EJOR.*
- Hauskrecht (2000) — VFA for POMDPs. *JAIR.*
- Faber (2007) — quantitative approach to TAA. *JoWM.*
- Hurst, Ooi, Pedersen (2017) — century of evidence on trend-following. *JPM.*
- Moskowitz, Ooi, Pedersen (2012) — time-series momentum. *JFE.*
- Asness, Frazzini, Pedersen (2012) — risk parity. *FAJ.*
- Maillard, Roncalli, Teiletche (2010) — ERC properties. *JPM.*
- DeMiguel, Garlappi, Uppal (2009) — 1/N beats MV. *RFS.*
- Black & Litterman (1992) — global portfolio optimization. *FAJ.*

---

## License

Coursework. Not licensed for redistribution outside the team and the course.
