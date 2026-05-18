# Regime-Switching Asset Allocation via a POMDP

**ENGS 177, Decision-Making Under Uncertainty, Spring 2026**
Dario Blanco Morales · Even Hogberget · Kyle David Ledda-Lewaren · Taka Khoo
Instructor: Prof. Wesley Marrero, Thayer School of Engineering, Dartmouth College

> A long-only investor must split capital between U.S. equities (SPY) and aggregate bonds (AGG) on a monthly schedule. We model the prevailing macro-financial regime as a hidden state, infer it from VIX and the 10Y–3M Treasury term spread via a Gaussian Hidden Markov Model, and use a QMDP approximation of the resulting POMDP to choose portfolio weights. The policy is backtested against a static 60/40 benchmark and a myopic trend-following baseline over 2003–2026.

## Deliverables (click to download)

| Item | Pages | Link |
|---|---|---|
| **Final report** (full math, methods, results) | 10 | [`report/report.pdf`](report/report.pdf) |
| **Presentation slides** (10-slide overview) | 11 | [`presentation/slides.pdf`](presentation/slides.pdf) |
| Original proposal (submitted to Canvas) | 2 | [`proposal/ENGS177_Term_Project_Proposal.docx`](proposal/ENGS177_Term_Project_Proposal.docx) |

## Headline result

Out-of-sample backtest, October 2003 through April 2026 (271 monthly observations), 5 bps transaction cost per side, monthly rebalancing:

| Policy | CAGR | Vol | Sharpe | Max DD | Calmar | Turnover |
|---|---:|---:|---:|---:|---:|---:|
| **Static 60/40** (benchmark) | 7.35% | 9.4% | 0.81 | −34.2% | 0.22 | 0.000 |
| **QMDP** (CRRA γ=2)           | 9.90% | 14.7% | 0.72 | −53.0% | 0.19 | 0.000 |
| **Myopic** 12-month trend     | **13.99%** | 11.3% | **1.22** | **−21.1%** | **0.66** | 0.236 |

Under log/γ=2 utility, the underlying MDP's optimal policy collapses to 100% stocks in every regime, so QMDP behaves like a leveraged equity strategy. Sweeping CRRA risk aversion (figures below) shows that QMDP's Sharpe surpasses the static benchmark once γ ≥ 8, but the optimal policy is the same in bull and bear regimes at every γ we tested. The two macro observables identify the regime correctly; they do not differentiate it enough to act on.

## Main figures

### Regime detection: HMM correctly recovers 2008, 2020 and 2022 stress
![Regime timeline across K=2,3,4 HMMs](figures/multistate_regime_timeline.png)

The bear regime (filled red region) is the state with the lowest mean SPY return. NBER recessions (gray bands) sit cleanly inside the bear bands at K=2. K=3 and K=4 collapse extra states.

### Backtest: equity curves across regime counts
![Equity curves across K=2,3,4](figures/multistate_equity_curves.png)

Terminal wealth from \$1 invested, log scale. The QMDP and static curves are nearly indistinguishable; the myopic trend follower captures recent first-moment information and dominates.

### Sensitivity: when does QMDP unlock?
![QMDP Sharpe vs CRRA risk aversion](figures/gamma_sensitivity_sharpe.png)

Sharpe ratio of each policy as CRRA risk aversion γ varies. The vertical red line marks the crossover (γ ≈ 8) where QMDP first beats the static benchmark.

### Sensitivity: full equity curves at four γ values
![Equity curves under varying gamma](figures/gamma_equity_curves.png)

Equity curves at γ ∈ {2, 5, 10, 20}. Higher γ shifts the QMDP policy bondward; the static and myopic policies do not depend on γ.

## What's in this repo

```
ENGS177_Final_Project/
├── README.md                 ← you are here
├── report/                   ← LaTeX final report
│   ├── report.tex
│   └── report.pdf
├── presentation/             ← beamer slides
│   ├── slides.tex
│   └── slides.pdf
├── proposal/                 ← submitted Canvas proposal
├── docs/                     ← agent / contributor briefs
│   ├── 00_agent_quickstart.md
│   ├── 01_project_overview.md
│   ├── 02_class_concepts.md
│   ├── 03_external_research.md
│   ├── 04_implementation_plan.md
│   ├── 05_experimental_design.md
│   └── 06_deliverables.md
├── homework/                 ← per-author HW1–3 PDFs and TeX
│   ├── taka/  kdll/  dario/  even/
├── data/                     ← FRED + Yahoo raw and aligned monthly panel
│   ├── README.md
│   ├── raw/
│   └── processed/monthly.csv
├── src/                      ← reusable Python modules
│   ├── data/fetch_data.py
│   ├── models/{hmm,mdp,qmdp}.py
│   └── utils/{utility,metrics,plotting}.py
├── experiments/              ← runnable pipeline scripts
│   ├── 00_synthetic_demo.py            (no-network smoke test)
│   ├── 01_fetch_data.py                (FRED + Yahoo download)
│   ├── 02_hmm_calibration.py           (Baum–Welch + BIC)
│   ├── 03_regime_interpretation.py     (timeline plot vs NBER)
│   ├── 04_qmdp_solve.py                (VI + PI + QMDP)
│   ├── 05_backtest_compare.py          (headline backtest)
│   ├── 06_multistate_comparison.py     (K ∈ {2,3,4})
│   └── 07_gamma_sensitivity.py         (CRRA γ sweep)
├── figures/                  ← PDF + PNG outputs
└── results/                  ← CSV tables and INITIAL_FINDINGS.md
```

## How to reproduce

```bash
# One-time setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Synthetic smoke test (no network)
python experiments/00_synthetic_demo.py

# Real-data pipeline (FRED + Yahoo)
python experiments/01_fetch_data.py
python experiments/02_hmm_calibration.py
python experiments/03_regime_interpretation.py
python experiments/04_qmdp_solve.py
python experiments/05_backtest_compare.py

# Sensitivity studies (used for the report's headline figures)
python experiments/06_multistate_comparison.py
python experiments/07_gamma_sensitivity.py
```

## How to load the data and a fitted HMM

After running steps 01 and 02 above:

```python
import pickle
from pathlib import Path
import pandas as pd

REPO = Path(".")
# Monthly aligned panel: VIX, term spread, NBER dummy, SPY and AGG log returns.
df = pd.read_csv(REPO / "data/processed/monthly.csv")
dc = df.columns[0]                       # FRED uses 'observation_date'
df[dc] = pd.to_datetime(df[dc]); df = df.set_index(dc)

# Pre-trained 2-state Gaussian HMM
with open(REPO / "data/processed/hmm_2state.pkl", "rb") as f:
    hmm = pickle.load(f)

print(df.tail())
print("transition matrix:\n", hmm.transmat_)
print("emission means (VIX, term spread):\n", hmm.means_)
```

The monthly panel columns are documented in [`data/README.md`](data/README.md). Models are `hmmlearn.GaussianHMM` instances and are fully picklable.

## Further reading inside the repo

- [`docs/01_project_overview.md`](docs/01_project_overview.md), canonical problem statement, POMDP tuple, solver choice
- [`docs/03_external_research.md`](docs/03_external_research.md), five-tier literature map (Hamilton, Kaelbling, Puterman, etc.)
- [`docs/04_implementation_plan.md`](docs/04_implementation_plan.md), pipeline diagram and per-step responsibilities
- [`docs/05_experimental_design.md`](docs/05_experimental_design.md), seven experiments with falsifiable predictions
- [`results/INITIAL_FINDINGS.md`](results/INITIAL_FINDINGS.md), narrative of the first real-data backtest and why the QMDP collapses at γ=2

## External references

The report cites the following papers and books in full: Hamilton (1989), Ang & Bekaert (2002), Guidolin & Timmermann (2007), Nystrup, Madsen & Lindström (2018), Kaelbling, Littman & Cassandra (1998), Littman, Cassandra & Kaelbling (1995), Rabiner (1989), Pineau, Gordon & Thrun (2003), Puterman (2005), Kochenderfer (2015), Sutton & Barto (2018), and López de Prado (2018). See [`report/report.pdf`](report/report.pdf) §References for citations.

## License

Coursework. Not licensed for redistribution outside the team and the course.
