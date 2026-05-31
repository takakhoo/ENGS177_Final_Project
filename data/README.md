# Data Directory

This directory holds raw input data and processed/aligned frames. **All files
here are committed to the repository** so teammates can run the full pipeline
end-to-end without first hitting FRED or Yahoo. They are still fully
reproducible by running [`../experiments/01_fetch_data.py`](../experiments/01_fetch_data.py) if you want to refresh
to the latest available data.

## Structure

- [`raw/`](raw/), one CSV per upstream source (FRED, Yahoo). Daily frequency where applicable.
- [`processed/`](processed/), aligned monthly frame used by all downstream experiments, plus pickled HMM models.

## Sources (clickable)

Every series has both a clickable link to its primary source page and a link to the raw CSV in this repo.

| Filename | Source page | Frequency | Identifier | Description |
|---|---|---|---|---|
| [`raw/vix.csv`](raw/vix.csv) | [FRED · VIXCLS](https://fred.stlouisfed.org/series/VIXCLS) | Daily → monthly EOM | `VIXCLS` | CBOE Volatility Index (VIX) close |
| [`raw/term_spread.csv`](raw/term_spread.csv) | [FRED · T10Y3M](https://fred.stlouisfed.org/series/T10Y3M) | Daily → monthly EOM | `T10Y3M` | 10-Year Treasury minus 3-Month Treasury yield (percent) |
| [`raw/hy_oas.csv`](raw/hy_oas.csv) | [FRED · BAMLH0A0HYM2](https://fred.stlouisfed.org/series/BAMLH0A0HYM2) | Daily → monthly EOM | `BAMLH0A0HYM2` | ICE BofA US High Yield option-adjusted spread (percent). **Truncated by FRED CSV endpoint → dropped from model.** |
| [`raw/nber.csv`](raw/nber.csv) | [NBER cycle dates](https://www.nber.org/research/business-cycle-dating) · [FRED · USREC](https://fred.stlouisfed.org/series/USREC) | Monthly | `USREC` | NBER recession indicator (1 = recession). Used for qualitative validation only, **never inside the model**. |
| [`raw/spy.csv`](raw/spy.csv) | [Yahoo Finance · SPY](https://finance.yahoo.com/quote/SPY/) | Daily → monthly EOM | `SPY` | SPDR S&P 500 ETF adjusted-close |
| [`raw/agg.csv`](raw/agg.csv) | [Yahoo Finance · AGG](https://finance.yahoo.com/quote/AGG/) | Daily → monthly EOM | `AGG` | iShares Core US Aggregate Bond ETF adjusted-close (starts Sep 2003 → sets the monthly-panel start date) |
| [`processed/monthly.csv`](processed/monthly.csv) | (derived) | Monthly EOM | aligned frame | What every downstream experiment reads. 271 rows, 2003-10-31 → 2026-04-30 |
| [`processed/hmm_2state.pkl`](processed/hmm_2state.pkl) | (derived) |, | trained model | `hmmlearn.GaussianHMM` with 2 regimes, fit on (VIX, term spread) |
| [`processed/hmm_3state.pkl`](processed/hmm_3state.pkl) | (derived) |, | trained model | 3-state variant (used for the K-sweep figure only) |
| [`processed/hmm_4state.pkl`](processed/hmm_4state.pkl) | (derived) |, | trained model | 4-state variant (used for the K-sweep figure only) |

## `processed/monthly.csv` schema

| Column | Type | Description |
|---|---|---|
| `DATE` (index) | datetime | End of month |
| `vix` | float | CBOE VIX level at month-end |
| `term_spread` | float | 10Y minus 3M Treasury yield, percent |
| `hy_oas` | float | ICE BofA US High Yield OAS, percent (column present but mostly empty due to truncation) |
| `nber_recession` | int | 1 during NBER-dated recession, else 0 |
| `spy_ret` | float | Monthly log return on SPY |
| `agg_ret` | float | Monthly log return on AGG (Sep 2003+) |

## Fetch script

[`../src/data/fetch_data.py`](../src/data/fetch_data.py) calls FRED's public CSV endpoint:
```
https://fred.stlouisfed.org/graph/fredgraph.csv?id=<series>&cosd=1990-01-01
```
and `yfinance.download(...)` for Yahoo tickers. No API key needed.

## Reproducibility

```bash
python experiments/01_fetch_data.py
# Downloads ~6 series totalling < 5 MB, writes CSVs in this directory.
# Network required (FRED + Yahoo).
```

The pipeline is fully deterministic given identical input data. Set random seeds in any script that uses Monte Carlo (`np.random.default_rng(42)` is the convention).
