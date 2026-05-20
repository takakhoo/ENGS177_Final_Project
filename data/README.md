# Data Directory

This directory holds raw input data and processed/aligned frames. **All files
here are committed to the repository** so teammates can run the full pipeline
end-to-end without first hitting FRED or Yahoo. They are still fully
reproducible by running `experiments/01_fetch_data.py` if you want to refresh
to the latest available data.

## Structure

- `raw/` — one CSV per upstream source (FRED, Yahoo). Daily frequency where applicable.
- `processed/` — aligned monthly frame used by all downstream experiments, plus pickled HMM models.

## Sources

| Filename | Source | Frequency | Identifier |
|---|---|---|---|
| `raw/vix.csv` | FRED | Daily → monthly | `VIXCLS` |
| `raw/term_spread.csv` | FRED | Daily → monthly | `T10Y3M` |
| `raw/hy_oas.csv` | FRED | Daily → monthly | `BAMLH0A0HYM2` |
| `raw/nber.csv` | FRED | Monthly | `USREC` |
| `raw/spy.csv` | Yahoo | Daily → monthly | `SPY` |
| `raw/agg.csv` | Yahoo | Daily → monthly | `AGG` |
| `processed/monthly.csv` | (derived) | Monthly EOM | aligned frame |
| `processed/hmm_2state.pkl` | (derived) | — | trained Gaussian HMM (2 states) |
| `processed/hmm_3state.pkl` | (derived) | — | trained Gaussian HMM (3 states) |

## `processed/monthly.csv` schema

| Column | Type | Description |
|---|---|---|
| `DATE` (index) | datetime | End of month |
| `vix` | float | CBOE VIX level at month-end |
| `term_spread` | float | 10Y minus 3M Treasury yield, percent |
| `hy_oas` | float | ICE BofA US High Yield option-adjusted spread, percent |
| `nber_recession` | int | 1 during NBER-dated recession, else 0 |
| `spy_ret` | float | Monthly log return on SPY |
| `agg_ret` | float | Monthly log return on AGG (Sep 2003+) |

## Reproducibility

```bash
python experiments/01_fetch_data.py
# Downloads ~6 series totalling < 5 MB, writes CSVs in this directory.
# Network required.
```

The pipeline is fully deterministic given identical input data. Set random seeds in any script that uses Monte Carlo (`np.random.default_rng(42)` is the convention).
