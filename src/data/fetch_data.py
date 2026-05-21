"""Fetch raw FRED and Yahoo data, write CSVs to data/raw/, and produce a clean
monthly aligned frame in data/processed/monthly.csv.

Run from repo root:
    python -m src.data.fetch_data
or
    python experiments/01_fetch_data.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data" / "raw"
PROCESSED = REPO_ROOT / "data" / "processed"

FRED_SERIES = {
    # ----- Headline observation channels (2-state HMM) -----
    "vix": "VIXCLS",          # CBOE Volatility Index — equity-vol regime signal
    "term_spread": "T10Y3M",  # 10Y minus 3M Treasury — yield-curve signal

    # ----- Original proposal third channel (truncated by FRED CSV endpoint) -----
    "hy_oas": "BAMLH0A0HYM2", # ICE BofA US HY OAS

    # ----- Recession label (qualitative validation only, never inside model) -----
    "nber": "USREC",          # NBER recession indicator

    # ----- Extension v2: richer macro-financial observables -----
    "term_spread_2y": "T10Y2Y",   # 10Y minus 2Y spread (more standard yield-curve metric)
    "nfci": "NFCI",                # Chicago Fed National Financial Conditions Index (weekly)
    "stlfsi": "STLFSI4",           # St. Louis Fed Financial Stress Index 4 (weekly)
    "umcsent": "UMCSENT",          # U Michigan Consumer Sentiment (monthly)
    "jobless_claims": "ICSA",      # Initial weekly jobless claims (weekly)
    "usd_index": "DTWEXBGS",       # Trade-weighted US dollar index (broad)
    "wti_oil": "DCOILWTICO",       # West Texas Intermediate crude oil price (daily)
    "fed_funds": "DFF",            # Effective federal funds rate (daily)
}
YF_TICKERS = ["SPY", "AGG"]


def _fred(series_id: str, start: str = "1990-01-01") -> pd.Series:
    """Fetch one FRED series via the public CSV endpoint.

    No pandas_datareader dependency — that library has compat issues with
    pandas>=3.0. The CSV endpoint is stable and well-documented.
    """
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        f"&cosd={start}"
    )
    df = pd.read_csv(url, parse_dates=["observation_date"], na_values=["."])
    s = df.set_index("observation_date")[series_id].astype(float)
    s.name = series_id
    return s


def _yahoo(ticker: str, start: str = "1990-01-01") -> pd.Series:
    """Fetch adjusted-close from Yahoo. Returns daily price series."""
    import yfinance as yf  # type: ignore[import-not-found]

    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    return df["Close"][ticker] if isinstance(df["Close"], pd.DataFrame) else df["Close"]


def fetch_raw() -> dict[str, pd.Series]:
    """Download every series and save to data/raw/. Returns the dict for chaining."""
    RAW.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.Series] = {}
    for name, code in FRED_SERIES.items():
        print(f"FRED  → {name} ({code})")
        s = _fred(code).rename(name)
        s.to_csv(RAW / f"{name}.csv")
        out[name] = s
    for t in YF_TICKERS:
        print(f"Yahoo → {t}")
        s = _yahoo(t).rename(t.lower())
        s.to_csv(RAW / f"{t.lower()}.csv")
        out[t.lower()] = s
    return out


def build_monthly(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Resample to end-of-month and align into one frame.

    Headline observation columns (always-on): vix, term_spread.
    Extension columns: term_spread_2y, nfci, stlfsi, umcsent, jobless_claims,
    usd_index, wti_oil, fed_funds — used by the 03 + 04 richer-observation
    experiments in the extended pipeline. Each is forward-filled at month-end.

    Optional: hy_oas — kept if FRED CSV gives us a long-enough series, else dropped.
    NBER recession indicator is loaded as a side channel for plotting only.

    Return columns are monthly *log* returns of SPY and AGG.
    """
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # ----- Headline observations -----
    parts = {
        "vix": series["vix"].resample("ME").last(),
        "term_spread": series["term_spread"].resample("ME").last(),
        "nber_recession": series["nber"].resample("ME").last().astype(int),
    }

    # ----- Optional HY OAS — drop if FRED truncated it -----
    hy = series["hy_oas"].resample("ME").last()
    vix_start = parts["vix"].dropna().index.min()
    if hy.dropna().index.min() - vix_start <= pd.Timedelta(days=365 * 10):
        parts["hy_oas"] = hy
    else:
        print(f"  [warn] hy_oas series starts {hy.dropna().index.min().date()} "
              f"vs vix {vix_start.date()} — dropping hy_oas from observations.")

    # ----- Extension v2 channels -----
    extension_keys = [
        "term_spread_2y", "nfci", "stlfsi", "umcsent",
        "jobless_claims", "usd_index", "wti_oil", "fed_funds",
    ]
    for k in extension_keys:
        if k in series:
            parts[k] = series[k].resample("ME").last()

    monthly_obs = pd.concat(parts, axis=1)

    # ----- Asset returns -----
    spy_m = series["spy"].resample("ME").last()
    agg_m = series["agg"].resample("ME").last()
    monthly_ret = pd.concat(
        {"spy_ret": np.log(spy_m / spy_m.shift(1)), "agg_ret": np.log(agg_m / agg_m.shift(1))},
        axis=1,
    )

    # Concat. Use 'outer' so we can keep partially-missing extension columns
    # (some FRED series start later than the headline panel — we'll drop NaNs on
    # the headline columns only).
    df = pd.concat([monthly_obs, monthly_ret], axis=1)
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp("M")

    # Drop rows where headline columns or returns are missing
    headline_required = ["vix", "term_spread", "spy_ret", "agg_ret"]
    df = df.dropna(subset=headline_required)

    out_path = PROCESSED / "monthly.csv"
    df.to_csv(out_path)
    print(f"\nWrote {len(df)} aligned monthly rows ({df.index.min().date()} → {df.index.max().date()}) "
          f"to {out_path.relative_to(REPO_ROOT)}")
    avail_cols = [c for c in extension_keys if c in df.columns and df[c].notna().any()]
    print(f"Extension columns present: {avail_cols}")
    return df


def main() -> None:
    series = fetch_raw()
    df = build_monthly(series)
    print("\nHead:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe().round(3))


if __name__ == "__main__":
    main()
