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
    "vix": "VIXCLS",
    "term_spread": "T10Y3M",
    "hy_oas": "BAMLH0A0HYM2",
    "nber": "USREC",
}
YF_TICKERS = ["SPY", "AGG"]


def _fred(series_id: str, start: str = "1990-01-01") -> pd.Series:
    """Fetch one FRED series. Lazy import for environments without the dep."""
    from pandas_datareader import data as pdr  # type: ignore[import-not-found]

    return pdr.DataReader(series_id, "fred", start=start).iloc[:, 0]


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

    Observation columns are *level* (vix, spread, hy_oas at month end).
    Return columns are monthly *log* returns of SPY and AGG.
    """
    PROCESSED.mkdir(parents=True, exist_ok=True)

    monthly_obs = pd.concat(
        {
            "vix": series["vix"].resample("ME").last(),
            "term_spread": series["term_spread"].resample("ME").last(),
            "hy_oas": series["hy_oas"].resample("ME").last(),
            "nber_recession": series["nber"].resample("ME").last().astype(int),
        },
        axis=1,
    )

    spy_m = series["spy"].resample("ME").last()
    agg_m = series["agg"].resample("ME").last()
    monthly_ret = pd.concat(
        {"spy_ret": np.log(spy_m / spy_m.shift(1)), "agg_ret": np.log(agg_m / agg_m.shift(1))},
        axis=1,
    )

    df = pd.concat([monthly_obs, monthly_ret], axis=1).dropna()
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp("M")
    out_path = PROCESSED / "monthly.csv"
    df.to_csv(out_path)
    print(f"\nWrote {len(df)} aligned monthly rows ({df.index.min().date()} → {df.index.max().date()}) "
          f"to {out_path.relative_to(REPO_ROOT)}")
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
