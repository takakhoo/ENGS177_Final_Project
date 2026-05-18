"""Performance metrics for monthly return series."""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 12  # monthly rebalancing


def cagr(returns: pd.Series) -> float:
    cum = (1.0 + returns).prod()
    yrs = len(returns) / PERIODS_PER_YEAR
    return cum ** (1.0 / yrs) - 1.0


def annualized_vol(returns: pd.Series) -> float:
    return returns.std(ddof=0) * np.sqrt(PERIODS_PER_YEAR)


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / PERIODS_PER_YEAR
    if excess.std(ddof=0) == 0:
        return 0.0
    return excess.mean() / excess.std(ddof=0) * np.sqrt(PERIODS_PER_YEAR)


def max_drawdown(returns: pd.Series) -> float:
    cum = (1.0 + returns).cumprod()
    running_max = cum.cummax()
    return (cum / running_max - 1.0).min()  # negative number


def calmar(returns: pd.Series) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0:
        return np.nan
    return cagr(returns) / abs(mdd)


def turnover(weights: pd.DataFrame) -> float:
    """Average monthly L1 turnover."""
    return (weights.diff().abs().sum(axis=1)).mean()


def summarize(returns: pd.Series, weights: pd.DataFrame | None = None, name: str = "") -> dict:
    out = {
        "name": name,
        "cagr": cagr(returns),
        "vol": annualized_vol(returns),
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar(returns),
    }
    if weights is not None:
        out["turnover_avg"] = turnover(weights)
    return out
