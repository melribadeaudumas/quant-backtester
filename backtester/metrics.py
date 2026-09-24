"""Performance and risk metrics computed from a daily return series."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def total_return(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() - 1.0)


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    n = returns.shape[0]
    if n == 0:
        return 0.0
    growth = (1.0 + returns).prod()
    if growth <= 0:
        return -1.0
    return float(growth ** (periods_per_year / n) - 1.0)


def annualised_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return float(returns.std(ddof=0) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualised Sharpe ratio. `risk_free` is an annual rate."""
    if returns.std(ddof=0) == 0:
        return 0.0
    rf_per_period = risk_free / periods_per_year
    excess = returns - rf_per_period
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std(ddof=0))


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough decline of the cumulative equity curve (a negative number)."""
    equity = (1.0 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return 0.0
    return cagr(returns, periods_per_year) / mdd


def performance_summary(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> dict[str, float]:
    """Return the full metric set as a dictionary."""
    returns = returns.dropna()
    return {
        "total_return": total_return(returns),
        "cagr": cagr(returns, periods_per_year),
        "ann_volatility": annualised_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, risk_free, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns, periods_per_year),
    }
