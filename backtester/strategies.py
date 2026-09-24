"""Signal generators.

Each strategy takes a price Series and returns a *position* Series in {0, 1}
(1 = fully long, 0 = flat) aligned to the price index. Positions are applied
with a one-day lag by the engine, so a signal computed on day t is traded on
day t+1 (no look-ahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ma_crossover(prices: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """Long when the fast moving average is above the slow one, else flat."""
    if fast >= slow:
        raise ValueError("`fast` window must be shorter than `slow`.")
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    position = (fast_ma > slow_ma).astype(float)
    position[slow_ma.isna()] = 0.0
    return position.rename("position")


def momentum(prices: pd.Series, lookback: int = 126) -> pd.Series:
    """Time-series momentum: long when trailing `lookback`-day return is positive."""
    trailing_return = prices.pct_change(lookback)
    position = (trailing_return > 0).astype(float)
    position[trailing_return.isna()] = 0.0
    return position.rename("position")


def _rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index."""
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.fillna(50.0)


def rsi_reversion(
    prices: pd.Series, window: int = 14, buy: float = 30.0, exit_level: float = 55.0
) -> pd.Series:
    """Mean-reversion: go long when RSI drops below `buy` (oversold),
    hold until RSI recovers above `exit_level`, then go flat.
    """
    rsi = _rsi(prices, window)
    position = np.zeros(len(prices))
    holding = False
    for i, value in enumerate(rsi.to_numpy()):
        if not holding and value < buy:
            holding = True
        elif holding and value > exit_level:
            holding = False
        position[i] = 1.0 if holding else 0.0
    return pd.Series(position, index=prices.index, name="position")


# Registry used by the CLI / ranking helper.
STRATEGIES = {
    "ma_crossover": ma_crossover,
    "momentum": momentum,
    "rsi_reversion": rsi_reversion,
}
