"""Price data loading.

Three sources, in order of convenience:
  * load_prices(ticker)      -> Yahoo Finance via yfinance (needs network + yfinance)
  * load_csv(path)           -> a CSV with a Date index and a 'Close' (or 'Adj Close') column
  * synthetic_prices(...)    -> reproducible geometric-Brownian-motion series (no dependencies)

Everything downstream only needs a pandas Series of prices indexed by date, so the
engine runs fully offline on synthetic data if you have no market feed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_prices(
    n_days: int = 2520,
    mu: float = 0.08,
    sigma: float = 0.20,
    s0: float = 100.0,
    seed: int | None = 42,
) -> pd.Series:
    """Generate a reproducible daily price series via geometric Brownian motion.

    Defaults: ~10 years of trading days, 8% drift, 20% annual volatility.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    shocks = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), size=n_days)
    prices = s0 * np.exp(np.cumsum(shocks))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    return pd.Series(prices, index=idx, name="Close")


def load_csv(path: str, price_col: str | None = None) -> pd.Series:
    """Load a price series from a CSV file with a date index."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if price_col is None:
        for candidate in ("Adj Close", "adj_close", "Close", "close", "price"):
            if candidate in df.columns:
                price_col = candidate
                break
    if price_col is None:
        raise ValueError(f"No price column found in {path}; columns = {list(df.columns)}")
    return df[price_col].dropna().rename("Close")


def load_prices(ticker: str, period: str = "10y", interval: str = "1d") -> pd.Series:
    """Download a price series from Yahoo Finance. Requires the optional `yfinance`."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "yfinance is not installed. Run `pip install yfinance`, "
            "or use synthetic_prices()/load_csv() instead."
        ) from exc
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker!r}.")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance sometimes returns a 1-col frame
        close = close.iloc[:, 0]
    return close.dropna().rename("Close")
