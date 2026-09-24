"""The backtest loop and a helper to rank several strategies."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .metrics import performance_summary
from .strategies import STRATEGIES


@dataclass
class BacktestResult:
    """Container for a single backtest run."""
    name: str
    returns: pd.Series          # daily strategy returns (net of costs)
    equity_curve: pd.Series     # cumulative growth of 1 unit
    position: pd.Series         # 0/1 position actually held each day
    stats: dict[str, float]     # performance_summary output

    def __repr__(self) -> str:  # pragma: no cover
        s = self.stats
        return (
            f"BacktestResult({self.name}: "
            f"CAGR={s['cagr']:.2%}, Sharpe={s['sharpe']:.2f}, "
            f"MaxDD={s['max_drawdown']:.2%})"
        )


def backtest(
    prices: pd.Series,
    position: pd.Series,
    name: str = "strategy",
    cost_bps: float = 1.0,
    risk_free: float = 0.0,
) -> BacktestResult:
    """Run a long/flat backtest.

    Parameters
    ----------
    prices : daily price series.
    position : 0/1 target position from a strategy (same index as prices).
    cost_bps : round-trip-agnostic transaction cost in basis points, charged on
        every change in position (|Δposition| * cost).
    risk_free : annual risk-free rate for the Sharpe ratio.

    The position is lagged by one day, so a signal from day t is only traded on
    day t+1 — this avoids look-ahead bias.
    """
    prices = prices.dropna()
    position = position.reindex(prices.index).fillna(0.0)

    asset_returns = prices.pct_change().fillna(0.0)
    held = position.shift(1).fillna(0.0)                 # trade with a one-day lag
    turnover = held.diff().abs().fillna(held.abs())      # position changes
    costs = turnover * (cost_bps / 10_000.0)

    strat_returns = held * asset_returns - costs
    equity = (1.0 + strat_returns).cumprod()
    stats = performance_summary(strat_returns, risk_free=risk_free)

    return BacktestResult(
        name=name,
        returns=strat_returns.rename(name),
        equity_curve=equity.rename(name),
        position=held.rename("position"),
        stats=stats,
    )


def rank_strategies(
    prices: pd.Series,
    strategies: dict | None = None,
    cost_bps: float = 1.0,
    risk_free: float = 0.0,
    sort_by: str = "sharpe",
) -> pd.DataFrame:
    """Backtest several strategies on the same series and rank them.

    `strategies` maps a name to either a callable(prices) -> position, or a
    (callable, kwargs) tuple. Defaults to the built-in registry with default
    parameters. Returns a DataFrame of metrics sorted by `sort_by`
    (Sharpe descending; max drawdown is ranked by smallest loss).
    """
    if strategies is None:
        strategies = STRATEGIES

    rows = {}
    for name, spec in strategies.items():
        func, kwargs = (spec if isinstance(spec, tuple) else (spec, {}))
        position = func(prices, **kwargs)
        result = backtest(prices, position, name=name, cost_bps=cost_bps, risk_free=risk_free)
        rows[name] = result.stats

    table = pd.DataFrame(rows).T
    ascending = sort_by in {"max_drawdown", "ann_volatility"}
    return table.sort_values(sort_by, ascending=ascending)
