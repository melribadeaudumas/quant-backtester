"""A small, dependency-light backtesting engine for long/flat systematic strategies.

Strategies: moving-average crossover, time-series momentum, RSI mean-reversion.
Metrics: total return, CAGR, annualised volatility, Sharpe ratio, maximum drawdown, Calmar.
"""
from .metrics import performance_summary
from .strategies import ma_crossover, momentum, rsi_reversion, STRATEGIES
from .engine import backtest, rank_strategies

__all__ = [
    "performance_summary",
    "ma_crossover",
    "momentum",
    "rsi_reversion",
    "STRATEGIES",
    "backtest",
    "rank_strategies",
]
__version__ = "0.1.0"
