"""Minimal end-to-end example: rank the three strategies on synthetic data.

Run from the repo root:
    python examples/run_example.py
"""
import sys
from pathlib import Path

# Allow running the file directly without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtester.data import synthetic_prices
from backtester.engine import backtest, rank_strategies
from backtester.strategies import ma_crossover

prices = synthetic_prices(seed=7)

# 1) Rank all built-in strategies by Sharpe ratio.
print("Strategy ranking (by Sharpe):")
print(rank_strategies(prices, sort_by="sharpe").to_string())

# 2) Inspect a single strategy in detail.
position = ma_crossover(prices, fast=50, slow=200)
result = backtest(prices, position, name="ma_50_200", cost_bps=1.0)
print("\nSingle run:", result)
print("Final equity multiple:", round(result.equity_curve.iloc[-1], 3))
