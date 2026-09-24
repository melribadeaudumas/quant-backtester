"""Command-line entry point.

Examples
--------
    python -m backtester.cli --synthetic
    python -m backtester.cli --ticker AAPL --period 10y --cost-bps 1.5
    python -m backtester.cli --csv prices.csv --sort-by max_drawdown
"""
from __future__ import annotations

import argparse

import pandas as pd

from .data import load_csv, load_prices, synthetic_prices
from .engine import rank_strategies

pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rank systematic strategies by risk-adjusted return.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--ticker", help="Yahoo Finance ticker (needs yfinance + network).")
    src.add_argument("--csv", help="Path to a CSV price file.")
    src.add_argument("--synthetic", action="store_true", help="Use built-in synthetic data.")
    p.add_argument("--period", default="10y", help="History for --ticker (default 10y).")
    p.add_argument("--cost-bps", type=float, default=1.0, help="Transaction cost, bps per trade.")
    p.add_argument("--risk-free", type=float, default=0.0, help="Annual risk-free rate.")
    p.add_argument("--sort-by", default="sharpe",
                   choices=["sharpe", "cagr", "total_return", "max_drawdown", "calmar"],
                   help="Ranking metric.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.ticker:
        prices = load_prices(args.ticker, period=args.period)
        label = args.ticker
    elif args.csv:
        prices = load_csv(args.csv)
        label = args.csv
    else:
        prices = synthetic_prices()
        label = "synthetic GBM (~10y)"

    table = rank_strategies(
        prices, cost_bps=args.cost_bps, risk_free=args.risk_free, sort_by=args.sort_by
    )
    print(f"\nInstrument: {label}   |   {prices.index[0].date()} to {prices.index[-1].date()}"
          f"   |   {len(prices)} days\n")
    print(table.to_string())
    print(f"\nRanked by {args.sort_by}. Transaction cost = {args.cost_bps} bps per trade.\n")


if __name__ == "__main__":
    main()
