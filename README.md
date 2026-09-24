# Quant Backtester

A small, readable backtesting engine for long/flat systematic equity strategies,
written in Python with only NumPy and pandas. I built it to check whether the
trading ideas I was running on my own portfolio actually held up — or whether I
was just remembering the trades that worked.

It implements three classic signals and ranks them on **risk-adjusted** metrics
(Sharpe ratio and maximum drawdown), rather than raw return.

## Strategies

| Name | Idea | Rule |
|------|------|------|
| `ma_crossover` | Trend following | Long when the fast moving average is above the slow one |
| `momentum` | Time-series momentum | Long when the trailing N-day return is positive |
| `rsi_reversion` | Mean reversion | Buy when RSI is oversold, hold until it recovers |

## Metrics

Total return, CAGR, annualised volatility, **Sharpe ratio**, **maximum drawdown**
and Calmar ratio — all computed from the daily strategy returns.

## Design notes

- **No look-ahead bias.** A signal computed on day *t* is only traded on day *t+1*
  (positions are lagged one day inside the engine).
- **Transaction costs.** A cost in basis points is charged on every change in
  position, so churny strategies are penalised.
- **Runs offline.** If you have no market feed, the engine generates reproducible
  synthetic prices via geometric Brownian motion, so the whole thing works with
  just NumPy and pandas.

## Install

```bash
pip install -r requirements.txt   # yfinance is optional; numpy + pandas are enough
```

## Usage

Command line:

```bash
# No network needed — runs on ~10 years of synthetic data
python -m backtester.cli --synthetic

# Real data (requires yfinance)
python -m backtester.cli --ticker AAPL --period 10y --cost-bps 1.5

# From your own CSV (needs a Date index and a Close / Adj Close column)
python -m backtester.cli --csv prices.csv --sort-by max_drawdown
```

In code:

```python
from backtester.data import synthetic_prices
from backtester.engine import backtest, rank_strategies
from backtester.strategies import ma_crossover

prices = synthetic_prices()
print(rank_strategies(prices, sort_by="sharpe"))          # compare all strategies

result = backtest(prices, ma_crossover(prices), name="ma_50_200")
print(result.stats["sharpe"], result.stats["max_drawdown"])
```

## Tests

```bash
python -m pytest -q
```

## Project layout

```
backtester/
  data.py         # price loading: yfinance, CSV, or synthetic
  strategies.py   # ma_crossover, momentum, rsi_reversion
  metrics.py      # returns, Sharpe, max drawdown, Calmar, ...
  engine.py       # backtest loop + rank_strategies()
  cli.py          # command-line runner
examples/
  run_example.py
tests/
  test_metrics.py
```

## Disclaimer

For research and educational use only. Nothing here is investment advice, and
past (or backtested) performance does not indicate future results.

## License

MIT — see [LICENSE](LICENSE).
