"""Sanity checks for the metric functions. Run with: python -m pytest -q"""
import numpy as np
import pandas as pd

from backtester.metrics import max_drawdown, sharpe_ratio, total_return


def _series(vals):
    idx = pd.bdate_range("2020-01-01", periods=len(vals))
    return pd.Series(vals, index=idx)


def test_total_return_simple():
    r = _series([0.10, -0.10])           # +10% then -10%
    assert abs(total_return(r) - (1.10 * 0.90 - 1.0)) < 1e-12


def test_zero_vol_sharpe_is_zero():
    r = _series([0.0, 0.0, 0.0])
    assert sharpe_ratio(r) == 0.0


def test_max_drawdown_is_negative_and_bounded():
    r = _series([0.05, -0.20, 0.03, -0.10])
    mdd = max_drawdown(r)
    assert -1.0 <= mdd < 0.0


def test_positive_mean_returns_have_positive_sharpe():
    rng = np.random.default_rng(0)
    r = _series(list(0.001 + 0.005 * rng.standard_normal(250)))  # positive mean, real variance
    assert sharpe_ratio(r) > 0.0


def test_constant_returns_have_zero_sharpe_by_convention():
    r = _series(list(np.full(50, 0.001)))   # zero volatility -> Sharpe undefined
    assert sharpe_ratio(r) == 0.0
