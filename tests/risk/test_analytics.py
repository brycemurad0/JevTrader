from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jevtrader.core.types import AssetClass
from jevtrader.risk.analytics import (
    historical_cvar,
    historical_var,
    marginal_risk_contribution,
    parametric_cvar,
    parametric_var,
    portfolio_var_cvar,
    portfolio_vol,
    risk_report_markdown,
    rolling_correlation,
    stress_scenarios,
)


@pytest.fixture
def returns_df():
    rng = np.random.default_rng(42)
    n = 500
    common = rng.normal(0, 0.01, n)
    spy = common + rng.normal(0, 0.002, n)
    qqq = common + rng.normal(0, 0.002, n)  # highly correlated with spy
    btc = rng.normal(0, 0.03, n)  # independent, higher vol
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"SPY": spy, "QQQ": qqq, "BTC/USD": btc}, index=idx)


# ------------------------------------------------------------------------------- single series


def test_historical_var_cvar_sane_and_ordered():
    returns = pd.Series(np.random.default_rng(1).normal(0, 0.02, 1000))
    var95 = historical_var(returns, level=0.95)
    cvar95 = historical_cvar(returns, level=0.95)
    assert var95 >= 0
    assert cvar95 >= var95  # expected shortfall is at least as bad as VaR


def test_parametric_var_cvar_sane_and_ordered():
    returns = pd.Series(np.random.default_rng(2).normal(0, 0.02, 1000))
    var95 = parametric_var(returns, level=0.95)
    cvar95 = parametric_cvar(returns, level=0.95)
    assert var95 >= 0
    assert cvar95 >= var95


def test_higher_confidence_level_means_larger_var():
    returns = pd.Series(np.random.default_rng(3).normal(0, 0.02, 1000))
    var95 = historical_var(returns, level=0.95)
    var99 = historical_var(returns, level=0.99)
    assert var99 >= var95


def test_var_on_empty_series_is_zero():
    assert historical_var(pd.Series(dtype=float)) == 0.0
    assert parametric_cvar(pd.Series(dtype=float)) == 0.0


# ------------------------------------------------------------------------------- portfolio level


def test_portfolio_var_cvar_scales_with_equity(returns_df):
    positions = {"SPY": 5_000.0, "QQQ": 3_000.0, "BTC/USD": 2_000.0}
    equity = 10_000.0
    result = portfolio_var_cvar(positions, returns_df, equity, level=0.95, method="historical")
    assert result["var_usd"] == pytest.approx(result["var_pct"] * equity)
    assert result["cvar_pct"] >= result["var_pct"]


def test_rolling_correlation_high_for_correlated_pair(returns_df):
    corr = rolling_correlation(returns_df, window=200)
    assert corr.loc["SPY", "QQQ"] > 0.8
    assert abs(corr.loc["SPY", "BTC/USD"]) < 0.8


def test_portfolio_vol_and_marginal_contribution_sum_correctly(returns_df):
    cov = returns_df.cov()
    weights = pd.Series({"SPY": 0.5, "QQQ": 0.3, "BTC/USD": 0.2})
    sigma_p_periodic = portfolio_vol(weights, cov, periods_per_year=1)
    mrc = marginal_risk_contribution(weights, cov)
    assert mrc.sum() == pytest.approx(sigma_p_periodic, rel=1e-6)


def test_stress_scenarios_apply_asset_class_shocks():
    positions = {"SPY": 10_000.0, "BTC/USD": 4_000.0}
    asset_class_of = {"SPY": AssetClass.EQUITY, "BTC/USD": AssetClass.CRYPTO}
    result = stress_scenarios(positions, asset_class_of)
    assert result["equities_down_10pct"] == pytest.approx(-1_000.0)
    assert result["crypto_down_25pct"] == pytest.approx(-1_000.0)
    assert result["combined"] == pytest.approx(-2_000.0)


def test_risk_report_markdown_contains_key_sections(returns_df):
    positions = {"SPY": 5_000.0, "QQQ": 3_000.0, "BTC/USD": 2_000.0}
    asset_class_of = {"SPY": AssetClass.EQUITY, "QQQ": AssetClass.EQUITY, "BTC/USD": AssetClass.CRYPTO}
    md = risk_report_markdown(positions, 10_000.0, returns_df, asset_class_of)
    assert "# Risk report" in md
    assert "VaR95" in md
    assert "Stress scenarios" in md
