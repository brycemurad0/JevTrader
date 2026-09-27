import numpy as np
import pandas as pd

from jevtrader.backtest.metrics import (
    compute_metrics,
    compute_trade_pnls,
    deflated_sharpe_ratio,
    max_drawdown,
    max_drawdown_duration,
    periods_per_year,
    probabilistic_sharpe_ratio,
)
from jevtrader.core.types import AssetClass


def _daily_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")


def test_max_drawdown_hand_computed():
    equity = pd.Series([100, 110, 105, 90, 95, 120], index=_daily_index(6))
    # running max: 100,110,110,110,110,120 -> dd: 0,0,-5/110,-20/110,-15/110,0
    assert abs(max_drawdown(equity) - (-20 / 110)) < 1e-9


def test_max_drawdown_duration_hand_computed():
    equity = pd.Series([100, 110, 105, 90, 95, 120], index=_daily_index(6))
    # below running peak at indices 2,3,4 (three in a row) -> longest run = 3
    assert max_drawdown_duration(equity) == 3


def test_total_return_and_sharpe_hand_computed():
    idx = _daily_index(6)
    equity = pd.Series([100.0, 101.0, 102.0, 101.0, 103.0, 104.0], index=idx)
    m = compute_metrics(equity, None, asset_class=AssetClass.EQUITY)

    returns = equity.pct_change().dropna()
    expected_sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
    assert abs(m["sharpe"] - expected_sharpe) < 1e-9
    assert abs(m["total_return"] - (104.0 / 100.0 - 1.0)) < 1e-9
    assert abs(m["annualized_vol"] - float(returns.std(ddof=1) * np.sqrt(252))) < 1e-9


def test_periods_per_year_daily_vs_intraday_vs_crypto():
    assert abs(periods_per_year(_daily_index(30), AssetClass.EQUITY) - 252) < 1e-6
    assert abs(periods_per_year(_daily_index(30), AssetClass.CRYPTO) - 365) < 1e-6

    idx_min = pd.date_range("2024-01-02 14:30", periods=200, freq="1min", tz="UTC")
    expected = 252 * 6.5 * 3600 / 60
    assert abs(periods_per_year(idx_min, AssetClass.EQUITY) - expected) < 1e-6

    idx_min_crypto = pd.date_range("2024-01-02 00:00", periods=200, freq="1min", tz="UTC")
    expected_crypto = 365 * 24 * 3600 / 60
    assert abs(periods_per_year(idx_min_crypto, AssetClass.CRYPTO) - expected_crypto) < 1e-6


def test_compute_trade_pnls_hand_computed_round_trip():
    fills = pd.DataFrame(
        [
            {"ts": pd.Timestamp("2024-01-01", tz="UTC"), "symbol": "AAPL", "side": "buy", "qty": 10.0, "price": 100.0, "fee": 1.0},
            {"ts": pd.Timestamp("2024-01-02", tz="UTC"), "symbol": "AAPL", "side": "sell", "qty": 10.0, "price": 105.0, "fee": 1.0},
        ]
    )
    pnls = compute_trade_pnls(fills)
    # gross realized pnl on the closing (sell) fill = qty * (sell_price - avg_buy_price) = 10*(105-100) = 50
    assert len(pnls) == 1
    assert abs(pnls.iloc[0] - 50.0) < 1e-6


def test_hit_rate_and_profit_factor_hand_computed():
    fills = pd.DataFrame(
        [
            {"ts": pd.Timestamp("2024-01-01", tz="UTC"), "symbol": "AAPL", "side": "buy", "qty": 10.0, "price": 100.0, "fee": 0.0, "notional": 1000.0},
            {"ts": pd.Timestamp("2024-01-02", tz="UTC"), "symbol": "AAPL", "side": "sell", "qty": 10.0, "price": 110.0, "fee": 0.0, "notional": 1100.0},  # +100 win
            {"ts": pd.Timestamp("2024-01-03", tz="UTC"), "symbol": "AAPL", "side": "buy", "qty": 10.0, "price": 110.0, "fee": 0.0, "notional": 1100.0},
            {"ts": pd.Timestamp("2024-01-04", tz="UTC"), "symbol": "AAPL", "side": "sell", "qty": 10.0, "price": 105.0, "fee": 0.0, "notional": 1050.0},  # -50 loss
        ]
    )
    equity = pd.Series([100_000, 100_100, 100_100, 100_050, 100_050], index=pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"))
    m = compute_metrics(equity, fills, asset_class=AssetClass.EQUITY)
    assert m["n_closed_trades"] == 2
    assert abs(m["hit_rate"] - 0.5) < 1e-9
    assert abs(m["profit_factor"] - (100.0 / 50.0)) < 1e-9
    assert abs(m["avg_trade"] - 25.0) < 1e-9  # (100 - 50) / 2


def test_fee_drag_and_cost_per_trade_bps_hand_computed():
    fills = pd.DataFrame(
        [
            {"ts": pd.Timestamp("2024-01-01", tz="UTC"), "symbol": "AAPL", "side": "buy", "qty": 10.0, "price": 100.0, "fee": 2.0, "notional": 1000.0},
        ]
    )
    equity = pd.Series([100_000, 99_998], index=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"))
    m = compute_metrics(equity, fills, asset_class=AssetClass.EQUITY)
    assert abs(m["fee_total"] - 2.0) < 1e-9
    assert abs(m["cost_per_trade_bps"] - (2.0 / 1000.0 * 1e4)) < 1e-9  # 20 bps


def test_probabilistic_sharpe_ratio_is_calibrated_under_the_null():
    # PSR(0) is a p-value-like quantity: under a true Sharpe of exactly 0, it should be
    # (approximately) uniformly distributed in [0, 1] across independent draws -- so on average
    # about half of many independent zero-skill trials should score above 0.5, not every single one.
    rng = np.random.default_rng(0)
    psrs = [probabilistic_sharpe_ratio(pd.Series(rng.normal(0.0, 0.01, 500)), benchmark_sr=0.0) for _ in range(300)]
    above_half = np.mean([p > 0.5 for p in psrs])
    assert 0.35 < above_half < 0.65
    assert 0.2 < float(np.mean(psrs)) < 0.8


def test_deflated_sharpe_ratio_more_trials_never_increases_dsr():
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.0006, 0.01, 500))
    dsr_1 = deflated_sharpe_ratio(returns, n_trials=1)
    dsr_100 = deflated_sharpe_ratio(returns, n_trials=100)
    assert dsr_100["expected_max_sr"] >= dsr_1["expected_max_sr"]
    assert dsr_100["dsr"] <= dsr_1["dsr"] + 1e-9


def test_deflated_sharpe_ratio_n_trials_one_matches_plain_psr():
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0.0004, 0.012, 300))
    dsr_info = deflated_sharpe_ratio(returns, n_trials=1)
    psr = probabilistic_sharpe_ratio(returns, benchmark_sr=0.0)
    assert abs(dsr_info["dsr"] - psr) < 1e-9


def test_compute_metrics_on_empty_or_tiny_equity_is_safe():
    empty = pd.Series(dtype=float)
    m = compute_metrics(empty)
    assert np.isnan(m["sharpe"])
    single = pd.Series([100.0], index=_daily_index(1))
    m2 = compute_metrics(single)
    assert np.isnan(m2["sharpe"])
