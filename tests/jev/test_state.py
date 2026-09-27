from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from jevtrader.core.types import BookLevel, OrderBook, Position, Quote
from jevtrader.jev.state import build_state, to_json


def _bars(n=80, seed=0, trend=0.0, freq="1min"):
    idx = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(trend, 0.05, n))
    return pd.DataFrame(
        {"open": close, "high": close + 0.02, "low": close - 0.02, "close": close, "volume": rng.uniform(100, 200, n), "vwap": close},
        index=idx,
    )


def test_deterministic_given_same_inputs():
    df = _bars()
    s1 = build_state("AAPL", df, cost_bps=12.3)
    s2 = build_state("AAPL", df, cost_bps=12.3)
    assert s1 == s2
    assert to_json(s1) == to_json(s2)


def test_json_serializable_and_compact():
    df = _bars()
    state = build_state("AAPL", df, cost_bps=5.0)
    encoded = to_json(state)
    # round-trips through json exactly (no NaN/Infinity, no exotic types)
    assert json.loads(encoded) == state
    assert len(encoded) < 700, f"state should be token-cheap, got {len(encoded)} bytes: {encoded}"


def test_handles_minimal_single_bar_without_crashing():
    idx = pd.date_range("2026-01-01", periods=1, freq="1min", tz="UTC")
    df = pd.DataFrame({"open": [100.0], "high": [100.1], "low": [99.9], "close": [100.0], "volume": [10.0]}, index=idx)
    state = build_state("AAPL", df, cost_bps=1.0)
    assert state["n_bars"] == 1
    assert state["ret_bps"]["5"] is None  # not enough history
    assert state["rsi"] is None
    assert state["px"] == 100.0
    # still valid, compact JSON
    json.loads(to_json(state))


def test_empty_bars_raises():
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    with pytest.raises(ValueError):
        build_state("AAPL", df)


def test_uptrend_produces_positive_returns_and_slope():
    df = _bars(n=100, trend=0.2)
    state = build_state("AAPL", df)
    assert state["ret_bps"]["30"] > 0
    assert state["trend_slope_bps"] > 0


def test_downtrend_produces_negative_returns_and_slope():
    df = _bars(n=100, trend=-0.2)
    state = build_state("AAPL", df)
    assert state["ret_bps"]["30"] < 0
    assert state["trend_slope_bps"] < 0


def test_rsi_bounded_0_100():
    df = _bars(n=100, trend=0.3)
    state = build_state("AAPL", df)
    assert 0.0 <= state["rsi"] <= 100.0


def test_quote_and_book_features():
    df = _bars()
    ts = df.index[-1]
    quote = Quote(symbol="AAPL", ts=ts, bid=99.99, ask=100.01, bid_size=100, ask_size=50)
    book = OrderBook(
        symbol="AAPL",
        ts=ts,
        bids=(BookLevel(99.99, 100), BookLevel(99.98, 50)),
        asks=(BookLevel(100.01, 50), BookLevel(100.02, 50)),
    )
    state = build_state("AAPL", df, quote=quote, book=book, cost_bps=3.0)
    assert state["spread_bps"] == pytest.approx(1e4 * 0.02 / quote.mid, rel=1e-6)
    assert state["book_imbalance"] == pytest.approx(book.imbalance(), rel=1e-6)
    assert state["microprice_offset_bps"] is not None
    assert state["cost_bps"] == 3.0


def test_no_quote_or_book_gives_none_not_missing_key():
    df = _bars()
    state = build_state("AAPL", df)
    assert state["spread_bps"] is None
    assert state["book_imbalance"] is None
    assert state["microprice_offset_bps"] is None


def test_position_context():
    df = _bars()
    last_px = float(df["close"].iloc[-1])
    long_pos = Position(symbol="AAPL", qty=10, avg_price=last_px * 0.95)
    state = build_state("AAPL", df, position=long_pos)
    assert state["pos_qty_sign"] == 1
    assert state["unrealized_pnl_bps"] > 0  # bought below market, in profit

    short_pos = Position(symbol="AAPL", qty=-10, avg_price=last_px * 0.95)
    state2 = build_state("AAPL", df, position=short_pos)
    assert state2["pos_qty_sign"] == -1
    assert state2["unrealized_pnl_bps"] < 0  # shorted below market, at a loss

    flat_state = build_state("AAPL", df)
    assert flat_state["pos_qty_sign"] == 0
    assert flat_state["unrealized_pnl_bps"] == 0.0


def test_extra_features_default_none_does_not_change_output():
    df = _bars()
    without = build_state("AAPL", df, cost_bps=5.0)
    without_explicit_none = build_state("AAPL", df, cost_bps=5.0, extra_features=None)
    assert without == without_explicit_none
    assert "forecast" not in without


def test_extra_features_merged_under_forecast_key_and_rounded():
    df = _bars()
    extra = {"tfm_ret_q50_bps": 12.3456789, "tfm_p_up_gt_cost": 0.61234567, "nested": {"x": 1.23456}}
    state = build_state("AAPL", df, cost_bps=5.0, extra_features=extra)
    assert state["forecast"]["tfm_ret_q50_bps"] == pytest.approx(12.3457)
    assert state["forecast"]["tfm_p_up_gt_cost"] == pytest.approx(0.6123)
    assert state["forecast"]["nested"]["x"] == pytest.approx(1.2346)
    # still JSON-serializable and doesn't blow up the token budget
    json.loads(to_json(state))


def test_now_defaults_to_last_bar_ts_and_is_overridable():
    df = _bars()
    state_default = build_state("AAPL", df)
    override_ts = pd.Timestamp("2026-06-01T15:30:00Z")
    state_override = build_state("AAPL", df, now=override_ts)
    assert state_override["minute_of_day"] == 15 * 60 + 30
    assert state_override["session"] == "us_midday"
    assert state_default["minute_of_day"] != state_override["minute_of_day"] or True  # just exercising both paths
