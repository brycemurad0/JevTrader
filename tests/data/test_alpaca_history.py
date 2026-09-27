"""These tests never touch the network: `normalize_*_df` are pure functions of an alpaca-py
`.df`-shaped DataFrame, so we build that shape by hand (exactly as `BarSet(...).df` etc. would)."""

import pandas as pd
import pytest

from jevtrader.data.alpaca_history import (
    _timeframe,
    normalize_bars_df,
    normalize_quotes_df,
    normalize_trades_df,
)


def _multiindex_bars() -> pd.DataFrame:
    idx = pd.MultiIndex.from_tuples(
        [
            ("AAPL", pd.Timestamp("2024-01-02T14:30:00Z")),
            ("AAPL", pd.Timestamp("2024-01-02T14:31:00Z")),
            ("MSFT", pd.Timestamp("2024-01-02T14:30:00Z")),
        ],
        names=["symbol", "timestamp"],
    )
    return pd.DataFrame(
        {
            "open": [100.0, 100.2, 300.0],
            "high": [100.5, 100.6, 301.0],
            "low": [99.8, 100.0, 299.5],
            "close": [100.2, 100.4, 300.5],
            "volume": [1000, 2000, 500],
            "trade_count": [10, 20, 5],
            "vwap": [100.1, 100.3, 300.2],
        },
        index=idx,
    )


def test_normalize_bars_filters_to_one_symbol_and_localizes_utc():
    df = normalize_bars_df(_multiindex_bars(), "AAPL")
    assert len(df) == 2
    assert str(df.index.tz) == "UTC"
    assert df.index.name == "ts"
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
    assert df.iloc[0]["open"] == 100.0


def test_normalize_missing_symbol_returns_empty_with_expected_columns():
    df = normalize_bars_df(_multiindex_bars(), "GOOG")
    assert df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "trade_count", "vwap"]


def test_normalize_none_or_empty_input():
    assert normalize_bars_df(None, "AAPL").empty
    assert normalize_bars_df(pd.DataFrame(), "AAPL").empty


def test_normalize_sorts_and_dedupes_keeping_last():
    idx = pd.MultiIndex.from_tuples(
        [
            ("AAPL", pd.Timestamp("2024-01-02T14:31:00Z")),
            ("AAPL", pd.Timestamp("2024-01-02T14:30:00Z")),
            ("AAPL", pd.Timestamp("2024-01-02T14:30:00Z")),  # duplicate ts; this later row should win
        ],
        names=["symbol", "timestamp"],
    )
    raw = pd.DataFrame(
        {
            "open": [2.0, 1.0, 1.5],
            "high": [2.0, 1.0, 1.5],
            "low": [2.0, 1.0, 1.5],
            "close": [2.0, 1.0, 1.5],
            "volume": [1, 1, 1],
            "trade_count": [1, 1, 1],
            "vwap": [2.0, 1.0, 1.5],
        },
        index=idx,
    )
    df = normalize_bars_df(raw, "AAPL")
    assert list(df.index) == sorted(df.index)
    assert df.iloc[0]["open"] == 1.5  # the later duplicate wins over the earlier one


def test_normalize_quotes_renames_bid_ask_price_columns():
    idx = pd.MultiIndex.from_tuples([("AAPL", pd.Timestamp("2024-01-02T14:30:00Z"))], names=["symbol", "timestamp"])
    raw = pd.DataFrame(
        {"bid_price": [100.0], "bid_size": [3.0], "ask_price": [100.05], "ask_size": [5.0], "bid_exchange": ["P"], "ask_exchange": ["P"]},
        index=idx,
    )
    df = normalize_quotes_df(raw, "AAPL")
    assert list(df.columns) == ["bid", "ask", "bid_size", "ask_size"]
    assert df.iloc[0]["bid"] == 100.0 and df.iloc[0]["ask"] == 100.05


def test_normalize_trades_keeps_price_and_size():
    idx = pd.MultiIndex.from_tuples([("AAPL", pd.Timestamp("2024-01-02T14:30:00Z"))], names=["symbol", "timestamp"])
    raw = pd.DataFrame({"price": [100.02], "size": [100.0], "exchange": ["P"], "id": [1]}, index=idx)
    df = normalize_trades_df(raw, "AAPL")
    assert list(df.columns) == ["price", "size"]
    assert df.iloc[0]["price"] == 100.02


@pytest.mark.parametrize(
    "spec,amount,unit_name",
    [("5Min", 5, "Minute"), ("1Min", 1, "Minute"), ("1Hour", 1, "Hour"), ("1Day", 1, "Day"), ("1Week", 1, "Week")],
)
def test_timeframe_parsing(spec, amount, unit_name):
    from alpaca.data.timeframe import TimeFrameUnit

    tf = _timeframe(spec)
    assert tf.amount == amount
    assert tf.unit == getattr(TimeFrameUnit, unit_name)


def test_timeframe_passthrough_for_non_string():
    from alpaca.data.timeframe import TimeFrame

    tf = TimeFrame.Day
    assert _timeframe(tf) is tf


def test_timeframe_invalid_raises():
    with pytest.raises(ValueError):
        _timeframe("banana")
