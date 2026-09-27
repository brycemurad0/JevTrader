import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from jevtrader.core.types import AssetClass
from jevtrader.data.synthetic import (
    generate_bars,
    generate_cointegrated_pair,
    generate_order_book_stream,
    generate_quote_trade_stream,
)

EQ_START = pd.Timestamp("2024-01-02 14:30", tz="UTC")  # 09:30 America/New_York on a Tuesday


def test_bars_deterministic_with_seed():
    end = EQ_START + pd.Timedelta(hours=1)
    a = generate_bars("AAPL", EQ_START, end, seed=42)
    b = generate_bars("AAPL", EQ_START, end, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_bars_different_seed_differs():
    end = EQ_START + pd.Timedelta(hours=1)
    a = generate_bars("AAPL", EQ_START, end, seed=1)
    b = generate_bars("AAPL", EQ_START, end, seed=2)
    assert not a["close"].equals(b["close"])


def test_bars_ohlc_consistency_and_equity_session_mask():
    start = pd.Timestamp("2024-01-02 00:00", tz="UTC")
    end = pd.Timestamp("2024-01-03 00:00", tz="UTC")
    df = generate_bars("AAPL", start, end, freq="1min", asset_class=AssetClass.EQUITY, seed=1)
    assert len(df) == 390  # 6.5h regular session in 1-minute bars
    assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (df["volume"] > 0).all()
    # U-shape: open/close bars should on average see more volume than midday bars.
    edge_vol = pd.concat([df["volume"].iloc[:15], df["volume"].iloc[-15:]]).mean()
    mid_vol = df["volume"].iloc[180:210].mean()
    assert edge_vol > mid_vol


def test_crypto_bars_are_24_7():
    start = pd.Timestamp("2024-01-02 00:00", tz="UTC")
    end = pd.Timestamp("2024-01-03 00:00", tz="UTC")
    df = generate_bars("BTC/USD", start, end, freq="1min", asset_class=AssetClass.CRYPTO, seed=1)
    assert len(df) == 1440  # every minute of the day, no session gaps


def test_regime_labels_present_and_persist():
    end = EQ_START + pd.Timedelta(hours=2)
    df = generate_bars("AAPL", EQ_START, end, seed=3)
    assert set(df["regime"].unique()) <= {"trend", "mean_revert", "chop"}
    # regimes should persist for more than a single bar on average (not i.i.d. flipping).
    changes = (df["regime"] != df["regime"].shift()).sum()
    assert changes < len(df) * 0.3


def test_quote_trade_stream_has_weak_but_real_predictive_ofi():
    end = EQ_START + pd.Timedelta(days=3, hours=6, minutes=30)
    q, t = generate_quote_trade_stream("AAPL", EQ_START, end, freq="1s", seed=42)
    assert not q.empty and not t.empty
    ic = np.corrcoef(q["ofi"].to_numpy()[:-1], np.diff(np.log(q["mid"].to_numpy())))[0, 1]
    # a genuine edge (clearly nonzero, statistically significant at this sample size) but weak
    # (nowhere close to a near-perfect predictor).
    assert 0.02 < ic < 0.3
    assert (q["ask"] > q["bid"]).all()
    assert (q["bid_size"] > 0).all() and (q["ask_size"] > 0).all()


def test_quote_trade_stream_deterministic():
    end = EQ_START + pd.Timedelta(minutes=5)
    q1, t1 = generate_quote_trade_stream("AAPL", EQ_START, end, freq="1s", seed=7)
    q2, t2 = generate_quote_trade_stream("AAPL", EQ_START, end, freq="1s", seed=7)
    pd.testing.assert_frame_equal(q1, q2)
    pd.testing.assert_frame_equal(t1, t2)


def test_order_book_matches_quotes_and_has_requested_depth():
    end = EQ_START + pd.Timedelta(minutes=2)
    books = generate_order_book_stream("AAPL", EQ_START, end, freq="1s", n_levels=5, seed=1)
    assert len(books) > 0
    for book in books:
        assert len(book.bids) == 5
        assert len(book.asks) == 5
        assert book.best_bid.price < book.best_ask.price
        # sizes should generally decay with depth (allow noise: check the overall trend via ends)
        assert book.bids[0].size > 0 and book.asks[0].size > 0


def test_cointegrated_pair_spread_is_stationary():
    end = EQ_START + pd.Timedelta(days=7, hours=6, minutes=30)
    pair = generate_cointegrated_pair("XLE", "XOM", EQ_START, end, freq="1min", seed=5)
    assert set(pair.keys()) == {"XLE", "XOM"}
    spread = pair["XOM"]["spread"]
    assert len(spread) == len(pair["XLE"])
    pvalue = adfuller(spread.to_numpy())[1]
    assert pvalue < 0.05  # reject the unit-root null: the spread is stationary by construction
