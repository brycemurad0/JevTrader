"""core_bar/core_quote/core_trade normalization (duck-typed alpaca-shaped fakes), StockStream /
CryptoStream subscription wiring against an injected fake client, and SimulatedStream replay."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pandas as pd
import pytest

from jevtrader.core.types import Bar, OrderBook, Quote, Trade
from jevtrader.live.streams import CryptoStream, SimulatedStream, StockStream, core_bar, core_quote, core_trade


def test_core_bar_mapping():
    raw = SimpleNamespace(symbol="AAPL", timestamp=pd.Timestamp("2026-01-01T00:00:00"), open=1, high=2, low=0.5,
                           close=1.5, volume=1000, vwap=1.2, trade_count=42)
    bar = core_bar(raw)
    assert bar.symbol == "AAPL" and bar.high == 2.0 and bar.trade_count == 42
    assert bar.ts.tzinfo is not None


def test_core_quote_mapping():
    raw = SimpleNamespace(symbol="AAPL", timestamp=pd.Timestamp("2026-01-01T00:00:00"),
                           bid_price=99.0, ask_price=100.0, bid_size=5, ask_size=7)
    quote = core_quote(raw)
    assert quote.bid == 99.0 and quote.ask == 100.0 and quote.ask_size == 7


def test_core_trade_mapping():
    raw = SimpleNamespace(symbol="BTC/USD", timestamp=pd.Timestamp("2026-01-01T00:00:00"), price=50000.0, size=0.01)
    trade = core_trade(raw)
    assert trade.price == 50000.0 and trade.size == 0.01


class _FakeStockClient:
    def __init__(self):
        self.subs = {}
        self.ran = False
        self.stopped = False

    def subscribe_bars(self, handler, *symbols):
        self.subs["bars"] = (handler, symbols)

    def subscribe_quotes(self, handler, *symbols):
        self.subs["quotes"] = (handler, symbols)

    def subscribe_trades(self, handler, *symbols):
        self.subs["trades"] = (handler, symbols)

    def run(self):
        self.ran = True

    def stop(self):
        self.stopped = True


class _FakeCryptoClient(_FakeStockClient):
    def subscribe_orderbooks(self, handler, *symbols):
        self.subs["orderbooks"] = (handler, symbols)


def test_stock_stream_subscribes_and_pushes_to_queue():
    queue: asyncio.Queue = asyncio.Queue()
    client = _FakeStockClient()
    stream = StockStream("k", "s", queue, ["AAPL", "MSFT"], client=client)
    stream.subscribe()
    assert client.subs["bars"][1] == ("AAPL", "MSFT")
    assert client.subs["quotes"][1] == ("AAPL", "MSFT")
    assert client.subs["trades"][1] == ("AAPL", "MSFT")

    bar_handler = client.subs["bars"][0]
    raw = SimpleNamespace(symbol="AAPL", timestamp=pd.Timestamp("2026-01-01"), open=1, high=1, low=1, close=1, volume=1)
    asyncio.run(bar_handler(raw))
    assert isinstance(queue.get_nowait(), Bar)


def test_stock_stream_run_calls_client_run_in_thread():
    queue: asyncio.Queue = asyncio.Queue()
    client = _FakeStockClient()
    stream = StockStream("k", "s", queue, ["AAPL"], client=client)
    asyncio.run(stream.run())
    assert client.ran
    stream.stop()
    assert client.stopped


def test_crypto_stream_subscribes_orderbooks_and_merges():
    queue: asyncio.Queue = asyncio.Queue()
    client = _FakeCryptoClient()
    stream = CryptoStream("k", "s", queue, ["BTC/USD"], client=client)
    stream.subscribe()
    assert "orderbooks" in client.subs

    ob_handler = client.subs["orderbooks"][0]
    raw = SimpleNamespace(
        symbol="BTC/USD", timestamp=pd.Timestamp("2026-01-01"),
        bids=[SimpleNamespace(price=100.0, size=1.0)], asks=[SimpleNamespace(price=101.0, size=1.0)], reset=True,
    )
    asyncio.run(ob_handler(raw))
    event = queue.get_nowait()
    assert isinstance(event, OrderBook)
    assert event.best_bid.price == 100.0


def test_simulated_stream_replays_in_order_with_no_delay():
    queue: asyncio.Queue = asyncio.Queue()
    events = [
        Trade(symbol="AAPL", ts=pd.Timestamp("2026-01-01T00:00:00Z"), price=100.0, size=1.0),
        Trade(symbol="AAPL", ts=pd.Timestamp("2026-01-01T00:00:05Z"), price=101.0, size=1.0),
    ]
    stream = SimulatedStream(events, queue, speed=0.0)
    asyncio.run(stream.run())
    assert queue.qsize() == 2
    assert queue.get_nowait().price == 100.0
    assert queue.get_nowait().price == 101.0
    assert stream.done.is_set()


def test_simulated_stream_from_dataframe_requires_symbol():
    df = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [10.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-01-01")]),
    )
    with pytest.raises(ValueError):
        SimulatedStream(df, asyncio.Queue())


def test_simulated_stream_from_dataframe_builds_bars():
    df = pd.DataFrame(
        {"open": [1.0, 2.0], "high": [1.5, 2.5], "low": [0.5, 1.5], "close": [1.2, 2.2], "volume": [10.0, 20.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")]),
    )
    queue: asyncio.Queue = asyncio.Queue()
    stream = SimulatedStream(df, queue, symbol="AAPL")
    asyncio.run(stream.run())
    bar = queue.get_nowait()
    assert isinstance(bar, Bar)
    assert bar.symbol == "AAPL"
    assert bar.close == 1.2


def test_simulated_stream_stop_marks_done():
    stream = SimulatedStream([], asyncio.Queue(), speed=0.0)
    stream.stop()
    assert stream.done.is_set()
