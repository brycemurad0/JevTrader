"""Focused tests for the quote/trade-mode maker queue-position model in SimBroker."""

import pandas as pd

from jevtrader.core.fees import ZeroFees
from jevtrader.core.types import Liquidity, Order, OrderStatus, OrderType, Quote, Side, TimeInForce, Trade
from jevtrader.backtest.sim_broker import SimBroker


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def test_queue_depletes_gradually_across_multiple_trades_before_filling():
    broker = SimBroker(fee_model=ZeroFees(), latency_ms=0)
    broker.on_market_event(Quote("AAPL", _ts("2024-01-02 14:30:00"), bid=100.0, ask=100.2, bid_size=30, ask_size=30))
    order = Order(symbol="AAPL", side=Side.BUY, qty=10, type=OrderType.LIMIT, limit_price=100.0, tif=TimeInForce.GTC)
    broker.submit(order)

    broker.on_market_event(Quote("AAPL", _ts("2024-01-02 14:30:01"), bid=100.0, ask=100.2, bid_size=30, ask_size=30))  # queue init = 30

    for i, size in enumerate([10, 10, 5]):  # 25 of 30 consumed; still nothing for us
        fills = broker.on_market_event(Trade("AAPL", _ts(f"2024-01-02 14:30:0{2+i}"), price=100.0, size=size))
        assert fills == []

    fills = broker.on_market_event(Trade("AAPL", _ts("2024-01-02 14:30:05"), price=100.0, size=8))  # overflow = 8-5 = 3
    assert len(fills) == 1
    assert abs(fills[0].qty - 3) < 1e-9
    assert order.status is OrderStatus.PARTIALLY_FILLED

    fills2 = broker.on_market_event(Trade("AAPL", _ts("2024-01-02 14:30:06"), price=100.0, size=20))  # queue already exhausted
    assert len(fills2) == 1
    assert abs(fills2[0].qty - 7) < 1e-9  # remaining 10 - 3
    assert order.status is OrderStatus.FILLED


def test_trade_strictly_through_limit_fills_immediately_regardless_of_queue():
    broker = SimBroker(fee_model=ZeroFees(), latency_ms=0)
    broker.on_market_event(Quote("AAPL", _ts("2024-01-02 14:30:00"), bid=100.0, ask=100.2, bid_size=500, ask_size=500))
    order = Order(symbol="AAPL", side=Side.BUY, qty=10, type=OrderType.LIMIT, limit_price=100.0, tif=TimeInForce.GTC)
    broker.submit(order)
    broker.on_market_event(Quote("AAPL", _ts("2024-01-02 14:30:01"), bid=100.0, ask=100.2, bid_size=500, ask_size=500))  # huge queue ahead

    # a print strictly BELOW our buy limit means the market traded through everyone ahead of us.
    fills = broker.on_market_event(Trade("AAPL", _ts("2024-01-02 14:30:02"), price=99.9, size=1))
    assert len(fills) == 1
    assert fills[0].liquidity is Liquidity.MAKER
    assert abs(fills[0].price - 100.0) < 1e-9  # fills at OUR limit price, not the trade price
    assert order.status is OrderStatus.FILLED


def test_sell_side_queue_tracks_ask_size_and_depletes_on_ask_prints():
    broker = SimBroker(fee_model=ZeroFees(), latency_ms=0)
    broker.on_market_event(Quote("AAPL", _ts("2024-01-02 14:30:00"), bid=99.8, ask=100.0, bid_size=40, ask_size=40))
    order = Order(symbol="AAPL", side=Side.SELL, qty=5, type=OrderType.LIMIT, limit_price=100.0, tif=TimeInForce.GTC)
    broker.submit(order)
    broker.on_market_event(Quote("AAPL", _ts("2024-01-02 14:30:01"), bid=99.8, ask=100.0, bid_size=40, ask_size=40))  # queue = 40

    fills = broker.on_market_event(Trade("AAPL", _ts("2024-01-02 14:30:02"), price=100.0, size=38))
    assert fills == []
    fills2 = broker.on_market_event(Trade("AAPL", _ts("2024-01-02 14:30:03"), price=100.0, size=10))  # overflow = 10-2=8
    assert len(fills2) == 1
    assert abs(fills2[0].qty - 5) < 1e-9  # capped at order qty
    assert fills2[0].side is Side.SELL


def test_marketable_limit_takes_liquidity_instead_of_queueing():
    broker = SimBroker(fee_model=ZeroFees(), latency_ms=0)
    broker.on_market_event(Quote("AAPL", _ts("2024-01-02 14:30:00"), bid=100.0, ask=100.2, bid_size=10, ask_size=10))
    order = Order(symbol="AAPL", side=Side.BUY, qty=5, type=OrderType.LIMIT, limit_price=100.5, tif=TimeInForce.GTC)  # crosses the ask
    broker.submit(order)
    fills = broker.on_market_event(Quote("AAPL", _ts("2024-01-02 14:30:01"), bid=100.0, ask=100.2, bid_size=10, ask_size=10))
    assert len(fills) == 1
    assert fills[0].liquidity is Liquidity.TAKER
    assert abs(fills[0].price - 100.2) < 1e-9
