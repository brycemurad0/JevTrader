"""Direct SimBroker unit tests for maker vs. taker fee application (no engine involved)."""

import pandas as pd

from jevtrader.core.fees import BpsFees
from jevtrader.core.types import Liquidity, Order, OrderType, Quote, Side, TimeInForce, Trade
from jevtrader.backtest.sim_broker import SimBroker


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def test_taker_fee_applied_on_marketable_quote_fill():
    broker = SimBroker(fee_model=BpsFees(maker_bps=5.0, taker_bps=10.0), latency_ms=0)
    q0 = Quote("AAPL", _ts("2024-01-02 14:30:00"), bid=99.9, ask=100.1, bid_size=100, ask_size=100)
    broker.on_market_event(q0)

    order = Order(symbol="AAPL", side=Side.BUY, qty=10, type=OrderType.MARKET)
    broker.submit(order)

    q1 = Quote("AAPL", _ts("2024-01-02 14:30:01"), bid=99.9, ask=100.1, bid_size=100, ask_size=100)
    fills = broker.on_market_event(q1)

    assert len(fills) == 1
    f = fills[0]
    assert f.liquidity is Liquidity.TAKER
    expected_fee = 10 * 100.1 * 10.0 / 1e4
    assert abs(f.fee - expected_fee) < 1e-9
    assert abs(f.price - 100.1) < 1e-9  # buy takes the ask


def test_maker_fee_applied_on_queue_depleted_limit_fill():
    broker = SimBroker(fee_model=BpsFees(maker_bps=5.0, taker_bps=10.0), latency_ms=0)
    q0 = Quote("AAPL", _ts("2024-01-02 14:30:00"), bid=100.0, ask=100.2, bid_size=50, ask_size=50)
    broker.on_market_event(q0)

    order = Order(symbol="AAPL", side=Side.BUY, qty=10, type=OrderType.LIMIT, limit_price=100.0, tif=TimeInForce.GTC)
    broker.submit(order)

    q1 = Quote("AAPL", _ts("2024-01-02 14:30:01"), bid=100.0, ask=100.2, bid_size=50, ask_size=50)
    assert broker.on_market_event(q1) == []  # not marketable; just initializes the queue at size 50

    t1 = Trade("AAPL", _ts("2024-01-02 14:30:02"), price=100.0, size=45)
    assert broker.on_market_event(t1) == []  # depletes 45 of the 50 ahead of us; still nothing for us

    t2 = Trade("AAPL", _ts("2024-01-02 14:30:03"), price=100.0, size=20)  # overflow = 20 - 5 = 15
    fills = broker.on_market_event(t2)

    assert len(fills) == 1
    f = fills[0]
    assert f.liquidity is Liquidity.MAKER
    assert abs(f.qty - 10) < 1e-9  # capped at the order's remaining qty even though overflow was 15
    expected_fee = 10 * 100.0 * 5.0 / 1e4
    assert abs(f.fee - expected_fee) < 1e-9


def test_bar_mode_market_fee_is_taker_and_limit_fee_is_maker():
    from jevtrader.core.types import Bar

    broker = SimBroker(fee_model=BpsFees(maker_bps=5.0, taker_bps=10.0), latency_ms=0)
    bar0 = Bar("AAPL", _ts("2024-01-02 14:30:00"), open=100.0, high=100.5, low=99.5, close=100.2, volume=10_000)
    broker.on_market_event(bar0)

    market_order = Order(symbol="AAPL", side=Side.BUY, qty=10, type=OrderType.MARKET)
    broker.submit(market_order)
    limit_order = Order(symbol="AAPL", side=Side.BUY, qty=5, type=OrderType.LIMIT, limit_price=98.0)
    broker.submit(limit_order)

    bar1 = Bar("AAPL", _ts("2024-01-02 14:31:00"), open=100.3, high=100.6, low=97.5, close=100.0, volume=10_000)
    fills = broker.on_market_event(bar1)  # low=97.5 trades strictly through the 98.0 buy limit

    by_type = {f.client_order_id: f for f in fills}
    assert by_type[market_order.client_order_id].liquidity is Liquidity.TAKER
    assert by_type[limit_order.client_order_id].liquidity is Liquidity.MAKER
    assert abs(by_type[limit_order.client_order_id].price - 98.0) < 1e-9


def test_post_only_order_that_would_cross_is_rejected():
    from jevtrader.core.types import OrderStatus

    broker = SimBroker(fee_model=BpsFees(), latency_ms=0)
    q0 = Quote("AAPL", _ts("2024-01-02 14:30:00"), bid=100.0, ask=100.2, bid_size=50, ask_size=50)
    broker.on_market_event(q0)

    crossing_buy = Order(symbol="AAPL", side=Side.BUY, qty=5, type=OrderType.LIMIT, limit_price=100.5, post_only=True)
    result = broker.submit(crossing_buy)
    assert result.status is OrderStatus.REJECTED
    assert result.reject_reason == "post_only_would_cross"
