"""AlpacaBroker.submit/cancel/cancel_all/reconcile against a fake TradingClient, and the
post-only emulation (Alpaca has no native post-only order type)."""

from __future__ import annotations

from jevtrader.core.broker import TradingMode
from jevtrader.core.types import Order, OrderStatus, OrderType, Quote, Side
from jevtrader.execution.alpaca_broker import AlpacaBroker
from tests.execution.fakes import ApiError, FakeTradingClient, alpaca_account, alpaca_position, make_settings


def _broker(tmp_path, **kwargs) -> tuple[AlpacaBroker, FakeTradingClient]:
    client = FakeTradingClient()
    settings = make_settings(tmp_path)
    broker = AlpacaBroker(TradingMode.PAPER, client=client, settings=settings, **kwargs)
    return broker, client


def test_submit_market_order_reaches_client(tmp_path):
    broker, client = _broker(tmp_path)
    order = Order(symbol="AAPL", side=Side.BUY, qty=10, type=OrderType.MARKET)
    result = broker.submit(order)
    assert len(client.submitted) == 1
    assert result.broker_order_id
    assert result.status is OrderStatus.NEW
    assert result.client_order_id in [o.client_order_id for o in broker.open_orders()]


def test_submit_rejected_by_api_error(tmp_path):
    broker, client = _broker(tmp_path)
    client.fail_next_submit = ApiError('{"code": 40310000, "message": "insufficient buying power"}')
    order = Order(symbol="AAPL", side=Side.BUY, qty=1_000_000, type=OrderType.MARKET)
    result = broker.submit(order)
    assert result.status is OrderStatus.REJECTED
    assert result.reject_reason
    assert result.client_order_id not in [o.client_order_id for o in broker.open_orders()]


def test_post_only_rejects_when_crossing(tmp_path):
    quote = Quote(symbol="AAPL", ts=None, bid=99.0, ask=100.0, bid_size=10, ask_size=10)
    broker, client = _broker(tmp_path, quote_provider=lambda s: quote)
    order = Order(symbol="AAPL", side=Side.BUY, qty=1, type=OrderType.LIMIT, limit_price=100.5, post_only=True)
    result = broker.submit(order)
    assert result.status is OrderStatus.REJECTED
    assert "post_only" in result.reject_reason
    assert client.submitted == []  # never actually sent to Alpaca


def test_post_only_passes_when_resting(tmp_path):
    quote = Quote(symbol="AAPL", ts=None, bid=99.0, ask=100.0, bid_size=10, ask_size=10)
    broker, client = _broker(tmp_path, quote_provider=lambda s: quote)
    order = Order(symbol="AAPL", side=Side.BUY, qty=1, type=OrderType.LIMIT, limit_price=98.5, post_only=True)
    result = broker.submit(order)
    assert result.status is OrderStatus.NEW
    assert len(client.submitted) == 1


def test_post_only_rejected_for_non_limit_order(tmp_path):
    broker, client = _broker(tmp_path)
    order = Order(symbol="AAPL", side=Side.BUY, qty=1, type=OrderType.MARKET, post_only=True)
    result = broker.submit(order)
    assert result.status is OrderStatus.REJECTED
    assert client.submitted == []


def test_liquidity_hint_maker_for_resting_limit(tmp_path):
    quote = Quote(symbol="AAPL", ts=None, bid=99.0, ask=100.0, bid_size=10, ask_size=10)
    broker, client = _broker(tmp_path, quote_provider=lambda s: quote)
    order = Order(symbol="AAPL", side=Side.BUY, qty=1, type=OrderType.LIMIT, limit_price=98.5)
    broker.submit(order)
    from jevtrader.core.types import Liquidity

    assert broker._liquidity_hint[order.client_order_id] is Liquidity.MAKER


def test_liquidity_hint_taker_for_crossing_limit_without_post_only(tmp_path):
    quote = Quote(symbol="AAPL", ts=None, bid=99.0, ask=100.0, bid_size=10, ask_size=10)
    broker, client = _broker(tmp_path, quote_provider=lambda s: quote)
    order = Order(symbol="AAPL", side=Side.BUY, qty=1, type=OrderType.LIMIT, limit_price=100.5)
    result = broker.submit(order)
    assert result.status is OrderStatus.NEW  # not post-only, so it's fine to cross
    from jevtrader.core.types import Liquidity

    assert broker._liquidity_hint[order.client_order_id] is Liquidity.TAKER


def test_liquidity_hint_taker_for_market_order(tmp_path):
    broker, client = _broker(tmp_path)
    order = Order(symbol="AAPL", side=Side.BUY, qty=1, type=OrderType.MARKET)
    broker.submit(order)
    from jevtrader.core.types import Liquidity

    assert broker._liquidity_hint[order.client_order_id] is Liquidity.TAKER


def test_cancel_and_cancel_all(tmp_path):
    broker, client = _broker(tmp_path)
    o1 = broker.submit(Order(symbol="AAPL", side=Side.BUY, qty=1, type=OrderType.MARKET))
    o2 = broker.submit(Order(symbol="MSFT", side=Side.BUY, qty=1, type=OrderType.MARKET))
    broker.cancel(o1.client_order_id)
    assert client.canceled_ids == [o1.broker_order_id]

    broker.cancel_all()
    assert client.canceled_all_count == 1
    assert broker.open_orders() == []


def test_reconcile_syncs_positions_and_account(tmp_path):
    broker, client = _broker(tmp_path)
    client.set_positions([alpaca_position(symbol="AAPL", qty="7", side="long", avg_entry_price="120.0")])
    client.set_account(alpaca_account(cash="9000", equity="10000", buying_power="20000"))

    broker.reconcile()

    positions = broker.positions()
    assert positions["AAPL"].qty == 7.0
    assert positions["AAPL"].avg_price == 120.0
    account = broker.account()
    assert account.cash == 9000.0
    assert account.equity == 10000.0


def test_reconcile_syncs_open_orders(tmp_path):
    broker, client = _broker(tmp_path)
    broker.submit(Order(symbol="AAPL", side=Side.BUY, qty=1, type=OrderType.MARKET))
    # simulate the broker having a resting order this process doesn't know about yet
    broker._open_orders.clear()
    broker.reconcile()
    assert len(broker.open_orders()) == 1
    assert broker.open_orders()[0].symbol == "AAPL"
