"""alpaca-py (duck-typed fake) model -> core type mapping."""

from __future__ import annotations

from jevtrader.core.types import OrderStatus, OrderType, Side, TimeInForce
from jevtrader.execution.alpaca_broker import (
    map_alpaca_account_to_core,
    map_alpaca_order_to_core,
    map_alpaca_position_to_core,
)
from tests.execution.fakes import alpaca_account, alpaca_order, alpaca_position


def test_map_order_basic_fields():
    resp = alpaca_order(
        symbol="AAPL", side="sell", type="limit", time_in_force="gtc",
        qty="10", filled_qty="3", filled_avg_price="101.5", limit_price="101.0", status="partially_filled",
    )
    order = map_alpaca_order_to_core(resp)
    assert order.symbol == "AAPL"
    assert order.side is Side.SELL
    assert order.type is OrderType.LIMIT
    assert order.tif is TimeInForce.GTC
    assert order.qty == 10.0
    assert order.filled_qty == 3.0
    assert order.avg_fill_price == 101.5
    assert order.limit_price == 101.0
    assert order.status is OrderStatus.PARTIALLY_FILLED
    assert order.broker_order_id == resp.id
    assert order.client_order_id == resp.client_order_id


def test_map_order_status_variants():
    cases = {
        "new": OrderStatus.NEW, "accepted": OrderStatus.NEW, "pending_new": OrderStatus.NEW,
        "filled": OrderStatus.FILLED, "canceled": OrderStatus.CANCELED, "expired": OrderStatus.EXPIRED,
        "rejected": OrderStatus.REJECTED,
    }
    for alpaca_status, core_status in cases.items():
        order = map_alpaca_order_to_core(alpaca_order(status=alpaca_status))
        assert order.status is core_status, alpaca_status


def test_map_order_rejected_gets_reject_reason():
    order = map_alpaca_order_to_core(alpaca_order(status="rejected"))
    assert order.status is OrderStatus.REJECTED
    assert order.reject_reason


def test_map_position_long():
    pos = map_alpaca_position_to_core(alpaca_position(symbol="AAPL", qty="12.5", side="long", avg_entry_price="100.0"))
    assert pos.symbol == "AAPL"
    assert pos.qty == 12.5
    assert pos.avg_price == 100.0


def test_map_position_short():
    pos = map_alpaca_position_to_core(alpaca_position(symbol="AAPL", qty="4", side="short", avg_entry_price="150.0"))
    assert pos.qty == -4.0


def test_map_account():
    account = map_alpaca_account_to_core(alpaca_account(cash="1000", equity="1500", buying_power="4000"), {})
    assert account.cash == 1000.0
    assert account.equity == 1500.0
    assert account.buying_power == 4000.0
