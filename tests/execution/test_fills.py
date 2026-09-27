"""Fills arrive only via TradingStream trade updates. Fee is always computed locally through the
FeeModel (Alpaca's TradeUpdate carries no fee/liquidity field for either asset class), and
liquidity is whatever was hinted at submission time (maker for a resting limit, taker otherwise).
"""

from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from jevtrader.core.broker import TradingMode
from jevtrader.core.fees import AlpacaEquityFees, CompositeFees
from jevtrader.core.types import Fill, Liquidity, Order, OrderStatus, OrderType, Quote, Side
from jevtrader.execution.alpaca_broker import AlpacaBroker
from tests.execution.fakes import FakeTradingClient, alpaca_order, alpaca_trade_update, make_settings


def _broker(tmp_path, **kwargs):
    client = FakeTradingClient()
    settings = make_settings(tmp_path)
    fee_model = CompositeFees(equity=AlpacaEquityFees(commission_per_share=0.0))
    broker = AlpacaBroker(TradingMode.PAPER, client=client, settings=settings, fee_model=fee_model, **kwargs)
    return broker, client, fee_model


def test_maker_fill_uses_fee_model_and_records_maker_liquidity(tmp_path):
    quote = Quote(symbol="AAPL", ts=pd.Timestamp.now("UTC"), bid=99.0, ask=100.0, bid_size=10, ask_size=10)
    broker, client, fee_model = _broker(tmp_path, quote_provider=lambda s: quote)
    order = Order(symbol="AAPL", side=Side.BUY, qty=10, type=OrderType.LIMIT, limit_price=98.5)
    submitted = broker.submit(order)

    fills: list[Fill] = []
    broker.on_fill(fills.append)

    update = alpaca_trade_update(
        event="fill",
        order=alpaca_order(client_order_id=submitted.client_order_id, symbol="AAPL", side="buy", status="filled",
                            filled_qty="10", filled_avg_price="98.5"),
        price=98.5, qty=10.0,
    )
    asyncio.run(broker.on_trade_update(update))

    assert len(fills) == 1
    fill = fills[0]
    assert fill.liquidity is Liquidity.MAKER
    expected_fee = fee_model.fee(broker._instrument("AAPL"), Side.BUY, 10.0, 98.5, Liquidity.MAKER)
    assert fill.fee == pytest.approx(expected_fee)
    assert fill.price == 98.5
    assert fill.qty == 10.0

    positions = broker.positions()
    assert positions["AAPL"].qty == 10.0


def test_taker_market_order_fill(tmp_path):
    broker, client, fee_model = _broker(tmp_path)
    order = Order(symbol="AAPL", side=Side.BUY, qty=5, type=OrderType.MARKET)
    submitted = broker.submit(order)

    fills: list[Fill] = []
    broker.on_fill(fills.append)

    update = alpaca_trade_update(
        event="fill",
        order=alpaca_order(client_order_id=submitted.client_order_id, symbol="AAPL", side="buy", status="filled",
                            filled_qty="5", filled_avg_price="150.0"),
        price=150.0, qty=5.0,
    )
    asyncio.run(broker.on_trade_update(update))

    assert fills[0].liquidity is Liquidity.TAKER
    expected_fee = fee_model.fee(broker._instrument("AAPL"), Side.BUY, 5.0, 150.0, Liquidity.TAKER)
    assert fills[0].fee == pytest.approx(expected_fee)


def test_partial_fill_then_full_fill_updates_position_incrementally(tmp_path):
    broker, client, fee_model = _broker(tmp_path)
    order = Order(symbol="AAPL", side=Side.BUY, qty=10, type=OrderType.MARKET)
    submitted = broker.submit(order)

    partial = alpaca_trade_update(
        event="partial_fill",
        order=alpaca_order(client_order_id=submitted.client_order_id, symbol="AAPL", side="buy",
                            status="partially_filled", filled_qty="4", filled_avg_price="100.0"),
        price=100.0, qty=4.0,
    )
    asyncio.run(broker.on_trade_update(partial))
    assert broker.positions()["AAPL"].qty == 4.0
    assert len(broker.open_orders()) == 1  # still open

    full = alpaca_trade_update(
        event="fill",
        order=alpaca_order(client_order_id=submitted.client_order_id, symbol="AAPL", side="buy",
                            status="filled", filled_qty="10", filled_avg_price="100.5"),
        price=101.0, qty=6.0,
    )
    asyncio.run(broker.on_trade_update(full))
    assert broker.positions()["AAPL"].qty == 10.0
    assert broker.open_orders() == []  # order is done


def test_order_update_dispatches_to_order_callback(tmp_path):
    broker, client, fee_model = _broker(tmp_path)
    order = Order(symbol="AAPL", side=Side.BUY, qty=1, type=OrderType.MARKET)
    submitted = broker.submit(order)

    updates: list[Order] = []
    broker.on_order_update(updates.append)

    update = alpaca_trade_update(
        event="canceled",
        order=alpaca_order(client_order_id=submitted.client_order_id, symbol="AAPL", side="buy", status="canceled"),
    )
    asyncio.run(broker.on_trade_update(update))

    assert updates[-1].status is OrderStatus.CANCELED
    assert broker.open_orders() == []
