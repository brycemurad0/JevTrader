"""Core Order -> alpaca-py *OrderRequest mapping: stock vs crypto, TIFs, fractional qty."""

from __future__ import annotations

import pytest
from alpaca.trading.enums import OrderSide as ASide
from alpaca.trading.enums import TimeInForce as ATIF
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopOrderRequest,
)

from jevtrader.core.types import Instrument, Order, OrderType, Side, TimeInForce
from jevtrader.execution.alpaca_broker import build_order_request


def test_market_order_stock():
    order = Order(symbol="AAPL", side=Side.BUY, qty=10, type=OrderType.MARKET, tif=TimeInForce.DAY)
    instrument = Instrument.infer("AAPL")
    req = build_order_request(order, instrument)
    assert isinstance(req, MarketOrderRequest)
    assert req.symbol == "AAPL"
    assert req.side == ASide.BUY
    assert req.qty == 10
    assert req.time_in_force == ATIF.DAY
    assert req.client_order_id == order.client_order_id


def test_limit_order_crypto_fractional_qty():
    order = Order(
        symbol="BTC/USD", side=Side.BUY, qty=0.123456789, type=OrderType.LIMIT,
        limit_price=30_000.0, tif=TimeInForce.GTC,
    )
    instrument = Instrument.infer("BTC/USD")
    assert instrument.lot_size == 1e-6
    req = build_order_request(order, instrument)
    assert isinstance(req, LimitOrderRequest)
    assert req.symbol == "BTC/USD"
    assert req.limit_price == 30_000.0
    # rounded to the instrument's lot size, not truncated arbitrarily
    assert abs(req.qty - 0.123457) < 1e-9


def test_tif_day_becomes_gtc_for_crypto():
    order = Order(symbol="BTC/USD", side=Side.SELL, qty=1, type=OrderType.MARKET, tif=TimeInForce.DAY)
    req = build_order_request(order, Instrument.infer("BTC/USD"))
    assert req.time_in_force == ATIF.GTC


def test_tif_fok_becomes_ioc_for_crypto():
    order = Order(symbol="ETH/USD", side=Side.BUY, qty=1, type=OrderType.MARKET, tif=TimeInForce.FOK)
    req = build_order_request(order, Instrument.infer("ETH/USD"))
    assert req.time_in_force == ATIF.IOC


def test_tif_ioc_stays_ioc_for_crypto():
    order = Order(symbol="ETH/USD", side=Side.BUY, qty=1, type=OrderType.MARKET, tif=TimeInForce.IOC)
    req = build_order_request(order, Instrument.infer("ETH/USD"))
    assert req.time_in_force == ATIF.IOC


@pytest.mark.parametrize("core_tif,alpaca_tif", [
    (TimeInForce.DAY, ATIF.DAY), (TimeInForce.GTC, ATIF.GTC), (TimeInForce.IOC, ATIF.IOC), (TimeInForce.FOK, ATIF.FOK),
])
def test_tif_passthrough_for_equities(core_tif, alpaca_tif):
    order = Order(symbol="AAPL", side=Side.BUY, qty=1, type=OrderType.MARKET, tif=core_tif)
    req = build_order_request(order, Instrument.infer("AAPL"))
    assert req.time_in_force == alpaca_tif


def test_stop_order_stock():
    order = Order(symbol="AAPL", side=Side.SELL, qty=5, type=OrderType.STOP, stop_price=90.0)
    req = build_order_request(order, Instrument.infer("AAPL"))
    assert isinstance(req, StopOrderRequest)
    assert req.stop_price == 90.0


def test_stop_limit_order_stock():
    order = Order(symbol="AAPL", side=Side.SELL, qty=5, type=OrderType.STOP_LIMIT, stop_price=90.0, limit_price=89.5)
    req = build_order_request(order, Instrument.infer("AAPL"))
    assert isinstance(req, StopLimitOrderRequest)
    assert req.stop_price == 90.0
    assert req.limit_price == 89.5


def test_crypto_plain_stop_is_converted_to_stop_limit():
    """Alpaca crypto has no plain 'stop' order type; we convert to stop_limit with limit pinned
    to the stop price, the standard workaround."""
    order = Order(symbol="BTC/USD", side=Side.SELL, qty=0.1, type=OrderType.STOP, stop_price=25_000.0)
    req = build_order_request(order, Instrument.infer("BTC/USD"))
    assert isinstance(req, StopLimitOrderRequest)
    assert req.stop_price == 25_000.0
    assert req.limit_price == 25_000.0


def test_limit_order_missing_price_raises():
    order = Order(symbol="AAPL", side=Side.BUY, qty=1, type=OrderType.LIMIT)
    with pytest.raises(ValueError):
        build_order_request(order, Instrument.infer("AAPL"))


def test_client_order_id_passthrough():
    order = Order(symbol="AAPL", side=Side.BUY, qty=1, type=OrderType.MARKET, client_order_id="my-custom-id")
    req = build_order_request(order, Instrument.infer("AAPL"))
    assert req.client_order_id == "my-custom-id"
