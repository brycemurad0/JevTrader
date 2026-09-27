import pandas as pd

from jevtrader.core.fees import ZeroFees
from jevtrader.core.types import AssetClass, Bar, Instrument, Order, OrderStatus, OrderType, Side
from jevtrader.backtest.sim_broker import SimBroker


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def test_sell_exceeding_flat_position_on_non_shortable_crypto_is_rejected():
    inst = Instrument("BTC/USD", AssetClass.CRYPTO, lot_size=1e-6, min_notional=1.0, shortable=False)
    broker = SimBroker(fee_model=ZeroFees(), instruments={"BTC/USD": inst}, initial_cash=1_000_000)

    bar = Bar("BTC/USD", _ts("2024-01-02 00:00:00"), open=50_000, high=50_100, low=49_900, close=50_000, volume=10)
    broker.on_market_event(bar)

    result = broker.submit(Order(symbol="BTC/USD", side=Side.SELL, qty=1.0, type=OrderType.MARKET))
    assert result.status is OrderStatus.REJECTED
    assert result.reject_reason == "would_short_non_shortable_instrument"
    assert broker.positions() == {}


def test_sell_up_to_existing_long_is_allowed_but_not_beyond_it():
    inst = Instrument("BTC/USD", AssetClass.CRYPTO, lot_size=1e-6, min_notional=1.0, shortable=False)
    broker = SimBroker(fee_model=ZeroFees(), instruments={"BTC/USD": inst}, initial_cash=1_000_000)

    bar0 = Bar("BTC/USD", _ts("2024-01-02 00:00:00"), open=50_000, high=50_100, low=49_900, close=50_000, volume=10)
    broker.on_market_event(bar0)
    buy = broker.submit(Order(symbol="BTC/USD", side=Side.BUY, qty=1.0, type=OrderType.MARKET))
    assert buy.status is OrderStatus.NEW

    bar1 = Bar("BTC/USD", _ts("2024-01-02 00:01:00"), open=50_000, high=50_100, low=49_900, close=50_050, volume=10)
    fills = broker.on_market_event(bar1)
    assert len(fills) == 1
    assert broker.positions()["BTC/USD"].qty == 1.0

    ok_sell = broker.submit(Order(symbol="BTC/USD", side=Side.SELL, qty=1.0, type=OrderType.MARKET))
    assert ok_sell.status is OrderStatus.NEW

    too_much_sell = broker.submit(Order(symbol="BTC/USD", side=Side.SELL, qty=0.5, type=OrderType.MARKET))
    # the first sell (qty=1.0) already consumes the whole long position for shorting purposes
    # once both are pending; the second must be rejected since it would take us net short.
    assert too_much_sell.status is OrderStatus.REJECTED
    assert too_much_sell.reject_reason == "would_short_non_shortable_instrument"


def test_shortable_equity_instrument_allows_short_sell():
    inst = Instrument("AAPL", AssetClass.EQUITY, shortable=True)
    broker = SimBroker(fee_model=ZeroFees(), instruments={"AAPL": inst}, initial_cash=1_000_000)
    bar = Bar("AAPL", _ts("2024-01-02 14:30:00"), open=100.0, high=101.0, low=99.0, close=100.0, volume=100_000)
    broker.on_market_event(bar)

    result = broker.submit(Order(symbol="AAPL", side=Side.SELL, qty=10, type=OrderType.MARKET))
    assert result.status is OrderStatus.NEW  # shortable instruments may open a short position
