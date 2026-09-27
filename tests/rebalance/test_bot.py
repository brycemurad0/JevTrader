from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pytest

from jevtrader.core.registry import get as registry_get
from jevtrader.core.types import (
    AccountState,
    AssetClass,
    Instrument,
    Order,
    OrderStatus,
    Position,
    Quote,
)
from jevtrader.rebalance.bot import SmartRebalanceBot

SYMBOLS = ["SPY", "TLT", "BTC/USD"]


class FakeContext:
    """Minimal offline stand-in for jevtrader.core.strategy.StrategyContext."""

    def __init__(self, bars: dict[str, pd.DataFrame], prices: dict[str, float], account: AccountState, now: pd.Timestamp):
        self._bars = bars
        self._prices = prices
        self._account = account
        self._now = now
        self.submitted: list[Order] = []
        self.logs: list[str] = []
        self.jev = None

    @property
    def now(self) -> pd.Timestamp:
        return self._now

    def instrument(self, symbol: str) -> Instrument:
        return Instrument.infer(symbol)

    def position(self, symbol: str) -> Position:
        return self._account.positions.get(symbol, Position(symbol, 0.0, 0.0))

    def account(self) -> AccountState:
        return self._account

    def last_quote(self, symbol: str) -> Optional[Quote]:
        price = self._prices.get(symbol)
        if price is None:
            return None
        return Quote(symbol, self._now, price * 0.999, price * 1.001, 1000, 1000)

    def last_book(self, symbol: str):
        return None

    def bars(self, symbol: str, n: int) -> pd.DataFrame:
        df = self._bars.get(symbol)
        if df is None:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return df.tail(n)

    def submit(self, order: Order) -> Optional[str]:
        order.status = OrderStatus.NEW
        self.submitted.append(order)
        return order.client_order_id

    def cancel(self, client_order_id: str) -> None:
        pass

    def cancel_all(self, symbol=None) -> None:
        pass

    def open_orders(self, symbol=None) -> list[Order]:
        return []

    def log(self, msg: str, **fields) -> None:
        self.logs.append(msg)


def _make_bars(n=120, seed=0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    out = {}
    starts = {"SPY": 400.0, "TLT": 90.0, "BTC/USD": 30_000.0}
    vols = {"SPY": 0.01, "TLT": 0.006, "BTC/USD": 0.03}
    for sym in SYMBOLS:
        rets = rng.normal(0, vols[sym], n)
        close = starts[sym] * np.cumprod(1 + rets)
        out[sym] = pd.DataFrame(
            {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1_000_000.0},
            index=idx,
        )
    return out


def test_smart_rebalance_registered():
    cls = registry_get("smart_rebalance")
    assert cls is SmartRebalanceBot
    assert cls.spec.style == "rebalance"
    assert cls.spec.asset_classes == ("equity", "crypto")


def test_bot_skips_when_not_enough_history():
    bars = {s: df.head(1) for s, df in _make_bars().items()}
    account = AccountState(cash=100_000.0, equity=100_000.0, buying_power=100_000.0, positions={})
    ctx = FakeContext(bars, {"SPY": 400.0, "TLT": 90.0, "BTC/USD": 30_000.0}, account, pd.Timestamp("2024-01-01", tz="UTC"))
    bot = SmartRebalanceBot(SYMBOLS, params={"lookback": 90})
    bot.on_day_end(ctx)
    assert ctx.submitted == []
    assert any("not enough" in m for m in ctx.logs)


def test_bot_no_trade_when_already_at_target():
    bars = _make_bars(n=120)
    prices = {s: float(df["close"].iloc[-1]) for s, df in bars.items()}
    equity = 100_000.0
    returns_df = pd.DataFrame({s: bars[s]["close"].pct_change().dropna() for s in SYMBOLS}).dropna()
    from jevtrader.rebalance.targets import target_weights

    weights = target_weights("inverse_vol", returns_df, min_weight=0.0, max_weight=0.4, cash_buffer_pct=0.02)
    positions = {s: Position(s, (weights[s] * equity) / prices[s], prices[s]) for s in SYMBOLS}
    account = AccountState(cash=equity * 0.02, equity=equity, buying_power=equity, positions=positions)
    ctx = FakeContext(bars, prices, account, pd.Timestamp("2024-06-01", tz="UTC"))

    bot = SmartRebalanceBot(SYMBOLS, params={"lookback": 90, "scheme": "inverse_vol", "max_weight": 0.4})
    bot.on_day_end(ctx)
    assert ctx.submitted == []


def test_bot_trades_and_respects_dry_run():
    bars = _make_bars(n=120)
    prices = {s: float(df["close"].iloc[-1]) for s, df in bars.items()}
    equity = 100_000.0
    # Deliberately concentrated in BTC/USD, far from any sane target -> should trigger a trade.
    positions = {
        "SPY": Position("SPY", 0.0, prices["SPY"]),
        "TLT": Position("TLT", 0.0, prices["TLT"]),
        "BTC/USD": Position("BTC/USD", (0.9 * equity) / prices["BTC/USD"], prices["BTC/USD"]),
    }
    account = AccountState(cash=0.1 * equity, equity=equity, buying_power=equity, positions=positions)
    now = pd.Timestamp("2024-06-01", tz="UTC")

    dry_ctx = FakeContext(bars, prices, account, now)
    dry_bot = SmartRebalanceBot(SYMBOLS, params={"lookback": 90, "scheme": "inverse_vol", "max_weight": 0.4, "dry_run": True})
    dry_bot.on_day_end(dry_ctx)
    assert dry_ctx.submitted == []
    assert any("plan" in m.lower() for m in dry_ctx.logs)

    live_ctx = FakeContext(bars, prices, account, now)
    live_bot = SmartRebalanceBot(SYMBOLS, params={"lookback": 90, "scheme": "inverse_vol", "max_weight": 0.4, "dry_run": False})
    live_bot.on_day_end(live_ctx)
    assert len(live_ctx.submitted) > 0
    for order in live_ctx.submitted:
        assert order.strategy_id == live_bot.id
        assert order.tag == "rebalance"


def test_bot_calendar_gate_prevents_re_trading_same_period():
    bars = _make_bars(n=120)
    prices = {s: float(df["close"].iloc[-1]) for s, df in bars.items()}
    equity = 100_000.0
    positions = {
        "SPY": Position("SPY", 0.0, prices["SPY"]),
        "TLT": Position("TLT", 0.0, prices["TLT"]),
        "BTC/USD": Position("BTC/USD", (0.9 * equity) / prices["BTC/USD"], prices["BTC/USD"]),
    }
    account = AccountState(cash=0.1 * equity, equity=equity, buying_power=equity, positions=positions)
    now = pd.Timestamp("2024-06-01", tz="UTC")
    ctx = FakeContext(bars, prices, account, now)
    bot = SmartRebalanceBot(SYMBOLS, params={"lookback": 90, "scheme": "inverse_vol", "max_weight": 0.4})
    bot.policy.frequency = "1w"

    bot.on_day_end(ctx)
    first_count = len(ctx.submitted)
    assert first_count > 0

    ctx2 = FakeContext(bars, prices, account, now + pd.Timedelta(days=1))
    bot.on_day_end(ctx2)
    assert ctx2.submitted == []  # within the same week -> calendar not due
