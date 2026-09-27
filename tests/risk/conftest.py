from __future__ import annotations

import pandas as pd
import pytest

from jevtrader.core.types import AccountState, Order, OrderType, Position, Quote, Side
from jevtrader.risk.limits import RiskLimits, TradingHours

# Tuesday 2024-01-02, 15:00 UTC = 10:00 America/New_York -> inside regular equity session.
NOW = pd.Timestamp("2024-01-02 15:00", tz="UTC")
# Same day, 22:00 UTC = 17:00 ET -> after the close.
AFTER_HOURS = pd.Timestamp("2024-01-02 22:00", tz="UTC")


def loose_limits(**overrides) -> RiskLimits:
    """RiskLimits with every cap wide open except whatever `overrides` sets, so a test can
    isolate exactly one limit at a time."""
    base = dict(
        max_position_notional_usd=1_000_000.0,
        max_position_pct_equity=1.0,
        max_gross_exposure_pct=10.0,
        max_net_exposure_pct=10.0,
        crypto_max_pct_equity=10.0,
        per_strategy_capital_pct={},
        default_strategy_capital_pct=10.0,
        max_open_orders_per_symbol=1000,
        max_order_notional_usd=1_000_000.0,
        max_orders_per_minute=1000,
        fat_finger_band_bps=10_000.0,
        min_notional_usd=0.01,
        max_daily_loss_pct=1.0,
        max_drawdown_pct=1.0,
        max_consecutive_losses=1_000_000,
        consecutive_loss_cooldown_minutes=1.0,
        daily_reset_hour_utc=0,
        enforce_long_only_for_non_shortable=True,
        trading_hours=TradingHours(enabled=False),
    )
    base.update(overrides)
    return RiskLimits(**base)


def make_account(equity: float, positions: dict | None = None, cash: float | None = None) -> AccountState:
    return AccountState(
        cash=cash if cash is not None else equity,
        equity=equity,
        buying_power=equity,
        positions=positions or {},
    )


def make_position(symbol: str, qty: float, avg_price: float) -> Position:
    return Position(symbol=symbol, qty=qty, avg_price=avg_price)


def make_quote(symbol: str, mid: float, spread: float = 0.02, ts: pd.Timestamp = NOW) -> Quote:
    half = spread / 2
    return Quote(symbol=symbol, ts=ts, bid=mid - half, ask=mid + half, bid_size=1000, ask_size=1000)


def make_order(
    symbol: str,
    side: Side,
    qty: float,
    type: OrderType = OrderType.MARKET,
    limit_price: float | None = None,
    strategy_id: str = "strat1",
    reduce_only: bool = False,
) -> Order:
    return Order(
        symbol=symbol,
        side=side,
        qty=qty,
        type=type,
        limit_price=limit_price,
        strategy_id=strategy_id,
        reduce_only=reduce_only,
    )
