"""Strategy contract. The SAME strategy class runs unchanged in backtest, paper and live.

A strategy never talks to a broker directly. It receives market events and a `StrategyContext`,
and expresses intent via `ctx.submit(...)` / `ctx.cancel(...)`. Every order it submits passes
through the RiskManager before reaching the broker (or the simulator), which may veto or resize.

Jev usage pattern ("Jev judges, code executes"): a strategy may call `ctx.jev` (a JevAdvisor,
possibly None) to get calibrated probabilities, but thresholds, sizing and risk stay in code.
Strategies MUST work when `ctx.jev is None` (fallback path), so they are backtestable offline.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Protocol

import pandas as pd

from jevtrader.core.types import (
    AccountState,
    Bar,
    Fill,
    Instrument,
    Order,
    OrderBook,
    Position,
    Quote,
    Trade,
)

if TYPE_CHECKING:  # pragma: no cover
    from jevtrader.jev.advisor import JevAdvisor


class StrategyContext(Protocol):
    """What a strategy can see and do. Implemented by the backtester and by the live runner."""

    @property
    def now(self) -> pd.Timestamp: ...

    @property
    def jev(self) -> Optional["JevAdvisor"]: ...

    def instrument(self, symbol: str) -> Instrument: ...

    def position(self, symbol: str) -> Position: ...

    def account(self) -> AccountState: ...

    def last_quote(self, symbol: str) -> Optional[Quote]: ...

    def last_book(self, symbol: str) -> Optional[OrderBook]: ...

    def bars(self, symbol: str, n: int) -> pd.DataFrame:
        """Last `n` completed bars (columns open, high, low, close, volume[, vwap]) indexed by ts."""
        ...

    def submit(self, order: Order) -> Optional[str]:
        """Submit an order. Returns client_order_id, or None if the risk layer rejected it."""
        ...

    def cancel(self, client_order_id: str) -> None: ...

    def cancel_all(self, symbol: Optional[str] = None) -> None: ...

    def open_orders(self, symbol: Optional[str] = None) -> list[Order]: ...

    def log(self, msg: str, **fields: Any) -> None: ...


@dataclass
class StrategySpec:
    """Metadata used by the registry, CLI, docs and the promotion pipeline."""

    name: str
    description: str
    asset_classes: tuple[str, ...]  # ("equity",), ("crypto",), ("equity", "crypto")
    frequency: str  # "tick", "book", "1s", "1min", "5min", "1h", "1d"
    style: str  # "market_making", "mean_reversion", "momentum", "stat_arb", "breakout", "rebalance"
    uses_jev: bool = False
    default_params: dict[str, Any] = field(default_factory=dict)
    # Honest notes: when it works, when it fails, fee sensitivity.
    notes: str = ""


class Strategy(ABC):
    """Base class. Override only the handlers you need."""

    spec: ClassVar[StrategySpec]

    def __init__(self, symbols: list[str], params: Optional[dict[str, Any]] = None, strategy_id: Optional[str] = None):
        self.symbols = list(symbols)
        self.params: dict[str, Any] = {**self.spec.default_params, **(params or {})}
        self.id = strategy_id or f"{self.spec.name}:{','.join(self.symbols)}"

    # lifecycle
    def on_start(self, ctx: StrategyContext) -> None: ...

    def on_stop(self, ctx: StrategyContext) -> None: ...

    # market data
    def on_bar(self, bar: Bar, ctx: StrategyContext) -> None: ...

    def on_quote(self, quote: Quote, ctx: StrategyContext) -> None: ...

    def on_trade(self, trade: Trade, ctx: StrategyContext) -> None: ...

    def on_book(self, book: OrderBook, ctx: StrategyContext) -> None: ...

    # order lifecycle
    def on_fill(self, fill: Fill, ctx: StrategyContext) -> None: ...

    def on_order_update(self, order: Order, ctx: StrategyContext) -> None: ...

    # daily hook (equities: after close; crypto: 00:00 UTC)
    def on_day_end(self, ctx: StrategyContext) -> None: ...
