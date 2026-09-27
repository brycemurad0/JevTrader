"""Core market and order types shared by every layer (backtest, paper, live).

All timestamps are timezone-aware UTC `pandas.Timestamp`s. Prices and quantities are floats;
quantities are always positive, direction is carried by `Side`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

import pandas as pd


class AssetClass(str, Enum):
    EQUITY = "equity"
    CRYPTO = "crypto"


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


class OrderStatus(str, Enum):
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"

    @property
    def is_open(self) -> bool:
        return self in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED)


class Liquidity(str, Enum):
    MAKER = "maker"
    TAKER = "taker"


@dataclass(frozen=True)
class Instrument:
    symbol: str  # "AAPL", "BTC/USD"
    asset_class: AssetClass
    tick_size: float = 0.01
    lot_size: float = 1.0  # min qty increment (crypto e.g. 1e-6)
    min_notional: float = 1.0
    shortable: bool = True  # Alpaca crypto is long-only -> False

    @staticmethod
    def infer(symbol: str) -> "Instrument":
        """Reasonable defaults: 'XXX/USD' is crypto (long-only, fractional), else US equity."""
        if "/" in symbol:
            return Instrument(symbol, AssetClass.CRYPTO, tick_size=0.01, lot_size=1e-6, min_notional=1.0, shortable=False)
        return Instrument(symbol, AssetClass.EQUITY, tick_size=0.01, lot_size=1.0, min_notional=1.0, shortable=True)


# ----------------------------------------------------------------------------- market data


@dataclass(frozen=True)
class Bar:
    symbol: str
    ts: pd.Timestamp  # bar close time
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    trade_count: Optional[int] = None


@dataclass(frozen=True)
class Quote:
    symbol: str
    ts: pd.Timestamp
    bid: float
    ask: float
    bid_size: float
    ask_size: float

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def microprice(self) -> float:
        tot = self.bid_size + self.ask_size
        if tot <= 0:
            return self.mid
        return (self.bid * self.ask_size + self.ask * self.bid_size) / tot


@dataclass(frozen=True)
class Trade:
    symbol: str
    ts: pd.Timestamp
    price: float
    size: float
    aggressor: Optional[Side] = None  # side of the taker, if known


@dataclass(frozen=True)
class BookLevel:
    price: float
    size: float


@dataclass(frozen=True)
class OrderBook:
    """L2 snapshot. bids sorted best (highest) first, asks best (lowest) first."""

    symbol: str
    ts: pd.Timestamp
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]

    @property
    def best_bid(self) -> Optional[BookLevel]:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[BookLevel]:
        return self.asks[0] if self.asks else None

    @property
    def mid(self) -> Optional[float]:
        if not self.bids or not self.asks:
            return None
        return 0.5 * (self.bids[0].price + self.asks[0].price)

    def imbalance(self, depth: int = 5) -> float:
        """(bidvol - askvol) / (bidvol + askvol) over the top `depth` levels, in [-1, 1]."""
        b = sum(l.size for l in self.bids[:depth])
        a = sum(l.size for l in self.asks[:depth])
        return 0.0 if a + b <= 0 else (b - a) / (b + a)

    def to_quote(self) -> Optional[Quote]:
        if not self.bids or not self.asks:
            return None
        return Quote(self.symbol, self.ts, self.bids[0].price, self.asks[0].price, self.bids[0].size, self.asks[0].size)


MarketEvent = Union[Bar, Quote, Trade, OrderBook]


# ----------------------------------------------------------------------------- orders

_order_ids = itertools.count(1)


def next_client_order_id(prefix: str = "jt") -> str:
    return f"{prefix}-{next(_order_ids)}"


@dataclass
class Order:
    symbol: str
    side: Side
    qty: float
    type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    tif: TimeInForce = TimeInForce.GTC
    post_only: bool = False  # reject/cancel instead of taking liquidity (simulated for venues w/o native support)
    reduce_only: bool = False
    strategy_id: str = ""
    tag: str = ""  # free text, e.g. "entry", "tp", "rebalance"
    client_order_id: str = field(default_factory=next_client_order_id)
    # mutable state, set by broker
    broker_order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.NEW
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    created_ts: Optional[pd.Timestamp] = None
    reject_reason: str = ""

    @property
    def remaining(self) -> float:
        return max(self.qty - self.filled_qty, 0.0)

    @property
    def signed_qty(self) -> float:
        return self.side.sign * self.qty


@dataclass(frozen=True)
class Fill:
    client_order_id: str
    symbol: str
    side: Side
    qty: float
    price: float
    fee: float  # in quote currency (USD), always >= 0
    liquidity: Liquidity
    ts: pd.Timestamp
    strategy_id: str = ""

    @property
    def notional(self) -> float:
        return self.qty * self.price


@dataclass
class Position:
    symbol: str
    qty: float = 0.0  # signed
    avg_price: float = 0.0
    realized_pnl: float = 0.0  # net of fees
    fees_paid: float = 0.0

    def apply_fill(self, fill: Fill) -> float:
        """Update position with a fill; returns realized PnL (gross, before fee) of this fill."""
        signed = fill.side.sign * fill.qty
        realized = 0.0
        if self.qty == 0 or (self.qty > 0) == (signed > 0):
            new_qty = self.qty + signed
            self.avg_price = (self.avg_price * abs(self.qty) + fill.price * abs(signed)) / abs(new_qty)
            self.qty = new_qty
        else:
            closing = min(abs(signed), abs(self.qty))
            direction = 1 if self.qty > 0 else -1
            realized = closing * (fill.price - self.avg_price) * direction
            new_qty = self.qty + signed
            if abs(new_qty) < 1e-12:
                self.qty, self.avg_price = 0.0, 0.0
            elif (new_qty > 0) != (self.qty > 0):  # flipped
                self.qty, self.avg_price = new_qty, fill.price
            else:
                self.qty = new_qty
        self.realized_pnl += realized - fill.fee
        self.fees_paid += fill.fee
        return realized

    def market_value(self, price: float) -> float:
        return self.qty * price

    def unrealized_pnl(self, price: float) -> float:
        return self.qty * (price - self.avg_price)


@dataclass
class AccountState:
    cash: float
    equity: float
    buying_power: float
    positions: dict[str, Position] = field(default_factory=dict)
    ts: Optional[pd.Timestamp] = None
