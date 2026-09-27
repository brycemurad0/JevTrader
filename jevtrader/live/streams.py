"""Market data adapters: normalize Alpaca stock/crypto live streams (bars, quotes, trades,
crypto L2 order books) into core `MarketEvent`s pushed onto an `asyncio.Queue`, plus a
`SimulatedStream` that replays recorded events for tests and for dry-run-on-recorded-data.

Every adapter exposes the same small surface the `LiveRunner` needs:

* `.queue`  -- an `asyncio.Queue` the adapter pushes normalized core events onto.
* `async .run()` -- subscribes (if applicable) and runs until cancelled/stopped.
* `.stop()` -- best-effort request to stop.

so the runner can treat a live Alpaca feed and a recorded replay identically.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence, Union

import pandas as pd

from jevtrader.core.types import Bar, BookLevel, MarketEvent, OrderBook, Quote, Trade

logger = logging.getLogger(__name__)


def _val(x: Any) -> Any:
    return x.value if hasattr(x, "value") else x


def _ts(x: Any) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


# --------------------------------------------------------------------------------- alpaca -> core

def core_bar(b: Any) -> Bar:
    return Bar(
        symbol=b.symbol,
        ts=_ts(b.timestamp),
        open=float(b.open),
        high=float(b.high),
        low=float(b.low),
        close=float(b.close),
        volume=float(b.volume),
        vwap=float(b.vwap) if getattr(b, "vwap", None) is not None else None,
        trade_count=int(b.trade_count) if getattr(b, "trade_count", None) is not None else None,
    )


def core_quote(q: Any) -> Quote:
    return Quote(
        symbol=q.symbol,
        ts=_ts(q.timestamp),
        bid=float(q.bid_price),
        ask=float(q.ask_price),
        bid_size=float(q.bid_size),
        ask_size=float(q.ask_size),
    )


def core_trade(t: Any) -> Trade:
    return Trade(symbol=t.symbol, ts=_ts(t.timestamp), price=float(t.price), size=float(t.size))


# --------------------------------------------------------------------------------- L2 book merge

@dataclass
class _BookSide:
    levels: dict[float, float] = field(default_factory=dict)

    def apply(self, price: float, size: float) -> None:
        if size <= 0:
            self.levels.pop(price, None)
        else:
            self.levels[price] = size


class OrderBookBuilder:
    """Maintains one L2 book per symbol from Alpaca's crypto orderbook stream.

    Alpaca sends `reset=True` as the initial full snapshot after subscribing, then incremental
    deltas; in both, a level with `size == 0` means "remove this price level" -- this mirrors
    that exactly rather than assuming every message is a full snapshot.
    """

    def __init__(self) -> None:
        self._bids: dict[str, _BookSide] = {}
        self._asks: dict[str, _BookSide] = {}

    def apply(self, ob: Any) -> OrderBook:
        symbol = ob.symbol
        bid_side = self._bids.setdefault(symbol, _BookSide())
        ask_side = self._asks.setdefault(symbol, _BookSide())
        if getattr(ob, "reset", False):
            bid_side.levels.clear()
            ask_side.levels.clear()
        for level in ob.bids:
            bid_side.apply(float(level.price), float(level.size))
        for level in ob.asks:
            ask_side.apply(float(level.price), float(level.size))
        return OrderBook(
            symbol=symbol,
            ts=_ts(ob.timestamp),
            bids=_sorted_levels(bid_side.levels, descending=True),
            asks=_sorted_levels(ask_side.levels, descending=False),
        )

    def snapshot(self, symbol: str) -> Optional[OrderBook]:
        if symbol not in self._bids:
            return None
        return OrderBook(
            symbol=symbol,
            ts=pd.Timestamp.utcnow(),
            bids=_sorted_levels(self._bids[symbol].levels, descending=True),
            asks=_sorted_levels(self._asks[symbol].levels, descending=False),
        )


def _sorted_levels(levels: dict[float, float], *, descending: bool) -> tuple[BookLevel, ...]:
    items = sorted(levels.items(), key=lambda kv: kv[0], reverse=descending)
    return tuple(BookLevel(price=p, size=s) for p, s in items)


# --------------------------------------------------------------------------------- live adapters

class StockStream:
    """Wraps `alpaca.data.live.stock.StockDataStream`: subscribes to bars/quotes/trades for
    `symbols` and normalizes each message onto `queue` as a core `Bar`/`Quote`/`Trade`."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        queue: asyncio.Queue,
        symbols: Sequence[str],
        feed: str = "iex",
        client: Optional[Any] = None,
    ) -> None:
        self.queue = queue
        self.symbols = list(symbols)
        if client is not None:
            self._client = client
        else:
            from alpaca.data.enums import DataFeed
            from alpaca.data.live.stock import StockDataStream

            self._client = StockDataStream(api_key, secret_key, feed=DataFeed(feed))

    async def _on_bar(self, bar: Any) -> None:
        await self.queue.put(core_bar(bar))

    async def _on_quote(self, quote: Any) -> None:
        await self.queue.put(core_quote(quote))

    async def _on_trade(self, trade: Any) -> None:
        await self.queue.put(core_trade(trade))

    def subscribe(self) -> None:
        if not self.symbols:
            return
        self._client.subscribe_bars(self._on_bar, *self.symbols)
        self._client.subscribe_quotes(self._on_quote, *self.symbols)
        self._client.subscribe_trades(self._on_trade, *self.symbols)

    async def run(self) -> None:
        self.subscribe()
        await asyncio.to_thread(self._client.run)

    def stop(self) -> None:
        self._client.stop()


class CryptoStream:
    """Wraps `alpaca.data.live.crypto.CryptoDataStream`: bars/quotes/trades like `StockStream`,
    plus L2 order books merged through an `OrderBookBuilder` into a maintained `OrderBook`."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        queue: asyncio.Queue,
        symbols: Sequence[str],
        feed: str = "us",
        client: Optional[Any] = None,
    ) -> None:
        self.queue = queue
        self.symbols = list(symbols)
        self._book = OrderBookBuilder()
        if client is not None:
            self._client = client
        else:
            from alpaca.data.enums import CryptoFeed
            from alpaca.data.live.crypto import CryptoDataStream

            self._client = CryptoDataStream(api_key, secret_key, feed=CryptoFeed(feed))

    async def _on_bar(self, bar: Any) -> None:
        await self.queue.put(core_bar(bar))

    async def _on_quote(self, quote: Any) -> None:
        await self.queue.put(core_quote(quote))

    async def _on_trade(self, trade: Any) -> None:
        await self.queue.put(core_trade(trade))

    async def _on_orderbook(self, ob: Any) -> None:
        await self.queue.put(self._book.apply(ob))

    def subscribe(self) -> None:
        if not self.symbols:
            return
        self._client.subscribe_bars(self._on_bar, *self.symbols)
        self._client.subscribe_quotes(self._on_quote, *self.symbols)
        self._client.subscribe_trades(self._on_trade, *self.symbols)
        self._client.subscribe_orderbooks(self._on_orderbook, *self.symbols)

    async def run(self) -> None:
        self.subscribe()
        await asyncio.to_thread(self._client.run)

    def stop(self) -> None:
        self._client.stop()


class SimulatedStream:
    """Replays a recorded sequence of core `MarketEvent`s (or a DataFrame of OHLCV bars) onto
    `queue`, either as fast as possible (`speed=0`, the default -- good for tests) or paced to
    real elapsed time between event timestamps divided by `speed` (`speed=60` replays an hour of
    recorded data in one minute). Used for offline tests and for "dry-run on recorded data"."""

    def __init__(
        self,
        events: Union[Iterable[MarketEvent], pd.DataFrame],
        queue: asyncio.Queue,
        speed: float = 0.0,
        symbol: Optional[str] = None,
    ) -> None:
        self.queue = queue
        self.speed = speed
        if isinstance(events, pd.DataFrame):
            self._events: list[MarketEvent] = _bars_from_dataframe(events, symbol)
        else:
            self._events = list(events)
        self._stopped = False
        self.done = asyncio.Event()

    async def run(self) -> None:
        prev_ts: Optional[pd.Timestamp] = None
        for event in self._events:
            if self._stopped:
                break
            if self.speed > 0 and prev_ts is not None:
                dt = (event.ts - prev_ts).total_seconds() / self.speed
                if dt > 0:
                    await asyncio.sleep(dt)
            prev_ts = event.ts
            await self.queue.put(event)
        self.done.set()

    def stop(self) -> None:
        self._stopped = True
        self.done.set()


def _bars_from_dataframe(df: pd.DataFrame, symbol: Optional[str]) -> list[Bar]:
    if symbol is None:
        raise ValueError("symbol is required when replaying a DataFrame of bars")
    bars: list[Bar] = []
    for ts, row in df.iterrows():
        vwap = row["vwap"] if "vwap" in row and pd.notna(row["vwap"]) else None
        bars.append(
            Bar(
                symbol=symbol,
                ts=_ts(ts),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0.0)),
                vwap=float(vwap) if vwap is not None else None,
            )
        )
    return bars
