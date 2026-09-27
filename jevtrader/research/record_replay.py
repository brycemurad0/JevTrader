"""Record real market microstructure during paper trading and replay it through the Backtester.

Why: we have no real L1/L2 data in the cache, so the quote/book strategies (`as_market_maker`,
`imbalance_alpha`, the VPIN gate) are validated for MECHANICS only. The live runner's journal
(`runs_dir/<run_id>/events.jsonl`) records orders, fills and Jev decisions but NOT market data,
so this module provides:

* `MarketDataRecorder` -- a no-trade `Strategy` you add to the `LiveRunner` alongside your real
  strategies. It appends every Quote / Trade / OrderBook / Bar it receives to
  `<dir>/market_<date>.jsonl` (one JSON object per line: `{"kind": "quote"|"trade"|"book"|"bar",
  "ts": iso, "symbol": ..., ...}`). A few days of Alpaca paper trading on 2-3 symbols is ~1-3 GB
  uncompressed; gzip them (`gzip market_*.jsonl`) -- the loader reads .gz transparently.
* `load_market_events(paths)` -- turns those files back into time-sorted `MarketEvent`
  iterators keyed by symbol, ready for `Backtester(strategies, data)`.
* `load_journal(path)` -- parses the runner's `events.jsonl` into DataFrames of fills, orders and
  Jev decisions, and `realized_cost_bps(fills)` computes the paper-trading `cost_bps` that
  `promote_to_live` compares against the backtest's modelled cost (max 15 bps deviation).
* `replay_strategy(...)` -- convenience: run a strategy over recorded events with the SimBroker
  (real fees, L1 queue model) so you can compare the sim's fills/PnL with the paper fills the
  journal actually recorded. Disagreement = the fill model is optimistic for that venue.

Everything here is offline and deterministic.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence, Union

import numpy as np
import pandas as pd

from jevtrader.backtest.engine import Backtester, BacktestResult
from jevtrader.core.fees import FeeModel
from jevtrader.core.strategy import Strategy, StrategyContext, StrategySpec
from jevtrader.core.types import Bar, BookLevel, OrderBook, Quote, Side, Trade

MarketEvent = Union[Bar, Quote, Trade, OrderBook]


# ----------------------------------------------------------------------------- recording


def event_to_record(event: MarketEvent) -> dict[str, Any]:
    ts = pd.Timestamp(event.ts).isoformat()
    if isinstance(event, Quote):
        return {"kind": "quote", "ts": ts, "symbol": event.symbol, "bid": event.bid, "ask": event.ask, "bid_size": event.bid_size, "ask_size": event.ask_size}
    if isinstance(event, Trade):
        return {"kind": "trade", "ts": ts, "symbol": event.symbol, "price": event.price, "size": event.size, "aggressor": event.aggressor.value if event.aggressor else None}
    if isinstance(event, OrderBook):
        return {"kind": "book", "ts": ts, "symbol": event.symbol, "bids": [[l.price, l.size] for l in event.bids], "asks": [[l.price, l.size] for l in event.asks]}
    if isinstance(event, Bar):
        return {"kind": "bar", "ts": ts, "symbol": event.symbol, "open": event.open, "high": event.high, "low": event.low, "close": event.close, "volume": event.volume, "vwap": event.vwap}
    raise TypeError(f"unsupported event {type(event)!r}")


def record_to_event(rec: dict[str, Any]) -> Optional[MarketEvent]:
    kind = rec.get("kind")
    ts = pd.Timestamp(rec["ts"])
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    sym = rec["symbol"]
    if kind == "quote":
        return Quote(sym, ts, float(rec["bid"]), float(rec["ask"]), float(rec["bid_size"]), float(rec["ask_size"]))
    if kind == "trade":
        agg = rec.get("aggressor")
        return Trade(sym, ts, float(rec["price"]), float(rec["size"]), Side(agg) if agg else None)
    if kind == "book":
        bids = tuple(BookLevel(float(p), float(s)) for p, s in rec["bids"])
        asks = tuple(BookLevel(float(p), float(s)) for p, s in rec["asks"])
        return OrderBook(sym, ts, bids, asks)
    if kind == "bar":
        vwap = rec.get("vwap")
        return Bar(sym, ts, float(rec["open"]), float(rec["high"]), float(rec["low"]), float(rec["close"]), float(rec["volume"]), vwap=float(vwap) if vwap is not None else None)
    return None


class MarketDataRecorder(Strategy):
    """No-trade strategy that journals every market event it sees to daily JSONL files."""

    spec = StrategySpec(
        name="market_recorder",
        description="Records quotes/trades/books/bars to JSONL for later replay through the Backtester. Never trades.",
        asset_classes=("equity", "crypto"),
        frequency="tick",
        style="tooling",
        default_params={"dir": "runs/market_data", "flush_every": 200},
        notes="Add to the LiveRunner next to your real strategies during paper trading; replay with research.record_replay.load_market_events.",
    )

    def __init__(self, symbols, params=None, strategy_id=None):
        super().__init__(symbols, params, strategy_id)
        self.dir = Path(self.params["dir"])
        self.dir.mkdir(parents=True, exist_ok=True)
        self._fh = None
        self._fh_date = None
        self._buf: list[str] = []
        self.n_recorded = 0

    def _write(self, event: MarketEvent) -> None:
        d = pd.Timestamp(event.ts).strftime("%Y%m%d")
        if self._fh is None or self._fh_date != d:
            self.flush()
            if self._fh is not None:
                self._fh.close()
            self._fh = open(self.dir / f"market_{d}.jsonl", "a", encoding="utf-8")
            self._fh_date = d
        self._buf.append(json.dumps(event_to_record(event), separators=(",", ":")))
        self.n_recorded += 1
        if len(self._buf) >= int(self.params["flush_every"]):
            self.flush()

    def flush(self) -> None:
        if self._fh is not None and self._buf:
            self._fh.write("\n".join(self._buf) + "\n")
            self._fh.flush()
        self._buf.clear()

    def on_quote(self, quote: Quote, ctx: StrategyContext) -> None:
        self._write(quote)

    def on_trade(self, trade: Trade, ctx: StrategyContext) -> None:
        self._write(trade)

    def on_book(self, book: OrderBook, ctx: StrategyContext) -> None:
        self._write(book)

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> None:
        self._write(bar)

    def on_stop(self, ctx: StrategyContext) -> None:
        self.flush()
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def write_events(path: Path, events: Iterable[MarketEvent]) -> int:
    """Write an iterable of events to one JSONL(.gz) file. Returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    n = 0
    with opener(path, "wt", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(event_to_record(ev), separators=(",", ":")) + "\n")
            n += 1
    return n


# ----------------------------------------------------------------------------- replay


def _iter_records(path: Path) -> Iterator[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_market_events(paths: Sequence[Union[str, Path]], kinds: Optional[set[str]] = None, symbols: Optional[set[str]] = None) -> dict[str, list[MarketEvent]]:
    """Read recorded files into `{symbol: [events sorted by ts]}` -- the `data` argument the
    Backtester takes. `kinds` (e.g. {"quote","trade"}) and `symbols` filter what is loaded."""
    out: dict[str, list[MarketEvent]] = {}
    for p in sorted(Path(x) for x in paths):
        for rec in _iter_records(p):
            if kinds is not None and rec.get("kind") not in kinds:
                continue
            if symbols is not None and rec.get("symbol") not in symbols:
                continue
            ev = record_to_event(rec)
            if ev is not None:
                out.setdefault(ev.symbol, []).append(ev)
    for sym in out:
        out[sym].sort(key=lambda e: e.ts)
    return out


def replay_strategy(strategy: Strategy, events: dict[str, list[MarketEvent]], fees: Optional[FeeModel] = None, latency_ms: float = 50.0, **bt_kwargs) -> BacktestResult:
    """Run `strategy` over recorded events with the SimBroker (real fee model, L1 queue model)."""
    bt = Backtester([strategy], events, fees=fees, latency_ms=latency_ms, **bt_kwargs)
    return bt.run()


# ----------------------------------------------------------------------------- journal parsing


def load_journal(path: Union[str, Path]) -> dict[str, pd.DataFrame]:
    """Parse a LiveRunner `events.jsonl` into DataFrames by kind: fills, orders, jev decisions,
    equity snapshots, everything else under 'other'."""
    rows: dict[str, list[dict]] = {"fill": [], "order": [], "jev": [], "equity": [], "other": []}
    for rec in _iter_records(Path(path)):
        kind = str(rec.get("kind", ""))
        if "fill" in kind:
            rows["fill"].append({**rec.get("fill", rec), "journal_ts": rec.get("ts")})
        elif "order" in kind:
            rows["order"].append({**rec.get("order", rec), "journal_ts": rec.get("ts")})
        elif "jev" in kind:
            rows["jev"].append(rec)
        elif "equity" in kind:
            rows["equity"].append(rec)
        else:
            rows["other"].append(rec)
    out = {k: pd.DataFrame(v) for k, v in rows.items()}
    return out


def realized_cost_bps(fills: pd.DataFrame, reference_mid: Optional[pd.Series] = None) -> float:
    """Realized fee + (optionally) spread/slippage cost in bps of traded notional -- the paper-
    trading `cost_bps` the promotion gate compares with the backtest's modelled cost.

    Fees come straight from the fill records. If `reference_mid` (a UTC-indexed Series of mids)
    is given, the signed distance between each fill price and the last mid before the fill is
    added as the spread+slippage component."""
    if fills is None or fills.empty:
        return float("nan")
    qty = fills["qty"].astype(float)
    price = fills["price"].astype(float)
    notional = (qty * price).abs()
    fee = fills["fee"].astype(float) if "fee" in fills else pd.Series(0.0, index=fills.index)
    total = float(fee.sum())
    if reference_mid is not None and "ts" in fills:
        ts = pd.to_datetime(fills["ts"], utc=True)
        mids = reference_mid.sort_index().reindex(ts, method="ffill").to_numpy()
        sign = np.where(fills["side"].astype(str).str.lower() == "buy", 1.0, -1.0)
        slip = sign * (price.to_numpy() - mids) * qty.to_numpy()
        total += float(np.nansum(slip))
    return 1e4 * total / float(notional.sum()) if notional.sum() > 0 else float("nan")


def compare_sim_vs_paper(sim: BacktestResult, paper_fills: pd.DataFrame) -> dict[str, float]:
    """Headline reconciliation numbers between a replay and the paper fills over the same period."""
    sim_cost = float(sim.metrics.get("cost_per_trade_bps", float("nan")))
    paper_cost = realized_cost_bps(paper_fills)
    return {
        "sim_n_fills": int(len(sim.fills)),
        "paper_n_fills": int(len(paper_fills)) if paper_fills is not None else 0,
        "sim_cost_bps": sim_cost,
        "paper_cost_bps": paper_cost,
        "cost_deviation_bps": abs(sim_cost - paper_cost) if np.isfinite(sim_cost) and np.isfinite(paper_cost) else float("nan"),
        "sim_maker_share": float((sim.fills["liquidity"] == "maker").mean()) if len(sim.fills) else float("nan"),
        "paper_maker_share": float((paper_fills["liquidity"] == "maker").mean()) if paper_fills is not None and len(paper_fills) and "liquidity" in paper_fills else float("nan"),
    }
