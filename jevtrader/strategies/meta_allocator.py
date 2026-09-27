"""Portfolio-level meta-allocator: runs several child strategies inside one `Strategy`, tracks
each child's *virtual* P&L stream, and sets risk-budget weights across children with the same
schemes `SmartRebalanceBot` uses (`jevtrader.rebalance.targets`: inverse_vol, hrp, risk_parity,
equal_weight, min_variance). Children see a proxy context whose `submit` scales their order size
by the child's weight and whose `position()` reports their own virtual position, so two children
can hold the same symbol without confusing each other.

Playing well with `SmartRebalanceBot`: give the rebalancer the *core* book (long-only ETFs +
BTC) and the meta-allocator a capital budget (`capital_frac`) for the alpha sleeve; both read the
same account and the risk manager applies gross-exposure caps across everything.

The children must be registered strategies. Weights are re-estimated every `rebalance_bars`
aggregated bars from the trailing `lookback` child returns; a child with no history yet gets
`1/n`. Weights change how *new* child orders are scaled -- existing positions are not forcibly
resized (that would create turnover for no signal reason).

Evidence: with only one VIABLE/MARGINAL sleeve found in the research (equity intraday mean
reversion), the allocator's benefit today is mainly risk control and having the plumbing ready
for when more sleeves pass validation. See research_reports/08_meta_allocator.md.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Optional

import numpy as np
import pandas as pd

from jevtrader.core.registry import get as registry_get
from jevtrader.core.registry import register
from jevtrader.core.strategy import Strategy, StrategyContext, StrategySpec
from jevtrader.core.types import AccountState, Bar, Fill, Instrument, Order, OrderBook, Position, Quote, Trade
from jevtrader.rebalance.targets import target_weights
from jevtrader.risk.sizing import lot_round


class _ChildContext:
    """StrategyContext proxy for one child: virtual positions, weight-scaled order size."""

    def __init__(self, meta: "MetaAllocator", child: Strategy) -> None:
        self._meta = meta
        self._child = child
        self.positions: dict[str, Position] = {}

    @property
    def now(self) -> pd.Timestamp:
        return self._meta._ctx.now

    @property
    def jev(self):
        return self._meta._ctx.jev

    def instrument(self, symbol: str) -> Instrument:
        return self._meta._ctx.instrument(symbol)

    def position(self, symbol: str) -> Position:
        return self.positions.get(symbol) or Position(symbol)

    def account(self) -> AccountState:
        acct = self._meta._ctx.account()
        w = self._meta.weight(self._child.id) * float(self._meta.params["capital_frac"])
        # child sees its slice of the equity so its alloc_frac sizing is relative to its budget
        return AccountState(cash=acct.cash * w, equity=acct.equity * w, buying_power=acct.buying_power * w, positions=dict(self.positions), ts=acct.ts)

    def last_quote(self, symbol: str) -> Optional[Quote]:
        return self._meta._ctx.last_quote(symbol)

    def last_book(self, symbol: str) -> Optional[OrderBook]:
        return self._meta._ctx.last_book(symbol)

    def bars(self, symbol: str, n: int) -> pd.DataFrame:
        return self._meta._ctx.bars(symbol, n)

    def submit(self, order: Order) -> Optional[str]:
        inst = self.instrument(order.symbol)
        qty = abs(lot_round(order.qty, inst))
        if qty <= 0:
            return None
        order.qty = qty
        order.strategy_id = self._meta.id
        order.tag = f"child:{self._child.id}|{order.tag}"
        oid = self._meta._ctx.submit(order)
        if oid is not None:
            self._meta._order_owner[oid] = self._child.id
        return oid

    def cancel(self, client_order_id: str) -> None:
        self._meta._ctx.cancel(client_order_id)

    def cancel_all(self, symbol: Optional[str] = None) -> None:
        for o in self.open_orders(symbol):
            self._meta._ctx.cancel(o.client_order_id)

    def open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        mine = {oid for oid, cid in self._meta._order_owner.items() if cid == self._child.id}
        return [o for o in self._meta._ctx.open_orders(symbol) if o.client_order_id in mine]

    def log(self, msg: str, **fields: Any) -> None:
        self._meta._ctx.log(f"[{self._child.id}] {msg}", **fields)


class MetaAllocator(Strategy):
    spec = StrategySpec(
        name="meta_allocator",
        description="Runs child strategies as sleeves with inverse-vol / HRP / risk-parity risk budgets estimated from their own virtual P&L streams.",
        asset_classes=("equity", "crypto"),
        frequency="15min",
        style="portfolio",
        uses_jev=True,
        default_params={
            "children": [  # list of {"name": registered strategy name, "symbols": [...], "params": {...}}
                {"name": "zscore_reversion", "symbols": ["SPY"], "params": {}},
            ],
            "scheme": "inverse_vol",
            "lookback": 200,  # child return observations used for weights
            "rebalance_bars": 50,
            "min_weight": 0.05,
            "max_weight": 0.60,
            "capital_frac": 1.0,  # share of account equity the alpha sleeve may use
        },
        notes=(
            "Portfolio plumbing, not an alpha source. Weights from jevtrader.rebalance.targets (shared with "
            "SmartRebalanceBot). Children hold virtual positions; the real account nets them. Use capital_frac < 1 "
            "when running next to smart_rebalance so the core book keeps its cash."
        ),
    )

    def __init__(self, symbols, params=None, strategy_id=None):
        super().__init__(symbols, params, strategy_id)
        self.children: list[Strategy] = []
        for spec in self.params["children"]:
            cls = registry_get(spec["name"])
            child = cls(list(spec["symbols"]), dict(spec.get("params", {})), strategy_id=spec.get("id"))
            self.children.append(child)
        for c in self.children:
            for s in c.symbols:
                if s not in self.symbols:
                    self.symbols.append(s)
        self._weights: dict[str, float] = {c.id: 1.0 / max(len(self.children), 1) for c in self.children}
        self._child_ctx: dict[str, _ChildContext] = {}
        self._order_owner: dict[str, str] = {}
        self._child_equity: dict[str, deque] = {c.id: deque(maxlen=int(self.params["lookback"]) + 1) for c in self.children}
        self._last_px: dict[str, float] = {}
        self._bars_seen = 0
        self._ctx: Optional[StrategyContext] = None
        self.weight_history: list[tuple[pd.Timestamp, dict[str, float]]] = []

    # ---------------------------------------------------------------- weights

    def weight(self, child_id: str) -> float:
        return self._weights.get(child_id, 0.0)

    def _child_mtm(self, child: Strategy) -> float:
        cc = self._child_ctx[child.id]
        total = 0.0
        for sym, pos in cc.positions.items():
            px = self._last_px.get(sym)
            total += pos.realized_pnl + (pos.unrealized_pnl(px) if px is not None else 0.0)
        return total

    def _reestimate_weights(self, ctx: StrategyContext) -> None:
        series = {}
        for c in self.children:
            eq = np.asarray(self._child_equity[c.id], dtype=float)
            if len(eq) < 10:
                continue
            series[c.id] = pd.Series(np.diff(eq))
        if len(series) < 2 or len(series) < len(self.children):
            return  # keep equal weights until every child has history
        df = pd.DataFrame(series)
        df = df.loc[:, df.std() > 0]
        if df.shape[1] < 2:
            return
        try:
            w = target_weights(str(self.params["scheme"]), df, min_weight=float(self.params["min_weight"]), max_weight=float(self.params["max_weight"]), cash_buffer_pct=0.0)
        except Exception as exc:  # pragma: no cover - defensive: keep previous weights
            ctx.log("meta_allocator: weight estimation failed", error=str(exc))
            return
        total = float(w.sum()) or 1.0
        for cid in self._weights:
            self._weights[cid] = float(w.get(cid, 0.0)) / total if cid in w.index else 0.0
        self.weight_history.append((ctx.now, dict(self._weights)))
        ctx.log("meta_allocator weights", **{k: round(v, 3) for k, v in self._weights.items()})

    # ---------------------------------------------------------------- lifecycle

    def on_start(self, ctx: StrategyContext) -> None:
        self._ctx = ctx
        for c in self.children:
            self._child_ctx[c.id] = _ChildContext(self, c)
            c.on_start(self._child_ctx[c.id])

    def on_stop(self, ctx: StrategyContext) -> None:
        self._ctx = ctx
        for c in self.children:
            c.on_stop(self._child_ctx[c.id])

    def _fanout(self, event, ctx: StrategyContext, handler: str) -> None:
        self._ctx = ctx
        for c in self.children:
            if event.symbol in c.symbols:
                getattr(c, handler)(event, self._child_ctx[c.id])

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> None:
        self._last_px[bar.symbol] = bar.close
        self._fanout(bar, ctx, "on_bar")
        self._bars_seen += 1
        for c in self.children:
            self._child_equity[c.id].append(self._child_mtm(c))
        if self._bars_seen % int(self.params["rebalance_bars"]) == 0:
            self._reestimate_weights(ctx)

    def on_quote(self, quote: Quote, ctx: StrategyContext) -> None:
        self._last_px[quote.symbol] = quote.mid
        self._fanout(quote, ctx, "on_quote")

    def on_trade(self, trade: Trade, ctx: StrategyContext) -> None:
        self._fanout(trade, ctx, "on_trade")

    def on_book(self, book: OrderBook, ctx: StrategyContext) -> None:
        self._fanout(book, ctx, "on_book")

    def on_fill(self, fill: Fill, ctx: StrategyContext) -> None:
        self._ctx = ctx
        cid = self._order_owner.get(fill.client_order_id)
        if cid is None:
            return
        child = next((c for c in self.children if c.id == cid), None)
        if child is None:
            return
        cc = self._child_ctx[cid]
        pos = cc.positions.setdefault(fill.symbol, Position(fill.symbol))
        pos.apply_fill(fill)
        child.on_fill(fill, cc)

    def on_order_update(self, order: Order, ctx: StrategyContext) -> None:
        self._ctx = ctx
        cid = self._order_owner.get(order.client_order_id)
        child = next((c for c in self.children if c.id == cid), None)
        if child is not None:
            child.on_order_update(order, self._child_ctx[cid])

    def on_day_end(self, ctx: StrategyContext) -> None:
        self._ctx = ctx
        for c in self.children:
            c.on_day_end(self._child_ctx[c.id])


register(MetaAllocator)
