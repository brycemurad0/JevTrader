"""Avellaneda-Stoikov market maker with inventory skew, vol-scaled spread, microprice/imbalance
skew, inventory limits, an optional VPIN toxicity gate, and a HARD rule that the quoted half-
spread must exceed maker fee + adverse-selection estimate + minimum edge.

Model (Avellaneda & Stoikov 2008, with the usual practitioner simplifications):
    reservation price r = mid - q * gamma * sigma^2 * T                       (inventory skew)
    optimal spread    d = gamma * sigma^2 * T + (2/gamma) * ln(1 + gamma/k)    (vol-scaled)
    r += skew_coeff * (microprice - mid)                                       (flow skew)
    bid = r - d/2, ask = r + d/2, rounded to tick, post-only.
`sigma` is the rolling std of mid changes per second (from quotes), `T` the quoting horizon in
seconds, `q` the signed inventory in units of `quote_size`, `k` the order-arrival decay.

Fee reality (the reason this strategy is mostly a *negative* result on Alpaca crypto): the
maker fee is 15 bps per side. A market maker earns the half-spread on each fill and loses to
adverse selection (the mid moves against the filled side by ~0.5-1x the 1-min vol, i.e. 3-6 bps
on BTC). To be profitable the quoted half-spread must exceed 15 + ~5 = 20 bps => a 40 bps
quoted spread on BTC when the venue's own inside spread is ~1-3 bps: almost never at the touch,
fills only arrive when the market moves through you (toxic fills). The strategy enforces this
floor (`min_half_spread_bps` = maker_fee + `adverse_selection_bps` + `min_edge_bps`) rather than
pretending, so on Alpaca crypto tier 1 it will quote wide and fill rarely. On US equities (0
commission, ~0.3 bps reg fee) the same logic quotes at/inside a 1-2 bp spread and can work in
principle -- but the fill-queue model here is L1-only and real edge can only be validated on
recorded Alpaca quotes during paper trading (see research/record_replay.py).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

from jevtrader.core.fees import FeeModel, default_fees
from jevtrader.core.registry import register
from jevtrader.core.strategy import Strategy, StrategyContext, StrategySpec
from jevtrader.core.types import AssetClass, Fill, Liquidity, Order, OrderBook, OrderType, Quote, Side, TimeInForce, Trade
from jevtrader.strategies.flow_toxicity import FlowToxicityGate


class AvellanedaStoikovMM(Strategy):
    spec = StrategySpec(
        name="as_market_maker",
        description="Avellaneda-Stoikov maker with inventory/microprice skew, vol-scaled spread, inventory caps, VPIN gate and a fee-aware minimum half-spread.",
        asset_classes=("equity", "crypto"),
        frequency="book",
        style="market_making",
        uses_jev=False,
        default_params={
            "quote_size_notional": 200.0,  # USD per quote
            "max_inventory_units": 5,  # in quote_size units, each side
            "gamma": 0.1,  # risk aversion
            "k": 1.5,  # order arrival intensity decay
            "horizon_s": 60.0,
            "vol_window": 120,  # quotes used for sigma
            "skew_coeff": 0.5,  # microprice skew weight
            "adverse_selection_bps": 2.0,
            "min_edge_bps": 0.5,
            "requote_bps": 0.5,  # re-quote only when desired price moves > this (fewer cancels)
            "vpin_gate": False,
            "vpin_bucket_volume": 1.0,
            "vpin_threshold": 0.7,
            "max_quotes_per_minute": 30,
        },
        notes=(
            "HONEST VERDICT: on Alpaca crypto tier-1 (15 bps maker) the fee-aware floor forces a >= 35-40 bps "
            "quoted spread on BTC -> fills are rare and mostly adverse; NOT VIABLE until fees are <= ~3 bps/side "
            "(tier >= $25M/30d or another venue). On liquid US equities the mechanics are sound and costs allow "
            "it in principle; the edge is unproven here because we have no real L2 data -- validate with "
            "record_replay on paper-trading books before sizing up. Never run on a symbol without a live quote."
        ),
    )

    def __init__(self, symbols, params=None, strategy_id=None, fee_model: Optional[FeeModel] = None):
        super().__init__(symbols, params, strategy_id)
        self._fees = fee_model or default_fees()
        self._mids: dict[str, deque] = {s: deque(maxlen=int(self.params["vol_window"])) for s in self.symbols}
        self._last_ts: dict[str, pd.Timestamp] = {}
        self._orders: dict[str, dict[str, Order]] = {s: {} for s in self.symbols}  # symbol -> {"bid": o, "ask": o}
        self._gate = {s: FlowToxicityGate(float(self.params["vpin_bucket_volume"]), toxic_threshold=float(self.params["vpin_threshold"])) for s in self.symbols}
        self._quote_times: deque = deque(maxlen=int(self.params["max_quotes_per_minute"]))
        self.n_requotes = 0
        self.n_fills = 0

    # ---------------------------------------------------------------- helpers

    def maker_fee_bps(self, ctx: StrategyContext, symbol: str, price: float) -> float:
        inst = ctx.instrument(symbol)
        qty = max(1000.0 / price, inst.lot_size)
        return self._fees.fee(inst, Side.BUY, qty, price, Liquidity.MAKER) / (qty * price) * 1e4

    def min_half_spread_bps(self, ctx: StrategyContext, symbol: str, price: float) -> float:
        return self.maker_fee_bps(ctx, symbol, price) + float(self.params["adverse_selection_bps"]) + float(self.params["min_edge_bps"])

    def _sigma_per_s(self, symbol: str) -> Optional[float]:
        m = self._mids[symbol]
        if len(m) < 10:
            return None
        arr = np.asarray(m, dtype=float)
        d = np.diff(arr[:, 0])
        dt = np.diff(arr[:, 1])
        dt = np.where(dt <= 0, 1.0, dt)
        s = float(np.std(d / np.sqrt(dt), ddof=1))
        return s if s > 0 else None

    def _inventory_units(self, ctx: StrategyContext, symbol: str, mid: float) -> float:
        pos = ctx.position(symbol)
        unit_qty = float(self.params["quote_size_notional"]) / mid
        return pos.qty / unit_qty if unit_qty > 0 else 0.0

    def desired_quotes(self, ctx: StrategyContext, symbol: str, quote: Quote) -> Optional[tuple[float, float]]:
        """Compute (bid, ask) or None when we should not quote."""
        mid = quote.mid
        sigma = self._sigma_per_s(symbol)
        if mid <= 0 or sigma is None:
            return None
        p = self.params
        gamma, k, T = float(p["gamma"]), float(p["k"]), float(p["horizon_s"])
        q = self._inventory_units(ctx, symbol, mid)
        # work in relative (bps) space so gamma/k are price-scale free
        sig_rel = sigma / mid
        r = mid * (1.0 - q * gamma * sig_rel**2 * T)
        spread_rel = gamma * sig_rel**2 * T + (2.0 / gamma) * math.log(1.0 + gamma / k) * sig_rel * math.sqrt(T)
        r += float(p["skew_coeff"]) * (quote.microprice - mid)
        half = 0.5 * spread_rel * mid
        floor = self.min_half_spread_bps(ctx, symbol, mid) / 1e4 * mid
        half = max(half, floor)
        tick = ctx.instrument(symbol).tick_size or 0.01
        bid = math.floor((r - half) / tick) * tick
        ask = math.ceil((r + half) / tick) * tick
        # never cross the touch (post-only semantics)
        bid = min(bid, quote.bid)
        ask = max(ask, quote.ask)
        return bid, ask

    # ---------------------------------------------------------------- events

    def on_trade(self, trade: Trade, ctx: StrategyContext) -> None:
        self._gate[trade.symbol].on_trade(trade)

    def on_book(self, book: OrderBook, ctx: StrategyContext) -> None:
        q = book.to_quote()
        if q is not None:
            self.on_quote(q, ctx)

    def on_quote(self, quote: Quote, ctx: StrategyContext) -> None:
        sym = quote.symbol
        ts_s = quote.ts.timestamp()
        self._mids[sym].append((quote.mid, ts_s))
        if self.params.get("vpin_gate") and self._gate[sym].toxic:
            self._cancel_side(ctx, sym, "bid")
            self._cancel_side(ctx, sym, "ask")
            return
        target = self.desired_quotes(ctx, sym, quote)
        if target is None:
            return
        bid, ask = target
        inst = ctx.instrument(sym)
        mid = quote.mid
        unit_qty = float(self.params["quote_size_notional"]) / mid
        qty = math.floor(unit_qty / inst.lot_size) * inst.lot_size
        if qty <= 0:
            return
        q_units = self._inventory_units(ctx, sym, mid)
        max_inv = float(self.params["max_inventory_units"])
        pos = ctx.position(sym)
        want_bid = q_units < max_inv
        want_ask = q_units > -max_inv and (inst.shortable or pos.qty >= qty - 1e-12)
        ask_qty = qty if inst.shortable else min(qty, max(pos.qty, 0.0))
        self._maintain(ctx, sym, "bid", Side.BUY, bid, qty if want_bid else 0.0, mid)
        self._maintain(ctx, sym, "ask", Side.SELL, ask, ask_qty if want_ask else 0.0, mid)

    def _cancel_side(self, ctx: StrategyContext, sym: str, side_key: str) -> None:
        o = self._orders[sym].pop(side_key, None)
        if o is not None and o.status.is_open:
            ctx.cancel(o.client_order_id)

    def _throttled(self, now: pd.Timestamp) -> bool:
        limit = int(self.params["max_quotes_per_minute"])
        if len(self._quote_times) < limit:
            return False
        return (now - self._quote_times[0]) < pd.Timedelta(minutes=1)

    def _maintain(self, ctx: StrategyContext, sym: str, key: str, side: Side, price: float, qty: float, mid: float) -> None:
        cur = self._orders[sym].get(key)
        if cur is not None and not cur.status.is_open:
            self._orders[sym].pop(key, None)
            cur = None
        if qty <= 0:
            if cur is not None:
                self._cancel_side(ctx, sym, key)
            return
        if cur is not None:
            moved_bps = abs(cur.limit_price - price) / mid * 1e4
            if moved_bps < float(self.params["requote_bps"]):
                return
            if self._throttled(ctx.now):
                return
            self._cancel_side(ctx, sym, key)
        elif self._throttled(ctx.now):
            return
        order = Order(sym, side, qty, OrderType.LIMIT, limit_price=price, tif=TimeInForce.GTC, post_only=True, strategy_id=self.id, tag=f"mm_{key}")
        if ctx.submit(order) is not None:
            self._orders[sym][key] = order
            self._quote_times.append(ctx.now)
            self.n_requotes += 1

    def on_fill(self, fill: Fill, ctx: StrategyContext) -> None:
        self.n_fills += 1

    def on_stop(self, ctx: StrategyContext) -> None:
        ctx.cancel_all()


register(AvellanedaStoikovMM)
