"""Order-book imbalance / microprice short-horizon alpha.

Signal: an EW-smoothed order-book imbalance I = (bid_vol - ask_vol)/(bid_vol + ask_vol) over the
top `depth` levels (or the L1 sizes when only quotes arrive), combined with the microprice offset
(microprice - mid). Predicted move over `hold_ticks` ticks, in bps: `beta_bps * I`. Decision:
  * |predicted| > taker round-trip cost (fee + spread): TAKE (IOC at the touch).
  * threshold_join < |I| but predicted < cost: JOIN the queue as a maker at the touch on the
    favoured side (post-only) -- earn the spread and the expected drift instead of paying it.
  * exit after `hold_ticks` quotes or when the signal flips.

`beta_bps` is *calibrated online* from realized (imbalance, forward return) pairs with a
recursive least squares fit when `calibrate=True`, so the strategy adapts to the venue's actual
IC instead of a hard-coded guess -- and refuses to take when the fitted beta is not significant.

Why it might work: L2 imbalance is the best-documented short-horizon predictor in the
microstructure literature (Cont, Kukanov & Stoikov 2014; IC of a few percent at 1-10 s).
Why it fails at retail: the edge is ~0.1-0.5 bps per event -- below the taker cost on any
venue and orders of magnitude below Alpaca crypto's 25 bps; the maker variant then lives or
dies on queue position, which a tens-of-ms REST/WS client loses to co-located makers.
Validation: MECHANICS only, on `jevtrader.data.synthetic.generate_quote_trade_stream` (which
embeds a weak, genuine lagged-OFI edge). Real edge can only be measured on recorded Alpaca
books during paper trading.
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
from jevtrader.core.types import Fill, Liquidity, Order, OrderBook, OrderType, Quote, Side, TimeInForce, Trade
from jevtrader.strategies.flow_toxicity import FlowToxicityGate


class ImbalanceAlpha(Strategy):
    spec = StrategySpec(
        name="imbalance_alpha",
        description="Order-book imbalance / microprice alpha: take when predicted move > taker cost, otherwise join as maker; online-calibrated beta.",
        asset_classes=("equity", "crypto"),
        frequency="book",
        style="microstructure",
        uses_jev=True,
        default_params={
            "depth": 5,
            "ema_alpha": 0.3,
            "hold_ticks": 10,
            "beta_bps": 2.0,  # prior: bps of move per unit imbalance over hold_ticks
            "calibrate": True,
            "min_t_stat": 2.0,
            "threshold_join": 0.3,
            "spread_bps_assumed": 2.0,
            "notional": 500.0,
            "max_position_notional": 2000.0,
            "vpin_gate": False,
            "vpin_bucket_volume": 1.0,
            "vpin_threshold": 0.7,
            "jev_gate": False,
            "jev_min_p": 0.55,
        },
        notes=(
            "MECHANICS validated on synthetic L1/L2 streams with a known weak OFI edge (the strategy recovers a "
            "positive fitted beta and is profitable at zero fees, unprofitable at Alpaca crypto fees). Real edge "
            "unproven: record real books in paper trading (research/record_replay.py) and replay. Retail latency "
            "means the TAKE branch will almost never clear cost honestly; the JOIN branch is the realistic one."
        ),
    )

    def __init__(self, symbols, params=None, strategy_id=None, fee_model: Optional[FeeModel] = None):
        super().__init__(symbols, params, strategy_id)
        self._fees = fee_model or default_fees()
        self._imb: dict[str, float] = {}
        self._mids: dict[str, deque] = {s: deque(maxlen=int(self.params["hold_ticks"]) + 1) for s in self.symbols}
        self._imb_hist: dict[str, deque] = {s: deque(maxlen=int(self.params["hold_ticks"]) + 1) for s in self.symbols}
        # RLS state for beta: sums for a 1-parameter no-intercept regression y = beta * x
        self._sxx: dict[str, float] = {s: 0.0 for s in self.symbols}
        self._sxy: dict[str, float] = {s: 0.0 for s in self.symbols}
        self._syy: dict[str, float] = {s: 0.0 for s in self.symbols}
        self._n: dict[str, int] = {s: 0 for s in self.symbols}
        self._entry_tick: dict[str, int] = {}
        self._tick: dict[str, int] = {s: 0 for s in self.symbols}
        self._resting: dict[str, Optional[Order]] = {s: None for s in self.symbols}
        self._gate = {s: FlowToxicityGate(float(self.params["vpin_bucket_volume"]), toxic_threshold=float(self.params["vpin_threshold"])) for s in self.symbols}
        self.n_takes = 0
        self.n_joins = 0

    # ---------------------------------------------------------------- signal + calibration

    def fitted_beta(self, symbol: str) -> tuple[float, float]:
        """(beta_bps, t_stat) of the online regression of forward return (bps) on imbalance."""
        n, sxx, sxy, syy = self._n[symbol], self._sxx[symbol], self._sxy[symbol], self._syy[symbol]
        if n < 30 or sxx <= 0:
            return float(self.params["beta_bps"]), 0.0
        beta = sxy / sxx
        resid_var = max(syy - beta * sxy, 1e-12) / max(n - 1, 1)
        t = beta / math.sqrt(resid_var / sxx)
        return beta, t

    def _update(self, symbol: str, imbalance: float, mid: float) -> None:
        a = float(self.params["ema_alpha"])
        prev = self._imb.get(symbol)
        self._imb[symbol] = imbalance if prev is None else (1 - a) * prev + a * imbalance
        self._mids[symbol].append(mid)
        self._imb_hist[symbol].append(self._imb[symbol])
        h = int(self.params["hold_ticks"])
        if self.params.get("calibrate") and len(self._mids[symbol]) > h:
            x = self._imb_hist[symbol][0]
            y = 1e4 * (mid / self._mids[symbol][0] - 1.0)
            # exponential forgetting keeps the fit adaptive
            lam = 0.999
            self._sxx[symbol] = lam * self._sxx[symbol] + x * x
            self._sxy[symbol] = lam * self._sxy[symbol] + x * y
            self._syy[symbol] = lam * self._syy[symbol] + y * y
            self._n[symbol] += 1

    def taker_cost_bps(self, ctx: StrategyContext, symbol: str, price: float, spread_bps: float) -> float:
        inst = ctx.instrument(symbol)
        qty = max(100.0 / price, inst.lot_size)
        fee = self._fees.round_trip_bps(inst, price, qty, maker=False)
        return fee + spread_bps

    # ---------------------------------------------------------------- events

    def on_trade(self, trade: Trade, ctx: StrategyContext) -> None:
        self._gate[trade.symbol].on_trade(trade)

    def on_book(self, book: OrderBook, ctx: StrategyContext) -> None:
        q = book.to_quote()
        if q is None:
            return
        self._step(q, book.imbalance(int(self.params["depth"])), ctx)

    def on_quote(self, quote: Quote, ctx: StrategyContext) -> None:
        if ctx.last_book(quote.symbol) is not None:
            return  # book handler drives when L2 is available
        tot = quote.bid_size + quote.ask_size
        imb = (quote.bid_size - quote.ask_size) / tot if tot > 0 else 0.0
        self._step(quote, imb, ctx)

    def _step(self, quote: Quote, imbalance: float, ctx: StrategyContext) -> None:
        sym = quote.symbol
        self._tick[sym] += 1
        mid = quote.mid
        if mid <= 0:
            return
        self._update(sym, imbalance, mid)
        pos = ctx.position(sym)
        sig = self._imb[sym]
        h = int(self.params["hold_ticks"])
        # exits
        if abs(pos.qty) > 0:
            aged = self._tick[sym] - self._entry_tick.get(sym, self._tick[sym]) >= h
            flipped = (pos.qty > 0 and sig < 0) or (pos.qty < 0 and sig > 0)
            if aged or flipped:
                self._cancel_resting(ctx, sym)
                side = Side.SELL if pos.qty > 0 else Side.BUY
                ctx.submit(Order(sym, side, abs(pos.qty), OrderType.MARKET, tif=TimeInForce.IOC, reduce_only=True, strategy_id=self.id, tag="exit"))
            return
        if self.params.get("vpin_gate") and self._gate[sym].toxic:
            self._cancel_resting(ctx, sym)
            return
        beta, t = self.fitted_beta(sym)
        if self.params.get("calibrate") and self._n[sym] >= 30 and t < float(self.params["min_t_stat"]):
            self._cancel_resting(ctx, sym)
            return  # no significant edge measured on this venue -> do nothing
        pred_bps = beta * sig
        spread_bps = 1e4 * quote.spread / mid if quote.spread > 0 else float(self.params["spread_bps_assumed"])
        cost = self.taker_cost_bps(ctx, sym, mid, spread_bps)
        inst = ctx.instrument(sym)
        qty = math.floor(float(self.params["notional"]) / mid / inst.lot_size) * inst.lot_size
        if qty <= 0:
            return
        side = Side.BUY if pred_bps > 0 else Side.SELL
        if side is Side.SELL and not inst.shortable:
            self._cancel_resting(ctx, sym)
            return
        if not self._jev_ok(ctx, sym, side, mid, cost):
            return
        if abs(pred_bps) > cost:
            self._cancel_resting(ctx, sym)
            o = Order(sym, side, qty, OrderType.MARKET, tif=TimeInForce.IOC, strategy_id=self.id, tag="take")
            if ctx.submit(o) is not None:
                self.n_takes += 1
                self._entry_tick[sym] = self._tick[sym]
        elif abs(sig) > float(self.params["threshold_join"]):
            price = quote.bid if side is Side.BUY else quote.ask
            resting = self._resting.get(sym)
            if resting is not None and resting.status.is_open and resting.side is side and abs(resting.limit_price - price) < 1e-12:
                return
            self._cancel_resting(ctx, sym)
            o = Order(sym, side, qty, OrderType.LIMIT, limit_price=price, tif=TimeInForce.GTC, post_only=True, strategy_id=self.id, tag="join")
            if ctx.submit(o) is not None:
                self._resting[sym] = o
                self.n_joins += 1
        else:
            self._cancel_resting(ctx, sym)

    def _jev_ok(self, ctx: StrategyContext, sym: str, side: Side, mid: float, cost: float) -> bool:
        if not self.params.get("jev_gate") or ctx.jev is None:
            return True
        bars = ctx.bars(sym, 60)
        if bars is None or len(bars) < 5:
            return True
        from jevtrader.jev.state import build_state

        state = build_state(sym, bars, quote=ctx.last_quote(sym), book=ctx.last_book(sym), position=ctx.position(sym), cost_bps=cost, now=ctx.now)
        view = ctx.jev.direction(sym, state, "5min")
        if view is None or view.late:
            return False
        p = view.p("direction", "up" if side is Side.BUY else "down")
        return p >= float(self.params["jev_min_p"])

    def _cancel_resting(self, ctx: StrategyContext, sym: str) -> None:
        o = self._resting.get(sym)
        if o is not None and o.status.is_open:
            ctx.cancel(o.client_order_id)
        self._resting[sym] = None

    def on_fill(self, fill: Fill, ctx: StrategyContext) -> None:
        if fill.liquidity is Liquidity.MAKER:
            self._entry_tick[fill.symbol] = self._tick[fill.symbol]

    def on_stop(self, ctx: StrategyContext) -> None:
        ctx.cancel_all()


register(ImbalanceAlpha)
