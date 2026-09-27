"""Shared plumbing for bar-driven strategies: N-minute bar aggregation from a 1-minute feed,
fraction-of-equity sizing that respects lot size / long-only instruments, a round-trip cost
estimate (fees + assumed spread) and the optional Jev gate ("Jev judges, code executes").

Nothing here places orders on its own; concrete strategies call `_set_target()` with a target
position expressed as a fraction of equity, exactly the unit the research fast-screen uses
(`jevtrader.research.fastscreen.simulate_positions`), so fast-path and event-path results are
comparable one-for-one.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from jevtrader.core.fees import FeeModel, default_fees
from jevtrader.core.strategy import Strategy, StrategyContext
from jevtrader.core.types import AssetClass, Bar, Order, OrderType, Side, TimeInForce
from jevtrader.risk.sizing import kelly_from_jev, lot_round

# Conservative assumed spread+slippage per side when no quote is available, by asset class
# (matches jevtrader.research.fastscreen ALPACA_* CostSpecs).
ASSUMED_HALF_SPREAD_BPS = {AssetClass.CRYPTO: 3.5, AssetClass.EQUITY: 1.5}

BASE_JEV_PARAMS: dict[str, Any] = {
    "jev_gate": False,  # ask ctx.jev.direction(...) before every ENTRY; falls back to base when ctx.jev is None
    "jev_min_p": 0.55,  # required P(move in our direction > cost) to enter
    "jev_horizon": "30min",
    "jev_kelly_fraction": 0.5,  # fractional Kelly applied to jev probabilities for sizing
    "jev_regime_veto": True,  # also ask regime(); veto entries when risk_off=yes
}


@dataclass
class _Agg:
    bucket_end: Optional[pd.Timestamp] = None
    open: float = 0.0
    high: float = -np.inf
    low: float = np.inf
    close: float = 0.0
    volume: float = 0.0


class BarAggregator:
    """Aggregates bar-close-stamped bars into `minutes`-minute bars, closing on the right
    (10:01..10:05 -> 10:05), identical to `research.loaders.resample(..., label="right",
    closed="right")`. If the feed already arrives at that cadence it passes bars through."""

    def __init__(self, minutes: int) -> None:
        self.minutes = max(int(minutes), 1)
        self._state: dict[str, _Agg] = {}

    def _bucket_end(self, ts: pd.Timestamp) -> pd.Timestamp:
        return ts.ceil(f"{self.minutes}min")

    def push(self, bar: Bar) -> list[Bar]:
        """Feed one bar; returns the list of completed aggregated bars (0, 1 or 2 -- two when a
        gap in the feed left a bucket open that this bar closes implicitly)."""
        if self.minutes == 1:
            return [bar]
        end = self._bucket_end(bar.ts)
        out: list[Bar] = []
        st = self._state.get(bar.symbol)
        if st is not None and st.bucket_end is not None and end > st.bucket_end:
            out.append(Bar(bar.symbol, st.bucket_end, st.open, st.high, st.low, st.close, st.volume))
            st = None
        if st is None:
            st = _Agg(bucket_end=end, open=bar.open, high=bar.high, low=bar.low, close=bar.close, volume=bar.volume)
            self._state[bar.symbol] = st
        else:
            st.high = max(st.high, bar.high)
            st.low = min(st.low, bar.low)
            st.close = bar.close
            st.volume += bar.volume
        if bar.ts == end:
            self._state.pop(bar.symbol, None)
            out.append(Bar(bar.symbol, end, st.open, st.high, st.low, st.close, st.volume))
        return out


class BarStrategy(Strategy):
    """Base for bar strategies. Subclasses implement `on_agg_bar(bar, hist_df, ctx)` and call
    `self._set_target(ctx, symbol, frac, price)`.

    Common params (merge `BASE_PARAMS` into your spec.default_params):
        bar_minutes: aggregate the incoming feed to this cadence (1 = native).
        alloc_frac: |target position| as a fraction of account equity when fully invested.
        history: number of aggregated bars retained for indicators.
        entry_style: "market" (fills next bar open, TAKER) or "limit" (post a limit at the
            close price, MAKER fill only if the next bar trades through it).
        min_delta_frac: skip re-targeting when |change| < this fraction of equity (kills churn).
        + the BASE_JEV_PARAMS jev gate.
    """

    BASE_PARAMS: dict[str, Any] = {
        "bar_minutes": 5,
        "alloc_frac": 0.25,
        "history": 400,
        "entry_style": "market",
        "min_delta_frac": 0.02,
        **BASE_JEV_PARAMS,
    }

    def __init__(self, symbols, params=None, strategy_id=None, fee_model: Optional[FeeModel] = None):
        super().__init__(symbols, params, strategy_id)
        self._agg = BarAggregator(int(self.params.get("bar_minutes", 1)))
        self._hist: dict[str, deque] = {s: deque(maxlen=int(self.params.get("history", 400))) for s in self.symbols}
        self._fee_model = fee_model or default_fees()
        self._target_frac: dict[str, float] = {s: 0.0 for s in self.symbols}
        self.jev_holds = 0  # count of entries suppressed by the jev gate (for reports/tests)

    # ---------------------------------------------------------------- data

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> None:
        for agg in self._agg.push(bar):
            if bar.symbol not in self._hist:
                self._hist[bar.symbol] = deque(maxlen=int(self.params.get("history", 400)))
                self._target_frac.setdefault(bar.symbol, 0.0)
            self._hist[bar.symbol].append((agg.ts, agg.open, agg.high, agg.low, agg.close, agg.volume))
            self.on_agg_bar(agg, ctx)

    def hist_df(self, symbol: str, n: Optional[int] = None) -> pd.DataFrame:
        rows = list(self._hist.get(symbol, ()))
        if n is not None:
            rows = rows[-n:]
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        arr = np.array([r[1:] for r in rows], dtype=float)
        df = pd.DataFrame(arr, columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex([r[0] for r in rows], name="ts"))
        return df

    def on_agg_bar(self, bar: Bar, ctx: StrategyContext) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    # ---------------------------------------------------------------- costs

    def round_trip_cost_bps(self, ctx: StrategyContext, symbol: str, price: float, maker: bool = False) -> float:
        inst = ctx.instrument(symbol)
        qty = max(1000.0 / price, inst.lot_size)
        fee = self._fee_model.round_trip_bps(inst, price, qty, maker=maker)
        q = ctx.last_quote(symbol)
        if q is not None and q.mid > 0 and q.spread >= 0:
            spread_bps = 1e4 * q.spread / q.mid
        else:
            spread_bps = 2.0 * ASSUMED_HALF_SPREAD_BPS[inst.asset_class]
        return fee + (0.0 if maker else spread_bps)

    # ---------------------------------------------------------------- jev gate

    def jev_gate(self, ctx: StrategyContext, symbol: str, side: Side, hist: pd.DataFrame, price: float) -> tuple[bool, float]:
        """Returns (allowed, size_scale). Base behaviour when the gate is off or `ctx.jev` is
        None: (True, 1.0). A `None` or `late` view is a HOLD (False, 0.0)."""
        if not self.params.get("jev_gate") or ctx.jev is None or hist.empty:
            return True, 1.0
        from jevtrader.jev.state import build_state

        cost = self.round_trip_cost_bps(ctx, symbol, price)
        state = build_state(symbol, hist, quote=ctx.last_quote(symbol), book=ctx.last_book(symbol), position=ctx.position(symbol), cost_bps=cost, now=ctx.now)
        view = ctx.jev.direction(symbol, state, str(self.params.get("jev_horizon", "30min")))
        if view is None or view.late:
            self.jev_holds += 1
            return False, 0.0
        p_up, p_down = view.p("direction", "up"), view.p("direction", "down")
        p_fav = p_up if side is Side.BUY else p_down
        if p_fav < float(self.params.get("jev_min_p", 0.55)):
            self.jev_holds += 1
            return False, 0.0
        if self.params.get("jev_regime_veto"):
            rv = ctx.jev.regime(symbol, state)
            if rv is not None and not rv.late and rv.p("risk_off", "yes") > 0.5:
                self.jev_holds += 1
                return False, 0.0
        # Expected move over the horizon ~ realized bar vol * sqrt(bars in horizon); Kelly-size it.
        vol_bps = float(state.get("vol_bps") or 0.0)
        bars_in_h = max(pd.Timedelta(str(self.params.get("jev_horizon", "30min"))) / pd.Timedelta(minutes=self._agg.minutes), 1.0)
        move = max(vol_bps * float(np.sqrt(bars_in_h)), cost + 1.0)
        alloc = float(self.params.get("alloc_frac", 0.25))
        if side is Side.BUY:
            f = kelly_from_jev(p_up, p_down, move, move, cost, fraction=float(self.params.get("jev_kelly_fraction", 0.5)), cap_pct_equity=alloc)
        else:
            f = kelly_from_jev(p_down, p_up, move, move, cost, fraction=float(self.params.get("jev_kelly_fraction", 0.5)), cap_pct_equity=alloc)
        if f <= 0:
            self.jev_holds += 1
            return False, 0.0
        return True, float(min(1.0, f / alloc))

    # ---------------------------------------------------------------- execution

    def _set_target(self, ctx: StrategyContext, symbol: str, frac: float, price: float, tag: str = "") -> Optional[str]:
        """Move the position in `symbol` to `frac` x equity (signed). Long-only instruments clip
        at 0. Returns the client_order_id of the order sent (or None if nothing to do / rejected)."""
        inst = ctx.instrument(symbol)
        if not inst.shortable:
            frac = max(frac, 0.0)
        equity = max(ctx.account().equity, 0.0)
        if price <= 0 or equity <= 0:
            return None
        pos = ctx.position(symbol)
        target_qty = frac * equity / price
        current = pos.qty
        delta = target_qty - current
        if abs(delta) * price < max(inst.min_notional, float(self.params.get("min_delta_frac", 0.0)) * equity):
            self._target_frac[symbol] = frac
            return None
        # cancel any resting order of ours on this symbol before re-targeting
        for o in ctx.open_orders(symbol):
            ctx.cancel(o.client_order_id)
        side = Side.BUY if delta > 0 else Side.SELL
        qty = abs(lot_round(delta, inst))
        if qty <= 0:
            return None
        if side is Side.SELL and not inst.shortable:
            qty = min(qty, max(current, 0.0))
            if qty <= 0:
                return None
        if side is Side.BUY:
            cash_cap = ctx.account().buying_power / price * 0.995
            qty = min(qty, abs(lot_round(cash_cap, inst)))
            if qty <= 0:
                return None
        reduce_only = (current > 0 and side is Side.SELL and qty <= current + 1e-12) or (current < 0 and side is Side.BUY and qty <= -current + 1e-12)
        style = str(self.params.get("entry_style", "market"))
        if style == "limit" and not reduce_only:
            order = Order(symbol, side, qty, OrderType.LIMIT, limit_price=price, tif=TimeInForce.GTC, strategy_id=self.id, tag=tag or "entry")
        else:
            order = Order(symbol, side, qty, OrderType.MARKET, tif=TimeInForce.IOC, reduce_only=reduce_only, strategy_id=self.id, tag=tag or ("exit" if reduce_only else "entry"))
        oid = ctx.submit(order)
        if oid is not None:
            self._target_frac[symbol] = frac
        return oid

    def flatten(self, ctx: StrategyContext, symbol: str, price: float) -> None:
        self._set_target(ctx, symbol, 0.0, price, tag="exit")

    def on_stop(self, ctx: StrategyContext) -> None:
        # never leave resting orders behind
        try:
            ctx.cancel_all()
        except Exception:  # pragma: no cover - defensive
            pass
