"""Opening-range / session breakout.

Equities: the range is the high/low of the first `range_minutes` after the 13:30 UTC (09:30 ET)
open; go long on a close above the range high (short below the low), one trade per session,
flat after `hold_minutes` or at the session close. Crypto: same mechanics anchored at a session
open of your choice (00:00 Asia, 07:00 EU, 13:30 US, in UTC).

Why it might work: overnight information gets priced in a burst of directional volume in the
first half hour; a break of that range signals which side won. Why it fails: false breaks in
low-range days (hence `min_range_bps`), and on 24/7 crypto there is no real "open" -- the
research harness found every session anchor NEGATIVE gross on BTC. On equities the edge is
small (about +1 bp/round trip on SPX500 proxy in-sample) -- honest verdict: NOT VIABLE / research.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from jevtrader.core.registry import register
from jevtrader.core.strategy import StrategyContext, StrategySpec
from jevtrader.core.types import Bar, Side
from jevtrader.strategies._base import BarStrategy


class RangeBreakout(BarStrategy):
    spec = StrategySpec(
        name="range_breakout",
        description="Opening-range breakout (equities) / session-open breakout (crypto) on 1-min bars, one trade per session.",
        asset_classes=("equity", "crypto"),
        frequency="1min",
        style="breakout",
        uses_jev=True,
        default_params={
            **BarStrategy.BASE_PARAMS,
            "bar_minutes": 1,
            "session_start_utc": 13,
            "session_start_minute": 30,
            "range_minutes": 30,
            "hold_minutes": 180,
            "min_range_bps": 0.0,
            "alloc_frac": 0.25,
        },
        notes=(
            "EVIDENCE (research_reports/04_breakouts.md): BTC session breakouts (Asia/EU/US anchors) all "
            "negative gross in-sample -> NOT VIABLE. Equities ORB: roughly zero net edge on the SPX500 proxy "
            "and negative on C except one parameter set (range 60 / hold 300) -> treat as NOT VIABLE / "
            "research; kept for completeness and as a Jev-gate testbed (the gate can only remove trades)."
        ),
    )

    def __init__(self, symbols, params=None, strategy_id=None, **kw):
        super().__init__(symbols, params, strategy_id, **kw)
        self._day: dict[str, object] = {}
        self._hi: dict[str, float] = {}
        self._lo: dict[str, float] = {}
        self._traded: dict[str, bool] = {}
        self._entry_ts: dict[str, Optional[pd.Timestamp]] = {}

    def on_agg_bar(self, bar: Bar, ctx: StrategyContext) -> None:
        sym = bar.symbol
        p = self.params
        start_m = int(p["session_start_utc"]) * 60 + int(p["session_start_minute"])
        m = bar.ts.hour * 60 + bar.ts.minute
        rel = (m - start_m) % 1440
        day = (bar.ts - pd.Timedelta(minutes=start_m)).date()
        if self._day.get(sym) != day:
            self._day[sym] = day
            self._hi[sym], self._lo[sym] = float("-inf"), float("inf")
            self._traded[sym] = False
        rng_min, hold = int(p["range_minutes"]), int(p["hold_minutes"])
        pos = ctx.position(sym)
        if rel < rng_min:
            self._hi[sym] = max(self._hi[sym], bar.high)
            self._lo[sym] = min(self._lo[sym], bar.low)
            if abs(pos.qty) > 0:
                self.flatten(ctx, sym, bar.close)
            return
        if rel >= rng_min + hold:
            if abs(pos.qty) > 0:
                self.flatten(ctx, sym, bar.close)
            return
        if abs(pos.qty) > 0 or self._traded.get(sym):
            return
        hi, lo = self._hi.get(sym, float("-inf")), self._lo.get(sym, float("inf"))
        if hi == float("-inf") or lo == float("inf"):
            return
        if 1e4 * (hi - lo) / bar.close < float(p["min_range_bps"]):
            return
        side: Optional[Side] = None
        if bar.close > hi:
            side = Side.BUY
        elif bar.close < lo and ctx.instrument(sym).shortable:
            side = Side.SELL
        if side is None:
            return
        self._traded[sym] = True
        hist = self.hist_df(sym, 120)
        allowed, scale = self.jev_gate(ctx, sym, side, hist, bar.close)
        if not allowed:
            return
        self._set_target(ctx, sym, float(p["alloc_frac"]) * scale * (1.0 if side is Side.BUY else -1.0), bar.close, tag="entry")


register(RangeBreakout)
