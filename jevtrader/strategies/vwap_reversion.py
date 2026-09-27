"""Intraday mean reversion to rolling VWAP (Bollinger z-score) with a volatility-regime filter.

Signal (identical to `jevtrader.research.signals.zscore_features`): z = (close - rolling
VWAP_w) / rolling std_w(close). Enter long when z < -entry_z, short (if shortable) when z >
+entry_z, exit when |z| < exit_z or after `hold_max` bars. Entries are suppressed when the
short-window realized vol exceeds `max_vol_mult` x the long-window vol -- the regime where
"cheap" keeps getting cheaper.

Why it might work: at 5-60 min horizons, liquid equities show negative return autocorrelation
(inventory effects of intermediaries; overreaction to order-flow shocks) -- the research
harness measured ac(1) of -0.03..-0.12 on SPX500/C 5-60 min bars. Why it fails: news/trend
regimes (hence the vol filter) and, above all, costs: the gross edge is only a few bps per
round trip, so it only survives sub-2 bps round-trip costs (liquid US equities), NOT Alpaca
crypto's 30-50 bps.
"""

from __future__ import annotations

from typing import Optional

from jevtrader.core.registry import register
from jevtrader.core.strategy import StrategyContext, StrategySpec
from jevtrader.core.types import Bar, Side
from jevtrader.research.signals import zscore_features
from jevtrader.strategies._base import BarStrategy


class ZScoreReversion(BarStrategy):
    spec = StrategySpec(
        name="zscore_reversion",
        description="Intraday mean reversion to rolling VWAP via Bollinger z-score, with a vol-regime filter and optional Jev gate.",
        asset_classes=("equity", "crypto"),
        frequency="5min",
        style="mean_reversion",
        uses_jev=True,
        default_params={
            **BarStrategy.BASE_PARAMS,
            "bar_minutes": 15,
            "window": 30,
            "entry_z": 2.0,
            "exit_z": 0.0,
            "vol_window": 120,
            "max_vol_mult": 1.5,
            "hold_max": None,
            "alloc_frac": 0.25,
            "session_only": True,  # equities: trade only inside 13:30-20:00 UTC and flatten at the close
        },
        notes=(
            "EVIDENCE (research_reports/02_intraday_mean_reversion.md): equities at 5-15 min -- positive "
            "net edge on SPX500-proxy and C in-sample; out-of-sample validation is the deciding evidence, "
            "see the report for the verdict. Crypto (BTC on Alpaca): gross edge 2-15 bps per round trip vs "
            "30-50 bps cost -> NOT VIABLE at tier-1 fees; break-even fee ~1-7 bps/side. Few parameters "
            "(window, entry_z, exit_z); neighbours of the chosen params must also be positive."
        ),
    )

    def __init__(self, symbols, params=None, strategy_id=None, **kw):
        super().__init__(symbols, params, strategy_id, **kw)
        self._held: dict[str, int] = {}

    def _in_session(self, bar: Bar, ctx: StrategyContext) -> bool:
        if not self.params.get("session_only"):
            return True
        if ctx.instrument(bar.symbol).asset_class.value != "equity":
            return True
        m = bar.ts.hour * 60 + bar.ts.minute
        return (13 * 60 + 30) < m <= (20 * 60)

    def on_agg_bar(self, bar: Bar, ctx: StrategyContext) -> None:
        sym = bar.symbol
        p = self.params
        w, vw = int(p["window"]), int(p["vol_window"])
        hist = self.hist_df(sym, max(w, vw) + 2)
        pos = ctx.position(sym)
        in_pos = abs(pos.qty) > 0
        if not self._in_session(bar, ctx):
            if in_pos:
                self.flatten(ctx, sym, bar.close)
            return
        if len(hist) < w + 1:
            return
        z_s, ok_s = zscore_features(hist, w, vw, p.get("max_vol_mult"))
        z = float(z_s.iloc[-1]) if z_s.notna().iloc[-1] else None
        ok = bool(ok_s.iloc[-1])
        if z is None:
            return
        entry, exit_ = float(p["entry_z"]), float(p["exit_z"])
        hold_max: Optional[int] = p.get("hold_max")
        if in_pos:
            self._held[sym] = self._held.get(sym, 0) + 1
            long = pos.qty > 0
            done = (long and z > -exit_) or ((not long) and z < exit_) or (hold_max is not None and self._held[sym] >= int(hold_max))
            if done:
                self.flatten(ctx, sym, bar.close)
                self._held[sym] = 0
            return
        if not ok:
            return
        side: Optional[Side] = None
        if z < -entry:
            side = Side.BUY
        elif z > entry and ctx.instrument(sym).shortable:
            side = Side.SELL
        if side is None:
            return
        allowed, scale = self.jev_gate(ctx, sym, side, hist, bar.close)
        if not allowed:
            return
        frac = float(p["alloc_frac"]) * scale * (1.0 if side is Side.BUY else -1.0)
        if self._set_target(ctx, sym, frac, bar.close, tag="entry") is not None:
            self._held[sym] = 0


register(ZScoreReversion)
