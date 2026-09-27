"""Volatility squeeze / breakout: Bollinger bands inside Keltner channel ("squeeze") followed by a
release; trade the direction of `bb_window`-bar momentum on the release bar, hold `hold` bars.

Why it might work: volatility clusters; a compression period is often followed by an expansion,
and the first move of the expansion carries direction information. Why it fails: the *direction*
is the hard part -- the expansion is real but its sign is close to a coin flip after costs. The
research harness found negative gross edges on BTC at every horizon and on SPX500/C 5-min bars;
verdict NOT VIABLE. Kept as a documented negative result.
"""

from __future__ import annotations

import numpy as np

from jevtrader.core.registry import register
from jevtrader.core.strategy import StrategyContext, StrategySpec
from jevtrader.core.types import Bar, Side
from jevtrader.strategies._base import BarStrategy


class VolSqueeze(BarStrategy):
    spec = StrategySpec(
        name="vol_squeeze",
        description="Bollinger-inside-Keltner squeeze release, traded in the direction of momentum for a fixed hold.",
        asset_classes=("equity", "crypto"),
        frequency="5min",
        style="breakout",
        uses_jev=True,
        default_params={**BarStrategy.BASE_PARAMS, "bar_minutes": 5, "bb_window": 20, "kc_mult": 1.5, "hold": 24, "alloc_frac": 0.25},
        notes=(
            "EVIDENCE (research_reports/04_breakouts.md): negative gross edge in-sample on BTC (5/15/60 min) "
            "and on SPX500/C 5-min -> NOT VIABLE. Included so the negative result is reproducible."
        ),
    )

    def __init__(self, symbols, params=None, strategy_id=None, **kw):
        super().__init__(symbols, params, strategy_id, **kw)
        self._prev_squeeze: dict[str, bool] = {}
        self._left: dict[str, int] = {}

    def on_agg_bar(self, bar: Bar, ctx: StrategyContext) -> None:
        sym = bar.symbol
        p = self.params
        w = int(p["bb_window"])
        hist = self.hist_df(sym, w + 2)
        pos = ctx.position(sym)
        if abs(pos.qty) > 0:
            self._left[sym] = self._left.get(sym, 0) - 1
            if self._left[sym] <= 0:
                self.flatten(ctx, sym, bar.close)
        if len(hist) < w + 1:
            return
        close = hist["close"]
        sd = float(close.iloc[-w:].std())
        prev_close = close.shift()
        tr = np.maximum.reduce([(hist["high"] - hist["low"]).to_numpy(), (hist["high"] - prev_close).abs().to_numpy(), (hist["low"] - prev_close).abs().to_numpy()])
        atr = float(np.nanmean(tr[-w:]))
        squeeze = (2 * sd) < (float(p["kc_mult"]) * atr)
        prev = self._prev_squeeze.get(sym, False)
        self._prev_squeeze[sym] = squeeze
        if abs(pos.qty) > 0 or not (prev and not squeeze):
            return
        mom = float(close.iloc[-1] - close.iloc[-1 - w]) if len(close) > w else 0.0
        if mom == 0:
            return
        side = Side.BUY if mom > 0 else Side.SELL
        if side is Side.SELL and not ctx.instrument(sym).shortable:
            return
        allowed, scale = self.jev_gate(ctx, sym, side, hist, bar.close)
        if not allowed:
            return
        if self._set_target(ctx, sym, float(p["alloc_frac"]) * scale * (1.0 if side is Side.BUY else -1.0), bar.close, tag="entry") is not None:
            self._left[sym] = int(p["hold"])


register(VolSqueeze)
