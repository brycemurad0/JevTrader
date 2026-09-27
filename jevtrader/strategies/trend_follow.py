"""Time-series momentum / trend following at 5-60 min with volatility targeting.

Two signals, one class: `mode="ema"` (fast/slow EMA crossover, incremental EMAs identical to
`pandas.ewm(span, adjust=False)`) and `mode="donchian"` (close breaks the prior `n`-bar high/low;
exit on the opposite `exit_n` channel). Position = sign(signal) x min(1, target_vol / realized
vol) x alloc_frac -- the vol-targeting keeps risk roughly constant across regimes and shrinks size
when vol spikes (exactly when trend systems bleed).

Why it might work: intraday trend persistence after information shocks; why it fails: at 5-15
min the signal flips several times a day and each flip costs a round trip. On BTC the research
harness found gross edges of -60..+6 bps per round trip (no consistent sign across horizons) vs
30-50 bps cost. On SPX500 15-60 min EMA crossover was positive in-sample over an up-trending
4-month window -- a drift artefact until proven on validation/holdout.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from jevtrader.core.registry import register
from jevtrader.core.strategy import StrategyContext, StrategySpec
from jevtrader.core.types import Bar, Side
from jevtrader.strategies._base import BarStrategy


class TrendFollow(BarStrategy):
    spec = StrategySpec(
        name="trend_follow",
        description="EMA-crossover or Donchian breakout trend following with volatility targeting (5-60 min bars).",
        asset_classes=("equity", "crypto"),
        frequency="15min",
        style="momentum",
        uses_jev=True,
        default_params={
            **BarStrategy.BASE_PARAMS,
            "bar_minutes": 15,
            "mode": "ema",  # "ema" | "donchian"
            "fast": 10,
            "slow": 50,
            "n": 60,
            "exit_n": 30,
            "vol_window": 60,
            "target_bar_vol_bps": 0.0,  # 0 = no vol targeting (binary position)
            "alloc_frac": 0.25,
        },
        notes=(
            "EVIDENCE (research_reports/03_trend_following.md): BTC intraday trend NOT VIABLE at Alpaca crypto "
            "fees (gross edge per trade below cost at every horizon tested; break-even fee <= ~3 bps/side). "
            "Equities: in-sample positive at 15-60 min on the SPX500 proxy but the window was a bull run; "
            "treat as MARGINAL at best pending validation. Long-only automatically on crypto."
        ),
    )

    def __init__(self, symbols, params=None, strategy_id=None, **kw):
        super().__init__(symbols, params, strategy_id, **kw)
        self._ema: dict[str, tuple[Optional[float], Optional[float]]] = {}
        self._rets: dict[str, deque] = {}
        self._last_close: dict[str, float] = {}
        self._hi: dict[str, deque] = {}
        self._lo: dict[str, deque] = {}

    def _update_ema(self, sym: str, close: float) -> tuple[float, float]:
        af, as_ = 2.0 / (int(self.params["fast"]) + 1), 2.0 / (int(self.params["slow"]) + 1)
        f, s = self._ema.get(sym, (None, None))
        f = close if f is None else (1 - af) * f + af * close
        s = close if s is None else (1 - as_) * s + as_ * close
        self._ema[sym] = (f, s)
        return f, s

    def _vol_scale(self, sym: str, close: float) -> float:
        tgt = float(self.params.get("target_bar_vol_bps", 0.0) or 0.0)
        w = int(self.params["vol_window"])
        d = self._rets.setdefault(sym, deque(maxlen=w))
        lc = self._last_close.get(sym)
        if lc is not None and lc > 0:
            d.append(np.log(close / lc))
        self._last_close[sym] = close
        if tgt <= 0:
            return 1.0
        if len(d) < w:
            return 0.0
        rv = float(np.std(np.asarray(d), ddof=1)) * 1e4
        return float(min(1.0, tgt / rv)) if rv > 0 else 0.0

    def on_agg_bar(self, bar: Bar, ctx: StrategyContext) -> None:
        sym = bar.symbol
        p = self.params
        hist = self.hist_df(sym, int(p["vol_window"]) + 2)
        scale = self._vol_scale(sym, bar.close)
        pos = ctx.position(sym)
        cur_sign = float(np.sign(pos.qty))
        desired = cur_sign
        if p["mode"] == "ema":
            f, s = self._update_ema(sym, bar.close)
            if len(hist) < int(p["slow"]):
                return
            desired = 1.0 if f > s else (-1.0 if f < s else 0.0)
        else:
            n, m = int(p["n"]), int(p["exit_n"])
            hi = self._hi.setdefault(sym, deque(maxlen=n))
            lo = self._lo.setdefault(sym, deque(maxlen=n))
            if len(hi) >= n:
                ch_hi, ch_lo = max(hi), min(lo)
                x_hi, x_lo = max(list(hi)[-m:]), min(list(lo)[-m:])
                if cur_sign == 0:
                    desired = 1.0 if bar.close > ch_hi else (-1.0 if bar.close < ch_lo else 0.0)
                elif cur_sign > 0 and bar.close < x_lo:
                    desired = 0.0
                elif cur_sign < 0 and bar.close > x_hi:
                    desired = 0.0
            hi.append(bar.high)
            lo.append(bar.low)
            if len(hi) < n:
                return
        if not ctx.instrument(sym).shortable and desired < 0:
            desired = 0.0
        if desired == cur_sign and cur_sign == 0:
            return
        if desired != 0.0 and (cur_sign == 0 or desired != cur_sign):
            side = Side.BUY if desired > 0 else Side.SELL
            allowed, jscale = self.jev_gate(ctx, sym, side, hist, bar.close)
            if not allowed:
                if cur_sign != 0:
                    self.flatten(ctx, sym, bar.close)
                return
            scale *= jscale
        self._set_target(ctx, sym, desired * float(p["alloc_frac"]) * scale, bar.close)


register(TrendFollow)
