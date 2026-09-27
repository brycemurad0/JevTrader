"""Trade-flow toxicity (VPIN-style) gate, usable by any strategy, plus a no-trade monitor strategy.

`FlowToxicityGate` implements Easley, Lopez de Prado & O'Hara's Volume-synchronized PIN with
bulk-volume classification: trades are grouped into volume buckets of `bucket_volume`; within a
bucket the buy fraction is Phi(dP / sigma_dP) (or the aggressor flag when the feed provides it),
and VPIN = mean over the last `n_buckets` of |V_buy - V_sell| / V. High VPIN = informed/toxic
flow = market makers get run over and short-horizon reversion fails.

Usage inside a strategy: call `gate.on_trade(trade)` (or `gate.on_bar(bar)` for bar-only
feeds) and check `gate.toxic` before quoting/entering. The market maker and imbalance strategies
in this package do exactly that when `vpin_gate=True`.

Evidence: this is a *filter*, not an alpha; it cannot be validated for edge on bars alone. The
monitor strategy logs VPIN so the user can record it on real Alpaca trade prints during paper
trading and then correlate it with market-maker fill P&L (`research.record_replay`).
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
from scipy.stats import norm

from jevtrader.core.registry import register
from jevtrader.core.strategy import Strategy, StrategyContext, StrategySpec
from jevtrader.core.types import Bar, Side, Trade


class FlowToxicityGate:
    def __init__(self, bucket_volume: float, n_buckets: int = 50, toxic_threshold: float = 0.7, use_aggressor: bool = True) -> None:
        self.bucket_volume = float(bucket_volume)
        self.n_buckets = int(n_buckets)
        self.toxic_threshold = float(toxic_threshold)
        self.use_aggressor = use_aggressor
        self._imb: deque = deque(maxlen=self.n_buckets)
        self._cur_vol = 0.0
        self._cur_buy = 0.0
        self._last_px: Optional[float] = None
        self._dp: deque = deque(maxlen=200)

    def _sigma(self) -> float:
        if len(self._dp) < 5:
            return float("nan")
        s = float(np.std(np.asarray(self._dp), ddof=1))
        return s if s > 0 else float("nan")

    def _classify(self, price: float, size: float, aggressor: Optional[Side]) -> float:
        if self.use_aggressor and aggressor is not None:
            return size if aggressor is Side.BUY else 0.0
        if self._last_px is None:
            return 0.5 * size
        dp = price - self._last_px
        s = self._sigma()
        p_buy = 0.5 if not np.isfinite(s) else float(norm.cdf(dp / s))
        return p_buy * size

    def _push(self, price: float, size: float, aggressor: Optional[Side]) -> None:
        if size <= 0:
            return
        remaining = size
        while remaining > 0:
            room = self.bucket_volume - self._cur_vol
            take = min(room, remaining)
            frac = take / size
            self._cur_buy += frac * self._classify(price, size, aggressor)
            self._cur_vol += take
            remaining -= take
            if self._cur_vol >= self.bucket_volume - 1e-12:
                v = self._cur_vol
                self._imb.append(abs(2 * self._cur_buy - v) / v)
                self._cur_vol, self._cur_buy = 0.0, 0.0
        if self._last_px is not None:
            self._dp.append(price - self._last_px)
        self._last_px = price

    def on_trade(self, trade: Trade) -> None:
        self._push(trade.price, trade.size, trade.aggressor)

    def on_bar(self, bar: Bar) -> None:
        """Bar-only feeds: treat the bar as one print at its close with its volume (BVC on dP)."""
        self._push(bar.close, bar.volume, None)

    @property
    def vpin(self) -> float:
        return float(np.mean(self._imb)) if len(self._imb) >= max(5, self.n_buckets // 5) else float("nan")

    @property
    def toxic(self) -> bool:
        v = self.vpin
        return bool(np.isfinite(v) and v > self.toxic_threshold)


class VpinMonitor(Strategy):
    """Never trades. Computes VPIN from trades (or bars) and logs it so paper-trading journals
    contain a toxicity time series to validate the gate against realized market-maker P&L."""

    spec = StrategySpec(
        name="vpin_monitor",
        description="No-trade monitor: computes a VPIN flow-toxicity series from trade prints (or bars) and logs it.",
        asset_classes=("equity", "crypto"),
        frequency="tick",
        style="filter",
        default_params={"bucket_volume": 1.0, "n_buckets": 50, "toxic_threshold": 0.7, "log_every_n": 20},
        notes="A filter, not an alpha. Its value can only be judged by conditioning another strategy's realized fills on it (record_replay).",
    )

    def __init__(self, symbols, params=None, strategy_id=None):
        super().__init__(symbols, params, strategy_id)
        self.gates = {s: FlowToxicityGate(self.params["bucket_volume"], int(self.params["n_buckets"]), float(self.params["toxic_threshold"])) for s in self.symbols}
        self._n = 0
        self.history: list[tuple] = []

    def on_trade(self, trade: Trade, ctx: StrategyContext) -> None:
        g = self.gates[trade.symbol]
        g.on_trade(trade)
        self._log(trade.symbol, trade.ts, g, ctx)

    def on_bar(self, bar: Bar, ctx: StrategyContext) -> None:
        g = self.gates[bar.symbol]
        g.on_bar(bar)
        self._log(bar.symbol, bar.ts, g, ctx)

    def _log(self, symbol, ts, g: FlowToxicityGate, ctx: StrategyContext) -> None:
        self._n += 1
        v = g.vpin
        if np.isfinite(v):
            self.history.append((ts, symbol, v))
            if self._n % int(self.params["log_every_n"]) == 0:
                ctx.log("vpin", symbol=symbol, vpin=round(v, 4), toxic=g.toxic)


register(VpinMonitor)
