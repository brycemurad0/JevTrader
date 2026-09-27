"""Cross-asset lead-lag: trade a follower in the direction of a leader's recent return.

The canonical retail version is BTC (24/7, reacts first) -> equity index during the US session
(or the reverse, ES -> BTC). The strategy watches the leader's trailing `lookback`-bar return
(bps); when it exceeds `threshold_bps` it takes a position in the follower for `hold` bars.

Why it might work: shared macro/risk-appetite factor with slightly different price-discovery
speeds. Why it fails: at 1-5 min the two mostly move *together* (contemporaneous correlation,
not lead), and the residual predictability is a fraction of a bp -- below equity costs and far
below crypto costs. Research verdict on BTC -> SPX500 proxy, 2026-03..07: NOT VIABLE (see
research_reports/06_stat_arb.md). Mechanics kept for the user to test other pairs.
"""

from __future__ import annotations

from collections import deque

from jevtrader.core.registry import register
from jevtrader.core.strategy import StrategyContext, StrategySpec
from jevtrader.core.types import Bar, Side
from jevtrader.strategies._base import BarStrategy


class LeadLag(BarStrategy):
    spec = StrategySpec(
        name="lead_lag",
        description="Trades `follower` in the direction of `leader`'s trailing return when it exceeds a threshold; holds a fixed number of bars.",
        asset_classes=("equity", "crypto"),
        frequency="5min",
        style="stat_arb",
        uses_jev=True,
        default_params={
            **BarStrategy.BASE_PARAMS,
            "bar_minutes": 5,
            "leader": "BTC/USD",
            "follower": "SPY",
            "lookback": 3,
            "threshold_bps": 15.0,
            "hold": 3,
            "alloc_frac": 0.25,
        },
        notes=(
            "EVIDENCE (research_reports/06_stat_arb.md): BTC->SPX500 5-min lead-lag has ~zero gross edge "
            "in-sample (contemporaneous correlation, no lead) -> NOT VIABLE. Symbols must include both leader "
            "and follower; only the follower is traded."
        ),
    )

    def __init__(self, symbols, params=None, strategy_id=None, **kw):
        super().__init__(symbols, params, strategy_id, **kw)
        self._lead_closes: deque = deque(maxlen=int(self.params["lookback"]) + 1)
        self._left = 0

    def on_agg_bar(self, bar: Bar, ctx: StrategyContext) -> None:
        p = self.params
        leader, follower = p["leader"], p["follower"]
        if bar.symbol == leader:
            self._lead_closes.append(bar.close)
            return
        if bar.symbol != follower:
            return
        pos = ctx.position(follower)
        if abs(pos.qty) > 0:
            self._left -= 1
            if self._left <= 0:
                self.flatten(ctx, follower, bar.close)
            return
        if len(self._lead_closes) < int(p["lookback"]) + 1:
            return
        lead_ret_bps = 1e4 * (self._lead_closes[-1] / self._lead_closes[0] - 1.0)
        thr = float(p["threshold_bps"])
        if lead_ret_bps > thr:
            side = Side.BUY
        elif lead_ret_bps < -thr and ctx.instrument(follower).shortable:
            side = Side.SELL
        else:
            return
        allowed, scale = self.jev_gate(ctx, follower, side, self.hist_df(follower, 120), bar.close)
        if not allowed:
            return
        if self._set_target(ctx, follower, float(p["alloc_frac"]) * scale * (1.0 if side is Side.BUY else -1.0), bar.close, tag="entry") is not None:
            self._left = int(p["hold"])


register(LeadLag)
