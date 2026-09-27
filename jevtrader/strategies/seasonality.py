"""Intraday seasonality: hold long (or short) during fixed UTC hours / weekdays.

The candidate effects: (a) BTC time-of-day returns (US afternoon vs Asia), (b) equity
first-hour drift and (c) the overnight-vs-intraday split (the well-documented finding that most
of the equity premium accrues overnight). The research harness measures these with per-hour
t-statistics; a strategy is only worth running when |t| is large *after* a multiple-testing
correction (24 hours x 2 datasets = many tests: |t| > 3 is the bar, not 2).

Evidence: BTC train hours -- best |t| = 2.3 (hour 22 UTC) out of 24 tests -> consistent with
noise; SPX500 proxy / C: first hour t~2, last bar t~3 on tiny n. Verdict NOT VIABLE as a
stand-alone; the overnight-drift effect is only tradable as a *holding* (buy close, sell open),
which costs a round trip per day (2.6-3.3 bps on equities) against an average overnight move
of a few bps -- MARGINAL at best. Implemented so the user can re-test on their own Alpaca data.
"""

from __future__ import annotations

from jevtrader.core.registry import register
from jevtrader.core.strategy import StrategyContext, StrategySpec
from jevtrader.core.types import Bar, Side
from jevtrader.strategies._base import BarStrategy


class TimeOfDay(BarStrategy):
    spec = StrategySpec(
        name="time_of_day",
        description="Holds a long (or short) position during configured UTC hours / weekdays; the seasonality testbed.",
        asset_classes=("equity", "crypto"),
        frequency="15min",
        style="seasonality",
        uses_jev=True,
        default_params={
            **BarStrategy.BASE_PARAMS,
            "bar_minutes": 15,
            "long_hours": [13, 14],  # UTC hours (bar close) during which to be long
            "short_hours": [],
            "weekdays": [0, 1, 2, 3, 4],
            "alloc_frac": 0.25,
        },
        notes=(
            "EVIDENCE (research_reports/05_seasonality.md): no BTC hour survives multiple-testing; equity "
            "first-hour/overnight effects are small vs a daily round trip -> NOT VIABLE / MARGINAL. Do not run "
            "with real money on the basis of these samples; re-measure on >= 2 years of your own data."
        ),
    )

    def on_agg_bar(self, bar: Bar, ctx: StrategyContext) -> None:
        sym = bar.symbol
        p = self.params
        h = bar.ts.hour
        want = 0.0
        if bar.ts.dayofweek in set(int(d) for d in p["weekdays"]):
            if h in set(int(x) for x in p["long_hours"]):
                want = 1.0
            elif h in set(int(x) for x in p["short_hours"]) and ctx.instrument(sym).shortable:
                want = -1.0
        pos = ctx.position(sym)
        cur = 0.0 if pos.qty == 0 else (1.0 if pos.qty > 0 else -1.0)
        if want == cur:
            return
        if want == 0.0:
            self.flatten(ctx, sym, bar.close)
            return
        side = Side.BUY if want > 0 else Side.SELL
        allowed, scale = self.jev_gate(ctx, sym, side, self.hist_df(sym, 120), bar.close)
        if not allowed:
            return
        self._set_target(ctx, sym, want * float(p["alloc_frac"]) * scale, bar.close)


register(TimeOfDay)
