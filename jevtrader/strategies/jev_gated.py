"""Jev-gated variants of the best base strategies ("Jev judges, code executes").

Each variant is the base strategy with `jev_gate=True`: before every ENTRY it builds the compact
state (`jevtrader.jev.state.build_state`, including the strategy's own round-trip `cost_bps`),
asks `ctx.jev.direction(symbol, state, horizon)` and only trades when P(move in our direction >
cost) >= `jev_min_p`; the position is then sized with `kelly_from_jev` (fractional Kelly, capped
at `alloc_frac`). `regime()` adds a risk-off veto. A `None` or `late` view is a HOLD. With
`ctx.jev is None` the variant behaves exactly like the base strategy (see tests).

Evaluation caveat, stated plainly: `OfflineJevAdvisor` is a transparent logistic heuristic with
NO forecasting skill -- it exists to exercise the code path. Backtests with it tell you how the
gate *changes turnover and sizing*, not whether Jev adds alpha. The real test is:
  1. paper-trade the variant with the real advisor and `DecisionLog` on,
  2. replay with `ReplayJevAdvisor.from_log(...)` over the recorded period (`research.experiments.
     jev_gate_replay`), and
  3. read the Kev/Laya/Jev calibration scoreboard (`jevtrader.jev.calibration`).
"""

from __future__ import annotations

from jevtrader.core.registry import register
from jevtrader.core.strategy import StrategySpec
from jevtrader.strategies.range_breakout import RangeBreakout
from jevtrader.strategies.trend_follow import TrendFollow
from jevtrader.strategies.vwap_reversion import ZScoreReversion


def _jev_spec(base: StrategySpec, name: str, horizon: str) -> StrategySpec:
    return StrategySpec(
        name=name,
        description=f"Jev-gated variant of `{base.name}`: entries require P(move > cost) >= jev_min_p, Kelly-sized from Jev probabilities.",
        asset_classes=base.asset_classes,
        frequency=base.frequency,
        style=base.style,
        uses_jev=True,
        default_params={**base.default_params, "jev_gate": True, "jev_min_p": 0.55, "jev_horizon": horizon, "jev_kelly_fraction": 0.5, "jev_regime_veto": True},
        notes=(
            "Falls back to the base strategy when ctx.jev is None; treats late/None views as HOLD. Evaluated with "
            "OfflineJevAdvisor for MECHANICS only (no skill by construction) -- research_reports/07_jev_gated.md "
            "reports the turnover reduction and the sizing behaviour, not alpha. Real test = ReplayJevAdvisor on "
            "recorded paper decisions + the calibration scoreboard. " + base.notes
        ),
    )


class ZScoreReversionJev(ZScoreReversion):
    spec = _jev_spec(ZScoreReversion.spec, "zscore_reversion_jev", "30min")


class TrendFollowJev(TrendFollow):
    spec = _jev_spec(TrendFollow.spec, "trend_follow_jev", "60min")


class RangeBreakoutJev(RangeBreakout):
    spec = _jev_spec(RangeBreakout.spec, "range_breakout_jev", "60min")


register(ZScoreReversionJev)
register(TrendFollowJev)
register(RangeBreakoutJev)
