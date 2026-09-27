"""`ForecastAdvisor`: a `JevAdvisorProtocol` implementation whose only inputs are TimesFM/baseline
quantile forecasts (the `tfm_*` keys `features.py` produces), no LLM call and no other model.

The point is that TimesFM can then sit on the exact same scoreboard as Jev/Kev/Laya: run the
same strategy with `ctx.jev = ForecastAdvisor(...)` in a backtest, and `calibration.py`/the
research harness can compare it head-to-head against the others on the exact same metric. It's
also usable standalone (no Jev, Kev or Laya running at all) since it needs nothing but the
`tfm_*` features already merged into the state dict by `build_state(..., extra_features=...)`.

`direction()` reads `tfm_p_up_gt_cost`/`tfm_p_down_gt_cost` (already "P(return clears cost)"
probabilities from `QuantileForecast.prob_above`/`prob_below`) and turns them into a proper
3-way distribution over `{up, down, flat}` -- the two inputs are independent one-sided CDF
reads, not literal complements, so `flat = 1 - p_up - p_down` (clamped) and the whole triple is
renormalized to sum to exactly 1.

`regime()` is explicitly a heuristic (documented in `docs/FORECASTING.md#regime-heuristic`, not
a forecast): it only has a *volatility* forecast to go on (`tfm_vol_ratio` = forecast vol /
realized vol), so it can speak with some grounding to the volatile/quiet axis, and only weakly
tie-breaks trending/mean-reverting off the base state's `trend_slope_bps` when that key happens
to be present (it's optional -- `ForecastAdvisor` never requires the full Jev state, only its own
`tfm_*` keys).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from jevtrader.core.interfaces import JevView

DIRECTION_LABELS: tuple[str, ...] = ("up", "down", "flat")
REGIME_LABELS: tuple[str, ...] = ("trending_up", "trending_down", "mean_reverting", "volatile_chop", "quiet")


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(logits)] * len(logits)
    return [v / total for v in exps]


class ForecastAdvisor:
    """See module docstring. Never raises: returns `None` (protocol convention for "skip this
    cycle / no view") whenever the needed `tfm_*` features aren't present, e.g. too little
    history for `forecast_features` to have produced a forecast."""

    def __init__(self, *, model: str = "forecast-advisor-v1") -> None:
        self.model = model

    def direction(self, symbol: str, features: Mapping[str, Any], horizon: str) -> Optional[JevView]:
        raw_up = features.get("tfm_p_up_gt_cost")
        raw_down = features.get("tfm_p_down_gt_cost")
        if raw_up is None or raw_down is None:
            return None
        p_up = max(0.0, float(raw_up))
        p_down = max(0.0, float(raw_down))
        p_flat = max(0.0, 1.0 - p_up - p_down)
        total = p_up + p_down + p_flat
        if total <= 0:
            return None
        p_up, p_down, p_flat = p_up / total, p_down / total, p_flat / total
        probs = {"direction": {"up": p_up, "down": p_down, "flat": p_flat}}
        top = {"direction": max(probs["direction"], key=probs["direction"].get)}
        confidence = {"direction": max(probs["direction"].values())}
        return JevView(
            probs=probs,
            top=top,
            confidence=confidence,
            latency_ms=0.0,
            late=False,
            model=self.model,
            raw={"tfm_p_up_gt_cost": float(raw_up), "tfm_p_down_gt_cost": float(raw_down)},
        )

    def regime(self, symbol: str, features: Mapping[str, Any]) -> Optional[JevView]:
        raw_ratio = features.get("tfm_vol_ratio")
        if raw_ratio is None:
            return None
        vol_ratio = float(raw_ratio)
        trend = features.get("trend_slope_bps")
        trend = float(trend) if trend is not None else 0.0
        # Heuristic, not a forecast -- see module docstring and docs/FORECASTING.md.
        logits = {
            "volatile_chop": (vol_ratio - 1.0) * 3.0 - abs(trend) / 20.0,
            "quiet": (1.0 - vol_ratio) * 3.0,
            "trending_up": max(0.0, trend) / 10.0,
            "trending_down": max(0.0, -trend) / 10.0,
            "mean_reverting": -abs(trend) / 15.0,
        }
        names = list(logits.keys())
        dist = _softmax([logits[n] for n in names])
        regime_probs = {n: round(p, 4) for n, p in zip(names, dist)}
        risk_off_p = round(_sigmoid((vol_ratio - 1.2) * 2.0), 4)
        probs = {"regime": regime_probs, "risk_off": {"yes": risk_off_p, "no": round(1.0 - risk_off_p, 4)}}
        top = {"regime": max(regime_probs, key=regime_probs.get), "risk_off": "yes" if risk_off_p >= 0.5 else "no"}
        confidence = {"regime": max(regime_probs.values()), "risk_off": round(abs(risk_off_p - 0.5) * 2, 4)}
        return JevView(
            probs=probs,
            top=top,
            confidence=confidence,
            latency_ms=0.0,
            late=False,
            model=self.model,
            raw={"tfm_vol_ratio": vol_ratio, "trend_slope_bps": trend},
        )

    def ask(self, state: Mapping[str, Any], questions: Mapping[str, Any]) -> Optional[JevView]:
        """Limited fallback: `ForecastAdvisor` only knows how to answer a "direction"-shaped
        question (labels exactly `{"up", "down", "flat"}`) and only from `tfm_*` features already
        in `state`; anything else returns `None` rather than fabricate an opinion it has no
        forecast for. Prefer `direction()`/`regime()` directly wherever possible."""
        for name, question in questions.items():
            criteria = question.get("criteria") if isinstance(question, Mapping) else getattr(question, "criteria", None)
            if isinstance(criteria, Mapping) and set(criteria.keys()) == set(DIRECTION_LABELS):
                view = self.direction("", state, horizon="")
                if view is None:
                    return None
                probs = {name: view.probs["direction"]}
                top = {name: view.top["direction"]}
                confidence = {name: view.confidence["direction"]}
                return JevView(probs=probs, top=top, confidence=confidence, latency_ms=0.0, late=False, model=self.model, raw=dict(view.raw))
        return None
