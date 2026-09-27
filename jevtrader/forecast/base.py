"""Shared contract for every forecaster in `jevtrader.forecast`: the real TimesFM backend
(`timesfm_backend.py`), the honest statistical baselines (`baselines.py`), and anything a
future backend adds. Everything downstream (`features.py`, `advisor.py`, `evaluate.py`) is
written against this contract only, never against a specific model, so TimesFM can be swapped
for a baseline (or a future model) without touching a single other line.

`Forecaster.forecast` takes a *batch* of named 1-D series (already in whatever unit that
target uses -- see `targets.py`) and a horizon, and returns one `QuantileForecast` per series
name. Batching multiple symbols/targets into one call is what lets the TimesFM backend do a
single forward pass instead of one per symbol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

# TimesFM's continuous quantile head returns exactly these nine levels (plus a mean we drop --
# see `timesfm_backend.py`). Every forecaster in this package must produce this same grid so
# `QuantileForecast` consumers never have to special-case which backend produced a forecast.
QUANTILE_LEVELS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

_EPS = 1e-4  # tail clamp for prob_above/prob_below so callers never see exact 0/1


def _as_1d(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    return arr


@dataclass(frozen=True)
class QuantileForecast:
    """One series' forecast: a point path plus a quantile grid, both shape `(horizon,)`.

    `quantiles` must have exactly the keys in `QUANTILE_LEVELS` (0.1 .. 0.9), each an array of
    length `horizon`. Quantiles need not be strictly increasing across levels for a given step
    (a backend with `fix_quantile_crossing=False` could -- in principle -- emit a crossing); the
    CDF interpolation below tolerates that by nudging duplicates apart, so callers never see
    inconsistent probabilities even from an imperfectly calibrated backend.
    """

    point: np.ndarray
    quantiles: Mapping[float, np.ndarray]
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        point = _as_1d(self.point, "point")
        object.__setattr__(self, "point", point)
        h = point.shape[0]
        missing = set(QUANTILE_LEVELS) - set(self.quantiles.keys())
        if missing:
            raise ValueError(f"QuantileForecast missing quantile levels: {sorted(missing)}")
        clean: dict[float, np.ndarray] = {}
        for level in QUANTILE_LEVELS:
            arr = _as_1d(self.quantiles[level], f"quantiles[{level}]")
            if arr.shape[0] != h:
                raise ValueError(f"quantiles[{level}] has length {arr.shape[0]}, expected {h} (== len(point))")
            clean[level] = arr
        object.__setattr__(self, "quantiles", clean)

    @property
    def horizon(self) -> int:
        return int(self.point.shape[0])

    def quantile_at(self, level: float, h: int = -1) -> float:
        """The forecast value at quantile `level` (must be one of `QUANTILE_LEVELS`) for step
        `h` (default: the last/furthest horizon step, matching Python negative indexing)."""
        return float(self.quantiles[level][h])

    def cdf(self, threshold: float, h: int = -1) -> float:
        """P(value at horizon step `h` <= threshold), linearly interpolated from the quantile
        grid. Outside the q10..q90 range this linearly extrapolates from the nearest two
        quantiles (the honest thing to do with only nine points of a distribution) and then
        clamps to `[EPS, 1-EPS]` so a tail event is never reported as impossible or certain.
        """
        levels = list(QUANTILE_LEVELS)
        values = [float(self.quantiles[level][h]) for level in levels]
        # Enforce strict monotonicity for interpolation purposes only (a flat/crossed quantile
        # grid would otherwise make np.interp's assumptions undefined); this never changes the
        # reported point/quantile values themselves.
        for i in range(1, len(values)):
            if values[i] <= values[i - 1]:
                values[i] = values[i - 1] + 1e-9
        if threshold <= values[0]:
            if values[1] > values[0]:
                slope = (levels[1] - levels[0]) / (values[1] - values[0])
                p = levels[0] + slope * (threshold - values[0])
            else:
                p = levels[0]
        elif threshold >= values[-1]:
            if values[-1] > values[-2]:
                slope = (levels[-1] - levels[-2]) / (values[-1] - values[-2])
                p = levels[-1] + slope * (threshold - values[-1])
            else:
                p = levels[-1]
        else:
            p = float(np.interp(threshold, values, levels))
        return float(np.clip(p, _EPS, 1.0 - _EPS))

    def prob_above(self, threshold: float, h: int = -1) -> float:
        """P(value at horizon step `h` > threshold), e.g. P(return > +cost_bps)."""
        return float(np.clip(1.0 - self.cdf(threshold, h=h), _EPS, 1.0 - _EPS))

    def prob_below(self, threshold: float, h: int = -1) -> float:
        """P(value at horizon step `h` < threshold), e.g. P(return < -cost_bps)."""
        return float(np.clip(self.cdf(threshold, h=h), _EPS, 1.0 - _EPS))

    def scaled(self, factor: float) -> "QuantileForecast":
        """Elementwise-scale point and quantiles (e.g. fraction -> bps via factor=1e4)."""
        return QuantileForecast(
            point=self.point * factor,
            quantiles={level: arr * factor for level, arr in self.quantiles.items()},
            latency_ms=self.latency_ms,
        )


@runtime_checkable
class Forecaster(Protocol):
    """Batch forecaster contract. `series` maps an arbitrary key (typically
    `f"{symbol}:{target_name}"`) to a 1-D array of that target's history, oldest-first, with no
    look-ahead: the array must end at "now". Returns one `QuantileForecast` per key, or omits a
    key the forecaster could not produce a forecast for (e.g. too little history)."""

    def forecast(self, series: Mapping[str, np.ndarray], horizon: int) -> Dict[str, QuantileForecast]: ...


def constant_quantile_forecast(point: Sequence[float] | np.ndarray, spread: Mapping[float, Sequence[float] | np.ndarray], latency_ms: float = 0.0) -> QuantileForecast:
    """Small builder used by baselines/tests: `spread` maps quantile level -> array of the same
    length as `point` (already the *value*, not an offset)."""
    point_arr = np.asarray(point, dtype=float)
    quantiles = {level: np.asarray(spread[level], dtype=float) for level in QUANTILE_LEVELS}
    return QuantileForecast(point=point_arr, quantiles=quantiles, latency_ms=latency_ms)
