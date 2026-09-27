"""Time-series forecasting: a feature provider for the decision model and the risk/rebalance
layers, never a strategy or a trading decision-maker on its own. See `docs/FORECASTING.md`.

`timesfm`/`timesfm3` (the real TimesFM packages) are OPTIONAL dependencies of this package:
importing `jevtrader.forecast` (or `jevtrader.forecast.timesfm_backend`) never imports them --
that only happens lazily, inside `TimesFMForecaster.forecast()`, the first time it's actually
called. Everything else here (`baselines.py`, `targets.py`, `evaluate.py`, `features.py`,
`advisor.py`) has no optional dependencies at all and always works offline.

Modules:
    base.py             Forecaster protocol, QuantileForecast (prob_above/prob_below).
    targets.py          bars -> series transforms and their inverses (returns, vol, volume, spread).
    baselines.py        RandomWalkForecaster, EWMAVolForecaster, GARCH11Forecaster, SeasonalNaiveForecaster.
    timesfm_backend.py  TimesFMForecaster: real TimesFM 2.5/3.0, lazy-loaded, batched, license-guarded.
    features.py         forecast_features(...): compact tfm_* dict merged into jev.state.build_state.
    advisor.py          ForecastAdvisor: a JevAdvisorProtocol implementation from tfm_* features alone.
    evaluate.py         Rolling-origin (no-look-ahead) evaluation vs. the honest baselines.
"""

from __future__ import annotations

from jevtrader.forecast.advisor import ForecastAdvisor
from jevtrader.forecast.base import QUANTILE_LEVELS, Forecaster, QuantileForecast
from jevtrader.forecast.baselines import (
    EWMAVolForecaster,
    GARCH11Forecaster,
    GarchParams,
    RandomWalkForecaster,
    SeasonalNaiveForecaster,
    fit_garch11,
)
from jevtrader.forecast.evaluate import EvalResult, evaluate_returns, evaluate_volatility
from jevtrader.forecast.features import FEATURE_KEYS, forecast_features
from jevtrader.forecast.timesfm_backend import TimesFM3LicenseError, TimesFMForecaster, TimesFMNotInstalled

__all__ = [
    "QUANTILE_LEVELS",
    "EWMAVolForecaster",
    "EvalResult",
    "FEATURE_KEYS",
    "ForecastAdvisor",
    "Forecaster",
    "GARCH11Forecaster",
    "GarchParams",
    "QuantileForecast",
    "RandomWalkForecaster",
    "SeasonalNaiveForecaster",
    "TimesFM3LicenseError",
    "TimesFMForecaster",
    "TimesFMNotInstalled",
    "evaluate_returns",
    "evaluate_volatility",
    "fit_garch11",
    "forecast_features",
]
