"""Fast, honest, offline baselines implementing the same `Forecaster` protocol as TimesFM.

Per the honest prior in `docs/FORECASTING.md`: a foundation model has to beat these on the
`evaluate.py` scoreboard before anything in the strategy stack should trust it over them. None
of these needs a network call, a GPU, or even a slow fit -- they run in microseconds to
milliseconds per symbol, so they're always available as a fallback and as the null hypothesis.

- `RandomWalkForecaster`: the textbook honest baseline for *returns* -- zero expected drift,
  quantiles from the series' own empirical return distribution scaled by sqrt(horizon) (a
  Brownian-motion / i.i.d.-increments assumption).
- `EWMAVolForecaster`: RiskMetrics-style EWMA variance, forecast held flat over the horizon
  (a pure persistence forecast -- vol is much more persistent than returns, which is exactly
  why the honest prior expects a real edge here rather than for direction).
- `GARCH11Forecaster`: GARCH(1,1) fit by maximum likelihood with `scipy.optimize` (no `arch`
  dependency), forecasting the mean-reverting variance path implied by the fitted model.
- `SeasonalNaiveForecaster`: for volume/spread series with a daily (or other) seasonal period --
  "the value h steps from now is whatever it was one season ago", the standard baseline
  meteorologists and utilities forecast against before trusting anything fancier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np
from scipy.optimize import minimize

from jevtrader.forecast.base import QUANTILE_LEVELS, Forecaster, QuantileForecast, constant_quantile_forecast

_MIN_HISTORY = 20


# ----------------------------------------------------------------------------- random walk (returns)


@dataclass
class RandomWalkForecaster:
    """Zero-drift random walk over a *log-return* series. Quantiles are the series' own
    empirical return quantiles, scaled by `sqrt(h)` for horizon step `h` (1-indexed) -- the
    standard i.i.d.-increments scaling. `min_history` series shorter than this are skipped
    (omitted from the result) rather than given a degenerate forecast."""

    min_history: int = _MIN_HISTORY

    def forecast(self, series: Mapping[str, np.ndarray], horizon: int) -> Dict[str, QuantileForecast]:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        out: Dict[str, QuantileForecast] = {}
        for key, values in series.items():
            hist = np.asarray(values, dtype=float)
            hist = hist[~np.isnan(hist)]
            if hist.size < self.min_history:
                continue
            base_q = {level: float(np.quantile(hist, level)) for level in QUANTILE_LEVELS}
            steps = np.sqrt(np.arange(1, horizon + 1, dtype=float))
            point = np.zeros(horizon, dtype=float)
            quantiles = {level: base_q[level] * steps for level in QUANTILE_LEVELS}
            out[key] = constant_quantile_forecast(point, quantiles)
        return out


# ----------------------------------------------------------------------------- EWMA volatility


@dataclass
class EWMAVolForecaster:
    """RiskMetrics-style EWMA volatility forecast over a *log-return* series (NOT a
    pre-computed RV series -- it needs the raw returns to update its own recursive estimate).
    The point forecast is flat across the horizon (persistence: tomorrow's vol ~ today's EWMA
    vol). Quantiles come from the empirical distribution of `log(realized / ewma_at_t)` computed
    in-sample (one-step-ahead), so the band reflects how wrong this exact estimator has
    historically been on this series, not an assumed shape."""

    decay: float = 0.94  # RiskMetrics default (lambda)
    min_history: int = _MIN_HISTORY

    def _ewma_path(self, returns: np.ndarray) -> np.ndarray:
        var = np.empty(returns.shape[0], dtype=float)
        var[0] = returns[0] ** 2
        for i in range(1, returns.shape[0]):
            var[i] = self.decay * var[i - 1] + (1.0 - self.decay) * returns[i - 1] ** 2
        return var

    def forecast(self, series: Mapping[str, np.ndarray], horizon: int) -> Dict[str, QuantileForecast]:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        out: Dict[str, QuantileForecast] = {}
        for key, values in series.items():
            rets = np.asarray(values, dtype=float)
            rets = rets[~np.isnan(rets)]
            if rets.size < self.min_history:
                continue
            var_path = self._ewma_path(rets)
            vol_path = np.sqrt(var_path)
            forecast_vol = float(vol_path[-1])
            point = np.full(horizon, forecast_vol, dtype=float)

            # in-sample calibration: how far off was ewma[t] from |return[t+1]| historically?
            realized = np.abs(rets[1:])
            predicted = vol_path[:-1]
            valid = predicted > 1e-12
            if valid.sum() >= 5:
                log_ratio = np.log(np.clip(realized[valid], 1e-12, None) / predicted[valid])
                ratio_q = {level: float(np.exp(np.quantile(log_ratio, level))) for level in QUANTILE_LEVELS}
            else:
                # not enough in-sample residuals to calibrate: a symmetric +/-30% band per decile step
                ratio_q = {level: float(np.exp((level - 0.5) * 1.2)) for level in QUANTILE_LEVELS}
            quantiles = {level: point * ratio for level, ratio in ratio_q.items()}
            out[key] = constant_quantile_forecast(point, quantiles)
        return out


# ----------------------------------------------------------------------------- GARCH(1,1)


@dataclass(frozen=True)
class GarchParams:
    omega: float
    alpha: float
    beta: float

    @property
    def long_run_var(self) -> float:
        denom = 1.0 - self.alpha - self.beta
        return self.omega / denom if denom > 1e-8 else float("nan")


def fit_garch11(returns: np.ndarray, *, demean: bool = True) -> GarchParams:
    """Fit GARCH(1,1) by maximum likelihood: `sigma2_t = omega + alpha*eps_{t-1}^2 +
    beta*sigma2_{t-1}`, `eps_t ~ N(0, sigma2_t)`. Uses `scipy.optimize.minimize` (L-BFGS-B) on
    the negative log-likelihood, reparameterized so `alpha, beta >= 0` and `alpha + beta < 1`
    (stationarity) hold by construction. No `arch` package dependency.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if demean:
        r = r - r.mean()
    if r.size < 10:
        raise ValueError("need at least 10 return observations to fit GARCH(1,1)")
    sample_var = float(np.var(r)) or 1e-8

    def unpack(theta: np.ndarray) -> tuple[float, float, float]:
        # theta = [log_omega, logit_alpha_share, logit_persistence]
        omega = math.exp(theta[0])
        persistence = 1.0 / (1.0 + math.exp(-theta[2]))  # in (0, 1): alpha + beta
        alpha_share = 1.0 / (1.0 + math.exp(-theta[1]))  # in (0, 1): alpha / (alpha + beta)
        alpha = persistence * alpha_share
        beta = persistence * (1.0 - alpha_share)
        return omega, alpha, beta

    def neg_log_lik(theta: np.ndarray) -> float:
        omega, alpha, beta = unpack(theta)
        sigma2 = sample_var
        ll = 0.0
        for eps in r:
            sigma2 = max(sigma2, 1e-12)
            ll += -0.5 * (math.log(2 * math.pi) + math.log(sigma2) + eps * eps / sigma2)
            sigma2 = omega + alpha * eps * eps + beta * sigma2
        return -ll

    theta0 = np.array([math.log(sample_var * 0.05), 0.0, math.log(0.85 / 0.15)])
    result = minimize(neg_log_lik, theta0, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6})
    omega, alpha, beta = unpack(result.x)
    return GarchParams(omega=omega, alpha=alpha, beta=beta)


@dataclass
class GARCH11Forecaster:
    """Forecasts the volatility path implied by a GARCH(1,1) fit on a *log-return* series,
    refit every call (cheap: milliseconds for a few thousand points with Nelder-Mead). The
    variance path mean-reverts geometrically toward the model's long-run variance, which is the
    textbook GARCH forecast and is usually a better vol forecast than flat EWMA persistence at
    horizons beyond a few steps. Quantile band: same in-sample log-ratio calibration as
    `EWMAVolForecaster`, against this model's own fitted one-step-ahead vol."""

    min_history: int = 30

    def forecast(self, series: Mapping[str, np.ndarray], horizon: int) -> Dict[str, QuantileForecast]:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        out: Dict[str, QuantileForecast] = {}
        for key, values in series.items():
            rets = np.asarray(values, dtype=float)
            rets = rets[~np.isnan(rets)]
            if rets.size < self.min_history:
                continue
            try:
                params = fit_garch11(rets)
            except Exception:
                continue
            r = rets - rets.mean()
            sigma2 = float(np.var(r))
            fitted_vol = np.empty(r.shape[0], dtype=float)
            for i, eps in enumerate(r):
                fitted_vol[i] = math.sqrt(max(sigma2, 1e-12))
                sigma2 = params.omega + params.alpha * eps * eps + params.beta * sigma2
            last_sigma2 = sigma2
            long_run = params.long_run_var if math.isfinite(params.long_run_var) else float(np.var(r))
            persistence = params.alpha + params.beta
            # standard GARCH(1,1) multi-step variance forecast: geometric mean-reversion to
            # the long-run variance at rate `persistence` (alpha + beta), starting from the
            # last fitted conditional variance.
            var_path = long_run + (persistence ** np.arange(horizon)) * (last_sigma2 - long_run)
            var_path = np.clip(var_path, 1e-12, None)
            point = np.sqrt(var_path)

            realized = np.abs(r[1:])
            predicted = fitted_vol[:-1]
            valid = predicted > 1e-12
            if valid.sum() >= 5:
                log_ratio = np.log(np.clip(realized[valid], 1e-12, None) / predicted[valid])
                ratio_q = {level: float(np.exp(np.quantile(log_ratio, level))) for level in QUANTILE_LEVELS}
            else:
                ratio_q = {level: float(np.exp((level - 0.5) * 1.2)) for level in QUANTILE_LEVELS}
            quantiles = {level: point * ratio for level, ratio in ratio_q.items()}
            out[key] = constant_quantile_forecast(point, quantiles)
        return out


# ----------------------------------------------------------------------------- seasonal naive (volume)


@dataclass
class SeasonalNaiveForecaster:
    """"The value h steps from now equals its value one season ago" -- the standard baseline for
    a series with a strong daily/intraday seasonal pattern (e.g. per-minute volume: lunch-hour
    lull, open/close spikes). `period` is the seasonal length in bars (e.g. 1440 for 1-minute
    bars and a 24h cycle). Quantiles come from the empirical distribution of this predictor's
    own in-sample residuals (`actual - seasonal_lag`), additive and constant across the
    horizon (a conservative, non-widening band, appropriate for a bounded seasonal signal)."""

    period: int
    min_history: Optional[int] = None

    def forecast(self, series: Mapping[str, np.ndarray], horizon: int) -> Dict[str, QuantileForecast]:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.period < 1:
            raise ValueError("period must be >= 1")
        min_hist = self.min_history or (2 * self.period)
        out: Dict[str, QuantileForecast] = {}
        for key, values in series.items():
            arr = np.asarray(values, dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.size < min_hist:
                continue
            residuals = arr[self.period :] - arr[: -self.period]
            resid_q = {level: float(np.quantile(residuals, level)) for level in QUANTILE_LEVELS}
            point = np.empty(horizon, dtype=float)
            for h in range(horizon):
                lag_index = arr.size - self.period + h
                # wrap into the seasonal history once we run past the observed series (h >= period)
                while lag_index >= arr.size:
                    lag_index -= self.period
                point[h] = arr[lag_index] if lag_index >= 0 else float(arr[-1])
            quantiles = {level: point + resid_q[level] for level in QUANTILE_LEVELS}
            out[key] = constant_quantile_forecast(point, quantiles)
        return out
