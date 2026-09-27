from __future__ import annotations

import math

import numpy as np
import pytest

from jevtrader.forecast.baselines import (
    EWMAVolForecaster,
    GARCH11Forecaster,
    GarchParams,
    RandomWalkForecaster,
    SeasonalNaiveForecaster,
    fit_garch11,
)
from jevtrader.forecast.base import QUANTILE_LEVELS
from jevtrader.forecast.targets import cumulative_log_return_bps


# ----------------------------------------------------------------------------- random walk


def test_random_walk_forecaster_zero_drift_point():
    rng = np.random.default_rng(1)
    hist = rng.normal(0, 0.001, 500)
    out = RandomWalkForecaster().forecast({"r": hist}, horizon=5)
    qf = out["r"]
    assert np.allclose(qf.point, 0.0)


def test_random_walk_forecaster_skips_short_history():
    out = RandomWalkForecaster(min_history=50).forecast({"r": np.zeros(10)}, horizon=5)
    assert "r" not in out


def test_random_walk_forecaster_raw_output_is_flat_across_horizon():
    # per-step marginal quantiles are i.i.d. -> flat; sqrt(h) widening is applied downstream by
    # targets.cumulative_log_return_bps, not by this class (see its docstring).
    rng = np.random.default_rng(2)
    hist = rng.normal(0, 0.002, 1000)
    qf = RandomWalkForecaster().forecast({"r": hist}, horizon=9)["r"]
    q90 = qf.quantiles[0.9]
    assert np.allclose(q90, q90[0])


def test_random_walk_cumulative_widens_like_sqrt_horizon():
    rng = np.random.default_rng(2)
    hist = rng.normal(0, 0.002, 1000)
    qf = RandomWalkForecaster().forecast({"r": hist}, horizon=9)["r"]
    cum = cumulative_log_return_bps(qf)
    # log-scale cumulative quantile deviation from the point forecast should track sqrt(h):
    # this checks the SAME sqrt(9)~=3 widening as before, but at the composed (cumulative)
    # level where it actually belongs.
    single_step = RandomWalkForecaster().forecast({"r": hist}, horizon=1)["r"]
    cum1 = cumulative_log_return_bps(single_step)
    ratio = (cum.quantile_at(0.9) - cum.point[0]) / (cum1.quantile_at(0.9) - cum1.point[0])
    assert ratio == pytest.approx(3.0, rel=0.1)


def test_random_walk_forecaster_calibrated_on_gbm_monte_carlo():
    """80% band from a random-walk forecast fit on i.i.d. returns should cover ~80% of
    freshly-drawn (out-of-sample) paths from the same distribution."""
    rng = np.random.default_rng(42)
    sigma = 0.001
    hist = rng.normal(0, sigma, 3000)
    horizon = 10
    qf = RandomWalkForecaster().forecast({"r": hist}, horizon)["r"]
    cum = cumulative_log_return_bps(qf)
    lo, hi = cum.quantile_at(0.1), cum.quantile_at(0.9)

    n_trials = 5000
    paths = rng.normal(0, sigma, size=(n_trials, horizon))
    actual_bps = 1e4 * (np.exp(paths.sum(axis=1)) - 1.0)
    coverage = float(np.mean((actual_bps >= lo) & (actual_bps <= hi)))
    assert 0.68 <= coverage <= 0.92


# ----------------------------------------------------------------------------- EWMA vol


def test_ewma_vol_forecaster_flat_and_monotonic_quantiles():
    rng = np.random.default_rng(3)
    rets = rng.normal(0, 0.002, 400)
    qf = EWMAVolForecaster().forecast({"r": rets}, horizon=5)["r"]
    assert np.allclose(qf.point, qf.point[0])  # persistence: flat across horizon
    values = [qf.quantiles[level][0] for level in QUANTILE_LEVELS]
    assert all(a <= b + 1e-12 for a, b in zip(values, values[1:]))
    assert qf.point[0] > 0


def test_ewma_vol_forecaster_reacts_to_recent_vol_regime():
    quiet = np.random.default_rng(4).normal(0, 0.0002, 300)
    loud = np.concatenate([quiet, np.random.default_rng(5).normal(0, 0.01, 50)])
    qf_quiet = EWMAVolForecaster().forecast({"r": quiet}, horizon=1)["r"]
    qf_loud = EWMAVolForecaster().forecast({"r": loud}, horizon=1)["r"]
    assert qf_loud.point[0] > qf_quiet.point[0]


# ----------------------------------------------------------------------------- GARCH(1,1)


def _simulate_garch11(n: int, omega: float, alpha: float, beta: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigma2 = omega / (1.0 - alpha - beta)
    eps = np.empty(n)
    for t in range(n):
        e = rng.normal(0.0, math.sqrt(max(sigma2, 1e-12)))
        eps[t] = e
        sigma2 = omega + alpha * e * e + beta * sigma2
    return eps


def test_fit_garch11_recovers_params_roughly():
    true = GarchParams(omega=1e-6, alpha=0.10, beta=0.85)
    r = _simulate_garch11(4000, true.omega, true.alpha, true.beta, seed=7)
    fitted = fit_garch11(r, demean=False)
    assert 0.0 <= fitted.alpha <= 1.0
    assert 0.0 <= fitted.beta <= 1.0
    assert fitted.alpha + fitted.beta < 1.0
    assert abs(fitted.alpha - true.alpha) < 0.15
    assert abs(fitted.beta - true.beta) < 0.25


def test_garch11_forecaster_mean_reverts_toward_its_fitted_long_run_vol():
    true = GarchParams(omega=2e-6, alpha=0.08, beta=0.85)
    r = _simulate_garch11(3000, true.omega, true.alpha, true.beta, seed=11)
    fitted = fit_garch11(r, demean=True)  # GARCH11Forecaster demeans internally too
    qf = GARCH11Forecaster().forecast({"r": r}, horizon=50)["r"]
    long_run_vol = math.sqrt(fitted.long_run_var)
    # far-horizon forecast should be closer to the model's own long-run vol than the 1-step
    # forecast is -- the defining property of a mean-reverting GARCH forecast path.
    assert abs(qf.point[-1] - long_run_vol) <= abs(qf.point[0] - long_run_vol) + 1e-6


def test_garch11_forecaster_skips_short_history():
    out = GARCH11Forecaster(min_history=100).forecast({"r": np.random.default_rng(0).normal(0, 0.001, 20)}, horizon=5)
    assert "r" not in out


# ----------------------------------------------------------------------------- seasonal naive


def test_seasonal_naive_matches_one_period_ago():
    period = 4
    base = np.array([1.0, 2.0, 3.0, 4.0] * 20, dtype=float)
    out = SeasonalNaiveForecaster(period=period).forecast({"v": base}, horizon=period)["v"]
    # each forecast step h equals the value one full period back == the same repeating pattern
    assert np.allclose(out.point, base[-period:])


def test_seasonal_naive_skips_short_history():
    period = 100
    out = SeasonalNaiveForecaster(period=period).forecast({"v": np.arange(50, dtype=float)}, horizon=10)
    assert "v" not in out
