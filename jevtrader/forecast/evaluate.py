"""Rolling-origin (walk-forward) evaluation of a `Forecaster` on real bars, with no look-ahead:
at each origin `t` the forecaster only ever sees `bars[:t+1]`, and is scored against
`bars[t+1 : t+1+horizon]`, which it never received. This is what `scripts/forecast/eval_real.py`
runs against the cached BTC/USD minute bars, and what any strategy-facing scoreboard
(`docs/FORECASTING.md`) should run before trusting a forecaster over its baselines.

Metrics:

- **Pinball (quantile) loss** per quantile level, and a **CRPS approximation**: the standard
  identity `CRPS = 2 * integral_0^1 pinball_tau dtau`, discretized over the nine quantile levels
  we have (this under-covers the extreme tails outside q10/q90, so treat it as directional, not
  exact).
- **80% interval coverage**: the fraction of realized outcomes inside `[q10, q90]`; should be
  close to 0.80 for a well-calibrated forecaster (this is the standard sanity check quantile
  forecasters are held to).
- **MASE vs. a zero-drift random walk** (for returns) or **vs. persistence** (for vol): mean
  absolute error of the forecaster's point forecast, divided by the same for the honest naive
  baseline. Below 1.0 means "beats the naive baseline"; this is the number that decides whether
  a forecaster is worth using at all, per the honest-prior note in `docs/FORECASTING.md`.
- **Directional hit rate after costs** (returns only): among windows where the forecaster's
  P(up > cost) or P(down > cost) exceeded 50%, the fraction where the realized move actually
  cleared the same cost in that direction. This is the number `ForecastAdvisor`'s edge lives or
  dies on.
- **QLIKE and MSE vs. EWMA/GARCH** (volatility only): QLIKE (`log(sigma2_hat) +
  sigma2_actual/sigma2_hat`, lower is better) is the standard loss for comparing volatility
  forecasters because, unlike MSE, it penalizes underestimating volatility much more than
  overestimating it -- exactly the asymmetry that matters for risk sizing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from jevtrader.forecast.base import QUANTILE_LEVELS, Forecaster
from jevtrader.forecast.targets import cumulative_log_return_bps, log_return_series, realized_vol_series

_MIN_CONTEXT = 30


def pinball_loss(actual: float, predicted: float, level: float) -> float:
    diff = actual - predicted
    return float(max(level * diff, (level - 1.0) * diff))


def _crps_approx(pinball_by_level: Mapping[float, float]) -> float:
    """`CRPS ~= 2 * mean(pinball losses over the quantile grid)`. Exact in the limit of a dense
    quantile grid over `[0, 1]`; with only nine levels (0.1..0.9) this is a documented
    approximation that ignores the extreme tails."""
    if not pinball_by_level:
        return float("nan")
    return 2.0 * float(np.mean(list(pinball_by_level.values())))


def _origins(n: int, min_train: int, horizon: int, step: int, max_windows: Optional[int]) -> List[int]:
    last = n - horizon - 1
    if last < min_train:
        return []
    origins = list(range(min_train, last + 1, max(1, step)))
    if max_windows is not None and len(origins) > max_windows:
        # Evenly-spaced subsample so a huge dataset still gets coverage across its whole span,
        # not just its earliest windows.
        idx = np.linspace(0, len(origins) - 1, max_windows).round().astype(int)
        origins = [origins[i] for i in sorted(set(idx.tolist()))]
    return origins


@dataclass
class EvalResult:
    """Everything `evaluate_returns`/`evaluate_volatility` measured for one (forecaster,
    horizon) pair. `n_windows` is how many rolling origins actually produced a scored forecast
    (a forecaster may legitimately skip an origin, e.g. too little history)."""

    target: str
    horizon: int
    n_windows: int
    pinball_by_level: Dict[float, float] = field(default_factory=dict)
    crps: float = float("nan")
    coverage_80: float = float("nan")
    mase: float = float("nan")
    mae: float = float("nan")
    naive_mae: float = float("nan")
    directional_hit_rate: Optional[float] = None
    directional_n: int = 0
    qlike: Optional[float] = None
    mse: Optional[float] = None
    mse_vs_naive: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "horizon": self.horizon,
            "n_windows": self.n_windows,
            "pinball_by_level": dict(self.pinball_by_level),
            "crps": self.crps,
            "coverage_80": self.coverage_80,
            "mase": self.mase,
            "mae": self.mae,
            "naive_mae": self.naive_mae,
            "directional_hit_rate": self.directional_hit_rate,
            "directional_n": self.directional_n,
            "qlike": self.qlike,
            "mse": self.mse,
            "mse_vs_naive": self.mse_vs_naive,
        }


def evaluate_returns(
    bars: pd.DataFrame,
    forecaster: Forecaster,
    horizon: int,
    cost_bps: float,
    *,
    min_train: int = 500,
    step: int = 60,
    max_windows: Optional[int] = 500,
    series_key: str = "return",
) -> EvalResult:
    """Rolling-origin evaluation of a return forecaster. `horizon` is in bars. `cost_bps` is the
    round-trip cost used for the directional-hit-rate metric (same number `build_state` and
    `direction_questions` use)."""
    closes = bars["close"].astype(float).to_numpy()
    n = closes.shape[0]
    origins = _origins(n, max(min_train, _MIN_CONTEXT + 1), horizon, step, max_windows)

    pinball_sums = {level: 0.0 for level in QUANTILE_LEVELS}
    coverage_hits = 0
    abs_errors: List[float] = []
    naive_abs_errors: List[float] = []
    dir_hits = 0
    dir_n = 0
    used = 0

    for t in origins:
        context = closes[: t + 1]
        log_ctx = log_return_series(context)
        if log_ctx.size < _MIN_CONTEXT:
            continue
        try:
            raw = forecaster.forecast({series_key: log_ctx}, horizon)
        except Exception:
            continue
        qf = raw.get(series_key)
        if qf is None or qf.horizon < horizon:
            continue
        cum = cumulative_log_return_bps(qf)
        actual_bps = 1e4 * (math.log(closes[t + horizon]) - math.log(closes[t]))
        point_pred = float(cum.point[0])

        abs_errors.append(abs(actual_bps - point_pred))
        naive_abs_errors.append(abs(actual_bps))  # naive point forecast: zero drift
        for level in QUANTILE_LEVELS:
            pinball_sums[level] += pinball_loss(actual_bps, cum.quantile_at(level), level)
        lo, hi = cum.quantile_at(0.1), cum.quantile_at(0.9)
        if lo <= actual_bps <= hi:
            coverage_hits += 1

        p_up = cum.prob_above(float(cost_bps))
        p_down = cum.prob_below(-float(cost_bps))
        if p_up > 0.5 or p_down > 0.5:
            dir_n += 1
            predicted_dir = 1 if p_up > p_down else -1
            actual_dir = 1 if actual_bps > cost_bps else (-1 if actual_bps < -cost_bps else 0)
            if predicted_dir == actual_dir:
                dir_hits += 1
        used += 1

    if used == 0:
        return EvalResult(target="return", horizon=horizon, n_windows=0)

    pinball_by_level = {level: pinball_sums[level] / used for level in QUANTILE_LEVELS}
    mae = float(np.mean(abs_errors))
    naive_mae = float(np.mean(naive_abs_errors))
    return EvalResult(
        target="return",
        horizon=horizon,
        n_windows=used,
        pinball_by_level=pinball_by_level,
        crps=_crps_approx(pinball_by_level),
        coverage_80=coverage_hits / used,
        mase=mae / naive_mae if naive_mae > 1e-9 else float("nan"),
        mae=mae,
        naive_mae=naive_mae,
        directional_hit_rate=(dir_hits / dir_n) if dir_n > 0 else None,
        directional_n=dir_n,
    )


def evaluate_volatility(
    bars: pd.DataFrame,
    forecaster: Forecaster,
    horizon: int,
    *,
    vol_window: int = 20,
    min_train: int = 500,
    step: int = 60,
    max_windows: Optional[int] = 500,
    series_key: str = "vol",
) -> EvalResult:
    """Rolling-origin evaluation of a volatility forecaster. `horizon` is in bars *of the
    realized-vol series* (i.e. `realized_vol_series` steps, which are 1-step-return-aligned, so
    a `horizon` of 15 means "15 bars ahead", same units as `evaluate_returns`)."""
    closes = bars["close"].astype(float).to_numpy()
    rv = realized_vol_series(closes, vol_window)
    n = rv.shape[0]
    origins = _origins(n, max(min_train, _MIN_CONTEXT + vol_window), horizon, step, max_windows)

    se_list: List[float] = []
    naive_se_list: List[float] = []
    qlike_list: List[float] = []
    used = 0

    for t in origins:
        hist = rv[: t + 1]
        hist = hist[~np.isnan(hist)]
        if hist.size < _MIN_CONTEXT:
            continue
        target_idx = t + horizon
        if target_idx >= n or np.isnan(rv[target_idx]):
            continue
        try:
            raw = forecaster.forecast({series_key: hist}, horizon)
        except Exception:
            continue
        qf = raw.get(series_key)
        if qf is None or qf.horizon < horizon:
            continue
        forecast_var = max(float(qf.point[horizon - 1]) ** 2, 1e-14)
        actual_var = max(float(rv[target_idx]) ** 2, 1e-14)
        naive_var = max(float(hist[-1]) ** 2, 1e-14)  # persistence baseline

        se_list.append((actual_var - forecast_var) ** 2)
        naive_se_list.append((actual_var - naive_var) ** 2)
        qlike_list.append(math.log(forecast_var) + actual_var / forecast_var)
        used += 1

    if used == 0:
        return EvalResult(target="vol", horizon=horizon, n_windows=0)

    mse = float(np.mean(se_list))
    naive_mse = float(np.mean(naive_se_list))
    return EvalResult(
        target="vol",
        horizon=horizon,
        n_windows=used,
        mse=mse,
        mse_vs_naive=(mse / naive_mse) if naive_mse > 1e-18 else float("nan"),
        qlike=float(np.mean(qlike_list)),
        mae=float(np.sqrt(mse)),  # RMSE-of-variance, reported in the `mae` slot for the table
        naive_mae=float(np.sqrt(naive_mse)),
    )


def to_markdown(results: Mapping[str, Mapping[int, EvalResult]], *, title: str = "Forecast evaluation") -> str:
    """`results`: `{forecaster_name: {horizon: EvalResult}}`, all for the same target (call once
    per target -- returns, then vol -- and concatenate). Produces one Markdown table."""
    lines = [f"## {title}", ""]
    any_result = next((r for by_h in results.values() for r in by_h.values()), None)
    if any_result is None:
        return "\n".join(lines + ["_no results_"])
    if any_result.target == "return":
        lines.append("| forecaster | horizon | n | MASE | coverage_80 | CRPS | dir. hit rate (n) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for name, by_h in results.items():
            for h, r in sorted(by_h.items()):
                if r.n_windows == 0:
                    lines.append(f"| {name} | {h} | 0 | - | - | - | - |")
                    continue
                hit = f"{r.directional_hit_rate:.3f} (n={r.directional_n})" if r.directional_hit_rate is not None else "n/a"
                lines.append(f"| {name} | {h} | {r.n_windows} | {r.mase:.3f} | {r.coverage_80:.3f} | {r.crps:.2f} | {hit} |")
    else:
        lines.append("| forecaster | horizon | n | QLIKE | MSE (var) | MSE vs naive |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for name, by_h in results.items():
            for h, r in sorted(by_h.items()):
                if r.n_windows == 0:
                    lines.append(f"| {name} | {h} | 0 | - | - | - |")
                    continue
                lines.append(f"| {name} | {h} | {r.n_windows} | {r.qlike:.4f} | {r.mse:.3e} | {r.mse_vs_naive:.3f} |")
    return "\n".join(lines)
