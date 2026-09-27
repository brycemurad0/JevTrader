"""`Forecaster` backed by Google's TimesFM foundation model (github.com/google-research/timesfm).

`timesfm` (and `timesfm3`) are OPTIONAL dependencies: nothing at module import time touches
them, so `import jevtrader.forecast.timesfm_backend` always succeeds even on a machine that
never installed the package (e.g. this container, which also can't reach huggingface.co to
download weights). The import happens lazily, inside `_ensure_loaded`, the first time
`forecast()` is actually called -- and raises `TimesFMNotInstalled` with the exact `pip install`
line if it's missing.

License guard (see `docs/FORECASTING.md#license`): TimesFM 2.5 weights
(`google/timesfm-2.5-200m-pytorch`) are Apache-2.0 and are the production default. TimesFM 3.0
weights (`google/timesfm-3.0-pytorch`) are licensed **non-commercial, non-production only**.
`TimesFMForecaster(version="3.0", ...)` therefore refuses to construct unless BOTH:

1. `mode is TradingMode.BACKTEST` (never PAPER or LIVE), AND
2. the environment variable `TIMESFM3_LICENSE_ACK=research_only` is set, an explicit,
   deliberate opt-in that has to be set by a human, not a default in any config file.

Batching: `forecast()` takes the *whole* dict of `{key: series}` in one call and does a single
forward pass across all of them (one `model.forecast(inputs=[...])` call), which is what makes
per-bar-close, multi-symbol forecasting fast enough to run outside the tick hot path -- see the
latency notes in `docs/FORECASTING.md`.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

from jevtrader.core.broker import TradingMode
from jevtrader.forecast.base import QUANTILE_LEVELS, QuantileForecast

logger = logging.getLogger("jevtrader.forecast.timesfm")

TIMESFM3_LICENSE_ACK_ENV = "TIMESFM3_LICENSE_ACK"
TIMESFM3_LICENSE_ACK_VALUE = "research_only"

_INSTALL_HINT_25 = "pip install 'timesfm[torch]'   # TimesFM 2.5, Apache-2.0, production default"
_INSTALL_HINT_30_TORCH = "pip install timesfm3   # TimesFM 3.0, NON-COMMERCIAL weights, backtest-only"
_INSTALL_HINT_30_MLX = "pip install 'timesfm3[mlx]'   # TimesFM 3.0 on Apple Silicon, NON-COMMERCIAL weights, backtest-only"

DEFAULT_MODEL_ID_25 = "google/timesfm-2.5-200m-pytorch"
DEFAULT_MODEL_ID_30 = "google/timesfm-3.0-pytorch"


class TimesFMNotInstalled(ImportError):
    """Raised instead of a bare `ModuleNotFoundError` so callers get install instructions."""


class TimesFM3LicenseError(RuntimeError):
    """Raised when TimesFM 3.0 (non-commercial, non-production weights) is requested outside
    the one mode+ack combination it is licensed for in this codebase."""


def _default_device() -> str:
    """Best-effort `cuda` > `mps` > `cpu` autodetection. Never raises: any import/attribute
    problem just falls back to `cpu`, since torch itself may not be installed yet."""
    try:
        import torch
    except ImportError:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover - defensive against partial/odd torch builds
        pass
    return "cpu"


@dataclass
class TimesFMForecaster:
    """`Forecaster` protocol implementation backed by TimesFM.

    `version`: `"2.5"` (Apache-2.0, production default) or `"3.0"` (non-commercial, see module
    docstring -- requires `mode=TradingMode.BACKTEST` and the license-ack env var).
    `device`: `"auto"` (cuda > mps > cpu), or an explicit `"cuda"` / `"mps"` / `"cpu"`.
    `backend`: `"torch"` (both versions) or `"mlx"` (3.0 only, Apple Silicon).
    `max_context` / `max_horizon`: compiled context/horizon caps (see TimesFM's `ForecastConfig`).
    `mode`: the current `TradingMode`; only consulted for the 3.0 license guard, but always worth
    passing so a caller can't accidentally construct a backtest-only forecaster for a live run.
    """

    version: str = "2.5"
    device: str = "auto"
    max_context: int = 1024
    max_horizon: int = 256
    per_core_batch_size: int = 32
    backend: str = "torch"
    model_id: Optional[str] = None
    mode: Optional[TradingMode] = None

    def __post_init__(self) -> None:
        if self.version not in ("2.5", "3.0"):
            raise ValueError(f"unknown TimesFM version {self.version!r}; use '2.5' or '3.0'")
        if self.backend not in ("torch", "mlx"):
            raise ValueError(f"unknown backend {self.backend!r}; use 'torch' or 'mlx'")
        if self.backend == "mlx" and self.version != "3.0":
            raise ValueError("backend='mlx' is only available for version='3.0'")
        self._check_license()
        self._model: Any = None
        self._backend_kind: str = ""
        self._resolved_device: str = self.device

    # ------------------------------------------------------------------- license guard

    def _check_license(self) -> None:
        if self.version != "3.0":
            return
        if self.mode is None or self.mode is not TradingMode.BACKTEST:
            raise TimesFM3LicenseError(
                f"TimesFM 3.0 weights ({DEFAULT_MODEL_ID_30}) are licensed NON-COMMERCIAL, "
                f"non-production use only. JevTrader refuses to load them outside "
                f"TradingMode.BACKTEST (got mode={self.mode!r}). Use version='2.5' "
                f"(Apache-2.0) for paper/live trading -- see docs/FORECASTING.md#license."
            )
        ack = os.environ.get(TIMESFM3_LICENSE_ACK_ENV, "")
        if ack != TIMESFM3_LICENSE_ACK_VALUE:
            raise TimesFM3LicenseError(
                "TimesFM 3.0 weights are licensed NON-COMMERCIAL. Even in TradingMode.BACKTEST, "
                f"JevTrader requires an explicit human override: set "
                f"{TIMESFM3_LICENSE_ACK_ENV}={TIMESFM3_LICENSE_ACK_VALUE!r} to acknowledge you "
                "accept that license for research/backtest-only use. "
                "See docs/FORECASTING.md#license."
            )

    # ------------------------------------------------------------------- lazy load

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        device = self.device if self.device != "auto" else _default_device()
        self._resolved_device = device
        try:
            import torch

            torch.set_float32_matmul_precision("high")
        except Exception:  # pragma: no cover - torch may not be installed, or no matmul knob
            pass
        if self.version == "2.5":
            self._load_25(device)
        else:
            self._load_30(device)

    def _load_25(self, device: str) -> None:
        try:
            import timesfm
        except ImportError as e:
            raise TimesFMNotInstalled(
                "timesfm is not installed (it is an OPTIONAL dependency of "
                f"jevtrader.forecast). Install it with:\n\n    {_INSTALL_HINT_25}\n"
            ) from e
        model_id = self.model_id or DEFAULT_MODEL_ID_25
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)
        model.compile(
            timesfm.ForecastConfig(
                max_context=self.max_context,
                max_horizon=self.max_horizon,
                normalize_inputs=True,
                per_core_batch_size=self.per_core_batch_size,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                # Our inputs (targets.py) are always log returns, realized-vol levels, or
                # log-volume -- signed or not-obviously-positive after the log transform -- so
                # we never want the model silently clamping forecasts to >= 0.
                infer_is_positive=False,
                fix_quantile_crossing=True,
            )
        )
        if hasattr(model, "to"):
            try:
                model.to(device)
            except Exception:  # pragma: no cover - best-effort device placement
                logger.warning("TimesFM 2.5: could not move model to device=%s, continuing on default device", device)
        self._model = model
        self._backend_kind = "25-torch"

    def _load_30(self, device: str) -> None:
        model_id = self.model_id or DEFAULT_MODEL_ID_30
        if self.backend == "mlx":
            try:
                from timesfm3.mlx import TimesFM3Forecaster  # type: ignore[import-not-found]
            except ImportError as e:
                raise TimesFMNotInstalled(
                    f"timesfm3 (mlx backend) is not installed. Install it with:\n\n"
                    f"    {_INSTALL_HINT_30_MLX}\n\n"
                    f"NOTE: {model_id} is licensed non-commercial/non-production -- "
                    "see docs/FORECASTING.md#license."
                ) from e
            self._model = TimesFM3Forecaster.from_pretrained(model_id)
        else:
            try:
                from timesfm3 import ModelConfig, TimesFM3Evaluator  # type: ignore[import-not-found]
            except ImportError as e:
                raise TimesFMNotInstalled(
                    f"timesfm3 is not installed. Install it with:\n\n    {_INSTALL_HINT_30_TORCH}\n\n"
                    f"NOTE: {model_id} is licensed non-commercial/non-production -- "
                    "see docs/FORECASTING.md#license."
                ) from e
            self._model = TimesFM3Evaluator(ModelConfig(checkpoint_path=model_id))
        self._backend_kind = f"30-{self.backend}"

    # ------------------------------------------------------------------- forecast

    def forecast(self, series: Mapping[str, np.ndarray], horizon: int) -> Dict[str, QuantileForecast]:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if horizon > self.max_horizon:
            raise ValueError(f"horizon={horizon} exceeds max_horizon={self.max_horizon}; construct with a larger max_horizon")
        keys = list(series.keys())
        if not keys:
            return {}
        self._ensure_loaded()
        inputs = [np.asarray(series[k], dtype=np.float32)[-self.max_context :] for k in keys]
        t0 = time.perf_counter()
        if self._backend_kind == "25-torch":
            out = self._forecast_25(keys, inputs, horizon)
        else:
            out = self._forecast_30(keys, inputs, horizon)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        per_symbol_ms = elapsed_ms / len(keys)
        for qf in out.values():
            object.__setattr__(qf, "latency_ms", per_symbol_ms)
        return out

    def _forecast_25(self, keys: list[str], inputs: list[np.ndarray], horizon: int) -> Dict[str, QuantileForecast]:
        point, quantiles = self._model.forecast(horizon=horizon, inputs=inputs)
        point = np.asarray(point, dtype=float)
        quantiles = np.asarray(quantiles, dtype=float)
        if quantiles.ndim != 3 or quantiles.shape[2] < 10:
            raise ValueError(f"unexpected TimesFM 2.5 quantile shape {quantiles.shape}; expected (batch, horizon, 10) [mean, q10..q90]")
        out: Dict[str, QuantileForecast] = {}
        for i, key in enumerate(keys):
            # index 0 is the mean (dropped -- we use the median, quantiles[..., 5], as `point`
            # instead, consistent with what `point_forecast` itself returns); indices 1..9 are
            # q10..q90 in order.
            q = {level: quantiles[i, :, j + 1] for j, level in enumerate(QUANTILE_LEVELS)}
            out[key] = QuantileForecast(point=point[i], quantiles=q)
        return out

    def _forecast_30(self, keys: list[str], inputs: list[np.ndarray], horizon: int) -> Dict[str, QuantileForecast]:
        results = list(
            self._model.predict_batch(
                contexts=inputs,
                horizon=horizon,
                return_quantiles=True,
                sort_quantiles=True,
                make_positive=False,
            )
        )
        out: Dict[str, QuantileForecast] = {}
        for key, res in zip(keys, results):
            point = np.asarray(res.forecast, dtype=float).reshape(-1)
            q_arr = np.asarray(res.quantiles, dtype=float)
            # Per the TimesFM 3.0 reference source (`median_quantile_index=4`, 0-indexed, of 9
            # quantiles), the quantile axis is [0.1, 0.2, ..., 0.9] ascending -- no leading mean
            # column, unlike 2.5. We assert the shape rather than silently mis-map levels if a
            # future checkpoint changes the quantile grid.
            if q_arr.ndim != 2 or q_arr.shape[-1] != len(QUANTILE_LEVELS):
                raise ValueError(f"unexpected TimesFM 3.0 quantile shape {q_arr.shape}; expected (horizon, {len(QUANTILE_LEVELS)})")
            q = {level: q_arr[:, j] for j, level in enumerate(QUANTILE_LEVELS)}
            out[key] = QuantileForecast(point=point, quantiles=q)
        return out
