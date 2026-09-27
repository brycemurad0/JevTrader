"""Walk-forward evaluation, purged/embargoed cross-validation splits, parameter sweeps with a
trial-count-aware Deflated Sharpe Ratio, and a fee/slippage robustness stress test.

None of this runs a backtest itself -- every function here takes a `run_fn`/`data_slicer`
callback that the caller wires to `jevtrader.backtest.engine.Backtester`, so this module has no
dependency on how strategies or data are constructed. This keeps it trivially testable with tiny
fake `run_fn`s.
"""

from __future__ import annotations

import copy
import itertools
import random
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from jevtrader.backtest.metrics import compute_metrics, deflated_sharpe_ratio
from jevtrader.backtest.sim_broker import SlippageModel
from jevtrader.core.fees import AlpacaCryptoFees, AlpacaEquityFees, BpsFees, CompositeFees, FeeModel

# ----------------------------------------------------------------------------- splits


@dataclass(frozen=True)
class Split:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_splits(
    index: pd.DatetimeIndex,
    train_periods: int,
    test_periods: int,
    step_periods: Optional[int] = None,
    anchored: bool = False,
    embargo_periods: int = 0,
) -> list[Split]:
    """Rolling (default) or anchored train/test splits over `index`.

    Rolling: the train window has a fixed length `train_periods` and slides forward by
    `step_periods` (default `test_periods`, i.e. non-overlapping test windows) each split.
    Anchored: the train window always starts at `index[0]` and grows.

    `embargo_periods` inserts a gap of that many samples between the end of train and the start
    of test on every split (purging potential leakage from e.g. autocorrelated features/labels
    that span the boundary). Test windows never overlap each other or their own train window.
    """
    n = len(index)
    step = step_periods if step_periods is not None else test_periods
    splits: list[Split] = []
    if anchored:
        test_start_i = train_periods
        while test_start_i + test_periods <= n:
            train_end_i = test_start_i - embargo_periods
            if train_end_i <= 0:
                test_start_i += step
                continue
            test_end_i = min(test_start_i + test_periods, n) - 1
            splits.append(Split(index[0], index[train_end_i - 1], index[test_start_i], index[test_end_i]))
            test_start_i += step
    else:
        train_start_i = 0
        while train_start_i + train_periods + embargo_periods + test_periods <= n:
            train_end_i = train_start_i + train_periods
            test_start_i = train_end_i + embargo_periods
            test_end_i = min(test_start_i + test_periods, n) - 1
            splits.append(Split(index[train_start_i], index[train_end_i - 1], index[test_start_i], index[test_end_i]))
            train_start_i += step
    return splits


@dataclass(frozen=True)
class PurgedFold:
    fold: int
    train_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex


def purged_kfold_splits(index: pd.DatetimeIndex, n_splits: int, embargo_periods: int = 0) -> list[PurgedFold]:
    """Purged & embargoed K-fold (Lopez de Prado, "Advances in Financial Machine Learning", ch.
    7): `index` is cut into `n_splits` contiguous, equal-ish folds. For fold `i` as the test set,
    the training set is every OTHER sample, with `embargo_periods` samples immediately before and
    after the test fold's boundary also removed from training -- purging leakage that would
    otherwise leak from adjacent, autocorrelated samples straddling the train/test edge.
    """
    n = len(index)
    fold_positions = np.array_split(np.arange(n), n_splits)
    folds: list[PurgedFold] = []
    for i, test_idx in enumerate(fold_positions):
        if len(test_idx) == 0:
            continue
        test_start_i, test_end_i = int(test_idx[0]), int(test_idx[-1])
        purge_lo = max(0, test_start_i - embargo_periods)
        purge_hi = min(n - 1, test_end_i + embargo_periods)
        train_positions = np.array([j for j in range(n) if j < purge_lo or j > purge_hi])
        if len(train_positions) == 0:
            continue
        folds.append(PurgedFold(fold=i, train_index=index[train_positions], test_index=index[test_idx]))
    return folds


# ----------------------------------------------------------------------------- parameter sweeps


@dataclass
class Trial:
    params: dict
    metrics: dict
    sharpe: float


@dataclass
class SweepResult:
    trials: list[Trial]
    n_trials: int
    best: Optional[Trial]
    best_dsr: Optional[float]


def sweep(
    param_grid: dict[str, Sequence],
    run_fn: Callable[[dict], dict],
    n_trials: Optional[int] = None,
    seed: int = 0,
) -> SweepResult:
    """Evaluate `run_fn(params) -> metrics_dict` (typically a `BacktestResult.metrics`) over every
    combination in `param_grid`, or a random sample of `n_trials` of them if the grid is larger.

    Reports the Deflated Sharpe Ratio of the BEST observed in-sample Sharpe against the number of
    trials actually run -- the standard multiple-testing correction for a parameter search:
    trying more combinations makes even a lucky, skill-less best result look better, and the DSR
    discounts for exactly that.
    """
    keys = list(param_grid)
    combos = list(itertools.product(*(param_grid[k] for k in keys)))
    if n_trials is not None and n_trials < len(combos):
        rng = random.Random(seed)
        combos = rng.sample(combos, n_trials)
    trials: list[Trial] = []
    for combo in combos:
        params = dict(zip(keys, combo))
        metrics = run_fn(params)
        sharpe = float(metrics.get("sharpe", float("nan")))
        trials.append(Trial(params=params, metrics=metrics, sharpe=sharpe))

    finite = [t for t in trials if np.isfinite(t.sharpe)]
    best = max(finite, key=lambda t: t.sharpe) if finite else None
    best_dsr = None
    if best is not None:
        n_obs = int(best.metrics.get("n_fills", 0))  # not the OOS returns; caller can also pass a returns-based dsr already in metrics
        best_dsr = best.metrics.get("dsr")
    return SweepResult(trials=trials, n_trials=len(trials), best=best, best_dsr=best_dsr)


# ----------------------------------------------------------------------------- walk-forward run


@dataclass
class WalkForwardFoldResult:
    split: Split
    metrics: dict
    returns: pd.Series


@dataclass
class WalkForwardResult:
    folds: list[WalkForwardFoldResult]
    combined_returns: pd.Series
    dsr: dict
    n_trials: int


def walk_forward_run(
    splits: Sequence[Split],
    run_fn: Callable[[Split], "jevtrader.backtest.engine.BacktestResult"],  # noqa: F821
    n_trials: int = 1,
) -> WalkForwardResult:
    """Run `run_fn(split) -> BacktestResult` once per split (the caller decides what "run" means:
    fit params on `[split.train_start, split.train_end]`, then backtest
    `[split.test_start, split.test_end]` with those params -- this function only scores the
    resulting OUT-OF-SAMPLE equity curves) and aggregates their per-period returns into one
    combined OOS return series, scored with the Deflated Sharpe Ratio at `n_trials` (the number
    of parameterizations tried while fitting, if any -- pass 1 if there was no search).
    """
    folds: list[WalkForwardFoldResult] = []
    all_returns: list[pd.Series] = []
    for split in splits:
        result = run_fn(split)
        returns = result.equity_curve.pct_change().dropna()
        folds.append(WalkForwardFoldResult(split=split, metrics=result.metrics, returns=returns))
        all_returns.append(returns)
    combined = pd.concat(all_returns) if all_returns else pd.Series(dtype=float)
    dsr_info = deflated_sharpe_ratio(combined, n_trials=n_trials) if len(combined) >= 3 else {
        "dsr": float("nan"),
        "expected_max_sr": float("nan"),
        "sr": float("nan"),
        "n_trials": n_trials,
        "n_obs": len(combined),
    }
    return WalkForwardResult(folds=folds, combined_returns=combined, dsr=dsr_info, n_trials=n_trials)


# ----------------------------------------------------------------------------- robustness / fee sensitivity


def _scale_fee_model(fee_model: FeeModel, factor: float) -> FeeModel:
    fm = copy.deepcopy(fee_model)
    _scale_fee_fields(fm, factor)
    return fm


def _scale_fee_fields(fm: FeeModel, factor: float) -> None:
    if isinstance(fm, CompositeFees):
        _scale_fee_fields(fm.equity, factor)
        _scale_fee_fields(fm.crypto, factor)
    elif isinstance(fm, BpsFees):
        fm.maker_bps *= factor
        fm.taker_bps *= factor
    elif isinstance(fm, AlpacaEquityFees):
        fm.commission_per_share *= factor
        fm.sec_fee_per_million *= factor
        fm.taf_per_share *= factor
        fm.taf_max *= factor
    elif isinstance(fm, AlpacaCryptoFees):
        fm.tiers = [(vol, maker * factor, taker * factor) for vol, maker, taker in fm.tiers]
    # ZeroFees and anything else: nothing to scale.


def _scale_slippage_model(slippage: SlippageModel, factor: float) -> SlippageModel:
    return SlippageModel(
        half_spread_bps=slippage.half_spread_bps * factor,
        slippage_bps=slippage.slippage_bps * factor,
        impact_coeff_bps=slippage.impact_coeff_bps * factor,
        max_participation=slippage.max_participation,
    )


def robustness_report(
    run_fn: Callable[[FeeModel, SlippageModel], dict],
    base_fees: FeeModel,
    base_slippage: Optional[SlippageModel] = None,
    min_sharpe: float = 0.0,
    max_drawdown_floor: float = -0.5,
) -> dict:
    """Re-run `run_fn(fee_model, slippage_model) -> metrics_dict` under the base costs and three
    stress scenarios (fees x1.5, fees x2, slippage x2), and report a pass/fail verdict.

    Checks (all must pass for `passed=True`):
      * `base_sharpe_positive`: the base-cost Sharpe ratio is > 0.
      * `sharpe_above_min`: the base-cost Sharpe ratio is >= `min_sharpe`.
      * `survives_cost_stress`: Sharpe stays > 0 in every stress scenario (a strategy whose edge
        is an artifact of underpriced costs should fail this first).
      * `max_drawdown_within_floor`: base-cost max drawdown is no worse than `max_drawdown_floor`.
    """
    base_slippage = base_slippage or SlippageModel()
    scenarios = {
        "base": (_scale_fee_model(base_fees, 1.0), _scale_slippage_model(base_slippage, 1.0)),
        "fees_x1.5": (_scale_fee_model(base_fees, 1.5), _scale_slippage_model(base_slippage, 1.0)),
        "fees_x2": (_scale_fee_model(base_fees, 2.0), _scale_slippage_model(base_slippage, 1.0)),
        "slippage_x2": (_scale_fee_model(base_fees, 1.0), _scale_slippage_model(base_slippage, 2.0)),
    }
    outcomes: dict[str, dict] = {}
    for name, (fee_model, slip) in scenarios.items():
        outcomes[name] = run_fn(fee_model, slip)

    base_sharpe = float(outcomes["base"].get("sharpe", float("nan")))
    base_mdd = float(outcomes["base"].get("max_drawdown", float("nan")))
    stress_sharpes = [float(outcomes[k].get("sharpe", float("-inf"))) for k in ("fees_x1.5", "fees_x2", "slippage_x2")]

    checks = {
        "base_sharpe_positive": bool(np.isfinite(base_sharpe) and base_sharpe > 0),
        "sharpe_above_min": bool(np.isfinite(base_sharpe) and base_sharpe >= min_sharpe),
        "survives_cost_stress": bool(all(np.isfinite(s) and s > 0 for s in stress_sharpes)),
        "max_drawdown_within_floor": bool(np.isfinite(base_mdd) and base_mdd >= max_drawdown_floor),
    }
    return {"scenarios": outcomes, "checks": checks, "passed": all(checks.values())}
