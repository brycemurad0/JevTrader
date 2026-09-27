import pandas as pd

from jevtrader.backtest.sim_broker import SlippageModel
from jevtrader.backtest.walkforward import (
    purged_kfold_splits,
    robustness_report,
    sweep,
    walk_forward_run,
    walk_forward_splits,
)
from jevtrader.core.fees import BpsFees


def _idx(n=120):
    return pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")


def test_rolling_splits_are_non_overlapping_with_embargo_gap():
    idx = _idx()
    splits = walk_forward_splits(idx, train_periods=30, test_periods=10, embargo_periods=2)
    assert len(splits) > 1
    pos = {ts: i for i, ts in enumerate(idx)}
    for s in splits:
        assert s.train_end < s.test_start
        assert pos[s.test_start] - pos[s.train_end] >= 2  # the embargo gap
    for a, b in zip(splits, splits[1:]):
        assert a.test_end < b.test_start  # test windows never overlap


def test_rolling_splits_train_window_never_touches_its_own_test_window():
    idx = _idx()
    splits = walk_forward_splits(idx, train_periods=20, test_periods=5, embargo_periods=0)
    for s in splits:
        assert s.train_start <= s.train_end < s.test_start <= s.test_end


def test_anchored_splits_train_start_is_fixed_and_grows():
    idx = _idx()
    splits = walk_forward_splits(idx, train_periods=30, test_periods=10, anchored=True, embargo_periods=1)
    assert len(splits) > 1
    assert all(s.train_start == idx[0] for s in splits)
    for a, b in zip(splits, splits[1:]):
        assert b.train_end > a.train_end  # the anchored train window strictly grows


def test_purged_kfold_train_and_test_never_overlap_and_embargo_is_respected():
    idx = _idx(60)
    folds = purged_kfold_splits(idx, n_splits=5, embargo_periods=3)
    pos = {ts: i for i, ts in enumerate(idx)}
    assert len(folds) == 5
    for f in folds:
        assert set(f.train_index).isdisjoint(set(f.test_index))
        test_positions = {pos[ts] for ts in f.test_index}
        test_lo, test_hi = min(test_positions), max(test_positions)
        for ts in f.train_index:
            p = pos[ts]
            assert p not in test_positions
            # nothing within the embargo window of either boundary may appear in train
            assert not (test_lo - 3 <= p < test_lo) and not (test_hi < p <= test_hi + 3)


def test_sweep_records_n_trials_and_picks_best_by_sharpe():
    def run_fn(params):
        # a deterministic function of params so we know the best combo ahead of time
        sharpe = params["a"] * 1.0 + params["b"] * 0.1
        return {"sharpe": sharpe, "n_fills": 10, "dsr": 0.5}

    grid = {"a": [1, 2, 3], "b": [0, 1]}
    result = sweep(grid, run_fn)
    assert result.n_trials == 6
    assert result.best is not None
    assert result.best.params == {"a": 3, "b": 1}


def test_sweep_respects_n_trials_cap_with_random_sample():
    calls = []

    def run_fn(params):
        calls.append(params)
        return {"sharpe": sum(params.values())}

    grid = {"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]}  # 16 combos
    result = sweep(grid, run_fn, n_trials=5, seed=0)
    assert result.n_trials == 5
    assert len(calls) == 5


def test_walk_forward_run_aggregates_oos_returns_and_computes_dsr():
    import numpy as np

    class _FakeResult:
        def __init__(self, equity):
            self.equity_curve = equity
            self.metrics = {"sharpe": 1.0}

    rng = np.random.default_rng(3)

    def run_fn(split):
        idx = pd.date_range(split.test_start, split.test_end, freq="D", tz="UTC")
        rets = rng.normal(0.001, 0.01, len(idx))
        equity = pd.Series(100_000 * np.cumprod(1 + rets), index=idx)
        return _FakeResult(equity)

    idx = _idx(100)
    splits = walk_forward_splits(idx, train_periods=30, test_periods=10, embargo_periods=1)
    wf = walk_forward_run(splits, run_fn, n_trials=len(splits))
    assert len(wf.folds) == len(splits)
    assert len(wf.combined_returns) > 0
    assert 0.0 <= wf.dsr["dsr"] <= 1.0


def test_robustness_report_pass_when_edge_survives_cost_stress():
    def good_run_fn(fee_model, slippage):
        return {"sharpe": 2.0, "max_drawdown": -0.05}

    report = robustness_report(good_run_fn, BpsFees(maker_bps=1, taker_bps=2))
    assert report["passed"] is True
    assert set(report["scenarios"]) == {"base", "fees_x1.5", "fees_x2", "slippage_x2"}


def test_robustness_report_fail_when_edge_does_not_survive_cost_stress():
    def bad_run_fn(fee_model, slippage):
        return {"sharpe": -0.5, "max_drawdown": -0.6}

    report = robustness_report(bad_run_fn, BpsFees(maker_bps=1, taker_bps=2))
    assert report["passed"] is False
    assert report["checks"]["survives_cost_stress"] is False
    assert report["checks"]["max_drawdown_within_floor"] is False


def test_robustness_report_actually_scales_fee_model_passed_to_run_fn():
    seen_taker_bps = []

    def run_fn(fee_model, slippage):
        seen_taker_bps.append(fee_model.taker_bps)
        return {"sharpe": 1.0, "max_drawdown": -0.1}

    robustness_report(run_fn, BpsFees(maker_bps=1.0, taker_bps=10.0), SlippageModel())
    # base and slippage_x2 leave fees unscaled (10 bps); fees_x1.5 and fees_x2 scale them to 15/20.
    assert sorted(seen_taker_bps) == [10.0, 10.0, 15.0, 20.0]
