from __future__ import annotations

import pytest

from jevtrader.core.interfaces import JevView
from jevtrader.jev.advisor import OfflineJevAdvisor
from jevtrader.jev.finetune.baseline import fit_logistic_baseline
from jevtrader.jev.finetune.scoreboard import BackendResult, beats, results_to_json, results_to_markdown, run_scoreboard


def _example(ts, symbol, cost_bps, fwd_ret_bps, label, momentum=0.0):
    return {
        "ts": ts,
        "symbol": symbol,
        "horizon": "5min",
        "cost_bps": cost_bps,
        "state": {"ret_bps": {"5": momentum, "15": momentum, "30": momentum, "60": momentum}, "vol_bps": 5.0, "cost_bps": cost_bps},
        "questions": {},
        "labels": {"direction": label},
        "outcomes": {"fwd_ret_bps": fwd_ret_bps, "entry_px": 100.0, "exit_px": 100.0 * (1 + fwd_ret_bps / 1e4), "future_high": 101.0, "future_low": 99.0},
    }


class _PerfectPredictor:
    """Answers with certainty and always correctly -- an oracle, for testing the scoreboard's
    math on a known-perfect case."""

    def direction(self, symbol, features, horizon):
        label = features.get("_true_label", "flat")
        probs = {"up": 0.98 if label == "up" else 0.01, "down": 0.98 if label == "down" else 0.01, "flat": 0.98 if label == "flat" else 0.01}
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}
        return JevView(probs={"direction": probs}, top={"direction": label}, confidence={"direction": 0.98}, latency_ms=5.0, late=False, model="oracle")


class _UniformPredictor:
    def direction(self, symbol, features, horizon):
        probs = {"up": 1 / 3, "down": 1 / 3, "flat": 1 / 3}
        return JevView(probs={"direction": probs}, top={"direction": "flat"}, confidence={"direction": 0.33}, latency_ms=20.0, late=False, model="uniform")


class _ConfidentButWrongPredictor:
    """Always calls 'up' with high confidence, regardless of the truth: wrong 2/3 of the time on
    `_make_examples_for_oracle`'s balanced up/down/flat labels, so unlike `_UniformPredictor` it
    crosses the default 0.5 threshold and has a well-defined (negative) after-cost edge -- a
    proper incumbent for `beats()` tests, which need edge on both sides to be comparable."""

    def direction(self, symbol, features, horizon):
        probs = {"up": 0.9, "down": 0.05, "flat": 0.05}
        return JevView(probs={"direction": probs}, top={"direction": "up"}, confidence={"direction": 0.9}, latency_ms=20.0, late=False, model="confident-wrong")


class _LateAdvisor:
    def direction(self, symbol, features, horizon):
        return JevView(probs={}, top={}, confidence={}, latency_ms=999.0, late=True, model="slow")


class _NoneAdvisor:
    def direction(self, symbol, features, horizon):
        return None


def _make_examples_for_oracle(n=40, cost_bps=1.0):
    examples = []
    for i in range(n):
        label = ["up", "down", "flat"][i % 3]
        fwd = {"up": 20.0, "down": -20.0, "flat": 0.0}[label]
        ex = _example(f"t{i:03d}", "AAPL", cost_bps, fwd, label)
        ex["state"]["_true_label"] = label  # smuggle the ground truth in for _PerfectPredictor
        examples.append(ex)
    return examples


def test_scoreboard_oracle_gets_perfect_accuracy_and_low_brier():
    examples = _make_examples_for_oracle()
    results = run_scoreboard(examples, {"oracle": _PerfectPredictor()})
    r = results["oracle"]
    assert r.n == len(examples)
    assert r.accuracy == pytest.approx(1.0)
    assert r.brier_up < 0.05


def test_scoreboard_uniform_predictor_has_higher_brier_than_oracle():
    examples = _make_examples_for_oracle()
    results = run_scoreboard(examples, {"oracle": _PerfectPredictor(), "uniform": _UniformPredictor()})
    assert results["uniform"].brier_up > results["oracle"].brier_up


def test_scoreboard_latency_percentiles_reflect_reported_latency():
    examples = _make_examples_for_oracle(n=10)
    results = run_scoreboard(examples, {"uniform": _UniformPredictor()})
    r = results["uniform"]
    assert r.latency_p50_ms == pytest.approx(20.0, abs=0.5)
    assert r.latency_p95_ms >= r.latency_p50_ms


def test_scoreboard_late_predictor_has_zero_coverage_and_is_not_scored():
    examples = _make_examples_for_oracle(n=10)
    results = run_scoreboard(examples, {"late": _LateAdvisor()})
    r = results["late"]
    assert r.n == 0  # late views carry empty probs, so nothing was scored
    assert r.accuracy is None


def test_scoreboard_none_predictor_scores_nothing():
    examples = _make_examples_for_oracle(n=10)
    results = run_scoreboard(examples, {"none": _NoneAdvisor()})
    assert results["none"].n == 0


def test_scoreboard_accepts_a_logistic_baseline_predictor():
    rng_examples = []
    for i in range(200):
        momentum = (-1) ** i * (i % 5)
        label = "up" if momentum > 1 else ("down" if momentum < -1 else "flat")
        fwd = {"up": 20.0, "down": -20.0, "flat": 0.0}[label]
        rng_examples.append(_example(f"t{i:04d}", "AAPL", 1.0, fwd, label, momentum=momentum))
    baseline = fit_logistic_baseline(rng_examples)
    results = run_scoreboard(rng_examples, {"baseline": baseline})
    r = results["baseline"]
    assert r.n == len(rng_examples)
    assert r.accuracy is not None
    assert r.latency_p50_ms is not None  # measured wall time even though it's in-process


def test_scoreboard_offline_advisor_runs_without_error():
    examples = _make_examples_for_oracle(n=20)
    results = run_scoreboard(examples, {"offline": OfflineJevAdvisor()})
    assert results["offline"].n == 20


def test_scoreboard_rejects_unsupported_predictor():
    with pytest.raises(TypeError):
        run_scoreboard(_make_examples_for_oracle(n=1), {"bad": object()})


def test_scoreboard_requires_examples():
    with pytest.raises(ValueError):
        run_scoreboard([], {"oracle": _PerfectPredictor()})


def test_beats_true_when_candidate_wins_all_three_metrics():
    examples = _make_examples_for_oracle(n=60)
    results = run_scoreboard(examples, {"oracle": _PerfectPredictor(), "confident_wrong": _ConfidentButWrongPredictor()})
    verdict = beats(results["oracle"], results["confident_wrong"], n_boot=200)
    assert verdict["beats"] is True
    assert verdict["reasons"]["lower_brier"] is True
    assert verdict["reasons"]["higher_after_cost_edge"] is True
    assert verdict["reasons"]["lower_p95_latency"] is True
    assert "brier_up_bootstrap_ci95" in verdict["candidate"]


def test_beats_false_when_candidate_loses_any_one_metric():
    examples = _make_examples_for_oracle(n=60)
    results = run_scoreboard(examples, {"oracle": _PerfectPredictor(), "confident_wrong": _ConfidentButWrongPredictor()})
    # reversed: the worse predictor as "candidate" should not beat the better one
    verdict = beats(results["confident_wrong"], results["oracle"], n_boot=200)
    assert verdict["beats"] is False


def test_beats_handles_missing_metrics_gracefully():
    late_result = BackendResult(name="late", n=0, accuracy=None, brier_up=None, brier_down=None, log_loss_up=None, ece_up=None, latency_p50_ms=None, latency_p95_ms=None, latency_p99_ms=None, coverage_at_budget=None, edge_after_costs_bps=None)
    ok_result = BackendResult(name="ok", n=5, accuracy=0.9, brier_up=0.1, brier_down=0.1, log_loss_up=0.2, ece_up=0.05, latency_p50_ms=1.0, latency_p95_ms=2.0, latency_p99_ms=3.0, coverage_at_budget=1.0, edge_after_costs_bps={"edge_after_costs_bps": 5.0})
    verdict = beats(late_result, ok_result)
    assert verdict["beats"] is False


def test_results_to_json_is_serializable():
    import json

    examples = _make_examples_for_oracle(n=10)
    results = run_scoreboard(examples, {"oracle": _PerfectPredictor()})
    payload = results_to_json(results)
    json.dumps(payload)  # must not raise
    assert "oracle" in payload
    assert "raw" not in payload["oracle"]  # numpy arrays never leak into the JSON export


def test_results_to_markdown_contains_backend_names():
    examples = _make_examples_for_oracle(n=10)
    results = run_scoreboard(examples, {"oracle": _PerfectPredictor(), "uniform": _UniformPredictor()})
    md = results_to_markdown(results)
    assert "oracle" in md and "uniform" in md
    assert md.startswith("|")
