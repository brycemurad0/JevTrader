"""Evaluate any `/v1/systemone` endpoint (hosted Jev, a local Kev/Laya server) or in-process
advisor (`OfflineJevAdvisor`, a `jevtrader.forecast` model, `finetune.baseline.LogisticBaseline`,
...) on the SAME held-out test examples, and decide whether a candidate actually beats an
incumbent -- on calibration AND the trading metric that matters, not vibes.

A "predictor" here is either:
- anything implementing `JevAdvisorProtocol`'s `.direction(symbol, features, horizon)` (an
  `OfflineJevAdvisor`, a real `JevAdvisor` pointed at Jev/Kev/Laya via `backends.make_client`,
  a `ReplayJevAdvisor`, or any other in-process advisor with the same method -- this module
  never imports `jevtrader.forecast`, it just calls whatever `.direction` it's handed), or
- a `finetune.baseline.LogisticBaseline` (`.predict_one(state)`), which answers in-process with
  no notion of network latency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from jevtrader.jev.calibration import calibrate_direction

CallResult = tuple[Optional[dict[str, float]], Optional[float], bool]  # (probs, latency_ms, late)


def _make_caller(predictor: Any) -> Callable[[str, Mapping[str, Any], str], CallResult]:
    if hasattr(predictor, "predict_one"):

        def call_baseline(symbol: str, state: Mapping[str, Any], horizon: str) -> CallResult:
            t0 = time.perf_counter()
            probs = predictor.predict_one(state)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return (dict(probs) if probs else None), elapsed_ms, False

        return call_baseline

    if hasattr(predictor, "direction"):

        def call_advisor(symbol: str, state: Mapping[str, Any], horizon: str) -> CallResult:
            t0 = time.perf_counter()
            view = predictor.direction(symbol, state, horizon)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if view is None:
                return None, elapsed_ms, True
            probs = dict(view.probs.get("direction", {}))
            latency_ms = view.latency_ms if view.latency_ms else elapsed_ms
            return (probs or None), latency_ms, view.late

        return call_advisor

    raise TypeError(f"predictor {predictor!r} implements neither .predict_one(state) nor .direction(symbol, state, horizon)")


@dataclass
class BackendResult:
    """One predictor's scorecard on a test set. `raw` holds numpy arrays (per-example Brier
    terms and per-trade edge terms) used by `beats()`'s bootstrap CIs -- not meant for direct
    JSON export; use `results_to_json`/`results_to_markdown` for that."""

    name: str
    n: int
    accuracy: Optional[float]
    brier_up: Optional[float]
    brier_down: Optional[float]
    log_loss_up: Optional[float]
    ece_up: Optional[float]
    latency_p50_ms: Optional[float]
    latency_p95_ms: Optional[float]
    latency_p99_ms: Optional[float]
    coverage_at_budget: Optional[float]
    edge_after_costs_bps: Optional[dict[str, Any]]
    pnl_curve_bps: list[float] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, np.ndarray] = field(default_factory=dict)


def _score_one(
    name: str,
    predictor: Any,
    examples: Sequence[Mapping[str, Any]],
    horizon: str,
    threshold: float,
    n_bins: int,
    latency_budget_ms: Optional[float],
) -> BackendResult:
    caller = _make_caller(predictor)
    entries: list[dict[str, Any]] = []
    latencies: list[float] = []
    coverage_hits = 0
    correct = 0
    n_scored = 0
    pnl_terms: list[float] = []

    for ex in examples:
        probs, latency_ms, late = caller(ex["symbol"], ex["state"], horizon)
        if not probs:
            continue
        n_scored += 1
        if latency_ms is not None:
            latencies.append(latency_ms)
        within_budget = latency_budget_ms is None or (latency_ms is not None and latency_ms <= latency_budget_ms)
        if not late and within_budget:
            coverage_hits += 1

        true_label = ex["labels"].get("direction")
        top = max(probs, key=probs.get)
        correct += int(top == true_label)

        fwd = ex["outcomes"]["fwd_ret_bps"]
        cost = ex["cost_bps"]
        entries.append({"answers": {"probs": {"direction": probs}}, "fwd_ret_bps": fwd, "state": {"cost_bps": cost}})

        p_up, p_down = probs.get("up", 0.0), probs.get("down", 0.0)
        if p_up >= threshold and p_up >= p_down:
            pnl_terms.append(fwd - cost)
        elif p_down >= threshold and p_down > p_up:
            pnl_terms.append(-fwd - cost)
        else:
            pnl_terms.append(0.0)

    up_report = calibrate_direction(entries, label="up", threshold=threshold, n_bins=n_bins) if entries else None
    down_report = calibrate_direction(entries, label="down", threshold=threshold, n_bins=n_bins) if entries else None

    brier_terms_up = np.array(
        [(e["answers"]["probs"]["direction"].get("up", 0.0) - float(e["fwd_ret_bps"] > e["state"]["cost_bps"])) ** 2 for e in entries if e["fwd_ret_bps"] is not None]
    )
    edge_terms = np.array([e["fwd_ret_bps"] - e["state"]["cost_bps"] for e in entries if e["answers"]["probs"]["direction"].get("up", 0.0) >= threshold])
    latencies_arr = np.array(latencies, dtype=float)

    return BackendResult(
        name=name,
        n=n_scored,
        accuracy=(correct / n_scored) if n_scored else None,
        brier_up=up_report["brier"] if up_report else None,
        brier_down=down_report["brier"] if down_report else None,
        log_loss_up=up_report["log_loss"] if up_report else None,
        ece_up=up_report["ece"] if up_report else None,
        latency_p50_ms=float(np.percentile(latencies_arr, 50)) if latencies_arr.size else None,
        latency_p95_ms=float(np.percentile(latencies_arr, 95)) if latencies_arr.size else None,
        latency_p99_ms=float(np.percentile(latencies_arr, 99)) if latencies_arr.size else None,
        coverage_at_budget=(coverage_hits / n_scored) if n_scored else None,
        edge_after_costs_bps=up_report["edge_after_costs"] if up_report else None,
        pnl_curve_bps=[round(float(x), 3) for x in np.cumsum(pnl_terms)] if pnl_terms else [],
        details={"up": up_report, "down": down_report},
        raw={"brier_terms_up": brier_terms_up, "edge_terms": edge_terms},
    )


def run_scoreboard(
    test_examples: Sequence[Mapping[str, Any]],
    predictors: Mapping[str, Any],
    *,
    horizon: Optional[str] = None,
    threshold: float = 0.5,
    n_bins: int = 10,
    latency_budget_ms: Optional[float] = None,
) -> dict[str, BackendResult]:
    """Score every predictor in `predictors` (name -> predictor, see module docstring) against
    the identical `test_examples` (from `dataset.build_examples`/`build_dataset`, ideally its
    untouched `test` split). `horizon` defaults to the first example's `"horizon"`.

    The `pnl_curve_bps` in each result is a cumulative sum of per-example
    `(realized_forward_return - cost)` when that predictor would have traded, in test-example
    order. Because consecutive examples' forward-return windows can overlap (when the dataset's
    `stride` is smaller than the horizon in bars), this is an illustrative measure of
    directional consistency over time, NOT a literal, non-overlapping-position equity curve --
    treat `edge_after_costs_bps` (mean edge among acted-on examples) as the primary trading
    metric and the curve as a diagnostic.
    """
    if not test_examples:
        raise ValueError("run_scoreboard requires at least one test example")
    examples = sorted(test_examples, key=lambda e: e["ts"])
    resolved_horizon = horizon or examples[0].get("horizon", "5min")
    return {name: _score_one(name, predictor, examples, resolved_horizon, threshold, n_bins, latency_budget_ms) for name, predictor in predictors.items()}


def _bootstrap_mean_ci(values: np.ndarray, *, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    clean = values[~np.isnan(values)] if values.size else values
    if clean.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = clean.size
    means = clean[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), float(clean.mean())


def beats(candidate: BackendResult, incumbent: BackendResult, *, n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    """A candidate replaces an incumbent only if it wins on ALL THREE: lower Brier (`brier_up`),
    higher after-cost edge (`edge_after_costs_bps["edge_after_costs_bps"]`), and lower p95
    latency -- all on the SAME untouched test split. Also reports a (non-paired, per-metric)
    bootstrap 95% CI on each side's Brier and edge-per-trade means, so a "win" backed by a
    handful of test examples is visibly not one to trust.
    """
    c_edge = (candidate.edge_after_costs_bps or {}).get("edge_after_costs_bps")
    i_edge = (incumbent.edge_after_costs_bps or {}).get("edge_after_costs_bps")

    brier_ok = candidate.brier_up is not None and incumbent.brier_up is not None and candidate.brier_up < incumbent.brier_up
    edge_ok = c_edge is not None and i_edge is not None and c_edge > i_edge
    latency_ok = candidate.latency_p95_ms is not None and incumbent.latency_p95_ms is not None and candidate.latency_p95_ms < incumbent.latency_p95_ms

    c_brier_lo, c_brier_hi, c_brier_mean = _bootstrap_mean_ci(candidate.raw.get("brier_terms_up", np.array([])), n_boot=n_boot, seed=seed)
    i_brier_lo, i_brier_hi, i_brier_mean = _bootstrap_mean_ci(incumbent.raw.get("brier_terms_up", np.array([])), n_boot=n_boot, seed=seed + 1)
    c_edge_lo, c_edge_hi, c_edge_mean = _bootstrap_mean_ci(candidate.raw.get("edge_terms", np.array([])), n_boot=n_boot, seed=seed + 2)
    i_edge_lo, i_edge_hi, i_edge_mean = _bootstrap_mean_ci(incumbent.raw.get("edge_terms", np.array([])), n_boot=n_boot, seed=seed + 3)

    return {
        "beats": bool(brier_ok and edge_ok and latency_ok),
        "reasons": {"lower_brier": brier_ok, "higher_after_cost_edge": edge_ok, "lower_p95_latency": latency_ok},
        "candidate": {
            "name": candidate.name,
            "brier_up": candidate.brier_up,
            "brier_up_bootstrap_ci95": [c_brier_lo, c_brier_hi],
            "edge_after_costs_bps": c_edge,
            "edge_bootstrap_mean_bps": c_edge_mean,
            "edge_bootstrap_ci95_bps": [c_edge_lo, c_edge_hi],
            "latency_p95_ms": candidate.latency_p95_ms,
        },
        "incumbent": {
            "name": incumbent.name,
            "brier_up": incumbent.brier_up,
            "brier_up_bootstrap_ci95": [i_brier_lo, i_brier_hi],
            "edge_after_costs_bps": i_edge,
            "edge_bootstrap_mean_bps": i_edge_mean,
            "edge_bootstrap_ci95_bps": [i_edge_lo, i_edge_hi],
            "latency_p95_ms": incumbent.latency_p95_ms,
        },
        "n_bootstrap": n_boot,
    }


def results_to_json(results: Mapping[str, BackendResult]) -> dict[str, Any]:
    """JSON-serializable summary of `run_scoreboard`'s output (drops the numpy `raw` arrays)."""
    out: dict[str, Any] = {}
    for name, r in results.items():
        out[name] = {
            "name": r.name,
            "n": r.n,
            "accuracy": r.accuracy,
            "brier_up": r.brier_up,
            "brier_down": r.brier_down,
            "log_loss_up": r.log_loss_up,
            "ece_up": r.ece_up,
            "latency_p50_ms": r.latency_p50_ms,
            "latency_p95_ms": r.latency_p95_ms,
            "latency_p99_ms": r.latency_p99_ms,
            "coverage_at_budget": r.coverage_at_budget,
            "edge_after_costs": r.edge_after_costs_bps,
            "pnl_curve_bps": r.pnl_curve_bps,
        }
    return out


def results_to_markdown(results: Mapping[str, BackendResult]) -> str:
    lines = [
        "| backend | n | accuracy | brier(up) | log loss(up) | ECE(up) | p50 ms | p95 ms | p99 ms | coverage | edge after costs (bps) | final PnL (bps) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        edge = (r.edge_after_costs_bps or {}).get("edge_after_costs_bps")
        final_pnl = r.pnl_curve_bps[-1] if r.pnl_curve_bps else None
        lines.append(
            f"| {name} | {r.n} | {r.accuracy} | {r.brier_up} | {r.log_loss_up} | {r.ece_up} | "
            f"{r.latency_p50_ms} | {r.latency_p95_ms} | {r.latency_p99_ms} | {r.coverage_at_budget} | {edge} | {final_pnl} |"
        )
    return "\n".join(lines)
