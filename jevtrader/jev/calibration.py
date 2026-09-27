"""Calibration metrics from an annotated decision log.

This is what decides whether Jev actually adds value over `OfflineJevAdvisor`: run the same
metrics over a real Jev decision log and over an offline-heuristic log covering the same
period/symbols, and compare. Everything here operates on plain dicts (decision-log entries
already passed through `jevtrader.jev.log.annotate_outcomes`, so they carry `fwd_ret_bps`) --
no dependency on `advisor.py`, so it works equally on real, replayed, or offline logs.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Optional, Sequence

Pair = tuple[float, bool, float, float]  # (predicted_prob, outcome, fwd_ret_bps, cost_bps)


def _extract_pairs(
    entries: Sequence[Mapping[str, Any]],
    *,
    question: str,
    label: str,
    outcome_fn: Callable[[Mapping[str, Any]], Optional[bool]],
) -> list[Pair]:
    pairs: list[Pair] = []
    for entry in entries:
        p = entry.get("answers", {}).get("probs", {}).get(question, {}).get(label)
        if p is None:
            continue
        outcome = outcome_fn(entry)
        if outcome is None:
            continue
        fwd = entry.get("fwd_ret_bps")
        cost = float(entry.get("state", {}).get("cost_bps", 0.0) or 0.0)
        pairs.append((float(p), bool(outcome), float(fwd) if fwd is not None else float("nan"), cost))
    return pairs


def _brier(pairs: Sequence[Pair]) -> Optional[float]:
    if not pairs:
        return None
    return sum((p - float(y)) ** 2 for p, y, _, _ in pairs) / len(pairs)


def _log_loss(pairs: Sequence[Pair], eps: float = 1e-9) -> Optional[float]:
    if not pairs:
        return None
    total = 0.0
    for p, y, _, _ in pairs:
        pc = min(max(p, eps), 1 - eps)
        total += -(y * math.log(pc) + (1 - y) * math.log(1 - pc))
    return total / len(pairs)


def _reliability_curve(pairs: Sequence[Pair], n_bins: int) -> list[dict[str, Any]]:
    bins: list[list[Pair]] = [[] for _ in range(n_bins)]
    for pair in pairs:
        p = pair[0]
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append(pair)
    curve: list[dict[str, Any]] = []
    for i, bucket in enumerate(bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        n = len(bucket)
        mean_predicted = sum(p for p, _, _, _ in bucket) / n if n else None
        empirical_rate = sum(1.0 for _, y, _, _ in bucket if y) / n if n else None
        curve.append({"bin_lo": round(lo, 3), "bin_hi": round(hi, 3), "n": n, "mean_predicted": _r(mean_predicted), "empirical_rate": _r(empirical_rate)})
    return curve


def _r(x: Optional[float], ndigits: int = 4) -> Optional[float]:
    return None if x is None else round(x, ndigits)


def _expected_calibration_error(curve: Sequence[Mapping[str, Any]], total_n: int) -> Optional[float]:
    if total_n == 0:
        return None
    ece = 0.0
    for bucket in curve:
        n = bucket["n"]
        if n == 0 or bucket["mean_predicted"] is None or bucket["empirical_rate"] is None:
            continue
        ece += (n / total_n) * abs(bucket["mean_predicted"] - bucket["empirical_rate"])
    return round(ece, 4)


def _hit_rate_by_confidence(pairs: Sequence[Pair], edges: Sequence[float] = (0.0, 0.55, 0.65, 0.75, 0.85, 1.01)) -> list[dict[str, Any]]:
    """Buckets by |p - 0.5| * 2 (distance from a coin flip, rescaled to [0, 1]) and reports the
    fraction of entries in each bucket where the higher-probability side matched the outcome."""
    buckets: list[list[Pair]] = [[] for _ in range(len(edges) - 1)]
    for pair in pairs:
        p = pair[0]
        conf = abs(p - 0.5) * 2.0
        for i in range(len(edges) - 1):
            if edges[i] <= conf < edges[i + 1]:
                buckets[i].append(pair)
                break
    out = []
    for i, bucket in enumerate(buckets):
        n = len(bucket)
        if n == 0:
            out.append({"bucket": f"[{edges[i]:.2f},{edges[i+1]:.2f})", "n": 0, "hit_rate": None})
            continue
        hits = sum(1 for p, y, _, _ in bucket if (p >= 0.5) == y)
        out.append({"bucket": f"[{edges[i]:.2f},{edges[i+1]:.2f})", "n": n, "hit_rate": round(hits / n, 4)})
    return out


def _edge_after_costs(pairs: Sequence[Pair], threshold: float) -> Optional[dict[str, Any]]:
    selected = [pair for pair in pairs if pair[0] >= threshold and not math.isnan(pair[2])]
    if not selected:
        return {"n": 0, "avg_fwd_ret_bps": None, "avg_cost_bps": None, "edge_after_costs_bps": None}
    avg_fwd = sum(f for _, _, f, _ in selected) / len(selected)
    avg_cost = sum(c for _, _, _, c in selected) / len(selected)
    return {
        "n": len(selected),
        "avg_fwd_ret_bps": round(avg_fwd, 3),
        "avg_cost_bps": round(avg_cost, 3),
        "edge_after_costs_bps": round(avg_fwd - avg_cost, 3),
    }


def calibrate_binary(
    entries: Sequence[Mapping[str, Any]],
    *,
    question: str,
    label: str,
    outcome_fn: Callable[[Mapping[str, Any]], Optional[bool]],
    threshold: float = 0.5,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Calibration report for one question/label's probability against a boolean outcome.

    `outcome_fn(entry) -> True/False/None` decides the ground truth per entry (`None` skips
    it, e.g. missing forward return). Returns a dict with `n`, `brier`, `log_loss`, `ece`,
    `reliability` (binned curve), `hit_rate_by_confidence`, `edge_after_costs`, and a rendered
    `markdown` summary.
    """
    pairs = _extract_pairs(entries, question=question, label=label, outcome_fn=outcome_fn)
    n = len(pairs)
    reliability = _reliability_curve(pairs, n_bins)
    result: dict[str, Any] = {
        "question": question,
        "label": label,
        "n": n,
        "brier": _r(_brier(pairs)),
        "log_loss": _r(_log_loss(pairs)),
        "ece": _expected_calibration_error(reliability, n),
        "reliability": reliability,
        "hit_rate_by_confidence": _hit_rate_by_confidence(pairs),
        "edge_after_costs": _edge_after_costs(pairs, threshold),
        "threshold": threshold,
    }
    result["markdown"] = _to_markdown(result)
    return result


def calibrate_direction(
    entries: Sequence[Mapping[str, Any]],
    *,
    label: str = "up",
    threshold: float = 0.5,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Convenience wrapper of `calibrate_binary` for the `"direction"` question.

    Ground truth for `label="up"` is `fwd_ret_bps > cost_bps` (the move cleared round-trip
    cost in that direction); for `label="down"` it's `fwd_ret_bps < -cost_bps`. Entries must
    already be annotated by `jevtrader.jev.log.annotate_outcomes` (i.e. carry `fwd_ret_bps`).
    """
    if label not in ("up", "down"):
        raise ValueError("calibrate_direction only supports label='up' or label='down'")

    def outcome_fn(entry: Mapping[str, Any]) -> Optional[bool]:
        fwd = entry.get("fwd_ret_bps")
        if fwd is None:
            return None
        cost = float(entry.get("state", {}).get("cost_bps", 0.0) or 0.0)
        return float(fwd) > cost if label == "up" else float(fwd) < -cost

    return calibrate_binary(entries, question="direction", label=label, outcome_fn=outcome_fn, threshold=threshold, n_bins=n_bins)


def _to_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        f"### Calibration: `{result['question']}` = `{result['label']}` (n={result['n']})",
        "",
        f"- Brier score: {result['brier']}",
        f"- Log loss: {result['log_loss']}",
        f"- Expected calibration error: {result['ece']}",
        "",
        "| bin | n | mean predicted | empirical rate |",
        "|---|---|---|---|",
    ]
    for b in result["reliability"]:
        lines.append(f"| [{b['bin_lo']}, {b['bin_hi']}) | {b['n']} | {b['mean_predicted']} | {b['empirical_rate']} |")
    lines += ["", "| confidence bucket | n | hit rate |", "|---|---|---|"]
    for h in result["hit_rate_by_confidence"]:
        lines.append(f"| {h['bucket']} | {h['n']} | {h['hit_rate']} |")
    edge = result["edge_after_costs"]
    lines += [
        "",
        f"**Edge after costs** (p >= {result['threshold']}): n={edge['n']}, avg fwd return={edge['avg_fwd_ret_bps']} bps, "
        f"avg cost={edge['avg_cost_bps']} bps, edge={edge['edge_after_costs_bps']} bps",
    ]
    return "\n".join(lines)
