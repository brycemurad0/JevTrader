"""Fine-tune dataset construction from historical bars, using EXACTLY `build_state` and
`jevtrader.jev.questions` -- the same functions `JevAdvisor` calls at serve time -- so there is
no train/serve skew: a fine-tuned Kev/Laya checkpoint sees the identical state shape and
question wording in training as it will at inference.

Labels come from REALIZED forward outcomes, strictly after the decision bar (never from the
decision bar's own window, which only looks backward):

- `direction`: up/down/flat, net of `cost_bps`, exactly matching `questions.direction_questions`'
  criteria wording (a move must clear the cost hurdle to count as up/down).
- `regime` + `risk_off`: a documented HEURISTIC over forward volatility/trend/drawdown, not a
  ground truth (markets have no labeled "regime"). Useful as auxiliary training signal; treat
  `direction`, which has an objective, cost-aware definition, as the metric that decides
  whether a fine-tune actually helps.
- `entry_quality`: bucketed forward MFE/MAE ratio (the better of the two possible trade
  directions' favorable-vs-adverse excursion), independent of which side you'd have taken.

`from_decision_log` builds a SEPARATE kind of record: soft-label distillation from a teacher's
(e.g. hosted Jev's) logged answers, for when ground truth is scarce. These carry an explicit
`target` (soft) alongside `label` (the teacher's argmax) and must never be silently mixed with
ground-truth counts when reporting a dataset's size or class balance.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from jevtrader.jev.questions import direction_questions, entry_quality_questions, regime_questions
from jevtrader.jev.state import build_state, to_json

FeatureFn = Callable[[str, pd.DataFrame], Mapping[str, Any]]


# --------------------------------------------------------------------------- forward-window helpers


def _infer_bar_seconds(index: pd.DatetimeIndex) -> float:
    diffs = pd.Series(index).diff().dropna().dt.total_seconds()
    if diffs.empty:
        raise ValueError("cannot infer bar spacing from a single bar")
    return float(diffs.median())


def _horizon_bars(horizon: str, bar_seconds: float) -> int:
    n = round(pd.Timedelta(horizon).total_seconds() / bar_seconds)
    if n < 1:
        raise ValueError(f"horizon {horizon!r} ({pd.Timedelta(horizon)}) is shorter than one bar ({bar_seconds}s)")
    return int(n)


def _forward_vol_bps(future_closes: pd.Series) -> float:
    if len(future_closes) < 2:
        return 0.0
    rets = future_closes.pct_change().dropna()
    return float(1e4 * rets.std(ddof=0)) if not rets.empty else 0.0


def _forward_slope_bps(future_closes: pd.Series) -> float:
    n = len(future_closes)
    if n < 2:
        return 0.0
    y = future_closes.to_numpy(dtype=float)
    x = np.arange(n, dtype=float)
    x_mean, y_mean = x.mean(), y.mean()
    denom = float(((x - x_mean) ** 2).sum())
    if denom == 0 or y_mean == 0:
        return 0.0
    slope = float(((x - x_mean) * (y - y_mean)).sum()) / denom
    return 1e4 * slope / y_mean


# --------------------------------------------------------------------------- ground-truth labels


def _direction_label(fwd_ret_bps: float, cost_bps: float) -> str:
    if fwd_ret_bps > cost_bps:
        return "up"
    if fwd_ret_bps < -cost_bps:
        return "down"
    return "flat"


def _bucket(value: float, thresholds: Sequence[float]) -> int:
    level = 0
    for t in thresholds:
        if value >= t:
            level += 1
    return level


def _entry_quality_label(entry_px: float, future_high: float, future_low: float, thresholds: Sequence[float]) -> int:
    """0-4 bucket of the better of the two possible trade directions' MFE/MAE ratio (favorable
    excursion over adverse excursion): long's favorable = high-entry, adverse = entry-low;
    short's are the mirror image. Direction-agnostic on purpose -- this rates whether *now* is
    a good time to open ANY position, which is what `questions.entry_quality_questions` asks."""
    eps = 1e-9
    up_mfe = max(0.0, future_high - entry_px)
    up_mae = max(0.0, entry_px - future_low)
    long_ratio = up_mfe / max(up_mae, eps)
    short_ratio = up_mae / max(up_mfe, eps)
    return _bucket(max(long_ratio, short_ratio), thresholds)


def _risk_off_label(entry_px: float, future_low: float, drawdown_threshold_bps: float) -> bool:
    drawdown_bps = 1e4 * (entry_px - future_low) / entry_px if entry_px else 0.0
    return drawdown_bps > drawdown_threshold_bps


def _regime_label(fwd_ret_bps: float, forward_vol_bps: float, forward_slope_bps: float, cost_bps: float, quiet_vol_bps: float, chop_vol_bps: float) -> str:
    """A documented HEURISTIC proxy, not a ground truth: real markets have no labeled regime.
    `quiet_vol_bps`/`chop_vol_bps` are quantiles of the forward-vol distribution over the whole
    candidate set (see `build_examples`), a mild global look-ahead confined to calibrating this
    one heuristic label's thresholds -- it never leaks into any FEATURE, and never affects the
    `direction` label, which is the one used for the honest "does this add value" evaluation."""
    if forward_vol_bps >= chop_vol_bps and abs(fwd_ret_bps) < 2 * max(cost_bps, 1e-9):
        return "volatile_chop"
    if forward_vol_bps <= quiet_vol_bps:
        return "quiet"
    if fwd_ret_bps > cost_bps and forward_slope_bps > 0:
        return "trending_up"
    if fwd_ret_bps < -cost_bps and forward_slope_bps < 0:
        return "trending_down"
    return "mean_reverting"


# --------------------------------------------------------------------------- config / splits


@dataclass(frozen=True)
class DatasetConfig:
    horizon: str = "5min"
    cost_bps: float = 0.0
    lookback_bars: int = 120
    stride: int = 1
    include_regime: bool = True
    include_entry_quality: bool = True
    risk_off_drawdown_bps: float = 50.0
    regime_vol_quantiles: tuple[float, float] = (0.25, 0.75)
    entry_quality_thresholds: tuple[float, ...] = (1.0, 1.5, 2.5, 4.0)
    dedupe: bool = True
    max_examples: Optional[int] = None
    train_frac: float = 0.7
    calibration_frac: float = 0.1
    dev_frac: float = 0.1
    test_frac: float = 0.1
    balance_train_classes: bool = False
    balance_label: str = "direction"
    seed: int = 0

    def __post_init__(self) -> None:
        total = self.train_frac + self.calibration_frac + self.dev_frac + self.test_frac
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split fractions must sum to 1.0, got {total}")


@dataclass(frozen=True)
class DatasetSplits:
    train: list[dict[str, Any]] = field(default_factory=list)
    calibration: list[dict[str, Any]] = field(default_factory=list)
    dev: list[dict[str, Any]] = field(default_factory=list)
    test: list[dict[str, Any]] = field(default_factory=list)

    def __iter__(self):
        return iter((self.train, self.calibration, self.dev, self.test))

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "calibration": len(self.calibration), "dev": len(self.dev), "test": len(self.test)}


# --------------------------------------------------------------------------- building examples


def dedupe_examples(examples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Drop examples whose (rounded) `state` is byte-identical to one already kept. `build_state`
    rounds aggressively, so many raw windows collapse to the same state -- keeping only the
    first occurrence avoids massively overweighting quiet periods."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for ex in examples:
        key = to_json(ex["state"])
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(ex))
    return out


def balance_classes(examples: Sequence[Mapping[str, Any]], *, label_key: str = "direction", seed: int = 0) -> list[dict[str, Any]]:
    """Undersample every class to the size of the smallest one, by `labels[label_key]`. Intended
    for the TRAIN split only -- calibration/dev/test must stay a faithful, unbalanced sample of
    reality or every metric computed on them (base rates, edge-after-costs, ...) is a lie."""
    rng = random.Random(seed)
    buckets: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for ex in examples:
        buckets[ex["labels"].get(label_key)].append(ex)
    if len(buckets) <= 1:
        return [dict(e) for e in examples]
    min_n = min(len(v) for v in buckets.values())
    out: list[dict[str, Any]] = []
    for items in buckets.values():
        chosen = rng.sample(list(items), min_n) if len(items) > min_n else list(items)
        out.extend(dict(e) for e in chosen)
    out.sort(key=lambda e: e["ts"])
    return out


def build_examples(
    symbol: str,
    bars: pd.DataFrame,
    config: DatasetConfig,
    *,
    feature_fn: Optional[FeatureFn] = None,
) -> list[dict[str, Any]]:
    """One labelled example per eligible bar. `feature_fn(symbol, bars_window)`, if given, is
    called with bars up to and including the decision bar ONLY (never later) and its result is
    merged into the state under `"forecast"` via `build_state`'s `extra_features` -- e.g. a
    `jevtrader.forecast` quantile model's output. This module never imports that one; the
    caller wires it in.

    Returns internal example dicts (`ts, symbol, horizon, cost_bps, state, questions, labels,
    outcomes`) -- richer than the Kev-format JSONL `write_jsonl` produces, so splitting,
    deduping and balancing have what they need. `questions` values are real `typesafe_sdk`
    question objects, byte-identical to what `JevAdvisor` would send for the same state.
    """
    if bars is None or len(bars) < config.lookback_bars + 2:
        raise ValueError("not enough bars to build even one example")
    if not bars.index.is_monotonic_increasing:
        raise ValueError("bars must be sorted ascending by timestamp")

    bar_seconds = _infer_bar_seconds(bars.index)
    horizon_bars = _horizon_bars(config.horizon, bar_seconds)

    closes = bars["close"].astype(float)
    highs = bars["high"].astype(float) if "high" in bars.columns else closes
    lows = bars["low"].astype(float) if "low" in bars.columns else closes

    start = config.lookback_bars - 1
    stop = len(bars) - horizon_bars
    if stop <= start:
        raise ValueError("not enough forward bars for even one example at this horizon")
    stride = max(1, config.stride)
    positions = list(range(start, stop, stride))
    if config.max_examples is not None and len(positions) > config.max_examples > 0:
        step = len(positions) / config.max_examples
        positions = [positions[int(i * step)] for i in range(config.max_examples)]

    forward_vols: dict[int, float] = {}
    quiet_thr = chop_thr = 0.0
    if config.include_regime:
        for i in positions:
            forward_vols[i] = _forward_vol_bps(closes.iloc[i + 1 : i + 1 + horizon_bars])
        vol_values = np.array(list(forward_vols.values()), dtype=float)
        if len(vol_values):
            quiet_thr, chop_thr = (float(x) for x in np.quantile(vol_values, config.regime_vol_quantiles))

    examples: list[dict[str, Any]] = []
    for i in positions:
        window = bars.iloc[max(0, i - config.lookback_bars + 1) : i + 1]
        ts = bars.index[i]
        entry_px = float(closes.iloc[i])
        future_slice = slice(i + 1, i + 1 + horizon_bars)
        exit_px = float(closes.iloc[future_slice].iloc[-1])
        future_high = float(highs.iloc[future_slice].max())
        future_low = float(lows.iloc[future_slice].min())
        fwd_ret_bps = 1e4 * (exit_px / entry_px - 1.0) if entry_px else 0.0

        extra = feature_fn(symbol, window) if feature_fn is not None else None
        state = build_state(symbol, window, cost_bps=config.cost_bps, now=ts, extra_features=extra)

        questions: dict[str, Any] = dict(direction_questions(config.horizon, config.cost_bps, include_conviction=False))
        labels: dict[str, Any] = {"direction": _direction_label(fwd_ret_bps, config.cost_bps)}

        if config.include_regime:
            fslope = _forward_slope_bps(closes.iloc[future_slice])
            questions.update(regime_questions())
            labels["regime"] = _regime_label(fwd_ret_bps, forward_vols[i], fslope, config.cost_bps, quiet_thr, chop_thr)
            labels["risk_off"] = _risk_off_label(entry_px, future_low, config.risk_off_drawdown_bps)

        if config.include_entry_quality:
            questions.update(entry_quality_questions())
            labels["entry_quality"] = _entry_quality_label(entry_px, future_high, future_low, config.entry_quality_thresholds)

        examples.append(
            {
                "ts": ts.isoformat(),
                "symbol": symbol,
                "horizon": config.horizon,
                "cost_bps": config.cost_bps,
                "state": state,
                "questions": questions,
                "labels": labels,
                "outcomes": {
                    "fwd_ret_bps": round(fwd_ret_bps, 3),
                    "entry_px": entry_px,
                    "exit_px": exit_px,
                    "future_high": future_high,
                    "future_low": future_low,
                },
            }
        )

    return dedupe_examples(examples) if config.dedupe else examples


def split_chronological(
    examples: Sequence[Mapping[str, Any]],
    *,
    train_frac: float = 0.7,
    calibration_frac: float = 0.1,
    dev_frac: float = 0.1,
    test_frac: float = 0.1,
    embargo: int = 1,
) -> DatasetSplits:
    """Chronological train -> calibration -> dev -> test split, dropping `embargo` examples off
    the END of every split but the last so no split's LABEL (which looks forward `horizon` from
    its `ts`) depends on bars that fall inside the next split's time range. `embargo` should be
    at least `horizon_bars // stride`; `build_dataset` sizes it automatically from a
    `DatasetConfig`. Backward-looking FEATURES are never embargoed: reusing older history for a
    later split's state is ordinary walk-forward, not leakage.
    """
    total = train_frac + calibration_frac + dev_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0, got {total}")
    ordered = sorted(examples, key=lambda e: e["ts"])
    n = len(ordered)
    i1 = round(n * train_frac)
    i2 = i1 + round(n * calibration_frac)
    i3 = i2 + round(n * dev_frac)
    blocks = [ordered[:i1], ordered[i1:i2], ordered[i2:i3], ordered[i3:]]
    embargo = max(0, embargo)
    trimmed = [block[: max(0, len(block) - embargo)] for block in blocks[:-1]] + [blocks[-1]]
    return DatasetSplits(*trimmed)


def build_dataset(
    symbol: str,
    bars: pd.DataFrame,
    config: DatasetConfig,
    *,
    feature_fn: Optional[FeatureFn] = None,
) -> DatasetSplits:
    """`build_examples` + a leakage-safe `split_chronological` (embargo sized from
    `config.horizon`) + optional train-only class balancing, in one call."""
    bar_seconds = _infer_bar_seconds(bars.index)
    horizon_bars = _horizon_bars(config.horizon, bar_seconds)
    examples = build_examples(symbol, bars, config, feature_fn=feature_fn)
    embargo = max(1, -(-horizon_bars // max(1, config.stride)))  # ceil(horizon_bars / stride)
    splits = split_chronological(
        examples,
        train_frac=config.train_frac,
        calibration_frac=config.calibration_frac,
        dev_frac=config.dev_frac,
        test_frac=config.test_frac,
        embargo=embargo,
    )
    train = balance_classes(splits.train, label_key=config.balance_label, seed=config.seed) if config.balance_train_classes else list(splits.train)
    return DatasetSplits(train=train, calibration=list(splits.calibration), dev=list(splits.dev), test=list(splits.test))


# --------------------------------------------------------------------------- Kev-format JSONL


def _question_dict(question: Any) -> dict[str, Any]:
    raw = question.model_dump(exclude_none=True) if hasattr(question, "model_dump") else dict(question)
    return {k: v for k, v in raw.items() if k in ("type", "instructions", "criteria")}


def _to_kev_record(example: Mapping[str, Any], *, source: str, with_meta: bool = True) -> dict[str, Any]:
    """One example -> a Kev-format labelled request: `{"state", "questions": {qid: {type,
    instructions, criteria, "label", "src"}}}`, exactly `kev.data.load_records`'s schema (Laya's
    fine-tuning pipeline consumes the same `/v1/systemone` wire shape -- see
    `docs/DECISION_MODELS.md` for how far that equivalence was verified here).

    When `with_meta` (the default), also attaches a top-level `"_meta"` key with the realized
    outcome and bookkeeping `load_examples_jsonl` needs to reconstruct a scoreboard-ready
    example later, e.g. to evaluate a remote endpoint from a standalone JSONL file without
    rebuilding the dataset. `kev.data.load_records` and Laya's loader both only look at `state`
    and `questions`, so an extra top-level key is silently ignored by either -- this file is
    still valid, unmodified training data.
    """
    out_questions: dict[str, Any] = {}
    for qname, question in example["questions"].items():
        if qname not in example["labels"]:
            continue
        q = _question_dict(question)
        q["label"] = example["labels"][qname]
        q["src"] = f"{source}:{qname}"
        out_questions[qname] = q
    record: dict[str, Any] = {"state": example["state"], "questions": out_questions}
    if with_meta:
        record["_meta"] = {
            "ts": example["ts"],
            "symbol": example["symbol"],
            "horizon": example["horizon"],
            "cost_bps": example["cost_bps"],
            "labels": example["labels"],
            "outcomes": example["outcomes"],
        }
    return record


def write_records_jsonl(records: Sequence[Mapping[str, Any]], path: str | Path) -> int:
    """Write already Kev-shaped `{"state", "questions"}` records (e.g. from `from_decision_log`)
    as JSONL, one per line. Overwrites `path`. Returns the number of lines written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            if not rec.get("questions"):
                continue
            f.write(json.dumps(rec, sort_keys=True, default=str))
            f.write("\n")
            n += 1
    return n


def write_jsonl(examples: Sequence[Mapping[str, Any]], path: str | Path, *, source: str = "jevtrader", with_meta: bool = True) -> int:
    """Convert `build_examples` output to Kev-format labelled-request JSONL and write it. See
    `_to_kev_record` for what `with_meta` attaches (default on); pass `False` for a file meant
    only for `kev.train`/Laya's fine-tuner, with nothing beyond the wire schema."""
    return write_records_jsonl([_to_kev_record(ex, source=source, with_meta=with_meta) for ex in examples], path)


def load_examples_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Reconstruct scoreboard-ready examples from a JSONL file `write_jsonl` wrote (i.e. one
    that still carries `"_meta"`). Question objects come back as plain dicts (`type`,
    `instructions`, `criteria`) rather than `typesafe_sdk` objects -- fine for `run_scoreboard`,
    which only needs `example["state"]`, `["symbol"]`, `["labels"]` and `["outcomes"]`; it never
    resends `["questions"]` itself (a live `JevAdvisorProtocol` predictor builds its own
    questions from `state` and `horizon`).

    Raises `ValueError` on the first record missing `"_meta"` (e.g. a file written with
    `with_meta=False`, or one Kev/Laya trimmed while re-saving it) -- there is nothing to
    reconstruct labels/outcomes from without it.
    """
    examples: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for n, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = rec.get("_meta")
            if meta is None:
                raise ValueError(f"{path}:{n + 1}: record has no \"_meta\" -- was this written with with_meta=False?")
            examples.append(
                {
                    "ts": meta["ts"],
                    "symbol": meta["symbol"],
                    "horizon": meta["horizon"],
                    "cost_bps": meta["cost_bps"],
                    "state": rec["state"],
                    "questions": rec.get("questions", {}),
                    "labels": meta["labels"],
                    "outcomes": meta["outcomes"],
                }
            )
    return examples


# --------------------------------------------------------------------------- distillation from a decision log


def _hard_label(qtype: str, probs: Mapping[str, float], top: Optional[str]) -> Any:
    if qtype == "noul":
        p_yes = float(probs.get("yes", probs.get("true", 0.5)))
        return (top == "yes") if top is not None else (p_yes >= 0.5)
    if qtype == "score":
        if top is not None:
            try:
                return int(top)
            except ValueError:
                pass
        if probs:
            return int(max(probs, key=probs.get))
        return 0
    if top is not None:
        return top
    if probs:
        return max(probs, key=probs.get)
    raise ValueError("cannot derive a hard label: no top/probs given")


def _soft_target(qtype: str, probs: Mapping[str, float]) -> dict[str, float]:
    """Kev's `materialize()` expects a soft target keyed by the SAME option keys it reports
    probabilities under: option names for choice, string level indices for score (both already
    match our `JevView.probs` convention) -- but `"false"`/`"true"` for noul, NOT the
    `typesafe_sdk`/`JevView` convention of `"yes"`/`"no"`, which is translated here."""
    if qtype == "noul":
        p_yes = float(probs.get("yes", probs.get("true", 0.5)))
        return {"true": round(p_yes, 4), "false": round(1.0 - p_yes, 4)}
    return {k: round(float(v), 4) for k, v in probs.items()}


def from_decision_log(
    entries: Iterable[Mapping[str, Any]],
    *,
    source: str = "jevtrader_distill",
    include_soft_target: bool = True,
    skip_late: bool = True,
    only_backend: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Kev-format records distilled from a `DecisionLog`'s logged answers -- e.g. hosted Jev's
    real paper-trading decisions, or one `ShadowAdvisor` backend's (`only_backend="kev"` filters
    to `entry["extra"]["backend"] == "kev"`).

    This is a TEACHER signal, not ground truth: `label`/`target` are just what the teacher said,
    right or wrong, unlike `build_examples`' labels (what actually happened). Keep the two
    separate in accounting and prefer ground truth when both are available; mix in distillation
    records only to cover states ground truth is scarce for, or as an auxiliary loss term.

    `late` entries are skipped by default (`skip_late=True`): a slow answer that would have been
    treated as a HOLD in production is not a decision worth imitating.
    """
    records: list[dict[str, Any]] = []
    for entry in entries:
        if skip_late and entry.get("late"):
            continue
        if only_backend is not None and (entry.get("extra") or {}).get("backend") != only_backend:
            continue
        questions_meta = entry.get("questions") or {}
        probs_by_q = (entry.get("answers") or {}).get("probs", {})
        top_by_q = (entry.get("answers") or {}).get("top", {})
        if not questions_meta or not probs_by_q:
            continue
        out_questions: dict[str, Any] = {}
        for qname, qmeta in questions_meta.items():
            qtype = qmeta.get("type") if isinstance(qmeta, Mapping) else None
            if qtype not in ("choice", "noul", "score") or qname not in probs_by_q:
                continue
            probs = probs_by_q[qname]
            top = top_by_q.get(qname)
            try:
                label = _hard_label(qtype, probs, top)
            except ValueError:
                continue
            q_record = {k: v for k, v in qmeta.items() if k in ("type", "instructions", "criteria")}
            q_record["label"] = label
            q_record["src"] = f"{source}:{qtype}"
            if include_soft_target:
                q_record["target"] = _soft_target(qtype, probs)
            out_questions[qname] = q_record
        if out_questions:
            records.append({"state": entry.get("state", {}), "questions": out_questions})
    return records
