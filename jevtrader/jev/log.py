"""Append-only JSONL decision log: every Jev call, in or out of the trading loop, gets one line.

This is the honesty mechanism for the whole `jev` package: `calibration.py` can only tell you
whether Jev adds value if every call it ever made (state in, answers out, latency, whether it
was late) is recorded, not just the ones that happened to lead to a trade. `ReplayJevAdvisor`
(in `advisor.py`) reads this same format back to replay real paper-trading answers in a
backtest for an honest before/after comparison against `OfflineJevAdvisor`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional

import pandas as pd

from jevtrader.core.interfaces import JevView


def _serialize_question(question: Any) -> Any:
    if hasattr(question, "model_dump"):
        try:
            return question.model_dump(exclude_none=True)
        except Exception:
            pass
    if isinstance(question, Mapping):
        return dict(question)
    return str(question)


def _serialize_questions(questions: Mapping[str, Any]) -> dict[str, Any]:
    return {name: _serialize_question(q) for name, q in questions.items()}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return str(obj)


class DecisionLog:
    """An append-only JSONL file of Jev decisions.

    One line per call: `{ts, symbol, question_key, state, questions, answers, latency_ms,
    late, model}`. `answers` holds `{probs, top, confidence}` from the `JevView`; the raw
    per-question SDK answer detail (e.g. a Score's legend) is not duplicated here -- it lives
    in `JevView.raw` and callers that need it can log it via `extra`.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        symbol: str,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        view: JevView,
        question_key: str = "",
        ts: Optional[pd.Timestamp] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "ts": (ts if ts is not None else pd.Timestamp.now(tz="UTC")).isoformat(),
            "symbol": symbol,
            "question_key": question_key,
            "state": dict(state),
            "questions": _serialize_questions(questions),
            "answers": {
                "probs": {q: dict(labels) for q, labels in view.probs.items()},
                "top": dict(view.top),
                "confidence": dict(view.confidence),
            },
            "latency_ms": view.latency_ms,
            "late": view.late,
            "model": view.model,
        }
        if extra:
            entry["extra"] = dict(extra)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=_json_default, sort_keys=True))
            f.write("\n")
        return entry


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield decoded entries from a JSONL decision log, skipping blank lines."""
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _asof(series: pd.Series, ts: pd.Timestamp) -> Optional[float]:
    """Last observation at or before `ts`, or `None` if `ts` precedes the series."""
    idx = series.index
    pos = idx.searchsorted(ts, side="right") - 1
    if pos < 0:
        return None
    return float(series.iloc[pos])


def annotate_outcomes(
    entries: Iterable[Mapping[str, Any]],
    prices: Mapping[str, pd.Series],
    *,
    default_horizon: Optional[str] = None,
    horizon_of: Optional[Callable[[Mapping[str, Any]], Optional[str]]] = None,
) -> list[dict[str, Any]]:
    """Annotate decision-log entries with the realized forward return over their horizon.

    `prices` maps symbol -> a `close`-price `pd.Series` indexed by ascending, tz-aware
    timestamps covering at least the decision and its horizon. The horizon is parsed as a
    pandas timedelta string (e.g. "5min", "1h") taken, in order, from: `horizon_of(entry)` if
    given, else the entry's `question_key` when it looks like `"direction:<horizon>"`, else
    `default_horizon`. Entries with no resolvable horizon, no price series for their symbol,
    or a horizon running past the end of the series get `fwd_ret_bps: None`.

    Adds `fwd_ret_bps` (float or None) and `fwd_ts` (ISO string or None) to a shallow copy of
    each entry; the input is not mutated.
    """
    out: list[dict[str, Any]] = []
    for raw in entries:
        entry = dict(raw)
        symbol = entry.get("symbol")
        series = prices.get(symbol) if symbol is not None else None

        horizon_str: Optional[str] = None
        if horizon_of is not None:
            horizon_str = horizon_of(entry)
        if horizon_str is None:
            qkey = entry.get("question_key") or ""
            if qkey.startswith("direction:"):
                horizon_str = qkey.split(":", 1)[1]
        if horizon_str is None:
            horizon_str = default_horizon

        fwd_ret_bps: Optional[float] = None
        fwd_ts: Optional[str] = None
        if series is not None and horizon_str and len(series) > 0:
            try:
                horizon = pd.Timedelta(horizon_str)
                ts = pd.Timestamp(entry["ts"])
                target = ts + horizon
                entry_px = _asof(series, ts)
                if target <= series.index[-1]:
                    exit_px = _asof(series, target)
                    fwd_ts = target.isoformat()
                    if entry_px is not None and exit_px is not None and entry_px != 0:
                        fwd_ret_bps = round(1e4 * (exit_px / entry_px - 1.0), 3)
            except (ValueError, KeyError, TypeError):
                fwd_ret_bps = None

        entry["fwd_ret_bps"] = fwd_ret_bps
        entry["fwd_ts"] = fwd_ts
        out.append(entry)
    return out
