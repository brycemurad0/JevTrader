"""Summarize a paper/live run journal (runs/<run_id>/events.jsonl) into the metrics the
paper->live promotion gate expects (see jevtrader.live.promotion.evaluate_paper).

Equity snapshots are account-level, so run ONE candidate strategy per paper account/run when
you intend to promote it; fills are filtered by `strategy_id` when given.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


def read_journal(paths: Iterable[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in paths:
        p = Path(p)
        f = p / "events.jsonl" if p.is_dir() else p
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def summarize(events: list[dict[str, Any]], strategy_id: Optional[str] = None) -> dict[str, Any]:
    eq = [(pd.Timestamp(e.get("ts")), float(e["equity"])) for e in events if e.get("kind") == "equity"]
    fills = [e for e in events if e.get("kind") == "fill" and (strategy_id is None or e.get("strategy_id") == strategy_id)]

    summary: dict[str, Any] = {"strategy_id": strategy_id, "n_fills": len(fills)}
    # round trips ~ fills / 2 (entry + exit); conservative floor
    summary["n_trades"] = len(fills) // 2

    notional = sum(float(f["qty"]) * float(f["price"]) for f in fills)
    fees = sum(float(f.get("fee", 0.0)) for f in fills)
    summary["fees_paid"] = fees
    summary["notional_traded"] = notional
    # Fee cost per round trip in bps of one side's notional (comparable to backtest cost_bps).
    summary["cost_bps"] = (2e4 * fees / notional) if notional > 0 else None
    summary["maker_share"] = (sum(1 for f in fills if f.get("liquidity") == "maker") / len(fills)) if fills else None

    if len(eq) >= 2:
        s = pd.Series([v for _, v in eq], index=[t for t, _ in eq]).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        daily = s.resample("1D").last().dropna()
        summary["trading_days"] = int(len(daily))
        rets = daily.pct_change().dropna()
        summary["total_return"] = float(s.iloc[-1] / s.iloc[0] - 1.0)
        summary["sharpe"] = float(np.sqrt(252) * rets.mean() / rets.std()) if len(rets) > 1 and rets.std() > 0 else 0.0
        peak = s.cummax()
        summary["max_drawdown"] = float(((peak - s) / peak).max())
    else:
        summary.update(trading_days=0, total_return=0.0, sharpe=0.0, max_drawdown=0.0)
    return summary


def summarize_runs(run_dirs: Iterable[Path], strategy_id: Optional[str] = None) -> dict[str, Any]:
    return summarize(read_journal(run_dirs), strategy_id)
