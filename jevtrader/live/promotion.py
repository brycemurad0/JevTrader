"""The backtest -> paper -> live promotion gate (see docs/PROMOTION.md).

A "candidate" is a strategy+symbols+params combination, identified by a `candidate_id` string
(by convention, `Strategy.id`, e.g. `"meanrev:AAPL,MSFT"` -- include a params hash in your own
`strategy_id` if you need per-parameter-set promotion tracking).

Stages:

1. **backtest -> paper**: a backtest report (a dict of metrics -- OOS Sharpe after fees, DSR
   probability, max drawdown, trade count, a fee-stress pass flag) is checked against
   `config/promotion.yaml`'s `backtest` thresholds. On a pass, a promotion record is written.
2. **paper -> live**: a summary of the candidate's paper-trading journal (see
   `jevtrader.live.runner.Journal`) is checked against the `paper` thresholds: minimum trading
   days and trade count, paper Sharpe, max drawdown, and paper-vs-backtest cost/slippage
   deviation within tolerance. On a pass, the record is upgraded to `stage="live"` and a capital
   cap + scale-up schedule (from the `live` thresholds) starts ticking from `live_started_at`.

Records are stored as JSON under `settings.state_dir/promotions/<candidate_id>.json`.
`assert_live_allowed(strategy_id, settings)` is what `AlpacaBroker` calls before allowing any
LIVE-mode construction; it raises `PromotionError` unless a passing `stage="live"` record exists.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd
import yaml

from jevtrader.config import ROOT

DEFAULT_THRESHOLDS: dict[str, Any] = {
    "backtest": {
        "min_oos_sharpe_after_fees": 1.0,
        "min_dsr_prob": 0.95,
        "max_drawdown": 0.20,
        "min_trades": 30,
        "require_fee_stress_pass": True,
    },
    "paper": {
        "min_trading_days": 20,
        "min_trades": 30,
        "min_sharpe": 0.5,
        "max_drawdown": 0.20,
        "max_cost_deviation_bps": 15.0,
    },
    "live": {
        "initial_capital_frac": 0.10,
        "scale_schedule": [
            {"after_days": 0, "capital_frac": 0.10},
            {"after_days": 10, "capital_frac": 0.25},
            {"after_days": 20, "capital_frac": 0.50},
            {"after_days": 30, "capital_frac": 1.00},
        ],
    },
}


class PromotionError(RuntimeError):
    """Raised when live trading is requested for a candidate without a passing LIVE promotion."""


@dataclass
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed


# --------------------------------------------------------------------------------- thresholds

def load_thresholds(path: Optional[Path] = None) -> dict[str, Any]:
    path = Path(path) if path is not None else ROOT / "config" / "promotion.yaml"
    if not path.exists():
        return {k: dict(v) for k, v in DEFAULT_THRESHOLDS.items()}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    merged: dict[str, Any] = {}
    for stage, defaults in DEFAULT_THRESHOLDS.items():
        merged[stage] = {**defaults, **(data.get(stage) or {})}
    return merged


# --------------------------------------------------------------------------------- gate logic (pure)

def evaluate_backtest(report: Mapping[str, Any], thresholds: Optional[dict[str, Any]] = None) -> GateResult:
    th = (thresholds or load_thresholds())["backtest"]
    reasons: list[str] = []

    sharpe = float(report.get("oos_sharpe_after_fees", float("-inf")))
    if sharpe < th["min_oos_sharpe_after_fees"]:
        reasons.append(f"oos_sharpe_after_fees {sharpe:.2f} < required {th['min_oos_sharpe_after_fees']:.2f}")

    dsr = float(report.get("dsr_prob", 0.0))
    if dsr < th["min_dsr_prob"]:
        reasons.append(f"dsr_prob {dsr:.3f} < required {th['min_dsr_prob']:.3f}")

    max_dd = float(report.get("max_drawdown", 1.0))
    if max_dd > th["max_drawdown"]:
        reasons.append(f"max_drawdown {max_dd:.2%} > allowed {th['max_drawdown']:.2%}")

    n_trades = int(report.get("n_trades", 0))
    if n_trades < th["min_trades"]:
        reasons.append(f"n_trades {n_trades} < required {th['min_trades']}")

    if th.get("require_fee_stress_pass", True) and not report.get("fee_stress_pass", False):
        reasons.append("fee_stress_pass is not True in the backtest report")

    return GateResult(passed=not reasons, reasons=reasons, metrics=dict(report))


def evaluate_paper(
    paper_summary: Mapping[str, Any],
    backtest_report: Mapping[str, Any],
    thresholds: Optional[dict[str, Any]] = None,
) -> GateResult:
    th = (thresholds or load_thresholds())["paper"]
    reasons: list[str] = []

    days = int(paper_summary.get("trading_days", 0))
    if days < th["min_trading_days"]:
        reasons.append(f"trading_days {days} < required {th['min_trading_days']}")

    n_trades = int(paper_summary.get("n_trades", 0))
    if n_trades < th["min_trades"]:
        reasons.append(f"n_trades {n_trades} < required {th['min_trades']}")

    sharpe = float(paper_summary.get("sharpe", float("-inf")))
    if sharpe < th["min_sharpe"]:
        reasons.append(f"sharpe {sharpe:.2f} < required {th['min_sharpe']:.2f}")

    max_dd = float(paper_summary.get("max_drawdown", 1.0))
    if max_dd > th["max_drawdown"]:
        reasons.append(f"max_drawdown {max_dd:.2%} > allowed {th['max_drawdown']:.2%}")

    paper_cost = paper_summary.get("cost_bps")
    backtest_cost = backtest_report.get("cost_bps")
    if paper_cost is not None and backtest_cost is not None:
        deviation = abs(float(paper_cost) - float(backtest_cost))
        if deviation > th["max_cost_deviation_bps"]:
            reasons.append(
                f"paper cost {float(paper_cost):.1f}bps vs backtest {float(backtest_cost):.1f}bps "
                f"deviates {deviation:.1f}bps > allowed {th['max_cost_deviation_bps']:.1f}bps"
            )

    return GateResult(passed=not reasons, reasons=reasons, metrics=dict(paper_summary))


# --------------------------------------------------------------------------------- record storage

def _promotions_dir(settings: Any) -> Path:
    return Path(settings.state_dir) / "promotions"


def _record_path(candidate_id: str, settings: Any) -> Path:
    safe = candidate_id.replace("/", "_").replace(":", "_").replace(",", "_").replace(" ", "_")
    return _promotions_dir(settings) / f"{safe}.json"


def load_promotion(candidate_id: str, settings: Optional[Any] = None) -> Optional[dict[str, Any]]:
    settings = settings or _default_settings()
    path = _record_path(candidate_id, settings)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def list_promotions(settings: Optional[Any] = None) -> list[dict[str, Any]]:
    settings = settings or _default_settings()
    d = _promotions_dir(settings)
    if not d.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]


def _save_promotion(record: dict[str, Any], settings: Any) -> Path:
    path = _record_path(record["candidate_id"], settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str))
    return path


def _default_settings() -> Any:
    from jevtrader.config import load_settings

    return load_settings()


# --------------------------------------------------------------------------------- promotion actions

def promote_to_paper(
    candidate_id: str,
    backtest_report: Mapping[str, Any],
    *,
    thresholds: Optional[dict[str, Any]] = None,
    settings: Optional[Any] = None,
) -> GateResult:
    """Evaluate `backtest_report` against the backtest gate; on a pass, write (or refresh) a
    `stage="paper"` promotion record. Always returns the `GateResult` regardless of outcome."""
    settings = settings or _default_settings()
    result = evaluate_backtest(backtest_report, thresholds)
    if result.passed:
        existing = load_promotion(candidate_id, settings) or {}
        record = {
            "candidate_id": candidate_id,
            "stage": "paper",
            "promoted_at": pd.Timestamp.utcnow().isoformat(),
            "backtest_metrics": dict(backtest_report),
            "paper_metrics": existing.get("paper_metrics"),
            "live_started_at": None,
            "capital_frac": 0.0,
            "notes": "",
        }
        _save_promotion(record, settings)
    return result


def promote_to_live(
    candidate_id: str,
    paper_summary: Mapping[str, Any],
    *,
    thresholds: Optional[dict[str, Any]] = None,
    settings: Optional[Any] = None,
) -> GateResult:
    """Evaluate `paper_summary` (a summary of the candidate's paper journal, see
    `jevtrader.live.runner.Journal`) against the paper gate, including paper-vs-backtest cost
    deviation; on a pass, upgrade the record to `stage="live"` and start the capital scale-up
    schedule. Requires an existing `stage="paper"` (or already-`"live"`) record."""
    settings = settings or _default_settings()
    existing = load_promotion(candidate_id, settings)
    if existing is None or existing.get("stage") not in ("paper", "live"):
        return GateResult(passed=False, reasons=["candidate has not passed the backtest->paper gate yet"])

    result = evaluate_paper(paper_summary, existing.get("backtest_metrics", {}), thresholds)
    if result.passed:
        th_live = (thresholds or load_thresholds())["live"]
        record = dict(existing)
        record.update(
            stage="live",
            paper_metrics=dict(paper_summary),
            live_started_at=existing.get("live_started_at") or pd.Timestamp.utcnow().isoformat(),
            capital_frac=th_live["initial_capital_frac"],
        )
        _save_promotion(record, settings)
    return result


def current_capital_frac(
    candidate_id: str, settings: Optional[Any] = None, thresholds: Optional[dict[str, Any]] = None
) -> float:
    """Fraction of intended live size a promoted candidate should currently trade at, per the
    `live.scale_schedule`. 0.0 if the candidate is not (yet) live."""
    settings = settings or _default_settings()
    record = load_promotion(candidate_id, settings)
    if record is None or record.get("stage") != "live" or not record.get("live_started_at"):
        return 0.0
    th = (thresholds or load_thresholds())["live"]
    started = pd.Timestamp(record["live_started_at"])
    days = (pd.Timestamp.utcnow() - started).total_seconds() / 86400.0
    frac = th["scale_schedule"][0]["capital_frac"]
    for step in th["scale_schedule"]:
        if days >= step["after_days"]:
            frac = step["capital_frac"]
    return float(frac)


def assert_live_allowed(strategy_id: str, settings: Optional[Any] = None) -> None:
    """Raise `PromotionError` unless `strategy_id` has a passing `stage="live"` promotion
    record. Called by `AlpacaBroker` before it allows any LIVE-mode construction."""
    settings = settings or _default_settings()
    record = load_promotion(strategy_id, settings)
    if record is None or record.get("stage") != "live":
        raise PromotionError(
            f"{strategy_id!r} has no passing LIVE promotion record. Run "
            f"jevtrader.live.promotion.promote_to_live(...) for it first; refusing to trade live."
        )
