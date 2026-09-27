"""Event-driven backtester: engine, SimBroker fill models, metrics, walk-forward, and reports."""

from jevtrader.backtest.engine import Backtester, BacktestContext, BacktestResult
from jevtrader.backtest.metrics import (
    compute_metrics,
    compute_trade_pnls,
    deflated_sharpe_ratio,
    max_drawdown,
    max_drawdown_duration,
    periods_per_year,
    probabilistic_sharpe_ratio,
)
from jevtrader.backtest.report import fee_breakdown, render_html, render_markdown, save_report
from jevtrader.backtest.sim_broker import SimBroker, SlippageModel
from jevtrader.backtest.walkforward import (
    PurgedFold,
    Split,
    SweepResult,
    Trial,
    WalkForwardFoldResult,
    WalkForwardResult,
    purged_kfold_splits,
    robustness_report,
    sweep,
    walk_forward_run,
    walk_forward_splits,
)

__all__ = [
    "Backtester",
    "BacktestContext",
    "BacktestResult",
    "SimBroker",
    "SlippageModel",
    "compute_metrics",
    "compute_trade_pnls",
    "deflated_sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "max_drawdown",
    "max_drawdown_duration",
    "periods_per_year",
    "fee_breakdown",
    "render_markdown",
    "render_html",
    "save_report",
    "Split",
    "PurgedFold",
    "SweepResult",
    "Trial",
    "WalkForwardFoldResult",
    "WalkForwardResult",
    "walk_forward_splits",
    "purged_kfold_splits",
    "sweep",
    "walk_forward_run",
    "robustness_report",
]
