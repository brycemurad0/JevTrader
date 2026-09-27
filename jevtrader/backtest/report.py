"""Render a `BacktestResult` to Markdown and a self-contained HTML report.

No plotting library is required: the HTML report's equity curve and drawdown charts are hand-
rolled inline SVG polylines, so the report has zero external dependencies and opens as a single
file in any browser.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from jevtrader.backtest.engine import BacktestResult

_PCT_METRICS = {"total_return", "annualized_return", "annualized_vol", "max_drawdown", "hit_rate", "fee_drag", "exposure"}

_METRIC_ORDER = [
    "total_return",
    "annualized_return",
    "annualized_vol",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "max_drawdown_duration_periods",
    "hit_rate",
    "profit_factor",
    "avg_trade",
    "turnover",
    "exposure",
    "trades_per_day",
    "n_fills",
    "n_closed_trades",
    "fee_total",
    "fee_drag",
    "cost_per_trade_bps",
    "psr",
    "dsr",
    "n_trials",
    "periods_per_year",
]


def _format_metric(key: str, value) -> str:
    if isinstance(value, float):
        if not np.isfinite(value):
            return str(value)
        if key in _PCT_METRICS:
            return f"{value * 100:.2f}%"
        return f"{value:,.4f}"
    return str(value)


def fee_breakdown(fills: pd.DataFrame) -> dict:
    """Fees grouped by liquidity flag (maker/taker) and by symbol, plus the grand total."""
    if fills is None or fills.empty:
        return {"by_liquidity": {}, "by_symbol": {}, "total": 0.0}
    by_liq = {str(k): float(v) for k, v in fills.groupby("liquidity")["fee"].sum().items()}
    by_sym = {str(k): float(v) for k, v in fills.groupby("symbol")["fee"].sum().items()}
    return {"by_liquidity": by_liq, "by_symbol": by_sym, "total": float(fills["fee"].sum())}


def _svg_line_chart(values: np.ndarray, width: int = 760, height: int = 220, pad: int = 34, color: str = "#2563eb", fill: bool = True) -> str:
    n = len(values)
    if n < 2:
        return f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"></svg>'
    vmin, vmax = float(np.min(values)), float(np.max(values))
    vrange = (vmax - vmin) or 1.0
    xs = np.linspace(pad, width - pad, n)
    ys = height - pad - (values - vmin) / vrange * (height - 2 * pad)
    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))
    baseline_y = height - pad
    parts = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    if fill:
        area = f"{xs[0]:.2f},{baseline_y:.2f} {points} {xs[-1]:.2f},{baseline_y:.2f}"
        parts.append(f'<polygon points="{area}" fill="{color}" opacity="0.12"/>')
    parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    parts.append(f'<line x1="{pad}" y1="{baseline_y:.2f}" x2="{width - pad}" y2="{baseline_y:.2f}" stroke="#ccc" stroke-width="1"/>')
    parts.append(f'<text x="{pad}" y="{pad - 12}" font-size="11" fill="#666">max {vmax:,.2f}</text>')
    parts.append(f'<text x="{pad}" y="{height - pad + 16}" font-size="11" fill="#666">min {vmin:,.2f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def render_markdown(result: "BacktestResult", title: str = "Backtest Report") -> str:
    m = result.metrics
    lines = [f"# {title}", "", f"{m.get('n_fills', 0)} fills, {len(result.equity_curve)} equity marks.", "", "## Metrics", "", "| Metric | Value |", "|---|---|"]
    for k in _METRIC_ORDER:
        if k in m:
            lines.append(f"| {k.replace('_', ' ')} | {_format_metric(k, m[k])} |")

    fb = fee_breakdown(result.fills)
    lines += ["", "## Fee breakdown", "", f"Total fees: {fb['total']:.2f}", "", "| Liquidity | Fees |", "|---|---|"]
    lines += [f"| {k} | {v:.2f} |" for k, v in fb["by_liquidity"].items()]
    lines += ["", "| Symbol | Fees |", "|---|---|"]
    lines += [f"| {k} | {v:.2f} |" for k, v in fb["by_symbol"].items()]

    lines += ["", "## Per-strategy PnL", "", "| Strategy | PnL |", "|---|---|"]
    lines += [f"| {sid} | {pnl:,.2f} |" for sid, pnl in result.per_strategy_pnl.items()]
    return "\n".join(lines) + "\n"


def render_html(result: "BacktestResult", title: str = "Backtest Report") -> str:
    m = result.metrics
    equity = result.equity_curve.dropna()
    eq_svg = _svg_line_chart(equity.to_numpy(), color="#2563eb") if len(equity) > 1 else "<p>Not enough data for a chart.</p>"
    if len(equity) > 1:
        dd = (equity / equity.cummax() - 1.0).to_numpy()
        dd_svg = _svg_line_chart(dd, color="#dc2626")
    else:
        dd_svg = "<p>Not enough data for a chart.</p>"

    metrics_rows = "".join(f"<tr><td>{k.replace('_', ' ')}</td><td>{_format_metric(k, m[k])}</td></tr>" for k in _METRIC_ORDER if k in m)
    fb = fee_breakdown(result.fills)
    fee_liq_rows = "".join(f"<tr><td>{k}</td><td>{v:,.2f}</td></tr>" for k, v in fb["by_liquidity"].items())
    fee_sym_rows = "".join(f"<tr><td>{k}</td><td>{v:,.2f}</td></tr>" for k, v in fb["by_symbol"].items())
    strat_rows = "".join(f"<tr><td>{sid}</td><td>{pnl:,.2f}</td></tr>" for sid, pnl in result.per_strategy_pnl.items())

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif; max-width: 900px; margin: 32px auto; padding: 0 16px; color: #111; background: #fff; }}
h1, h2 {{ color: #111; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
td, th {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 14px; }}
th {{ background: #f3f4f6; }}
.section {{ margin-bottom: 32px; }}
</style></head>
<body>
<h1>{title}</h1>
<div class="section"><h2>Equity curve</h2>{eq_svg}</div>
<div class="section"><h2>Drawdown</h2>{dd_svg}</div>
<div class="section"><h2>Metrics</h2><table><tr><th>Metric</th><th>Value</th></tr>{metrics_rows}</table></div>
<div class="section"><h2>Fee breakdown</h2><p>Total fees: {fb['total']:,.2f}</p>
<table><tr><th>Liquidity</th><th>Fees</th></tr>{fee_liq_rows}</table>
<table><tr><th>Symbol</th><th>Fees</th></tr>{fee_sym_rows}</table>
</div>
<div class="section"><h2>Per-strategy PnL</h2><table><tr><th>Strategy</th><th>PnL</th></tr>{strat_rows}</table></div>
</body></html>
"""


def save_report(result: "BacktestResult", out_dir, basename: str = "report", title: str = "Backtest Report") -> tuple[Path, Path]:
    """Write `<out_dir>/<basename>.md` and `.html`, creating `out_dir` if needed. Returns their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{basename}.md"
    html_path = out_dir / f"{basename}.html"
    md_path.write_text(render_markdown(result, title))
    html_path.write_text(render_html(result, title))
    return md_path, html_path
