import json

from typer.testing import CliRunner

from jevtrader.cli import app
from jevtrader.live.journal_summary import summarize

runner = CliRunner()


def test_strategies_lists_rebalancer():
    r = runner.invoke(app, ["strategies"])
    assert r.exit_code == 0, r.output
    assert "smart_rebalance" in r.output


def test_backtest_synthetic_writes_metrics(tmp_path):
    r = runner.invoke(app, ["backtest", "smart_rebalance", "-s", "SPY", "-s", "BTC/USD", "--timeframe", "1H",
                            "--start", "2025-01-01", "--end", "2025-07-01", "--out", str(tmp_path)])
    assert r.exit_code == 0, r.output
    metrics = [p for p in tmp_path.glob("*.json")]
    assert metrics and "sharpe" in json.loads(metrics[0].read_text())


def test_journal_summary_counts_and_drawdown():
    ev = [
        {"kind": "equity", "ts": "2026-01-01T00:00:00Z", "equity": 100.0, "cash": 100.0},
        {"kind": "equity", "ts": "2026-01-02T00:00:00Z", "equity": 110.0, "cash": 100.0},
        {"kind": "equity", "ts": "2026-01-03T00:00:00Z", "equity": 99.0, "cash": 100.0},
        {"kind": "fill", "strategy_id": "s", "qty": 1, "price": 100, "fee": 0.15, "liquidity": "maker"},
        {"kind": "fill", "strategy_id": "s", "qty": 1, "price": 101, "fee": 0.15, "liquidity": "taker"},
        {"kind": "fill", "strategy_id": "other", "qty": 5, "price": 100, "fee": 9.0, "liquidity": "taker"},
    ]
    s = summarize(ev, strategy_id="s")
    assert s["n_trades"] == 1 and s["trading_days"] == 3
    assert abs(s["max_drawdown"] - 0.1) < 1e-9
    assert abs(s["cost_bps"] - 2e4 * 0.30 / 201) < 1e-9
