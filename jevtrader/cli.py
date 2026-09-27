"""`jev` command line: backtest -> paper -> live, all local.

    jev strategies                         list registered strategies
    jev fetch BTC/USD --start 2026-01-01   download Alpaca history into the local cache
    jev backtest <strategy> -s BTC/USD --data cache --start ... --end ...
    jev paper <strategy> -s SPY -s BTC/USD [--dashboard]
    jev live  <strategy> ...               (requires promotion + JEV_LIVE_CONFIRM)
    jev dashboard-demo                     UI with synthetic data, no keys needed
    jev ping [--backend kev]               one decision-model call, reports latency
    jev promote paper <id> --report bt.json
    jev promote live  <id> --run runs/<run_id>
    jev promotions                         show promotion records
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import typer

from jevtrader.config import ROOT, load_settings
from jevtrader.core.broker import TradingMode

app = typer.Typer(add_completion=False, no_args_is_help=True, help="JevTrader: local fee-aware quant trading.")
promote_app = typer.Typer(no_args_is_help=True, help="Backtest->paper->live promotion gate.")
app.add_typer(promote_app, name="promote")


def _parse_params(params: Optional[str]) -> dict[str, Any]:
    if not params:
        return {}
    p = Path(params)
    return json.loads(p.read_text()) if p.exists() else json.loads(params)


def _ts(s: Optional[str], default: pd.Timestamp) -> pd.Timestamp:
    if not s:
        return default
    t = pd.Timestamp(s)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _load_csv_bars(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        idx = pd.to_datetime(df["timestamp"], unit="s", utc=True) + pd.Timedelta(minutes=1)  # open time -> close time
    else:
        col = "datetime" if "datetime" in df.columns else df.columns[0]
        idx = pd.to_datetime(df[col], utc=True)
    df.index = idx
    return df[[c for c in ("open", "high", "low", "close", "volume", "vwap") if c in df.columns]].sort_index()


def _risk_manager():
    from jevtrader.risk.limits import RiskLimits
    from jevtrader.risk.manager import RiskManager

    return RiskManager(RiskLimits.from_yaml(ROOT / "config" / "risk.yaml"))


@app.command()
def strategies() -> None:
    """List registered strategies."""
    from jevtrader.core.registry import all_strategies

    for name, cls in sorted(all_strategies().items()):
        s = cls.spec
        typer.echo(f"{name:32s} {s.style:15s} {s.frequency:6s} {','.join(s.asset_classes):14s} jev={s.uses_jev}  {s.description}")


@app.command()
def fetch(
    symbol: str,
    start: str = typer.Option(..., help="UTC start, e.g. 2026-01-01"),
    end: Optional[str] = typer.Option(None, help="UTC end (default now)"),
    timeframe: str = typer.Option("1Min", help="1Min, 5Min, 1Hour, 1Day"),
) -> None:
    """Download Alpaca bars into the local cache (crypto needs no keys)."""
    from jevtrader.data import alpaca_history as ah
    from jevtrader.data.store import load_bars

    settings = load_settings()
    fn = ah.load_crypto_bars if "/" in symbol else ah.load_stock_bars
    df = load_bars(
        symbol, _ts(start, pd.Timestamp.now("UTC")), _ts(end, pd.Timestamp.now("UTC")), timeframe, settings.data_dir,
        fetch_fn=lambda s, a, b, tf: fn(s, a, b, tf, settings=settings),
    )
    typer.echo(f"{symbol}: {len(df)} bars cached ({df.index.min()} -> {df.index.max()})")


@app.command()
def backtest(
    strategy: str,
    symbol: list[str] = typer.Option(..., "--symbol", "-s"),
    data: str = typer.Option("synthetic", help="synthetic | cache | path/to/file.csv"),
    start: Optional[str] = typer.Option(None),
    end: Optional[str] = typer.Option(None),
    timeframe: str = typer.Option("1Min"),
    params: Optional[str] = typer.Option(None, help="JSON string or path to JSON file"),
    cash: float = typer.Option(10_000.0),
    jev: str = typer.Option("offline", help="offline | none (backtests never call a paid API)"),
    risk: bool = typer.Option(True, help="route orders through config/risk.yaml"),
    out: Path = typer.Option(ROOT / "runs" / "backtests"),
    seed: int = typer.Option(0),
) -> None:
    """Event-driven, fee-aware backtest. Writes metrics JSON + HTML/markdown report."""
    from jevtrader.backtest import Backtester, save_report
    from jevtrader.core.registry import get
    from jevtrader.core.types import AssetClass
    from jevtrader.data.synthetic import bars_to_events, generate_bars

    settings = load_settings()
    cls = get(strategy)
    end_ts = _ts(end, pd.Timestamp.now("UTC").floor("min"))
    start_ts = _ts(start, end_ts - pd.Timedelta(days=5))
    feeds = {}
    for i, sym in enumerate(symbol):
        if data == "synthetic":
            ac = AssetClass.CRYPTO if "/" in sym else AssetClass.EQUITY
            df = generate_bars(sym, start_ts, end_ts, freq=timeframe.lower(), asset_class=ac, seed=seed + i)
        elif data == "cache":
            from jevtrader.data.store import load_bars

            df = load_bars(sym, start_ts, end_ts, timeframe, settings.data_dir)
        else:
            df = _load_csv_bars(Path(data)).loc[start_ts:end_ts] if (start or end) else _load_csv_bars(Path(data))
        if df is None or df.empty:
            raise typer.BadParameter(f"no data for {sym} from {data}")
        feeds[sym] = bars_to_events(df, sym)

    advisor = None
    if jev == "offline":
        from jevtrader.jev.advisor import OfflineJevAdvisor

        advisor = OfflineJevAdvisor()
    strat = cls(list(symbol), _parse_params(params))
    result = Backtester([strat], feeds, risk=_risk_manager() if risk else None, jev=advisor, initial_cash=cash).run()

    out.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%S")
    base = f"{strategy}_{'-'.join(s.replace('/', '') for s in symbol)}_{stamp}"
    save_report(result, out, basename=base, title=f"{strategy} {','.join(symbol)}")
    (out / f"{base}.json").write_text(json.dumps(result.metrics, indent=2, default=float))
    m = result.metrics
    for k in ("total_return", "sharpe", "max_drawdown", "n_closed_trades", "trades_per_day", "fee_total", "fee_drag", "cost_per_trade_bps", "psr"):
        if k in m:
            typer.echo(f"{k:22s} {m[k]:.4f}" if isinstance(m[k], float) else f"{k:22s} {m[k]}")
    typer.echo(f"report: {out / base}.html")


def _run_trading(strategy: str, symbols: list[str], params: Optional[str], mode: TradingMode, dashboard: bool, port: int, shadow: Optional[list[str]] = None, record: bool = False) -> None:
    from jevtrader.core.fees import default_fees
    from jevtrader.core.registry import get
    from jevtrader.execution.alpaca_broker import AlpacaBroker
    from jevtrader.jev.advisor import make_advisor
    from jevtrader.jev.log import DecisionLog
    from jevtrader.live.runner import LiveRunner
    from jevtrader.live.streams import CryptoStream, StockStream

    settings = load_settings()
    if not settings.has_alpaca:
        raise typer.BadParameter("set ALPACA_API_KEY / ALPACA_SECRET_KEY in .env (see .env.example)")
    strat = get(strategy)(symbols, _parse_params(params))
    broker = AlpacaBroker(mode, settings=settings, fee_model=default_fees(), strategy_ids=[strat.id])

    q: asyncio.Queue = asyncio.Queue()
    key, secret = settings.keys_for(mode)
    crypto = [s for s in symbols if "/" in s]
    stocks = [s for s in symbols if "/" not in s]
    streams = []
    if stocks:
        streams.append(StockStream(key, secret, q, stocks, feed=settings.alpaca_data_feed))
    if crypto:
        streams.append(CryptoStream(key, secret, q, crypto))

    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    dlog = DecisionLog(settings.runs_dir / "decisions.jsonl")
    advisor = make_advisor(settings, mode, decision_log=dlog, backend=settings.decision_backend)
    if shadow:
        # Answer with the primary backend; fire identical questions at shadow backends in the
        # background and log all answers -> head-to-head data for the Jev/Kev/Laya scoreboard.
        from jevtrader.jev.advisor import ShadowAdvisor

        shadows = {b: make_advisor(settings, mode, backend=b) for b in shadow if b != settings.decision_backend}
        advisor = ShadowAdvisor(advisor, shadows, decision_log=dlog)
    state = None
    if dashboard:
        from jevtrader.dashboard.state import DashboardState

        state = DashboardState()
    strats = [strat]
    if record:
        # Journal real quotes/trades/L2 books so microstructure strategies can be validated offline
        # (jevtrader.research.record_replay.load_market_events -> Backtester).
        from jevtrader.research.record_replay import MarketDataRecorder

        strats.append(MarketDataRecorder(symbols, {"dir": str(settings.runs_dir / "market_data")}))
    runner = LiveRunner(strats, broker, streams, _risk_manager(), jev=advisor, runs_dir=settings.runs_dir, dashboard_state=state)
    typer.echo(f"[{mode.value.upper()}] {strat.id} | decision backend={settings.decision_backend} | journal={settings.runs_dir}")

    async def main() -> None:
        tasks = [asyncio.create_task(runner.run())]
        if state is not None:
            import uvicorn

            from jevtrader.dashboard.app import create_app

            state.bind_actions(submit_order=runner.submit_manual, cancel_order=broker.cancel, kill_switch=runner.kill_switch)
            server = uvicorn.Server(uvicorn.Config(create_app(state, runner), host="127.0.0.1", port=port, log_level="warning"))
            tasks.append(asyncio.create_task(server.serve()))
            typer.echo(f"dashboard: http://127.0.0.1:{port}")
        tasks.append(asyncio.create_task(broker.run_stream()))
        try:
            await asyncio.gather(*tasks)
        finally:
            runner.stop()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        typer.echo("stopped")


@app.command()
def paper(
    strategy: str,
    symbol: list[str] = typer.Option(..., "--symbol", "-s"),
    params: Optional[str] = typer.Option(None),
    dashboard: bool = typer.Option(True),
    port: int = typer.Option(8765),
    shadow: Optional[list[str]] = typer.Option(None, help="extra decision backends to log head-to-head (kev, laya, jev)"),
    record: bool = typer.Option(True, help="record real quotes/trades/books to runs/market_data for replay"),
) -> None:
    """Forward-test on Alpaca PAPER. Journals everything to runs/ for the promotion gate."""
    _run_trading(strategy, symbol, params, TradingMode.PAPER, dashboard, port, shadow, record)


@app.command()
def live(
    strategy: str,
    symbol: list[str] = typer.Option(..., "--symbol", "-s"),
    params: Optional[str] = typer.Option(None),
    dashboard: bool = typer.Option(True),
    port: int = typer.Option(8765),
) -> None:
    """REAL MONEY. Refuses unless the strategy passed promotion AND JEV_LIVE_CONFIRM is set."""
    typer.confirm("This routes REAL orders with REAL money. Continue?", abort=True)
    _run_trading(strategy, symbol, params, TradingMode.LIVE, dashboard, port)


@app.command("dashboard-demo")
def dashboard_demo(port: int = typer.Option(8765)) -> None:
    """Open the local dashboard with synthetic data and a fake broker (no keys needed)."""
    from jevtrader.dashboard.demo import run_demo

    run_demo(port=port)


@app.command()
def ping(backend: Optional[str] = typer.Option(None, help="jev | kev | laya (default: DECISION_BACKEND)")) -> None:
    """One decision-model call; prints answer + latency."""
    from jevtrader.jev.ping import ping as _ping

    typer.echo(json.dumps(_ping(load_settings(), backend=backend), indent=2, default=str))


@promote_app.command("paper")
def promote_paper(candidate_id: str, report: Path = typer.Option(..., help="JSON with oos_sharpe_after_fees, dsr_prob, max_drawdown, n_trades, fee_stress_pass, cost_bps")) -> None:
    """Backtest -> paper gate."""
    from jevtrader.live.promotion import promote_to_paper

    r = promote_to_paper(candidate_id, json.loads(report.read_text()))
    typer.echo(f"{'PASSED' if r.passed else 'FAILED'}: {r.reasons or 'all checks passed'}")


@promote_app.command("live")
def promote_live(candidate_id: str, run: list[Path] = typer.Option(..., help="paper run dir(s) under runs/")) -> None:
    """Paper -> live gate, computed from paper-trading journals."""
    from jevtrader.live.journal_summary import summarize_runs
    from jevtrader.live.promotion import promote_to_live

    summary = summarize_runs(run, strategy_id=candidate_id)
    typer.echo(json.dumps(summary, indent=2, default=str))
    r = promote_to_live(candidate_id, summary)
    typer.echo(f"{'PASSED' if r.passed else 'FAILED'}: {r.reasons or 'all checks passed'}")


@app.command()
def promotions() -> None:
    """Show promotion records."""
    from jevtrader.live.promotion import list_promotions

    for rec in list_promotions(load_settings()):
        typer.echo(f"{rec['candidate_id']:40s} stage={rec['stage']:6s} capital_frac={rec.get('capital_frac')}")


if __name__ == "__main__":
    app()
