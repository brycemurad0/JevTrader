"""A full, tiny end-to-end backtest run through `Backtester`, with an inline test strategy
(deliberately not a `jevtrader.strategies` module -- backtest/ owns this test, not the strategy
library)."""

import pandas as pd

from jevtrader.backtest import Backtester, render_html, render_markdown
from jevtrader.core.strategy import Strategy, StrategySpec
from jevtrader.core.types import Order, OrderType, Side
from jevtrader.data.synthetic import bars_to_events, generate_bars

EQ_START = pd.Timestamp("2024-01-02 14:30", tz="UTC")


class _TinyMomentum(Strategy):
    """Buys after a lookback-window gain, sells (and flips short) after a lookback-window loss."""

    spec = StrategySpec(
        name="_test_tiny_momentum",
        description="toy momentum strategy for engine e2e testing",
        asset_classes=("equity",),
        frequency="1min",
        style="momentum",
        default_params={"lookback": 5, "qty": 5, "threshold": 0.0008},
    )

    def on_start(self, ctx):
        self.fills_seen = 0

    def on_bar(self, bar, ctx):
        hist = ctx.bars(bar.symbol, self.params["lookback"] + 1)
        if len(hist) < self.params["lookback"] + 1:
            return
        ret = hist["close"].iloc[-1] / hist["close"].iloc[0] - 1.0
        pos = ctx.position(bar.symbol)
        qty = self.params["qty"]
        threshold = self.params["threshold"]
        if ret > threshold and pos.qty <= 0:
            ctx.submit(Order(symbol=bar.symbol, side=Side.BUY, qty=qty + abs(pos.qty), type=OrderType.MARKET, strategy_id=self.id))
        elif ret < -threshold and pos.qty >= 0:
            ctx.submit(Order(symbol=bar.symbol, side=Side.SELL, qty=qty + abs(pos.qty), type=OrderType.MARKET, strategy_id=self.id))

    def on_fill(self, fill, ctx):
        self.fills_seen += 1


def test_end_to_end_backtest_runs_and_produces_a_coherent_result():
    end = EQ_START + pd.Timedelta(days=3, hours=6, minutes=30)
    df = generate_bars("AAPL", EQ_START, end, freq="1min", seed=11)
    data = {"AAPL": list(bars_to_events(df, "AAPL"))}

    strat = _TinyMomentum(["AAPL"])
    bt = Backtester([strat], data, initial_cash=100_000)
    result = bt.run()

    assert len(result.equity_curve) == len(df)
    assert result.equity_curve.index.is_monotonic_increasing
    assert result.metrics["n_fills"] == len(result.fills)
    assert strat.fills_seen == len(result.fills)
    assert set(result.per_strategy_pnl) <= {strat.id}
    for key in ("sharpe", "sortino", "calmar", "max_drawdown", "psr", "dsr", "fee_drag", "turnover"):
        assert key in result.metrics

    # accounting sanity carried over from the dedicated accounting tests
    account = result.final_account
    reconstructed = account.cash + sum(
        pos.qty * (bt.broker.mark_price(sym) or pos.avg_price) for sym, pos in result.positions.items()
    )
    assert abs(reconstructed - account.equity) < 1e-6

    # the report renders without needing matplotlib or any other plotting dependency.
    md = render_markdown(result, title="Tiny Momentum")
    assert "Sharpe" in md.replace("sharpe", "Sharpe") or "sharpe" in md
    html = render_html(result, title="Tiny Momentum")
    assert "<svg" in html
    assert "Tiny Momentum" in html


def test_multiple_strategies_on_different_symbols_do_not_cross_wires():
    class _AlwaysBuyOnce(Strategy):
        spec = StrategySpec(name="_test_always_buy_once", description="", asset_classes=("equity",), frequency="1min", style="momentum")

        def on_start(self, ctx):
            self.done = False

        def on_bar(self, bar, ctx):
            if not self.done:
                ctx.submit(Order(symbol=bar.symbol, side=Side.BUY, qty=3, type=OrderType.MARKET))
                self.done = True

    end = EQ_START + pd.Timedelta(hours=1)
    df_a = generate_bars("AAPL", EQ_START, end, freq="1min", seed=1)
    df_b = generate_bars("MSFT", EQ_START, end, freq="1min", seed=2, start_price=300.0)
    data = {"AAPL": list(bars_to_events(df_a, "AAPL")), "MSFT": list(bars_to_events(df_b, "MSFT"))}

    strat_a = _AlwaysBuyOnce(["AAPL"], strategy_id="strat_a")
    strat_b = _AlwaysBuyOnce(["MSFT"], strategy_id="strat_b")
    bt = Backtester([strat_a, strat_b], data, initial_cash=200_000)
    result = bt.run()

    assert set(result.fills["symbol"]) == {"AAPL", "MSFT"}
    assert set(result.fills["strategy_id"]) == {"strat_a", "strat_b"}
    aapl_fill = result.fills[result.fills["symbol"] == "AAPL"].iloc[0]
    msft_fill = result.fills[result.fills["symbol"] == "MSFT"].iloc[0]
    assert aapl_fill["strategy_id"] == "strat_a"
    assert msft_fill["strategy_id"] == "strat_b"
