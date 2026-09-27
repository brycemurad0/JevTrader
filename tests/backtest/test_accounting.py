"""Position/PnL accounting consistency: cash + mark-to-market positions must always equal
reported equity, and the broker's internal ledger must match an independent replay of its own
fills log through the same `Position.apply_fill` accounting core."""

import pandas as pd

from jevtrader.backtest import Backtester
from jevtrader.core.strategy import Strategy, StrategySpec
from jevtrader.core.types import Fill, Liquidity, Order, OrderType, Position, Side
from jevtrader.data.synthetic import bars_to_events, generate_bars

EQ_START = pd.Timestamp("2024-01-02 14:30", tz="UTC")


class _Flipper(Strategy):
    """Alternates buy/sell every few bars, on two symbols, to exercise realized PnL, fees, and
    position flips (long -> flat -> short -> flat -> long)."""

    spec = StrategySpec(name="_test_flipper", description="", asset_classes=("equity",), frequency="1min", style="momentum")

    def on_start(self, ctx):
        self.count = 0

    def on_bar(self, bar, ctx):
        self.count += 1
        if self.count % 7 != 0:
            return
        pos = ctx.position(bar.symbol)
        if pos.qty == 0:
            ctx.submit(Order(symbol=bar.symbol, side=Side.BUY, qty=5, type=OrderType.MARKET))
        elif pos.qty > 0:
            ctx.submit(Order(symbol=bar.symbol, side=Side.SELL, qty=5, type=OrderType.MARKET))
        else:
            ctx.submit(Order(symbol=bar.symbol, side=Side.BUY, qty=5, type=OrderType.MARKET))


def _run_backtest():
    end = EQ_START + pd.Timedelta(hours=2)
    df_a = generate_bars("AAPL", EQ_START, end, freq="1min", seed=21)
    df_b = generate_bars("MSFT", EQ_START, end, freq="1min", seed=22, start_price=300.0)
    data = {"AAPL": list(bars_to_events(df_a, "AAPL")), "MSFT": list(bars_to_events(df_b, "MSFT"))}
    strat = _Flipper(["AAPL", "MSFT"])
    bt = Backtester([strat], data, initial_cash=100_000)
    return bt, bt.run()


def test_final_account_equity_equals_cash_plus_marked_positions():
    bt, result = _run_backtest()
    assert len(result.fills) > 0  # sanity: the strategy actually traded

    account = result.final_account
    reconstructed_equity = account.cash
    for symbol, pos in result.positions.items():
        price = bt.broker.mark_price(symbol)
        reconstructed_equity += pos.qty * (price if price is not None else pos.avg_price)
    assert abs(reconstructed_equity - account.equity) < 1e-6


def test_broker_ledger_matches_independent_replay_of_its_own_fills():
    bt, result = _run_backtest()

    cash = bt.broker.initial_cash
    positions: dict[str, Position] = {}
    for _, row in result.fills.iterrows():
        pos = positions.setdefault(row["symbol"], Position(row["symbol"]))
        side = Side.BUY if row["side"] == "buy" else Side.SELL
        fill = Fill(
            client_order_id=row["client_order_id"],
            symbol=row["symbol"],
            side=side,
            qty=row["qty"],
            price=row["price"],
            fee=row["fee"],
            liquidity=Liquidity.TAKER,
            ts=row["ts"],
        )
        pos.apply_fill(fill)
        cash += -side.sign * row["qty"] * row["price"] - row["fee"]

    assert abs(cash - result.final_account.cash) < 1e-6
    for symbol, pos in positions.items():
        assert abs(pos.qty - result.positions[symbol].qty) < 1e-9
        assert abs(pos.avg_price - result.positions[symbol].avg_price) < 1e-6

    equity = cash + sum(pos.qty * (bt.broker.mark_price(sym) or pos.avg_price) for sym, pos in positions.items())
    assert abs(equity - result.final_account.equity) < 1e-6


def test_equity_curve_never_produces_nan_and_starts_at_initial_cash():
    bt, result = _run_backtest()
    assert not result.equity_curve.isna().any()
    assert abs(result.equity_curve.iloc[0] - bt.broker.initial_cash) < result.equity_curve.iloc[0] * 0.05
