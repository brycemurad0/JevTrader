"""Vectorized fast-screen backtester for parameter sweeps over hundreds of thousands of bars.

It reproduces the `SimBroker` bar-mode cost model exactly, but for *position paths* instead of
individual orders, so a sweep over 100 parameter sets on 900k BTC bars runs in seconds:

* A strategy is expressed as a target position series `target[t]` in units of "fraction of
  equity" (or signed units if `units=True`), decided at bar `t`'s CLOSE using only information
  available at that close.
* Any change in target at bar `t` is executed at bar `t+1`'s OPEN (never bar `t` -- same
  no-lookahead rule as `SimBroker._process_bar_fills`), paying `per_side_bps` of traded notional:
  `fee_bps + half_spread_bps + slippage_bps` (the sqrt-participation impact term is dropped;
  for the tiny retail sizes considered here it is ~0 -- the event-driven confirmation run keeps
  it).
* P&L for the target decided at `t` accrues from open[t+1] to open[t+2] -- i.e. positions are
  marked at the price they could actually be traded at.

`compare_with_event_engine` (in tests) verifies this path agrees with the `Backtester` on a
sample to within the impact term.

The single most important number for the user is `edge_vs_cost`: gross edge per round trip in
bps vs cost per round trip in bps, and the implied *break-even fee* -- the per-side fee at which
the strategy's net expectancy hits zero given the observed gross edge and the spread/slippage
assumptions.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from jevtrader.backtest.metrics import compute_metrics, deflated_sharpe_ratio
from jevtrader.backtest.sim_broker import SlippageModel
from jevtrader.core.fees import FeeModel, default_fees
from jevtrader.core.types import AssetClass, Instrument, Liquidity, Side

# ----------------------------------------------------------------------------- costs


@dataclass(frozen=True)
class CostSpec:
    """Per-side execution cost in bps of notional, decomposed the way the SimBroker charges it."""

    fee_bps: float
    half_spread_bps: float = 1.0
    slippage_bps: float = 0.5
    label: str = ""

    @property
    def per_side_bps(self) -> float:
        return self.fee_bps + self.half_spread_bps + self.slippage_bps

    @property
    def round_trip_bps(self) -> float:
        return 2.0 * self.per_side_bps

    @property
    def non_fee_round_trip_bps(self) -> float:
        return 2.0 * (self.half_spread_bps + self.slippage_bps)

    def with_fee(self, fee_bps: float, label: str = "") -> "CostSpec":
        return CostSpec(fee_bps, self.half_spread_bps, self.slippage_bps, label or f"{fee_bps:g}bps")

    def scaled(self, fee_factor: float = 1.0, slip_factor: float = 1.0) -> "CostSpec":
        return CostSpec(self.fee_bps * fee_factor, self.half_spread_bps * slip_factor, self.slippage_bps * slip_factor, self.label)

    @staticmethod
    def from_fee_model(
        fee_model: FeeModel,
        instrument: Instrument,
        price: float,
        maker: bool = False,
        slippage: Optional[SlippageModel] = None,
        label: str = "",
    ) -> "CostSpec":
        """Pull the per-side fee (in bps) out of any `FeeModel` for a $10k reference ticket and
        pair it with a `SlippageModel`'s spread/slippage terms."""
        slippage = slippage or SlippageModel()
        qty = 10_000.0 / price
        liq = Liquidity.MAKER if maker else Liquidity.TAKER
        fee = fee_model.fee(instrument, Side.BUY, qty, price, liq) + fee_model.fee(instrument, Side.SELL, qty, price, liq)
        per_side = 1e4 * fee / (2 * qty * price)
        return CostSpec(per_side, slippage.half_spread_bps, slippage.slippage_bps, label)


# Alpaca defaults (from jevtrader/core/fees.py) plus a conservative spread model per venue.
ALPACA_CRYPTO_TAKER = CostSpec(25.0, half_spread_bps=2.5, slippage_bps=1.0, label="alpaca_crypto_taker_t1")
ALPACA_CRYPTO_MAKER = CostSpec(15.0, half_spread_bps=0.0, slippage_bps=0.0, label="alpaca_crypto_maker_t1")
ALPACA_EQUITY_TAKER = CostSpec(0.15, half_spread_bps=1.0, slippage_bps=0.5, label="alpaca_equity_taker")  # 0.3 bps reg fee round trip
ALPACA_EQUITY_LIQUID_ETF = CostSpec(0.15, half_spread_bps=0.25, slippage_bps=0.25, label="alpaca_spy_like")

# Alternative crypto fee levels for break-even analysis (per side, taker).
CRYPTO_FEE_LADDER_BPS: tuple[float, ...] = (25.0, 15.0, 10.0, 5.0, 2.0, 0.0)


def alpaca_crypto_costs(fee_model: Optional[FeeModel] = None, price: float = 100_000.0) -> CostSpec:
    fm = fee_model or default_fees()
    inst = Instrument("BTC/USD", AssetClass.CRYPTO, lot_size=1e-6, shortable=False)
    base = CostSpec.from_fee_model(fm, inst, price, maker=False, slippage=SlippageModel(half_spread_bps=2.5, slippage_bps=1.0))
    return CostSpec(base.fee_bps, base.half_spread_bps, base.slippage_bps, "alpaca_crypto_taker_t1")


# ----------------------------------------------------------------------------- simulation


@dataclass
class FastResult:
    equity: pd.Series  # marked at fill prices (index = open time of bar t+1 ~ close of bar t)
    position: pd.Series  # executed position path (fraction of equity), aligned to `equity`
    gross_returns: pd.Series
    net_returns: pd.Series
    turnover: pd.Series  # |delta position| per bar
    cost: CostSpec
    n_round_trips: float
    metrics: dict = field(default_factory=dict)

    # --- the numbers that matter ---------------------------------------------------------
    @property
    def gross_edge_bps_per_rt(self) -> float:
        """Gross P&L per unit of round-trip turnover, in bps. Compare with `cost.round_trip_bps`."""
        t = float(self.turnover.sum())
        return float(1e4 * self.gross_returns.sum() / (t / 2.0)) if t > 0 else float("nan")

    @property
    def net_edge_bps_per_rt(self) -> float:
        return self.gross_edge_bps_per_rt - self.cost.round_trip_bps

    @property
    def breakeven_fee_bps(self) -> float:
        """Per-side fee at which net expectancy is zero, holding spread/slippage fixed."""
        g = self.gross_edge_bps_per_rt
        if not np.isfinite(g):
            return float("nan")
        return (g - self.cost.non_fee_round_trip_bps) / 2.0

    def summary(self) -> dict:
        m = self.metrics
        return {
            "sharpe": m.get("sharpe", float("nan")),
            "total_return": m.get("total_return", float("nan")),
            "max_drawdown": m.get("max_drawdown", float("nan")),
            "psr": m.get("psr", float("nan")),
            "dsr": m.get("dsr", float("nan")),
            "n_round_trips": self.n_round_trips,
            "trades_per_day": self.n_round_trips / max((self.equity.index[-1] - self.equity.index[0]).total_seconds() / 86400.0, 1e-9) if len(self.equity) > 1 else 0.0,
            "gross_edge_bps_per_rt": self.gross_edge_bps_per_rt,
            "cost_bps_per_rt": self.cost.round_trip_bps,
            "net_edge_bps_per_rt": self.net_edge_bps_per_rt,
            "breakeven_fee_bps_per_side": self.breakeven_fee_bps,
            "exposure": float((self.position.abs() > 1e-12).mean()) if len(self.position) else 0.0,
        }


def simulate_positions(
    bars: pd.DataFrame,
    target: pd.Series | np.ndarray,
    cost: CostSpec,
    asset_class: AssetClass = AssetClass.CRYPTO,
    n_trials: int = 1,
    long_only: bool = False,
    max_abs_position: float = 1.0,
    compute_full_metrics: bool = True,
) -> FastResult:
    """Simulate a target-position path (fraction of equity, decided at each bar's close) with
    next-open fills and SimBroker-identical per-side costs. See module docstring.

    `long_only=True` clips negative targets to 0 (Alpaca crypto). `max_abs_position` caps
    leverage (1.0 = no leverage; the SimBroker rejects orders beyond cash unless leverage is on).
    """
    tgt = pd.Series(np.asarray(target, dtype=float), index=bars.index).fillna(0.0)
    if long_only:
        tgt = tgt.clip(lower=0.0)
    tgt = tgt.clip(-max_abs_position, max_abs_position)

    opn = bars["open"].to_numpy(dtype=float)
    n = len(opn)
    if n < 3:
        raise ValueError("need at least 3 bars")
    # position decided at t is held from open[t+1] to open[t+2]; execution at open[t+1].
    pos_exec = tgt.to_numpy()[:-2]  # length n-2, executed at open[1..n-1)
    exec_open = opn[1:-1]
    next_open = opn[2:]
    bar_ret = next_open / exec_open - 1.0
    gross = pos_exec * bar_ret
    prev_pos = np.concatenate([[0.0], pos_exec[:-1]])
    turnover = np.abs(pos_exec - prev_pos)
    cost_ret = turnover * cost.per_side_bps / 1e4
    net = gross - cost_ret
    idx = bars.index[1:-1]
    equity = pd.Series(np.cumprod(1.0 + net), index=idx, name="equity")
    n_rt = float(turnover.sum() / 2.0)
    res = FastResult(
        equity=equity,
        position=pd.Series(pos_exec, index=idx),
        gross_returns=pd.Series(gross, index=idx),
        net_returns=pd.Series(net, index=idx),
        turnover=pd.Series(turnover, index=idx),
        cost=cost,
        n_round_trips=n_rt,
    )
    if compute_full_metrics:
        res.metrics = compute_metrics(equity, None, asset_class=asset_class, initial_cash=1.0, n_trials=n_trials)
    else:
        res.metrics = _quick_metrics(net, idx, asset_class, n_trials)
    return res


def _quick_metrics(net: np.ndarray, idx: pd.DatetimeIndex, asset_class: AssetClass, n_trials: int) -> dict:
    from jevtrader.backtest.metrics import max_drawdown, periods_per_year

    ppy = periods_per_year(idx, asset_class)
    std = net.std(ddof=1) if len(net) > 1 else 0.0
    sharpe = float(net.mean() / std * np.sqrt(ppy)) if std > 0 else 0.0
    eq = pd.Series(np.cumprod(1.0 + net), index=idx)
    return {
        "sharpe": sharpe,
        "total_return": float(eq.iloc[-1] - 1.0),
        "max_drawdown": max_drawdown(eq),
        "psr": float("nan"),
        "dsr": float("nan"),
        "periods_per_year": ppy,
        "n_trials": n_trials,
    }


# ----------------------------------------------------------------------------- sweeps / stability


def sweep_grid(
    grid: dict[str, Sequence],
    run_fn: Callable[[dict], FastResult],
    sort_by: str = "sharpe",
) -> pd.DataFrame:
    """Evaluate every combination in `grid` with `run_fn(params) -> FastResult` and return a
    tidy DataFrame (one row per trial, params + summary). `n_trials` = len(rows): report it."""
    keys = list(grid)
    rows = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        res = run_fn(params)
        rows.append({**params, **res.summary()})
    df = pd.DataFrame(rows)
    df.attrs["n_trials"] = len(df)
    return df.sort_values(sort_by, ascending=False).reset_index(drop=True)


def stability_table(sweep_df: pd.DataFrame, row_param: str, col_param: str, value: str = "sharpe") -> pd.DataFrame:
    """Pivot a sweep into a parameter-neighbourhood table (a text 'heatmap')."""
    return sweep_df.pivot_table(index=row_param, columns=col_param, values=value, aggfunc="mean").round(2)


def fee_ladder(
    bars: pd.DataFrame,
    target: pd.Series | np.ndarray,
    base: CostSpec,
    fees_bps: Sequence[float] = CRYPTO_FEE_LADDER_BPS,
    asset_class: AssetClass = AssetClass.CRYPTO,
    long_only: bool = False,
) -> pd.DataFrame:
    """Re-cost the same position path at several per-side fee levels: shows where a strategy
    becomes viable (e.g. Alpaca tier 1 = 25 taker / 15 maker; higher tiers 10/5/2; other venues 0-5)."""
    rows = []
    for f in fees_bps:
        c = base.with_fee(f)
        r = simulate_positions(bars, target, c, asset_class=asset_class, long_only=long_only, compute_full_metrics=False)
        rows.append({"fee_bps_per_side": f, "round_trip_cost_bps": c.round_trip_bps, "sharpe": r.metrics["sharpe"], "total_return": r.metrics["total_return"], "max_drawdown": r.metrics["max_drawdown"]})
    return pd.DataFrame(rows)


def robustness(
    bars: pd.DataFrame,
    target: pd.Series | np.ndarray,
    base: CostSpec,
    asset_class: AssetClass = AssetClass.CRYPTO,
    long_only: bool = False,
) -> dict:
    """Mirror of `jevtrader.backtest.walkforward.robustness_report`'s scenarios for the fast path."""
    scen = {
        "base": base,
        "fees_x1.5": base.scaled(fee_factor=1.5),
        "fees_x2": base.scaled(fee_factor=2.0),
        "slippage_x2": base.scaled(slip_factor=2.0),
    }
    out = {}
    for k, c in scen.items():
        r = simulate_positions(bars, target, c, asset_class=asset_class, long_only=long_only, compute_full_metrics=False)
        out[k] = {"sharpe": r.metrics["sharpe"], "total_return": r.metrics["total_return"], "max_drawdown": r.metrics["max_drawdown"]}
    stress = [out[k]["sharpe"] for k in ("fees_x1.5", "fees_x2", "slippage_x2")]
    out["passed"] = bool(out["base"]["sharpe"] > 0 and all(s > 0 for s in stress))
    return out


def dsr_for(result: FastResult, n_trials: int) -> float:
    """Deflated Sharpe of a result's net return stream given the number of parameterizations tried."""
    return float(deflated_sharpe_ratio(result.net_returns, n_trials=n_trials)["dsr"])


def verdict(oos_sharpe: float, dsr: float, net_edge_bps: float, robust_pass: bool) -> str:
    """The promotion-gate-aligned label used across research_reports/."""
    if np.isfinite(oos_sharpe) and oos_sharpe >= 1.0 and dsr >= 0.95 and net_edge_bps > 0 and robust_pass:
        return "VIABLE"
    if np.isfinite(oos_sharpe) and oos_sharpe > 0.3 and net_edge_bps > 0:
        return "MARGINAL"
    return "NOT VIABLE"


def cost_spec_dict(c: CostSpec) -> dict:
    return {**asdict(c), "per_side_bps": c.per_side_bps, "round_trip_bps": c.round_trip_bps}
