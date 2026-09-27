"""Backtest performance metrics.

Every ratio here is computed on the equity curve's own per-step returns and then annualized with
`periods_per_year`, which is aware of the equity curve's actual sampling frequency:

* If the median spacing between equity-curve timestamps is a day or coarser, each sample is
  treated as one trading period and annualization uses trading days/year (252 equity, 365
  crypto) -- because you cannot map calendar seconds to trading seconds once you are below
  intraday resolution (a "1 day" gap and a "3 day" weekend gap are both just "the next trading
  day").
* Otherwise (intraday) it divides the asset class's total trading seconds in a year (a 6.5h
  session x 252 days for equities, 24h x 365 for always-on crypto) by the median inter-sample
  spacing in seconds.

`compute_trade_pnls` reconstructs realized, fee-inclusive trade-level P&L by replaying fills
through `Position.apply_fill` (the same accounting core the backtester itself uses), so hit rate,
profit factor and average trade are computed on money the system itself would report as
realized, not on some separate ad hoc definition of "a trade".

`probabilistic_sharpe_ratio` and `deflated_sharpe_ratio` implement Bailey & Lopez de Prado's PSR
and DSR (see "The Sharpe Ratio Efficient Frontier", 2012, and "The Deflated Sharpe Ratio", 2014):
the DSR corrects the observed Sharpe ratio for selection bias from having tried `n_trials`
parameterizations (as in a walk-forward / parameter sweep) by benchmarking it against the
*expected maximum* Sharpe ratio one would see from that many trials under a null of zero true
skill, then reporting a probabilistic Sharpe ratio (a p-value-like probability in [0, 1], not a
ratio) against that inflated benchmark.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import norm
from scipy.stats import skew as _skew

from jevtrader.core.types import AssetClass, Fill, Liquidity, Position, Side

_EQUITY_TRADING_DAYS = 252.0
_CRYPTO_DAYS = 365.0
_EQUITY_SESSION_SECONDS = 6.5 * 3600.0
_DAY_SECONDS = 86400.0
_EULER_GAMMA = 0.5772156649015329


def periods_per_year(index: pd.DatetimeIndex, asset_class: AssetClass = AssetClass.EQUITY) -> float:
    """Infer an annualization factor from the equity curve's own sampling frequency. See module
    docstring for the daily-vs-intraday heuristic."""
    base_days = _EQUITY_TRADING_DAYS if asset_class is AssetClass.EQUITY else _CRYPTO_DAYS
    if len(index) < 2:
        return base_days
    # np.diff on the raw datetime64 values (whatever resolution pandas stores, ns or us) divided
    # by a 1-second timedelta64 gives seconds regardless of that resolution.
    diffs = np.diff(index.values) / np.timedelta64(1, "s")
    dt = float(np.median(diffs))
    if dt <= 0:
        return base_days
    if dt >= 0.9 * _DAY_SECONDS:
        return base_days * (_DAY_SECONDS / dt)
    if asset_class is AssetClass.CRYPTO:
        return (_CRYPTO_DAYS * _DAY_SECONDS) / dt
    return (_EQUITY_TRADING_DAYS * _EQUITY_SESSION_SECONDS) / dt


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min()) if len(dd) else 0.0


def max_drawdown_duration(equity: pd.Series) -> int:
    """Longest run of consecutive samples strictly below the running peak, in # of samples."""
    running_max = equity.cummax()
    in_dd = (equity < running_max).to_numpy()
    longest = current = 0
    for flag in in_dd:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return int(longest)


def compute_trade_pnls(fills_df: pd.DataFrame) -> pd.Series:
    """Replay fills (as produced by `Backtester`'s fills DataFrame) through `Position.apply_fill`
    per symbol, collecting the non-zero realized (gross-of-nothing-else, fee-inclusive via
    `Position`) P&L of each fill that closes part of an existing position. That is this module's
    definition of one "trade" for hit-rate/profit-factor purposes."""
    if fills_df is None or fills_df.empty:
        return pd.Series(dtype=float)
    positions: dict[str, Position] = {}
    pnls: list[float] = []
    for _, row in fills_df.sort_values("ts").iterrows():
        symbol = row["symbol"]
        pos = positions.setdefault(symbol, Position(symbol))
        side = Side.BUY if row["side"] == "buy" else Side.SELL
        fill = Fill(
            client_order_id=str(row.get("client_order_id", "")),
            symbol=symbol,
            side=side,
            qty=float(row["qty"]),
            price=float(row["price"]),
            fee=float(row.get("fee", 0.0)),
            liquidity=Liquidity.TAKER,  # not used by Position.apply_fill; fee is already the row's actual fee
            ts=row["ts"],
        )
        realized = pos.apply_fill(fill)  # gross realized P&L of this fill's closing portion (fees tracked separately via fee_drag)
        if abs(realized) > 1e-12:
            pnls.append(realized)
    return pd.Series(pnls, dtype=float)


def compute_exposure(fills_df: pd.DataFrame, index: pd.DatetimeIndex) -> float:
    """Fraction of the backtest's wall-clock span during which at least one symbol had a
    non-zero net position, reconstructed from fill timestamps (a coarse but dependency-free
    proxy for "time in market")."""
    if fills_df is None or fills_df.empty or len(index) < 2:
        return 0.0
    span = (index[-1] - index[0]).total_seconds()
    if span <= 0:
        return 0.0
    qty: dict[str, float] = {}
    prev_ts = index[0]
    prev_flag = False
    exposed = 0.0
    for _, row in fills_df.sort_values("ts").iterrows():
        ts = row["ts"]
        exposed += max(0.0, (ts - prev_ts).total_seconds()) * (1.0 if prev_flag else 0.0)
        signed = row["qty"] if row["side"] == "buy" else -row["qty"]
        qty[row["symbol"]] = qty.get(row["symbol"], 0.0) + signed
        prev_ts = ts
        prev_flag = any(abs(q) > 1e-9 for q in qty.values())
    exposed += max(0.0, (index[-1] - prev_ts).total_seconds()) * (1.0 if prev_flag else 0.0)
    return float(exposed / span)


def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sr: float = 0.0) -> float:
    """PSR(SR*): probability that the true Sharpe ratio exceeds `benchmark_sr`, given the
    observed per-period returns (Bailey & Lopez de Prado, 2012). `returns` and `benchmark_sr`
    must be in the SAME (per-period, non-annualized) units."""
    r = pd.Series(returns).dropna().to_numpy(dtype=float)
    n = len(r)
    if n < 3:
        return float("nan")
    std = r.std(ddof=1)
    if std <= 0:
        return float("nan")
    sr = r.mean() / std
    g3 = float(_skew(r, bias=False))
    g4 = float(_kurtosis(r, fisher=False, bias=False))  # Pearson kurtosis; normal == 3
    denom = np.sqrt(max(1e-12, 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2))
    z = (sr - benchmark_sr) * np.sqrt(n - 1) / denom
    return float(norm.cdf(z))


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int = 1) -> dict:
    """DSR: PSR benchmarked against the expected MAXIMUM Sharpe ratio one would observe by chance
    across `n_trials` independent trials under a null of zero skill (Bailey & Lopez de Prado,
    2014). `n_trials=1` reduces exactly to the ordinary PSR against a zero benchmark.

    Returns a dict with `dsr` (probability in [0, 1]), `expected_max_sr` (the inflated benchmark,
    per-period units), `sr` (the observed per-period Sharpe ratio) and echoes `n_trials`/`n_obs`.
    """
    r = pd.Series(returns).dropna().to_numpy(dtype=float)
    n = len(r)
    if n < 3:
        return {"dsr": float("nan"), "expected_max_sr": float("nan"), "sr": float("nan"), "n_trials": n_trials, "n_obs": n}
    std = r.std(ddof=1)
    sr = float(r.mean() / std) if std > 0 else 0.0
    g3 = float(_skew(r, bias=False))
    g4 = float(_kurtosis(r, fisher=False, bias=False))
    var_sr = max(1e-12, (1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2) / max(n - 1, 1))
    std_sr = float(np.sqrt(var_sr))
    if n_trials <= 1:
        sr0 = 0.0
    else:
        z1 = norm.ppf(1.0 - 1.0 / n_trials)
        z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        sr0 = std_sr * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2)
    dsr = probabilistic_sharpe_ratio(pd.Series(r), benchmark_sr=sr0)
    return {"dsr": dsr, "expected_max_sr": float(sr0), "sr": sr, "n_trials": int(n_trials), "n_obs": n}


def compute_metrics(
    equity_curve: pd.Series,
    fills_df: Optional[pd.DataFrame] = None,
    asset_class: AssetClass = AssetClass.EQUITY,
    initial_cash: Optional[float] = None,
    n_trials: int = 1,
    risk_free_rate_annual: float = 0.0,
) -> dict:
    """Full metrics summary for one backtest run. `fills_df` is the `Backtester` result's
    `fills` DataFrame (columns: ts, symbol, side, qty, price, fee, liquidity, notional, ...);
    pass `None` or an empty DataFrame to get equity-curve-only metrics (fee/trade stats become
    0/NaN)."""
    fills_df = fills_df if fills_df is not None else pd.DataFrame(columns=["ts", "symbol", "side", "qty", "price", "fee", "notional"])
    equity = pd.Series(equity_curve, dtype=float).dropna()
    if len(equity) < 2:
        return _empty_metrics(n_trials)

    ppy = periods_per_year(equity.index, asset_class)
    returns = equity.pct_change().dropna()
    n = len(returns)
    rf_per_period = risk_free_rate_annual / ppy if ppy > 0 else 0.0
    excess = returns - rf_per_period

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    annualized_return = float((1.0 + total_return) ** (ppy / n) - 1.0) if n > 0 else 0.0
    annualized_vol = float(returns.std(ddof=1) * np.sqrt(ppy)) if n > 1 else 0.0

    mean_ex, std_ex = excess.mean(), excess.std(ddof=1)
    sharpe = float(mean_ex / std_ex * np.sqrt(ppy)) if std_ex > 0 else 0.0
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
    sortino = float(mean_ex / downside_std * np.sqrt(ppy)) if downside_std > 0 else 0.0

    mdd = max_drawdown(equity)
    mdd_duration = max_drawdown_duration(equity)
    if mdd < 0:
        calmar = float(annualized_return / abs(mdd))
    else:
        calmar = float("inf") if annualized_return > 0 else 0.0

    trade_pnls = compute_trade_pnls(fills_df)
    if len(trade_pnls) > 0:
        hit_rate = float((trade_pnls > 0).mean())
        gains = float(trade_pnls[trade_pnls > 0].sum())
        losses = float(trade_pnls[trade_pnls < 0].sum())
        profit_factor = float(gains / abs(losses)) if losses < 0 else float("inf") if gains > 0 else float("nan")
        avg_trade = float(trade_pnls.mean())
    else:
        hit_rate = float("nan")
        profit_factor = float("nan")
        avg_trade = 0.0

    fee_total = float(fills_df["fee"].sum()) if not fills_df.empty else 0.0
    net_pnl = float(equity.iloc[-1] - equity.iloc[0])
    gross_pnl = net_pnl + fee_total
    fee_drag = float(fee_total / gross_pnl) if abs(gross_pnl) > 1e-9 else float("nan")

    if fills_df.empty:
        total_notional = 0.0
    elif "notional" in fills_df:
        total_notional = float(fills_df["notional"].abs().sum())
    else:
        total_notional = float((fills_df["qty"].abs() * fills_df["price"].abs()).sum())
    cost_per_trade_bps = float(fee_total / total_notional * 1e4) if total_notional > 0 else 0.0
    mean_equity = float(equity.mean())
    turnover = float(total_notional / mean_equity) if mean_equity > 0 else 0.0

    span_days = max((equity.index[-1] - equity.index[0]).total_seconds() / _DAY_SECONDS, 1e-9)
    trades_per_day = float(len(fills_df) / span_days) if not fills_df.empty else 0.0
    exposure = compute_exposure(fills_df, equity.index)

    psr = probabilistic_sharpe_ratio(excess, benchmark_sr=0.0)
    dsr_info = deflated_sharpe_ratio(excess, n_trials=n_trials)

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": mdd,
        "max_drawdown_duration_periods": mdd_duration,
        "hit_rate": hit_rate,
        "profit_factor": profit_factor,
        "avg_trade": avg_trade,
        "turnover": turnover,
        "fee_total": fee_total,
        "fee_drag": fee_drag,
        "cost_per_trade_bps": cost_per_trade_bps,
        "exposure": exposure,
        "trades_per_day": trades_per_day,
        "n_fills": int(len(fills_df)),
        "n_closed_trades": int(len(trade_pnls)),
        "periods_per_year": float(ppy),
        "psr": psr,
        "dsr": dsr_info["dsr"],
        "dsr_expected_max_sr_per_period": dsr_info["expected_max_sr"],
        "sharpe_per_period": dsr_info["sr"],
        "n_trials": int(n_trials),
        "initial_cash": float(initial_cash) if initial_cash is not None else float(equity.iloc[0]),
        "final_equity": float(equity.iloc[-1]),
    }


def _empty_metrics(n_trials: int) -> dict:
    keys = [
        "total_return", "annualized_return", "annualized_vol", "sharpe", "sortino", "calmar", "max_drawdown",
        "hit_rate", "profit_factor", "avg_trade", "turnover", "fee_total", "fee_drag", "cost_per_trade_bps",
        "exposure", "trades_per_day", "psr", "dsr", "dsr_expected_max_sr_per_period", "sharpe_per_period",
        "initial_cash", "final_equity",
    ]
    out = {k: float("nan") for k in keys}
    out.update({"max_drawdown_duration_periods": 0, "n_fills": 0, "n_closed_trades": 0, "periods_per_year": 0.0, "n_trials": int(n_trials)})
    return out
