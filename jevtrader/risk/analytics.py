"""Pre-trade and portfolio risk analytics: VaR/CVaR, correlation, vol, marginal risk
contribution, stress scenarios. Pure functions over numpy/pandas - no broker/account coupling,
so they work identically in backtest, paper and live, and are trivial to unit test offline.

Conventions:
    - `returns_df`: pandas DataFrame of periodic (e.g. daily) simple returns, columns = symbols.
    - VaR/CVaR are returned as POSITIVE numbers representing a loss magnitude (as a fraction of
      the base, e.g. 0.03 = 3% loss), at the given confidence `level` (e.g. 0.95, 0.99).
    - `positions`: mapping symbol -> signed USD notional (qty * price).
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd
from scipy import stats

from jevtrader.core.types import AssetClass


# ----------------------------------------------------------------------------- single-series VaR/CVaR


def historical_var(returns: pd.Series, level: float = 0.95) -> float:
    """Empirical VaR: the `level`-quantile of the loss distribution. Positive = loss magnitude."""
    r = pd.Series(returns).dropna()
    if r.empty:
        return 0.0
    losses = -r.to_numpy()
    return float(max(np.percentile(losses, level * 100), 0.0))


def historical_cvar(returns: pd.Series, level: float = 0.95) -> float:
    """Empirical CVaR (expected shortfall): mean loss beyond the VaR threshold."""
    r = pd.Series(returns).dropna()
    if r.empty:
        return 0.0
    losses = -r.to_numpy()
    var = np.percentile(losses, level * 100)
    tail = losses[losses >= var]
    if tail.size == 0:
        return float(max(var, 0.0))
    return float(max(tail.mean(), 0.0))


def parametric_var(returns: pd.Series, level: float = 0.95) -> float:
    """Gaussian (variance-covariance) VaR from the sample mean/std of `returns`."""
    r = pd.Series(returns).dropna()
    if r.empty:
        return 0.0
    mu, sigma = float(r.mean()), float(r.std(ddof=1)) if len(r) > 1 else 0.0
    z = stats.norm.ppf(level)
    return float(max(z * sigma - mu, 0.0))


def parametric_cvar(returns: pd.Series, level: float = 0.95) -> float:
    """Gaussian CVaR: mu, sigma from sample; closed-form expected shortfall of the normal."""
    r = pd.Series(returns).dropna()
    if r.empty:
        return 0.0
    mu, sigma = float(r.mean()), float(r.std(ddof=1)) if len(r) > 1 else 0.0
    z = stats.norm.ppf(level)
    es = sigma * stats.norm.pdf(z) / (1 - level)
    return float(max(es - mu, 0.0))


# ----------------------------------------------------------------------------- portfolio level


def _weights_from_positions(positions: Mapping[str, float], equity: float) -> pd.Series:
    if equity <= 0:
        return pd.Series({k: 0.0 for k in positions})
    return pd.Series({k: v / equity for k, v in positions.items()})


def portfolio_returns(positions: Mapping[str, float], returns_df: pd.DataFrame, equity: float) -> pd.Series:
    """Historical portfolio return series given fixed (current) dollar positions and equity."""
    w = _weights_from_positions(positions, equity)
    cols = [c for c in returns_df.columns if c in w.index]
    if not cols:
        return pd.Series(dtype=float)
    aligned = returns_df[cols].fillna(0.0)
    return aligned.dot(w.reindex(cols).fillna(0.0))


def portfolio_var_cvar(
    positions: Mapping[str, float],
    returns_df: pd.DataFrame,
    equity: float,
    level: float = 0.95,
    method: str = "historical",
) -> dict:
    """VaR/CVaR of the whole book, in both fraction-of-equity and USD terms."""
    pr = portfolio_returns(positions, returns_df, equity)
    if method == "parametric":
        var_f, cvar_f = parametric_var(pr, level), parametric_cvar(pr, level)
    else:
        var_f, cvar_f = historical_var(pr, level), historical_cvar(pr, level)
    return {
        "level": level,
        "method": method,
        "var_pct": var_f,
        "cvar_pct": cvar_f,
        "var_usd": var_f * equity,
        "cvar_usd": cvar_f * equity,
    }


def rolling_correlation(returns_df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Correlation matrix over the trailing `window` periods (or all data if shorter)."""
    df = returns_df.tail(window) if len(returns_df) > window else returns_df
    return df.corr()


def portfolio_vol(weights: pd.Series, cov: pd.DataFrame, periods_per_year: int = 252) -> float:
    """Annualized portfolio volatility from weights and a (periodic) covariance matrix."""
    w = weights.reindex(cov.index).fillna(0.0).to_numpy()
    var = float(w @ cov.to_numpy() @ w)
    return float(np.sqrt(max(var, 0.0)) * np.sqrt(periods_per_year))


def marginal_risk_contribution(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Component contribution to (periodic, non-annualized) portfolio vol.

    MRC_i = (Σw)_i / σ_p; component contribution CC_i = w_i * MRC_i; sum(CC) == σ_p.
    """
    w = weights.reindex(cov.index).fillna(0.0)
    sigma_w = cov.to_numpy() @ w.to_numpy()
    port_var = float(w.to_numpy() @ sigma_w)
    sigma_p = float(np.sqrt(max(port_var, 0.0)))
    if sigma_p <= 0:
        return pd.Series(0.0, index=cov.index)
    mrc = sigma_w / sigma_p
    cc = w.to_numpy() * mrc
    return pd.Series(cc, index=cov.index)


# ----------------------------------------------------------------------------- stress scenarios

DEFAULT_SHOCKS: dict[AssetClass, float] = {
    AssetClass.EQUITY: -0.10,
    AssetClass.CRYPTO: -0.25,
}


def stress_scenarios(
    positions: Mapping[str, float],
    asset_class_of: Mapping[str, AssetClass],
    returns_df: Optional[pd.DataFrame] = None,
    shocks: Optional[Mapping[AssetClass, float]] = None,
) -> dict:
    """Apply simple directional shocks per asset class, plus a "correlation -> 1" shock.

    - "equities_down"/"crypto_down": pnl impact of an instantaneous shock to that asset class only.
    - "combined": both shocks applied simultaneously.
    - "correlation_one": worst-case portfolio vol if every pairwise correlation went to 1, i.e.
      sum(|position| * per-asset vol) instead of the diversified sqrt(w'Σw); requires `returns_df`
      to estimate per-asset vol (skipped, reported as None, if not supplied).
    """
    shocks = dict(DEFAULT_SHOCKS if shocks is None else shocks)
    out: dict = {"positions": dict(positions)}

    def pnl_for(shock_map: Mapping[AssetClass, float]) -> float:
        total = 0.0
        for sym, notional in positions.items():
            ac = asset_class_of.get(sym)
            shock = shock_map.get(ac, 0.0)
            total += notional * shock
        return total

    out["equities_down_10pct"] = pnl_for({AssetClass.EQUITY: shocks.get(AssetClass.EQUITY, -0.10)})
    out["crypto_down_25pct"] = pnl_for({AssetClass.CRYPTO: shocks.get(AssetClass.CRYPTO, -0.25)})
    out["combined"] = pnl_for(shocks)

    if returns_df is not None and not returns_df.empty:
        vols = returns_df.std(ddof=1)
        corr_one_vol_usd = sum(abs(notional) * float(vols.get(sym, 0.0)) for sym, notional in positions.items())
        equity = sum(abs(v) for v in positions.values()) or 1.0
        diversified = float(
            np.sqrt(
                max(
                    sum(
                        positions.get(a, 0.0)
                        * positions.get(b, 0.0)
                        * float(returns_df[[a, b]].cov().iloc[0, 1])
                        for a in positions
                        for b in positions
                        if a in returns_df.columns and b in returns_df.columns
                    ),
                    0.0,
                )
            )
        ) if all(s in returns_df.columns for s in positions) else None
        out["correlation_one_daily_vol_usd"] = corr_one_vol_usd
        out["diversified_daily_vol_usd"] = diversified
    else:
        out["correlation_one_daily_vol_usd"] = None
        out["diversified_daily_vol_usd"] = None

    return out


def risk_report_markdown(
    positions: Mapping[str, float],
    equity: float,
    returns_df: pd.DataFrame,
    asset_class_of: Mapping[str, AssetClass],
) -> str:
    """Human-readable risk report combining VaR/CVaR, correlation, vol contribution and stress."""
    lines = ["# Risk report", ""]
    lines.append(f"Equity: ${equity:,.2f}  |  Gross exposure: ${sum(abs(v) for v in positions.values()):,.2f}")
    lines.append("")

    for level in (0.95, 0.99):
        h = portfolio_var_cvar(positions, returns_df, equity, level=level, method="historical")
        p = portfolio_var_cvar(positions, returns_df, equity, level=level, method="parametric")
        lines.append(
            f"- VaR{int(level*100)} (hist): {h['var_pct']*100:.2f}% (${h['var_usd']:,.0f})  "
            f"CVaR{int(level*100)} (hist): {h['cvar_pct']*100:.2f}% (${h['cvar_usd']:,.0f})"
        )
        lines.append(
            f"- VaR{int(level*100)} (param): {p['var_pct']*100:.2f}% (${p['var_usd']:,.0f})  "
            f"CVaR{int(level*100)} (param): {p['cvar_pct']*100:.2f}% (${p['cvar_usd']:,.0f})"
        )
    lines.append("")

    cols = [c for c in returns_df.columns if c in positions]
    if cols:
        w = _weights_from_positions(positions, equity).reindex(cols).fillna(0.0)
        cov = returns_df[cols].cov()
        vol = portfolio_vol(w, cov)
        lines.append(f"Annualized portfolio vol: {vol*100:.1f}%")
        mrc = marginal_risk_contribution(w, cov)
        lines.append("")
        lines.append("## Risk contribution by position")
        for sym, val in mrc.sort_values(ascending=False).items():
            lines.append(f"- {sym}: {val*100:.2f}% (periodic)")
        lines.append("")
        lines.append("## Correlation (trailing)")
        corr = rolling_correlation(returns_df[cols])
        lines.append(corr.round(2).to_string())
        lines.append("")

    lines.append("## Stress scenarios")
    stress = stress_scenarios(positions, asset_class_of, returns_df)
    lines.append(f"- Equities -10%: ${stress['equities_down_10pct']:,.0f}")
    lines.append(f"- Crypto -25%: ${stress['crypto_down_25pct']:,.0f}")
    lines.append(f"- Combined: ${stress['combined']:,.0f}")
    if stress.get("correlation_one_daily_vol_usd") is not None:
        lines.append(
            f"- Correlation->1 daily vol: ${stress['correlation_one_daily_vol_usd']:,.0f} "
            f"(vs. diversified ${stress['diversified_daily_vol_usd']:,.0f})"
        )
    return "\n".join(lines)
