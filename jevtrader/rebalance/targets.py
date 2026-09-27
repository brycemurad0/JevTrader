"""Target-weight schemes from a returns DataFrame. All schemes are long-only, respect
`[min_weight, max_weight]` bounds and an optional cash buffer, and return a `pd.Series` indexed
by symbol that sums to `1 - cash_buffer_pct`.

Covariance estimation uses a hand-rolled Ledoit-Wolf shrinkage-to-identity estimator (the
"Honey, I Shrunk the Covariance Matrix" target = scaled identity), so we don't need scikit-learn.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform


# ----------------------------------------------------------------------------- covariance


def ledoit_wolf_cov(returns_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Ledoit-Wolf shrinkage covariance, shrinking the sample covariance towards a scaled
    identity target `mu * I` (mu = average sample variance). Returns (shrunk_cov, shrinkage).

    Reference: Ledoit & Wolf (2004), "Honey, I Shrunk the Sample Covariance Matrix". This is the
    same target used by `sklearn.covariance.ledoit_wolf`, reimplemented here to avoid the
    scikit-learn dependency.
    """
    X = returns_df.to_numpy(dtype=float)
    n, p = X.shape
    if n < 2 or p < 1:
        cov = returns_df.cov()
        return cov, 0.0

    X = X - X.mean(axis=0, keepdims=True)
    emp_cov = (X.T @ X) / n
    mu = np.trace(emp_cov) / p

    delta_mat = emp_cov.copy()
    delta_mat[np.diag_indices(p)] -= mu
    delta = float((delta_mat**2).sum()) / p

    X2 = X**2
    phi_mat = (X2.T @ X2) / n - emp_cov**2
    beta = float(phi_mat.sum()) / (p * n)
    beta = min(beta, delta)

    shrinkage = 0.0 if delta == 0 else beta / delta
    shrinkage = float(np.clip(shrinkage, 0.0, 1.0))

    target = mu * np.eye(p)
    shrunk = (1 - shrinkage) * emp_cov + shrinkage * target
    return pd.DataFrame(shrunk, index=returns_df.columns, columns=returns_df.columns), shrinkage


def cov_matrix(returns_df: pd.DataFrame, shrinkage: bool = True) -> pd.DataFrame:
    """Sample covariance, or Ledoit-Wolf shrunk covariance if `shrinkage` (default)."""
    if shrinkage:
        cov, _ = ledoit_wolf_cov(returns_df)
        return cov
    return returns_df.cov()


# ----------------------------------------------------------------------------- bounded normalization


def _project_to_bounds(w: np.ndarray, lo: float, hi: float, total: float, iters: int = 200) -> np.ndarray:
    """Water-filling projection of `w` onto {w : lo <= w_i <= hi, sum(w) == total}.

    Feasibility requires `lo * n <= total <= hi * n`; if infeasible we clip and renormalize
    proportionally as a best effort (documented behaviour, not silently wrong).
    """
    n = len(w)
    if n == 0:
        return w
    if not (lo * n - 1e-9 <= total <= hi * n + 1e-9):
        w = np.clip(w, lo, hi)
        s = w.sum()
        return w * (total / s) if s > 0 else np.full(n, total / n)

    w = np.clip(w, lo, hi).astype(float)
    for _ in range(iters):
        s = w.sum()
        if abs(s - total) < 1e-10:
            break
        free = (w > lo + 1e-12) & (w < hi - 1e-12)
        if not free.any():
            free = np.ones(n, dtype=bool)
        w[free] += (total - s) / free.sum()
        w = np.clip(w, lo, hi)
    return w


def _bounded_series(raw: pd.Series, min_weight: float, max_weight: float, cash_buffer_pct: float) -> pd.Series:
    budget = 1.0 - cash_buffer_pct
    w = _project_to_bounds(raw.to_numpy(dtype=float), min_weight, max_weight, budget)
    return pd.Series(w, index=raw.index)


# ----------------------------------------------------------------------------- schemes


def static_weights(
    weights: Mapping[str, float],
    symbols: Sequence[str],
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    cash_buffer_pct: float = 0.0,
) -> pd.Series:
    """User-specified target weights, reindexed to `symbols` (missing -> 0), bounded and
    renormalized to `1 - cash_buffer_pct`."""
    raw = pd.Series({s: float(weights.get(s, 0.0)) for s in symbols})
    if raw.sum() <= 0:
        raw = pd.Series(1.0, index=symbols)
    return _bounded_series(raw, min_weight, max_weight, cash_buffer_pct)


def equal_weight(symbols: Sequence[str], cash_buffer_pct: float = 0.0) -> pd.Series:
    n = len(symbols)
    budget = 1.0 - cash_buffer_pct
    return pd.Series(budget / n if n else 0.0, index=list(symbols))


def inverse_vol_weights(
    returns_df: pd.DataFrame,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    cash_buffer_pct: float = 0.0,
) -> pd.Series:
    vol = returns_df.std(ddof=1).replace(0.0, np.nan)
    inv = 1.0 / vol
    inv = inv.fillna(0.0)
    if inv.sum() <= 0:
        return equal_weight(list(returns_df.columns), cash_buffer_pct)
    raw = inv / inv.sum()
    return _bounded_series(raw, min_weight, max_weight, cash_buffer_pct)


def min_variance_weights(
    returns_df: pd.DataFrame,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    cash_buffer_pct: float = 0.0,
    shrinkage: bool = True,
) -> pd.Series:
    cov = cov_matrix(returns_df, shrinkage=shrinkage)
    symbols = list(cov.columns)
    n = len(symbols)
    budget = 1.0 - cash_buffer_pct
    cov_np = cov.to_numpy()

    def objective(w: np.ndarray) -> float:
        return float(w @ cov_np @ w)

    w0 = np.full(n, budget / n)
    bounds = [(min_weight, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - budget}]
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 1000, "ftol": 1e-14})
    w = res.x if res.success else w0
    w = _project_to_bounds(w, min_weight, max_weight, budget)
    return pd.Series(w, index=symbols)


def risk_parity_weights(
    returns_df: pd.DataFrame,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    cash_buffer_pct: float = 0.0,
    shrinkage: bool = True,
) -> pd.Series:
    """Equal risk contribution (ERC) via numerical optimization: minimize the dispersion of
    each asset's contribution to total portfolio variance."""
    cov = cov_matrix(returns_df, shrinkage=shrinkage)
    symbols = list(cov.columns)
    n = len(symbols)
    budget = 1.0 - cash_buffer_pct
    cov_np = cov.to_numpy()

    def objective(w: np.ndarray) -> float:
        port_var = w @ cov_np @ w
        mrc = cov_np @ w
        rc = w * mrc
        target = port_var / n
        return float(np.sum((rc - target) ** 2))

    w0 = np.full(n, budget / n)
    bounds = [(max(min_weight, 1e-6), max_weight)] * n  # keep strictly positive for the ERC objective
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - budget}]
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 2000, "ftol": 1e-16})
    w = res.x if res.success else w0
    w = _project_to_bounds(w, min_weight, max_weight, budget)
    return pd.Series(w, index=symbols)


# ----------------------------------------------------------------------------- HRP (Lopez de Prado)


def _hrp_quasi_diag(link: np.ndarray) -> list[int]:
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i, j = df0.index, df0.to_numpy() - num_items
        sort_ix[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def _hrp_cluster_var(cov: pd.DataFrame, items: list) -> float:
    sub = cov.loc[items, items]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub.to_numpy() @ ivp)


def _hrp_recursive_bisection(cov: pd.DataFrame, sort_ix: list) -> pd.Series:
    w = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]
    while clusters:
        clusters = [c[j:k] for c in clusters for j, k in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            var_left = _hrp_cluster_var(cov, left)
            var_right = _hrp_cluster_var(cov, right)
            alpha = 1.0 - var_left / (var_left + var_right)
            w[left] *= alpha
            w[right] *= 1.0 - alpha
    return w


def hrp_weights(
    returns_df: pd.DataFrame,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    cash_buffer_pct: float = 0.0,
    linkage_method: str = "single",
) -> pd.Series:
    """Hierarchical Risk Parity (Lopez de Prado, 2016): cluster assets by correlation distance,
    quasi-diagonalize, then recursively split capital between clusters in inverse proportion to
    their variance. Naturally long-only and diversifies across correlated blocks without an
    optimizer (robust to a near-singular covariance matrix, unlike mean-variance methods).
    """
    symbols = list(returns_df.columns)
    corr = returns_df.corr()
    dist = np.sqrt(np.clip(0.5 * (1 - corr.to_numpy()), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method=linkage_method)
    sort_ix = _hrp_quasi_diag(link)
    ordered_symbols = [symbols[i] for i in sort_ix]

    cov = returns_df.cov()
    w = _hrp_recursive_bisection(cov, ordered_symbols)
    w = w.reindex(symbols)

    budget = 1.0 - cash_buffer_pct
    w = _project_to_bounds(w.to_numpy(dtype=float) * budget, min_weight, max_weight, budget)
    return pd.Series(w, index=symbols)


SCHEMES = {
    "equal_weight": lambda returns_df, **kw: equal_weight(list(returns_df.columns), **{k: v for k, v in kw.items() if k == "cash_buffer_pct"}),
    "inverse_vol": inverse_vol_weights,
    "min_variance": min_variance_weights,
    "risk_parity": risk_parity_weights,
    "hrp": hrp_weights,
}


def target_weights(scheme: str, returns_df: pd.DataFrame, **kwargs) -> pd.Series:
    """Dispatch to a named scheme (see `SCHEMES`)."""
    if scheme not in SCHEMES:
        raise ValueError(f"unknown target-weight scheme {scheme!r}; known: {sorted(SCHEMES)}")
    return SCHEMES[scheme](returns_df, **kwargs)
