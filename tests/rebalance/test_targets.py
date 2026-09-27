from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jevtrader.rebalance.targets import (
    equal_weight,
    hrp_weights,
    inverse_vol_weights,
    ledoit_wolf_cov,
    min_variance_weights,
    risk_parity_weights,
    static_weights,
    target_weights,
)

SYMBOLS = ["SPY", "QQQ", "TLT", "GLD", "BTC/USD", "ETH/USD"]


@pytest.fixture
def returns_df():
    rng = np.random.default_rng(7)
    n = 400
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    # Two correlated equity-ish blocks + two crypto assets highly correlated with each other,
    # and two "diversifier" assets (bonds/gold) largely independent - a realistic block structure.
    equity_common = rng.normal(0, 0.008, n)
    crypto_common = rng.normal(0, 0.03, n)
    data = {
        "SPY": equity_common + rng.normal(0, 0.002, n),
        "QQQ": equity_common + rng.normal(0, 0.003, n),
        "TLT": rng.normal(0, 0.006, n),
        "GLD": rng.normal(0, 0.005, n),
        "BTC/USD": crypto_common + rng.normal(0, 0.005, n),
        "ETH/USD": crypto_common + rng.normal(0, 0.008, n),
    }
    return pd.DataFrame(data, index=idx)


def _assert_valid_weights(w: pd.Series, min_weight=0.0, max_weight=1.0, total=1.0):
    assert w.sum() == pytest.approx(total, abs=1e-6)
    assert (w >= min_weight - 1e-9).all()
    assert (w <= max_weight + 1e-9).all()


# ----------------------------------------------------------------------------- Ledoit-Wolf


def test_ledoit_wolf_shrinkage_between_zero_and_one(returns_df):
    cov, shrinkage = ledoit_wolf_cov(returns_df)
    assert 0.0 <= shrinkage <= 1.0
    assert cov.shape == (len(SYMBOLS), len(SYMBOLS))
    # symmetric, positive diagonal
    assert np.allclose(cov.to_numpy(), cov.to_numpy().T)
    assert (np.diag(cov.to_numpy()) > 0).all()


def test_ledoit_wolf_shrinks_off_diagonals_towards_zero(returns_df):
    sample_cov = returns_df.cov()
    shrunk, shrinkage = ledoit_wolf_cov(returns_df)
    assert shrinkage > 0.0  # finite sample -> some shrinkage expected
    off_sample = sample_cov.to_numpy()[~np.eye(len(SYMBOLS), dtype=bool)]
    off_shrunk = shrunk.to_numpy()[~np.eye(len(SYMBOLS), dtype=bool)]
    assert np.abs(off_shrunk).sum() < np.abs(off_sample).sum()


# ----------------------------------------------------------------------------- basic schemes


def test_static_weights_sum_and_bounds():
    w = static_weights({"SPY": 0.6, "TLT": 0.4}, SYMBOLS, min_weight=0.0, max_weight=0.5)
    _assert_valid_weights(w, max_weight=0.5)


def test_equal_weight_sums_to_budget():
    w = equal_weight(SYMBOLS, cash_buffer_pct=0.1)
    _assert_valid_weights(w, total=0.9)
    assert w.nunique() == 1


def test_inverse_vol_weights_sum_and_favor_low_vol(returns_df):
    w = inverse_vol_weights(returns_df)
    _assert_valid_weights(w)
    # TLT/GLD are lower vol than BTC/ETH -> should get more weight
    assert w["TLT"] > w["BTC/USD"]


def test_inverse_vol_respects_max_weight_bound(returns_df):
    w = inverse_vol_weights(returns_df, max_weight=0.2)
    _assert_valid_weights(w, max_weight=0.2)


def test_min_variance_weights_sum_and_bounds(returns_df):
    w = min_variance_weights(returns_df, min_weight=0.02, max_weight=0.4)
    _assert_valid_weights(w, min_weight=0.02, max_weight=0.4)


def test_min_variance_prefers_low_vol_assets(returns_df):
    w = min_variance_weights(returns_df, max_weight=1.0)
    assert w["BTC/USD"] + w["ETH/USD"] < w["TLT"] + w["GLD"]


def test_risk_parity_weights_sum_and_bounds(returns_df):
    w = risk_parity_weights(returns_df, min_weight=0.01, max_weight=0.5)
    _assert_valid_weights(w, min_weight=0.01, max_weight=0.5)


def test_risk_parity_gives_lower_weight_to_higher_vol_assets(returns_df):
    w = risk_parity_weights(returns_df)
    assert w["BTC/USD"] < w["TLT"]


def test_cash_buffer_reduces_invested_budget(returns_df):
    w = inverse_vol_weights(returns_df, cash_buffer_pct=0.25)
    assert w.sum() == pytest.approx(0.75, abs=1e-6)


# ----------------------------------------------------------------------------- HRP


def test_hrp_sums_to_one_and_long_only(returns_df):
    w = hrp_weights(returns_df)
    _assert_valid_weights(w)
    assert (w >= 0).all()


def test_hrp_respects_bounds(returns_df):
    w = hrp_weights(returns_df, min_weight=0.02, max_weight=0.35)
    _assert_valid_weights(w, min_weight=0.02, max_weight=0.35)


def test_hrp_on_block_correlated_matrix_diversifies_across_blocks(returns_df):
    """Two nearly-identical assets within a block should each get roughly the same weight as
    each other, and the crypto block (higher vol) should get less aggregate weight than the
    lower-vol diversifiers, matching HRP's inverse-variance recursive bisection."""
    w = hrp_weights(returns_df)
    # SPY/QQQ share a common factor and similar vol -> comparable individual weights.
    assert w["SPY"] == pytest.approx(w["QQQ"], rel=0.6)
    crypto_weight = w["BTC/USD"] + w["ETH/USD"]
    defensive_weight = w["TLT"] + w["GLD"]
    assert crypto_weight < defensive_weight


def test_target_weights_dispatch(returns_df):
    w1 = target_weights("hrp", returns_df)
    w2 = hrp_weights(returns_df)
    pd.testing.assert_series_equal(w1.sort_index(), w2.sort_index())

    with pytest.raises(ValueError):
        target_weights("not_a_scheme", returns_df)
