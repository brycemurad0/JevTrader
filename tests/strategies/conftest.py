"""Shared fixtures for the strategy library tests. Everything is synthetic, seeded and small so
the whole directory runs in well under a minute."""

from __future__ import annotations

import pandas as pd
import pytest

from jevtrader.core.types import AssetClass
from jevtrader.data.synthetic import bars_to_events, generate_bars

CRYPTO_START = pd.Timestamp("2024-03-01 00:00", tz="UTC")
EQ_START = pd.Timestamp("2024-03-04 14:30", tz="UTC")


@pytest.fixture(scope="session")
def btc_bars() -> pd.DataFrame:
    """2 days of 1-min synthetic 24/7 crypto bars (2880 bars)."""
    return generate_bars("BTC/USD", CRYPTO_START, CRYPTO_START + pd.Timedelta(days=2), freq="1min", asset_class=AssetClass.CRYPTO, seed=21, start_price=60_000.0, annual_vol=0.6)


@pytest.fixture(scope="session")
def spy_bars() -> pd.DataFrame:
    """~4 equity sessions of 1-min synthetic bars."""
    return generate_bars("SPY", EQ_START, EQ_START + pd.Timedelta(days=4, hours=6), freq="1min", asset_class=AssetClass.EQUITY, seed=5, start_price=500.0, annual_vol=0.2)


def events(df: pd.DataFrame, symbol: str):
    return list(bars_to_events(df, symbol))
