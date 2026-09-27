"""Historical data loaders backed by alpaca-py, normalized to UTC DataFrames and cached as
parquet (or gzip CSV, see `jevtrader.data.store`) under `settings.data_dir`.

Design: every alpaca-py response object of the historical data clients (`BarSet`, `QuoteSet`,
`TradeSet`) exposes a `.df` property -- a plain pandas DataFrame, multi-indexed by
`(symbol, timestamp)` when built for one or more symbols. All the actual normalization logic
here (`normalize_raw_df` and friends) is a pure function of that DataFrame, so it is unit
testable by constructing a fake `.df` directly, with no network and no alpaca-py client
involved. Pagination across pages of results is handled internally by alpaca-py's HTTP client
(it walks `next_page_token` until exhausted), so a single `get_*` call already returns the full
requested range.

Crypto data needs no API key (Alpaca crypto market data is public); equities need
`ALPACA_API_KEY`/`ALPACA_SECRET_KEY` from `jevtrader.config.load_settings()`.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import pandas as pd

from jevtrader.config import Settings, load_settings

_BAR_COLUMNS = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
_QUOTE_RENAME = {"bid_price": "bid", "ask_price": "ask", "bid_size": "bid_size", "ask_size": "ask_size"}
_QUOTE_COLUMNS = ["bid", "ask", "bid_size", "ask_size"]
_TRADE_RENAME = {"price": "price", "size": "size"}
_TRADE_COLUMNS = ["price", "size"]


def _tz_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def normalize_raw_df(raw_df: pd.DataFrame, symbol: str, columns: Sequence[str], rename: Optional[dict] = None) -> pd.DataFrame:
    """Pure normalization: `raw_df` is an alpaca-py `.df` (single- or multi-indexed by
    `(symbol, timestamp)`). Returns a single-symbol, UTC-tz, `ts`-indexed DataFrame with just
    `columns` (renamed via `rename` first, if given), sorted and de-duplicated by timestamp.
    """
    if raw_df is None or len(raw_df) == 0:
        return pd.DataFrame(columns=list(columns))
    df = raw_df
    if isinstance(df.index, pd.MultiIndex):
        level0 = df.index.get_level_values(0)
        if symbol in set(level0):
            df = df.xs(symbol, level=0, drop_level=True)
        else:
            return pd.DataFrame(columns=list(columns))
    df = df.copy()
    if rename:
        df = df.rename(columns=rename)
    df = _tz_utc_index(df)
    df.index.name = "ts"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    keep = [c for c in columns if c in df.columns]
    return df[keep]


def normalize_bars_df(raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    return normalize_raw_df(raw_df, symbol, _BAR_COLUMNS)


def normalize_quotes_df(raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    return normalize_raw_df(raw_df, symbol, _QUOTE_COLUMNS, rename=_QUOTE_RENAME)


def normalize_trades_df(raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    return normalize_raw_df(raw_df, symbol, _TRADE_COLUMNS, rename=_TRADE_RENAME)


def _stock_client(settings: Optional[Settings] = None):
    from alpaca.data.historical.stock import StockHistoricalDataClient

    s = settings or load_settings()
    return StockHistoricalDataClient(api_key=s.alpaca_key or None, secret_key=s.alpaca_secret or None)


def _crypto_client(settings: Optional[Settings] = None):
    from alpaca.data.historical.crypto import CryptoHistoricalDataClient

    s = settings or load_settings()
    # crypto market data does not require keys; pass through if present (higher rate limits).
    return CryptoHistoricalDataClient(api_key=s.alpaca_key or None, secret_key=s.alpaca_secret or None)


def _timeframe(tf: Union[str, "object"]):
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    if not isinstance(tf, str):
        return tf
    s = tf.strip()
    unit_map = {
        "min": TimeFrameUnit.Minute,
        "minute": TimeFrameUnit.Minute,
        "h": TimeFrameUnit.Hour,
        "hour": TimeFrameUnit.Hour,
        "d": TimeFrameUnit.Day,
        "day": TimeFrameUnit.Day,
        "w": TimeFrameUnit.Week,
        "week": TimeFrameUnit.Week,
        "m": TimeFrameUnit.Month,
        "month": TimeFrameUnit.Month,
    }
    for suffix, unit in sorted(unit_map.items(), key=lambda kv: -len(kv[0])):
        if s.lower().endswith(suffix):
            amount_str = s[: -len(suffix)] or "1"
            return TimeFrame(int(amount_str), unit)
    raise ValueError(f"cannot parse timeframe {tf!r}; use e.g. '1Min', '5Min', '1Hour', '1Day'")


def load_stock_bars(symbol: str, start: pd.Timestamp, end: pd.Timestamp, timeframe: str = "1Min", settings: Optional[Settings] = None) -> pd.DataFrame:
    """Fetch equity bars for one symbol from Alpaca and return a normalized UTC DataFrame.
    Requires `ALPACA_API_KEY`/`ALPACA_SECRET_KEY`."""
    from alpaca.data.requests import StockBarsRequest

    client = _stock_client(settings)
    req = StockBarsRequest(symbol_or_symbols=symbol, start=start, end=end, timeframe=_timeframe(timeframe))
    barset = client.get_stock_bars(req)
    return normalize_bars_df(barset.df, symbol)


def load_stock_quotes(symbol: str, start: pd.Timestamp, end: pd.Timestamp, settings: Optional[Settings] = None) -> pd.DataFrame:
    from alpaca.data.requests import StockQuotesRequest

    client = _stock_client(settings)
    req = StockQuotesRequest(symbol_or_symbols=symbol, start=start, end=end)
    qs = client.get_stock_quotes(req)
    return normalize_quotes_df(qs.df, symbol)


def load_stock_trades(symbol: str, start: pd.Timestamp, end: pd.Timestamp, settings: Optional[Settings] = None) -> pd.DataFrame:
    from alpaca.data.requests import StockTradesRequest

    client = _stock_client(settings)
    req = StockTradesRequest(symbol_or_symbols=symbol, start=start, end=end)
    ts = client.get_stock_trades(req)
    return normalize_trades_df(ts.df, symbol)


def load_crypto_bars(symbol: str, start: pd.Timestamp, end: pd.Timestamp, timeframe: str = "1Min", settings: Optional[Settings] = None) -> pd.DataFrame:
    """Fetch crypto bars (e.g. 'BTC/USD') from Alpaca. No API key required."""
    from alpaca.data.requests import CryptoBarsRequest

    client = _crypto_client(settings)
    req = CryptoBarsRequest(symbol_or_symbols=symbol, start=start, end=end, timeframe=_timeframe(timeframe))
    barset = client.get_crypto_bars(req)
    return normalize_bars_df(barset.df, symbol)


def load_crypto_quotes(symbol: str, start: pd.Timestamp, end: pd.Timestamp, settings: Optional[Settings] = None) -> pd.DataFrame:
    from alpaca.data.requests import CryptoQuoteRequest

    client = _crypto_client(settings)
    req = CryptoQuoteRequest(symbol_or_symbols=symbol, start=start, end=end)
    qs = client.get_crypto_quotes(req)
    return normalize_quotes_df(qs.df, symbol)


def load_crypto_trades(symbol: str, start: pd.Timestamp, end: pd.Timestamp, settings: Optional[Settings] = None) -> pd.DataFrame:
    from alpaca.data.requests import CryptoTradesRequest

    client = _crypto_client(settings)
    req = CryptoTradesRequest(symbol_or_symbols=symbol, start=start, end=end)
    ts = client.get_crypto_trades(req)
    return normalize_trades_df(ts.df, symbol)
