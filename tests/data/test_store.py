import pandas as pd

from jevtrader.data.store import cache_path, load_bars, load_cache, merge_and_save, save_cache


def _bars_df(idx: pd.DatetimeIndex, base: float = 0.0) -> pd.DataFrame:
    n = len(idx)
    return pd.DataFrame(
        {"open": [base + i for i in range(n)], "high": [base + i for i in range(n)], "low": [base + i for i in range(n)], "close": [base + i for i in range(n)], "volume": [100.0] * n},
        index=idx,
    )


def test_cache_roundtrip(tmp_path):
    idx = pd.date_range("2024-01-02 14:30", periods=5, freq="1min", tz="UTC")
    df = _bars_df(idx)
    path = cache_path(tmp_path, "AAPL", "1Min", "bars")
    save_cache(path, df)
    back = load_cache(path)
    assert back is not None
    pd.testing.assert_frame_equal(back.astype(float), df.astype(float), check_freq=False, check_names=False)
    assert str(back.index.tz) == "UTC"


def test_load_cache_missing_file_returns_none(tmp_path):
    assert load_cache(tmp_path / "does-not-exist.parquet") is None


def test_cache_path_handles_crypto_slash_symbols(tmp_path):
    path = cache_path(tmp_path, "BTC/USD", "1Min", "bars")
    assert "/" not in path.name


def test_merge_and_save_new_data_wins_on_overlap(tmp_path):
    path = cache_path(tmp_path, "AAPL", "1Min", "bars")
    idx1 = pd.date_range("2024-01-02 14:30", periods=3, freq="1min", tz="UTC")
    save_cache(path, _bars_df(idx1, base=1.0))

    idx2 = pd.date_range("2024-01-02 14:32", periods=3, freq="1min", tz="UTC")  # overlaps the last bar of idx1
    df2 = _bars_df(idx2, base=100.0)
    merged = merge_and_save(path, df2)

    assert len(merged) == 5  # 3 + 3 - 1 overlapping timestamp
    assert merged.loc[idx1[2], "close"] == 100.0  # the newer write wins
    reloaded = load_cache(path)
    assert len(reloaded) == 5


def test_load_bars_uses_cache_without_a_fetch_fn(tmp_path):
    idx = pd.date_range("2024-01-02 14:30", periods=10, freq="1min", tz="UTC")
    path = cache_path(tmp_path, "AAPL", "1Min", "bars")
    save_cache(path, _bars_df(idx))

    loaded = load_bars("AAPL", idx[2], idx[5], timeframe="1Min", data_dir=tmp_path)
    assert len(loaded) == 4
    assert loaded.index.min() == idx[2] and loaded.index.max() == idx[5]


def test_load_bars_returns_empty_without_cache_or_fetch_fn(tmp_path):
    loaded = load_bars("NOPE", pd.Timestamp("2024-01-02", tz="UTC"), pd.Timestamp("2024-01-03", tz="UTC"), data_dir=tmp_path)
    assert loaded.empty


def test_load_bars_calls_fetch_fn_when_cache_is_insufficient(tmp_path):
    calls = []

    def fetch(symbol, start, end, timeframe):
        calls.append((symbol, start, end, timeframe))
        idx = pd.date_range(start, end, freq="1min", tz="UTC")
        return _bars_df(idx, base=5.0)

    start = pd.Timestamp("2024-01-02 14:30", tz="UTC")
    end = pd.Timestamp("2024-01-02 14:35", tz="UTC")
    loaded = load_bars("MSFT", start, end, timeframe="1Min", data_dir=tmp_path, fetch_fn=fetch)
    assert len(calls) == 1
    assert not loaded.empty

    # a second call within the now-cached range must not call fetch_fn again.
    loaded_again = load_bars("MSFT", start, end, timeframe="1Min", data_dir=tmp_path, fetch_fn=fetch)
    assert len(calls) == 1
    assert len(loaded_again) == len(loaded)
