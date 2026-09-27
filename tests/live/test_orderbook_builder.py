"""OrderBookBuilder: incremental L2 merge for Alpaca's crypto order book stream, where a level
with size 0 means "remove this price level", and `reset=True` means "this is a fresh snapshot"."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from jevtrader.live.streams import OrderBookBuilder


def _level(price, size):
    return SimpleNamespace(price=price, size=size)


def _ob(symbol, bids, asks, reset=False, ts=None):
    return SimpleNamespace(
        symbol=symbol,
        timestamp=ts or pd.Timestamp("2026-01-01T00:00:00Z"),
        bids=[_level(p, s) for p, s in bids],
        asks=[_level(p, s) for p, s in asks],
        reset=reset,
    )


def test_initial_snapshot_sorted_correctly():
    builder = OrderBookBuilder()
    book = builder.apply(_ob("BTC/USD", [(100.0, 1.0), (99.0, 2.0)], [(101.0, 1.5), (102.0, 0.5)], reset=True))
    assert [lvl.price for lvl in book.bids] == [100.0, 99.0]  # best (highest) bid first
    assert [lvl.price for lvl in book.asks] == [101.0, 102.0]  # best (lowest) ask first
    assert book.best_bid.price == 100.0
    assert book.best_ask.price == 101.0


def test_delta_updates_existing_level():
    builder = OrderBookBuilder()
    builder.apply(_ob("BTC/USD", [(100.0, 1.0)], [(101.0, 1.0)], reset=True))
    book = builder.apply(_ob("BTC/USD", [(100.0, 3.5)], []))
    assert book.bids[0].size == 3.5
    assert book.asks[0].price == 101.0  # untouched side/levels persist


def test_zero_size_removes_level():
    builder = OrderBookBuilder()
    builder.apply(_ob("BTC/USD", [(100.0, 1.0), (99.0, 2.0)], [(101.0, 1.0)], reset=True))
    book = builder.apply(_ob("BTC/USD", [(99.0, 0.0)], []))
    assert [lvl.price for lvl in book.bids] == [100.0]


def test_delta_adds_new_level():
    builder = OrderBookBuilder()
    builder.apply(_ob("BTC/USD", [(100.0, 1.0)], [(101.0, 1.0)], reset=True))
    book = builder.apply(_ob("BTC/USD", [(98.0, 5.0)], []))
    assert sorted([lvl.price for lvl in book.bids], reverse=True) == [100.0, 98.0]


def test_reset_clears_prior_state():
    builder = OrderBookBuilder()
    builder.apply(_ob("BTC/USD", [(100.0, 1.0), (99.0, 2.0)], [(101.0, 1.0)], reset=True))
    book = builder.apply(_ob("BTC/USD", [(50.0, 1.0)], [(60.0, 1.0)], reset=True))
    assert [lvl.price for lvl in book.bids] == [50.0]
    assert [lvl.price for lvl in book.asks] == [60.0]


def test_independent_books_per_symbol():
    builder = OrderBookBuilder()
    builder.apply(_ob("BTC/USD", [(100.0, 1.0)], [(101.0, 1.0)], reset=True))
    builder.apply(_ob("ETH/USD", [(2000.0, 1.0)], [(2001.0, 1.0)], reset=True))
    assert builder.snapshot("BTC/USD").best_bid.price == 100.0
    assert builder.snapshot("ETH/USD").best_bid.price == 2000.0


def test_snapshot_before_any_data_is_none():
    builder = OrderBookBuilder()
    assert builder.snapshot("BTC/USD") is None


def test_imbalance_reflects_book_state():
    builder = OrderBookBuilder()
    book = builder.apply(_ob("BTC/USD", [(100.0, 8.0)], [(101.0, 2.0)], reset=True))
    assert book.imbalance() > 0  # bid-heavy
