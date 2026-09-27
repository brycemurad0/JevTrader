# JevTrader architecture

Local-only quant trading stack. Nothing runs in the cloud; keys live in `.env` on your machine.

```
            ┌──────────── market data (Alpaca stocks + crypto: bars, quotes, trades, L2 books) ────────────┐
            ▼                                                                                               │
  Strategy (jevtrader/strategies)  ──ask──▶  JevAdvisor (jevtrader/jev)  ── typed probs (Choice/Score/Noul)  │
      │  "Jev judges, code executes": thresholds, sizing, stops are code                                   │
      ▼ ctx.submit(order)                                                                                   │
  RiskManager (jevtrader/risk)  — pre-trade veto/resize, kill switch, drawdown/daily-loss halts             │
      ▼                                                                                                     │
  Broker:  SimBroker (backtest)  |  AlpacaBroker paper  |  AlpacaBroker live (gated)  ──fills──▶ ledger ────┘
```

The **same Strategy class** runs in all three modes. Promotion is one-way and gated:

`backtest (fee+slippage aware, walk-forward, out-of-sample) → paper (forward test, N days) → live (small size, then scale)`

## Package layout / ownership

| path | purpose |
|---|---|
| `jevtrader/core/` | contracts: types, fees, Strategy, Broker, RiskGate, JevView, registry. **Stable – do not change signatures.** |
| `jevtrader/data/` | synthetic market generator (GBM + regimes + microstructure), Alpaca historical loaders, parquet cache |
| `jevtrader/backtest/` | event-driven engine, SimBroker with maker/taker fill simulation, latency, metrics, walk-forward, parameter sweeps w/ deflated Sharpe |
| `jevtrader/jev/` | TypeSafe Jev client wrapper w/ latency budget, compact state builder, question library, offline advisor, decision log, calibration |
| `jevtrader/risk/` | RiskManager implementing `RiskGate`, sizing (vol-target, fractional Kelly), kill switch |
| `jevtrader/rebalance/` | smart rebalancer: target-weight schemes, cost-aware no-trade bands, Jev regime tilt |
| `jevtrader/execution/` | AlpacaBroker (paper/live), order manager, reconciliation |
| `jevtrader/live/` | async live/paper runner, market data streams, promotion gate |
| `jevtrader/dashboard/` | local FastAPI UI (127.0.0.1): order books, strategies, PnL, manual ticket, kill switch |
| `jevtrader/strategies/` | strategy library (registered via `@register`) |
| `jevtrader/research/` | research harness & reports used to validate strategies |
| `jevtrader/cli.py` | `jev` command |

## Fee reality check (drives strategy design)

Computed from `jevtrader/core/fees.py` defaults:

* **Alpaca crypto tier 1:** 15 bps maker / 25 bps taker → **50 bps taker round trip, 30 bps maker round trip.**
  Any crypto strategy on Alpaca must expect > 30–50 bps per trade *before* spread. Classic HFT (sub-bp edges) is impossible there;
  crypto strategies must be low-turnover, or maker-only with large expected moves.
* **Alpaca equities:** $0 commission; SEC + FINRA TAF on sells ≈ 0.3 bps round trip on a $20k ticket. The dominant cost is the
  **spread + slippage + adverse selection**, which the simulator models explicitly. Liquid names (SPY, QQQ, AAPL, …) have 1-cent
  spreads (~0.2–2 bps).
* Retail REST/WebSocket latency to Alpaca is tens of ms; true co-located HFT is not achievable. We target
  **"mid-frequency" (seconds → hours)** microstructure-aware strategies and make that explicit in every strategy's notes.

## Conventions

* Python ≥ 3.10, type hints, dataclasses, numpy/pandas. Timestamps are tz-aware UTC `pd.Timestamp`.
* No network in tests. Everything testable offline with `jevtrader.data.synthetic` and `OfflineJevAdvisor`.
* Tests live in `tests/<area>/test_*.py`. Run `pytest`.
* Never place live orders from tests or examples. Live routing requires `JEV_LIVE_CONFIRM=I_UNDERSTAND_THIS_IS_REAL_MONEY`
  AND a passing promotion record for that strategy.
