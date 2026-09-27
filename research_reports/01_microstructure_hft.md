# 01 - Microstructure / HFT-style strategies: what is achievable at retail cost

_Generated 2026-09-27._

## The arithmetic first (no simulation needed)

| venue / fee tier | maker fee per side | required half-spread (fee + ~2 bps adverse selection + 0.5 edge) | typical BTC inside spread | typical SPY inside spread | verdict |
|---|---|---|---|---|---|
| Alpaca crypto tier 1 (<$100k/30d) | 15.00 bps | 17.5 bps (35 bps quoted spread) | ~1-3 bps | ~0.2-1 bps | NOT VIABLE |
| Alpaca crypto tier 3 ($500k) | 10.00 bps | 12.5 bps (25 bps quoted spread) | ~1-3 bps | ~0.2-1 bps | NOT VIABLE |
| Alpaca crypto tier 5 ($10M) | 5.00 bps | 7.5 bps (15 bps quoted spread) | ~1-3 bps | ~0.2-1 bps | MARGINAL |
| Alpaca crypto tier 6 ($25M) | 2.00 bps | 4.5 bps (9 bps quoted spread) | ~1-3 bps | ~0.2-1 bps | MARGINAL |
| Alpaca crypto top tier / other venue maker rebate | 0.00 bps | 2.5 bps (5 bps quoted spread) | ~1-3 bps | ~0.2-1 bps | possible |
| Alpaca US equities | 0.15 bps | 2.6 bps (5 bps quoted spread) | ~1-3 bps | ~0.2-1 bps | possible |

BTC 1-min realized vol is ~6 bps/bar and the mid moves ~3-4 bps per minute on average (Bitstamp 2025-26). A maker who must quote a 35-40 bps spread to cover 15 bps fees is only filled when the price runs through the quote, i.e. by informed flow. This is why the Avellaneda-Stoikov strategy enforces `min_half_spread = maker_fee + adverse_selection + min_edge` and why it is a *negative* result on Alpaca crypto tier 1.

Retail latency (tens of ms REST/WebSocket, no co-location) removes latency arbitrage and most queue-priority games entirely; 'HFT' at retail means quote-driven strategies at 1 s - 5 min horizons with maker fills. Order-book imbalance at those horizons has an IC of a few percent: ~0.1-0.5 bps expected move per event. That clears NO taker fee anywhere and only clears maker costs on zero-commission equities.

## Simulation: synthetic L1 stream with a known weak OFI edge (MECHANICS ONLY)

`jevtrader.data.synthetic.generate_quote_trade_stream` embeds `ret[t] = 0.12 * vol * ofi[t-1] + noise` (a few % IC). We run both strategies through the SimBroker's L1 queue model at several fee levels. This validates that the code does what it says; it does not validate that Alpaca books have this edge -- record real books during paper trading (`research.record_replay`) and replay.

|  | net_pnl_usd | n_fills | maker_share | fees_usd | gross_pnl_usd | sharpe(per-step, indicative) |
|---|---|---|---|---|---|---|
| zero fees / as_market_maker | 2584.69 | 1846 | 1.00 | 0.00 | 2584.69 | 96.90 |
| zero fees / imbalance_alpha | 0.00 | 0 | nan | 0.00 | 0.00 | 0.00 |
| equity-like (0.15) / as_market_maker | 2470.79 | 1846 | 1.00 | 113.89 | 2584.69 | 92.66 |
| equity-like (0.15) / imbalance_alpha | 0.00 | 0 | nan | 0.00 | 0.00 | 0.00 |
| crypto tier1 maker 15 / taker 25 / as_market_maker | -8804.56 | 1846 | 1.00 | 11389.25 | 2584.69 | -307.15 |
| crypto tier1 maker 15 / taker 25 / imbalance_alpha | 0.00 | 0 | nan | 0.00 | 0.00 | 0.00 |
| crypto 2 / 12 / as_market_maker | 1066.12 | 1846 | 1.00 | 1518.57 | 2584.69 | 40.08 |
| crypto 2 / 12 / imbalance_alpha | 0.00 | 0 | nan | 0.00 | 0.00 | 0.00 |

Sample: 240 minutes of 1-second synthetic quotes (14,400 quotes, 8,684 prints). Gross P&L positive and fee-eaten at crypto tiers = the expected picture: the mechanics work, the edge is too small for 15-25 bps fees.

## VPIN toxicity gate

`flow_toxicity.FlowToxicityGate` (bulk-volume classified VPIN) is a filter; it has no stand-alone P&L. Its validation plan: run `vpin_monitor` next to `as_market_maker` in paper trading, then regress the maker's per-fill markout (mid 60 s after fill minus fill price) on VPIN at fill time. A significantly negative slope justifies the gate; otherwise leave it off (it only removes fills).

## Verdicts

* `as_market_maker` on Alpaca crypto tier 1: **NOT VIABLE** (fee floor forces uncompetitive quotes). On US equities: **RESEARCH** -- mechanics sound, edge unproven without real L2; validate via record/replay in paper.
* `imbalance_alpha`: TAKE branch **NOT VIABLE** at any Alpaca fee level (predicted moves are sub-bp); JOIN branch **RESEARCH** on equities, **NOT VIABLE** on crypto tier 1.
* True HFT (sub-second, latency-sensitive) is **not achievable** from a retail Alpaca account: it needs co-location, direct market access, exchange-member fee tiers/rebates and sub-100 us stacks. What would change the picture: crypto fees <= 2-3 bps/side (Alpaca >= $25M/30d tier, or a venue with maker rebates), and a colocated feed.