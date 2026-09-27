"""Turns target weights into an ordered, cash- and lot-aware list of orders.

Sells are planned before buys (to free cash first), quantities are lot-rounded and floored to
each instrument's `min_notional`, non-shortable instruments are never sold past a flat position,
and total buy notional never exceeds the cash freed by the sells plus the account's starting cash.
Orders are LIMIT at mid +/- an offset by default (never MARKET - the planner always produces a
plan for a human, or a bot, to review before it hits a broker).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import pandas as pd

from jevtrader.core.fees import FeeModel
from jevtrader.core.types import AccountState, Instrument, Liquidity, Order, OrderType, Side
from jevtrader.risk.sizing import lot_round


@dataclass
class RebalancePlanItem:
    symbol: str
    side: Side
    qty: float
    limit_price: float
    notional: float
    est_cost_usd: float
    reason: str = ""

    def to_order(self, strategy_id: str = "smart_rebalance", tif=None) -> Order:
        kwargs = dict(
            symbol=self.symbol,
            side=self.side,
            qty=self.qty,
            type=OrderType.LIMIT,
            limit_price=self.limit_price,
            strategy_id=strategy_id,
            tag="rebalance",
        )
        if tif is not None:
            kwargs["tif"] = tif
        return Order(**kwargs)


@dataclass
class RebalancePlan:
    ts: pd.Timestamp
    items: list[RebalancePlanItem] = field(default_factory=list)
    total_est_cost_usd: float = 0.0
    cash_before: float = 0.0
    cash_after_sells: float = 0.0
    cash_after_buys: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_orders(self, strategy_id: str = "smart_rebalance") -> list[Order]:
        return [item.to_order(strategy_id) for item in self.items]

    def to_markdown(self) -> str:
        lines = [f"# Rebalance plan - {self.ts}", ""]
        if not self.items:
            lines.append("No trades. " + ("; ".join(self.notes) if self.notes else "Portfolio is within tolerance."))
            return "\n".join(lines)

        lines.append(f"Cash before: ${self.cash_before:,.2f}  |  after sells: ${self.cash_after_sells:,.2f}  |  after buys: ${self.cash_after_buys:,.2f}")
        lines.append(f"Total estimated cost: ${self.total_est_cost_usd:,.2f}")
        lines.append("")
        lines.append("| # | Side | Symbol | Qty | Limit price | Notional | Est. cost |")
        lines.append("|---|------|--------|-----|-------------|----------|-----------|")
        for i, it in enumerate(self.items, 1):
            lines.append(
                f"| {i} | {it.side.value.upper()} | {it.symbol} | {it.qty:.6g} | ${it.limit_price:,.4f} | "
                f"${it.notional:,.2f} | ${it.est_cost_usd:,.2f} |"
            )
        if self.notes:
            lines.append("")
            lines.append("Notes:")
            for n in self.notes:
                lines.append(f"- {n}")
        lines.append("")
        lines.append("**This is a dry-run plan. Nothing has been submitted.**")
        return "\n".join(lines)


class RebalancePlanner:
    def __init__(
        self,
        instruments: Mapping[str, Instrument],
        fee_model: FeeModel,
        limit_offset_bps: float = 5.0,
    ):
        self.instruments = dict(instruments)
        self.fee_model = fee_model
        self.limit_offset_bps = limit_offset_bps

    def _instrument(self, symbol: str) -> Instrument:
        return self.instruments.get(symbol) or Instrument.infer(symbol)

    def plan(
        self,
        account: AccountState,
        prices: Mapping[str, float],
        targets: Mapping[str, float],
        now: Optional[pd.Timestamp] = None,
    ) -> RebalancePlan:
        now = now if now is not None else pd.Timestamp.now(tz="UTC")
        equity = account.equity
        notes: list[str] = []

        target_notional = {s: w * equity for s, w in targets.items()}
        current_notional = {
            sym: pos.qty * prices.get(sym, pos.avg_price) for sym, pos in account.positions.items()
        }
        symbols = sorted(set(target_notional) | set(current_notional))
        delta = {s: target_notional.get(s, 0.0) - current_notional.get(s, 0.0) for s in symbols}

        items: list[RebalancePlanItem] = []
        cash = account.cash

        # ------------------------------------------------------------ sells first, to free cash
        for sym in sorted((s for s, d in delta.items() if d < 0), key=lambda s: delta[s]):
            price = prices.get(sym)
            if price is None or price <= 0:
                notes.append(f"{sym}: no price available, skipped")
                continue
            inst = self._instrument(sym)
            pos = account.positions.get(sym)
            current_qty = pos.qty if pos else 0.0
            desired_sell_qty = abs(delta[sym]) / price
            max_sell_qty = current_qty if inst.shortable else max(current_qty, 0.0)
            qty = min(desired_sell_qty, max_sell_qty)
            qty = lot_round(qty, inst)
            if qty <= 0 or qty * price < inst.min_notional:
                if desired_sell_qty > 0:
                    notes.append(f"{sym}: sell skipped (below lot size / min notional, or no sellable position)")
                continue
            limit_price = price * (1 - self.limit_offset_bps / 1e4)
            notional = qty * price
            cost = self.fee_model.fee(inst, Side.SELL, qty, limit_price, Liquidity.TAKER)
            items.append(RebalancePlanItem(sym, Side.SELL, qty, limit_price, notional, cost, "reduce to target"))
            cash += notional

        cash_after_sells = cash

        # ------------------------------------------------------------ buys, never exceeding cash
        buy_symbols = [s for s, d in delta.items() if d > 0]
        total_buy_notional = sum(delta[s] for s in buy_symbols)
        scale = 1.0
        if total_buy_notional > cash_after_sells and total_buy_notional > 0:
            scale = max(cash_after_sells / total_buy_notional, 0.0)
            notes.append(
                f"buys scaled to {scale:.1%} of target to stay within available cash "
                f"(${cash_after_sells:,.2f} available, ${total_buy_notional:,.2f} requested)"
            )

        remaining_cash = cash_after_sells
        for sym in sorted(buy_symbols, key=lambda s: -delta[s]):
            price = prices.get(sym)
            if price is None or price <= 0:
                notes.append(f"{sym}: no price available, skipped")
                continue
            inst = self._instrument(sym)
            notional_target = delta[sym] * scale
            qty = notional_target / price
            qty = lot_round(qty, inst)
            notional = qty * price
            if qty <= 0 or notional < inst.min_notional:
                if notional_target > 0:
                    notes.append(f"{sym}: buy skipped (below lot size / min notional)")
                continue
            if notional > remaining_cash + 1e-6:
                qty = lot_round(remaining_cash / price, inst)
                notional = qty * price
                if qty <= 0 or notional < inst.min_notional:
                    notes.append(f"{sym}: buy skipped (insufficient remaining cash)")
                    continue
                notes.append(f"{sym}: buy trimmed to fit remaining cash")
            limit_price = price * (1 + self.limit_offset_bps / 1e4)
            cost = self.fee_model.fee(inst, Side.BUY, qty, limit_price, Liquidity.TAKER)
            items.append(RebalancePlanItem(sym, Side.BUY, qty, limit_price, notional, cost, "raise to target"))
            remaining_cash -= notional

        total_cost = sum(it.est_cost_usd for it in items)
        return RebalancePlan(
            ts=now,
            items=items,
            total_est_cost_usd=total_cost,
            cash_before=account.cash,
            cash_after_sells=cash_after_sells,
            cash_after_buys=remaining_cash,
            notes=notes,
        )
