"""RiskLimits: every number RiskManager enforces, loadable from YAML.

Philosophy: capital preservation first. Every limit here is explicit, documented, and has a
conservative default suitable for a small account. Nothing is hard-coded in `manager.py` -
if a behaviour needs a threshold, that threshold lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml


@dataclass
class TradingHours:
    """Equities trading-hours guard. Crypto ignores this entirely (24/7 market).

    Times are "HH:MM" in `tz` (a zoneinfo key, e.g. "America/New_York" for US equities).
    """

    enabled: bool = True
    tz: str = "America/New_York"
    regular_start: str = "09:30"
    regular_end: str = "16:00"
    allow_extended_hours: bool = False
    extended_start: str = "04:00"
    extended_end: str = "20:00"
    trade_weekends: bool = False  # equities never trade weekends in practice; kept explicit

    @classmethod
    def from_dict(cls, d: Optional[Mapping[str, Any]]) -> "TradingHours":
        if not d:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class RiskLimits:
    """Conservative defaults for a small account. See config/risk.yaml for the annotated version
    that ships with the repo; that file is the source of truth an operator actually edits.
    """

    # ---------------------------------------------------------------- position & exposure caps
    # Hard ceiling on any single symbol's position, in USD notional and as % of equity.
    # The tighter of the two always wins.
    max_position_notional_usd: float = 5_000.0
    max_position_pct_equity: float = 0.20  # 20% of equity in any one name

    # Total absolute exposure (sum of |position notional|) across the whole book.
    max_gross_exposure_pct: float = 1.00  # 100% of equity gross (no leverage by default)
    # Net (signed) exposure - caps directional bias even if gross is within budget.
    max_net_exposure_pct: float = 0.75

    # Crypto is a separate, smaller risk bucket: higher vol, thin liquidity, weekend gaps.
    crypto_max_pct_equity: float = 0.25

    # Per-strategy capital allocation: RiskManager tracks a shadow notional ledger per
    # strategy_id (Order.strategy_id) from fills and caps it here. Unlisted strategies fall
    # back to `default_strategy_capital_pct`.
    per_strategy_capital_pct: dict[str, float] = field(default_factory=dict)
    default_strategy_capital_pct: float = 0.50

    # ---------------------------------------------------------------- order-level guards
    max_open_orders_per_symbol: int = 5
    max_order_notional_usd: float = 2_000.0  # single-order ceiling, independent of position caps
    max_orders_per_minute: int = 20  # throttle: runaway-loop / fat-finger-spam protection
    # Fat-finger band: reject a LIMIT/STOP_LIMIT order whose limit price is more than this many
    # bps away from the last quoted mid. Does not apply to MARKET orders (no limit price).
    fat_finger_band_bps: float = 100.0  # 1.00%
    min_notional_usd: float = 5.0  # reject dust orders below this (never applies to reduce_only)

    # ---------------------------------------------------------------- capital preservation
    # Halt NEW risk for the rest of the trading day once realized+unrealized daily PnL drops
    # this much. Existing positions are left alone (this is not a kill switch).
    max_daily_loss_pct: float = 0.03  # 3%
    # Kill switch: halts everything AND sets `flatten` so the caller (bot/dashboard) flattens
    # every position. Measured from the all-time high-water mark of account equity.
    max_drawdown_pct: float = 0.15  # 15%
    # After this many consecutive losing (realized) trades, halt new risk for a cooldown period.
    # A cheap circuit breaker against a strategy that has started misbehaving.
    max_consecutive_losses: int = 5
    consecutive_loss_cooldown_minutes: float = 60.0
    # Hour (0-23, UTC) at which the daily PnL baseline resets. Real markets differ (crypto is
    # 00:00 UTC, US equities are the 9:30 ET open) but we keep ONE configurable hour for
    # simplicity, as directed; the default (0 = UTC midnight) suits crypto-heavy books.
    daily_reset_hour_utc: int = 0

    # ---------------------------------------------------------------- instrument rules
    enforce_long_only_for_non_shortable: bool = True
    trading_hours: TradingHours = field(default_factory=TradingHours)

    # ------------------------------------------------------------------------------ loading
    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RiskLimits":
        d = dict(d)
        th = TradingHours.from_dict(d.pop("trading_hours", None))
        known = {f.name for f in fields(cls)} - {"trading_hours"}
        kwargs = {k: v for k, v in d.items() if k in known}
        return cls(trading_hours=th, **kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RiskLimits":
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}
        # Allow either a bare mapping or {"risk": {...}}.
        raw = raw.get("risk", raw) if isinstance(raw, dict) else raw
        return cls.from_dict(raw)

    def to_dict(self) -> dict:
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = v.to_dict() if is_dataclass(v) and not isinstance(v, RiskLimits) else v
        return out

    @classmethod
    def conservative_default(cls) -> "RiskLimits":
        """Alias for the dataclass defaults - kept explicit for readability at call sites."""
        return cls()
