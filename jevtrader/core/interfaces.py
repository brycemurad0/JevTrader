"""Cross-module contracts so subsystems can be built and tested independently.

- RiskGate: implemented by jevtrader.risk.RiskManager; called by backtester AND live runner
  on every order before it reaches a broker.
- JevAdvisorProtocol: implemented by jevtrader.jev.advisor.JevAdvisor (real TypeSafe API) and
  jevtrader.jev.advisor.OfflineJevAdvisor (deterministic stand-in for backtests/tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol

import pandas as pd

from jevtrader.core.types import AccountState, Fill, Order, Quote


# ----------------------------------------------------------------------------- risk


@dataclass
class RiskDecision:
    approved: bool
    order: Optional[Order]  # possibly resized/modified; None if rejected
    reason: str = ""


class RiskGate(Protocol):
    def check(self, order: Order, account: AccountState, quote: Optional[Quote], now: pd.Timestamp) -> RiskDecision: ...

    def on_fill(self, fill: Fill, account: AccountState) -> None: ...

    def on_mark(self, account: AccountState, now: pd.Timestamp) -> None:
        """Called on every equity mark; may trip the kill switch."""
        ...

    @property
    def halted(self) -> bool: ...


# ----------------------------------------------------------------------------- jev


@dataclass(frozen=True)
class JevView:
    """A typed, calibrated answer set from Jev for one market state.

    `probs` maps question name -> {label: probability} (for Noul: {"yes": p, "no": 1-p};
    for Score: {"0": p0, "1": p1, ...}). `top` maps question -> most-likely label.
    `latency_ms` is wall time; `late` means it exceeded budget and MUST be ignored (treat as HOLD).
    """

    probs: Mapping[str, Mapping[str, float]]
    top: Mapping[str, str]
    confidence: Mapping[str, float]
    latency_ms: float
    late: bool = False
    model: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)

    def p(self, question: str, label: str, default: float = 0.0) -> float:
        return float(self.probs.get(question, {}).get(label, default))


class JevAdvisorProtocol(Protocol):
    def direction(self, symbol: str, features: Mapping[str, Any], horizon: str) -> Optional[JevView]:
        """Ask: over `horizon` will the price move up, down or stay flat (net of costs)?
        Questions returned: "direction" with labels {"up","down","flat"}."""
        ...

    def regime(self, symbol: str, features: Mapping[str, Any]) -> Optional[JevView]:
        """Ask: current regime. "regime" labels {"trending_up","trending_down","mean_reverting","volatile_chop","quiet"};
        "risk_off" noul {"yes","no"}."""
        ...

    def ask(self, state: Mapping[str, Any], questions: Mapping[str, Any]) -> Optional[JevView]:
        """Low-level: arbitrary state + typesafe_sdk questions (Choice/Score/Noul)."""
        ...
