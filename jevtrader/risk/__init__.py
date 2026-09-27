"""Risk engine: capital preservation first.

`RiskManager` (manager.py) implements `jevtrader.core.interfaces.RiskGate` and is the single
pre-trade gate between a Strategy's intent and any Broker (sim, paper or live). Every limit it
enforces is explicit and loaded from `config/risk.yaml`; nothing is a magic number in code.

Modules:
    limits.py    RiskLimits: everything that can be configured, with conservative defaults.
    manager.py   RiskManager: check/on_fill/on_mark/halted/kill/reset + audit trail.
    sizing.py    Position sizing math (fixed-fractional, vol-target, fractional Kelly, lot rounding).
    analytics.py VaR/CVaR, correlation, vol, marginal risk contribution, stress tests.
"""

from jevtrader.risk.limits import RiskLimits, TradingHours
from jevtrader.risk.manager import RiskManager

__all__ = ["RiskLimits", "TradingHours", "RiskManager"]
