"""Smart rebalancer: target-weight schemes, cost-aware no-trade bands, optional Jev regime tilt.

Modules:
    targets.py  Target-weight schemes from a returns DataFrame (equal weight, inverse vol,
                risk parity, HRP, min variance), all long-only with bounds + cash buffer, using
                a hand-rolled Ledoit-Wolf shrinkage covariance (no sklearn dependency).
    policy.py   Cost-aware rebalance decision: 5/25 tolerance bands, calendar check, and a
                trade-only-if-benefit-exceeds-cost gate; optional bounded Jev regime tilt.
    planner.py  Turns a decision into an ordered, cash- and lot-aware list of orders
                (RebalancePlanner) with a human-readable `RebalancePlan.to_markdown()`.
    bot.py      SmartRebalanceBot(Strategy), registered as "smart_rebalance".
"""

from jevtrader.rebalance.policy import RebalanceDecision, RebalancePolicy
from jevtrader.rebalance.planner import RebalancePlan, RebalancePlanItem, RebalancePlanner

__all__ = [
    "RebalanceDecision",
    "RebalancePolicy",
    "RebalancePlan",
    "RebalancePlanItem",
    "RebalancePlanner",
]
