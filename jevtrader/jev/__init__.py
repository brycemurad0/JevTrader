"""TypeSafe Jev integration: state builder, question library, advisors, decision log, calibration.

See `jevtrader.core.interfaces.JevAdvisorProtocol` for the contract every advisor here
implements, and `docs/ARCHITECTURE.md` for how this package fits into the rest of the stack.
"""

from __future__ import annotations

from jevtrader.jev.advisor import (
    AsyncJevAdvisor,
    JevAdvisor,
    OfflineJevAdvisor,
    OfflineWeights,
    ReplayJevAdvisor,
    ShadowAdvisor,
    SpeculativeDirection,
    make_advisor,
)
from jevtrader.jev.backends import DecisionBackend, make_client
from jevtrader.jev.calibration import calibrate_binary, calibrate_direction
from jevtrader.jev.log import DecisionLog, annotate_outcomes, read_jsonl
from jevtrader.jev.ping import ping
from jevtrader.jev.questions import (
    direction_questions,
    entry_quality_questions,
    rebalance_tilt_questions,
    regime_questions,
)
from jevtrader.jev.state import build_state, to_json

__all__ = [
    "AsyncJevAdvisor",
    "DecisionBackend",
    "DecisionLog",
    "JevAdvisor",
    "OfflineJevAdvisor",
    "OfflineWeights",
    "ReplayJevAdvisor",
    "ShadowAdvisor",
    "SpeculativeDirection",
    "annotate_outcomes",
    "build_state",
    "calibrate_binary",
    "calibrate_direction",
    "direction_questions",
    "entry_quality_questions",
    "make_advisor",
    "make_client",
    "ping",
    "read_jsonl",
    "rebalance_tilt_questions",
    "regime_questions",
    "to_json",
]
