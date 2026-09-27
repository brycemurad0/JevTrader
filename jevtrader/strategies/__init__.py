"""Strategy library. Every module registers its strategies with `jevtrader.core.registry`;
`jevtrader.core.registry.all_strategies()` autoloads this package.

See docs/STRATEGIES.md for the catalogue, evidence and verdicts, and research_reports/ for the
per-study numbers. Honest labelling matters more than a long list: a strategy in here is NOT an
endorsement -- read its `spec.notes`.
"""

from jevtrader.strategies import (  # noqa: F401  (import for registration side effects)
    flow_toxicity,
    imbalance_alpha,
    jev_gated,
    lead_lag,
    market_maker,
    meta_allocator,
    pairs_kalman,
    range_breakout,
    seasonality,
    trend_follow,
    vol_squeeze,
    vwap_reversion,
)
