"""Fine-tuning support for local decision models (Kev, Laya): dataset construction from
historical bars (`dataset.py`), an honest quant baseline to beat (`baseline.py`), and a
scoreboard that evaluates any `/v1/systemone` endpoint or in-process advisor on identical held-
out data (`scoreboard.py`). See `docs/DECISION_MODELS.md` for the end-to-end workflow.
"""

from __future__ import annotations

from jevtrader.jev.finetune.baseline import LogisticBaseline
from jevtrader.jev.finetune.dataset import (
    DatasetConfig,
    DatasetSplits,
    build_examples,
    from_decision_log,
    split_chronological,
    write_jsonl,
)
from jevtrader.jev.finetune.scoreboard import BackendResult, beats, run_scoreboard

__all__ = [
    "BackendResult",
    "DatasetConfig",
    "DatasetSplits",
    "LogisticBaseline",
    "beats",
    "build_examples",
    "from_decision_log",
    "run_scoreboard",
    "split_chronological",
    "write_jsonl",
]
