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
    balance_classes,
    build_dataset,
    build_examples,
    dedupe_examples,
    from_decision_log,
    load_examples_jsonl,
    split_chronological,
    write_jsonl,
    write_records_jsonl,
)
from jevtrader.jev.finetune.baseline import calibrate_temperature, fit_logistic_baseline, load_baseline, save_baseline
from jevtrader.jev.finetune.scoreboard import BackendResult, beats, results_to_json, results_to_markdown, run_scoreboard

__all__ = [
    "BackendResult",
    "DatasetConfig",
    "DatasetSplits",
    "LogisticBaseline",
    "balance_classes",
    "beats",
    "build_dataset",
    "build_examples",
    "calibrate_temperature",
    "dedupe_examples",
    "fit_logistic_baseline",
    "from_decision_log",
    "load_baseline",
    "load_examples_jsonl",
    "results_to_json",
    "results_to_markdown",
    "run_scoreboard",
    "save_baseline",
    "split_chronological",
    "write_jsonl",
    "write_records_jsonl",
]
