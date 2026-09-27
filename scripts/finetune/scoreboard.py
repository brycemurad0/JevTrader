#!/usr/bin/env python
"""CLI: score any mix of Jev / a local Kev server / a local Laya server / OfflineJevAdvisor /
a saved quant baseline on the SAME held-out test set, and (with exactly two backends) check
whether the second `beats()` the first.

Usage (after `scripts/finetune/build_dataset.py` and, on your own GPU/Apple Silicon machine,
`scripts/finetune/kev_finetune.sh` or the Laya notebook in `scripts/finetune/laya_finetune.md`):

    . .venv/bin/activate
    python scripts/finetune/scoreboard.py \\
        --test data/cache/finetune/btcusd_5min_taker50bps/test.jsonl \\
        --offline --baseline runs/baseline.json --kev http://127.0.0.1:8009 --jev \\
        --out runs/btcusd_5min_taker50bps-scoreboard

Writes `scoreboard.json` and `scoreboard.md` to `--out`; with exactly two `--*` backends given,
also writes `beats.json` (the `beats()` verdict of the SECOND named backend against the FIRST).
`--test` must be a JSONL file `dataset.write_jsonl` wrote (i.e. carrying `"_meta"`); a file
trimmed for training only (`--no-meta`) can't be scored this way.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `jevtrader` imports without an install

from jevtrader.jev.advisor import JevAdvisor, OfflineJevAdvisor  # noqa: E402
from jevtrader.jev.backends import DecisionBackend, make_client  # noqa: E402
from jevtrader.jev.finetune.baseline import load_baseline  # noqa: E402
from jevtrader.jev.finetune.dataset import load_examples_jsonl  # noqa: E402
from jevtrader.jev.finetune.scoreboard import beats, results_to_json, results_to_markdown, run_scoreboard  # noqa: E402


def _remote_predictor(backend: str, url: str, *, latency_budget_ms: float, settings=None) -> JevAdvisor:
    client = make_client(backend, settings=settings, base_url=(url or None), timeout=max(1.0, latency_budget_ms / 1000.0) + 5.0)
    # a generous latency_budget_ms here so a slow-but-correct answer is still scored (tagged
    # late, excluded only from `coverage_at_budget`) rather than discarded outright.
    return JevAdvisor(client, latency_budget_ms=latency_budget_ms, cache_ttl_s=0.0)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", required=True, help="test.jsonl written by build_dataset.py (carries _meta)")
    ap.add_argument("--jev", action="store_true", help="include hosted Jev (needs TYPESAFE_API_KEY)")
    ap.add_argument("--kev", nargs="?", const="", default=None, metavar="URL", help="include a Kev server (default $KEV_BASE_URL or http://127.0.0.1:8009)")
    ap.add_argument("--laya", nargs="?", const="", default=None, metavar="URL", help="include a Laya server (default $LAYA_BASE_URL or http://127.0.0.1:8000)")
    ap.add_argument("--offline", action="store_true", help="include OfflineJevAdvisor (the NOT-Jev heuristic)")
    ap.add_argument("--baseline", metavar="PATH", help="include a saved LogisticBaseline (finetune.baseline.save_baseline)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--latency-budget-ms", type=float, default=2000.0, help="generous budget for scoring, NOT the trading-loop one")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args(argv)

    examples = load_examples_jsonl(args.test)
    if not examples:
        ap.error(f"{args.test}: no examples")

    settings = None
    if args.jev:
        from jevtrader.config import load_settings

        settings = load_settings()

    predictors: dict[str, object] = {}
    if args.offline:
        predictors["offline"] = OfflineJevAdvisor()
    if args.baseline:
        predictors["baseline"] = load_baseline(args.baseline)
    if args.jev:
        predictors["jev"] = _remote_predictor("jev", "", latency_budget_ms=args.latency_budget_ms, settings=settings)
    if args.kev is not None:
        predictors["kev"] = _remote_predictor("kev", args.kev, latency_budget_ms=args.latency_budget_ms)
    if args.laya is not None:
        predictors["laya"] = _remote_predictor("laya", args.laya, latency_budget_ms=args.latency_budget_ms)

    if not predictors:
        ap.error("give at least one of --jev / --kev / --laya / --offline / --baseline")

    print(f"scoring {list(predictors)} on {len(examples)} examples from {args.test}")
    results = run_scoreboard(examples, predictors, threshold=args.threshold, n_bins=args.n_bins, latency_budget_ms=args.latency_budget_ms)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = results_to_markdown(results)
    (out_dir / "scoreboard.json").write_text(json.dumps(results_to_json(results), indent=2))
    (out_dir / "scoreboard.md").write_text(md)
    print(md)

    if len(predictors) == 2:
        incumbent_name, candidate_name = list(predictors)
        verdict = beats(results[candidate_name], results[incumbent_name])
        (out_dir / "beats.json").write_text(json.dumps(verdict, indent=2))
        print(f"\ndoes {candidate_name!r} beat {incumbent_name!r}? {verdict['beats']} -- {verdict['reasons']}")

    for advisor in predictors.values():
        close = getattr(advisor, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
