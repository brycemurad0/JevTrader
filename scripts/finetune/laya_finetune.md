# Fine-tuning Laya for JevTrader

Laya (`github.com/NandhaKishorM/laya`, Apache-2.0) ships two checkpoints on Hugging Face
(`convaiinnovations/laya`, 421M, and `convaiinnovations/laya-multilingual`, 322M) and, zero-shot,
answers near chance on domain-specific typed questions like ours -- **it must be fine-tuned**
before it's useful here. Its README's own RLCD (reinforcement learning from contrastive
distillation) fine-tuning notebook is
[`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
-- 4-5 hours on 2x T4 GPUs (a free-tier Kaggle notebook) for ~30k questions.

## 1. Build the dataset

```bash
. .venv/bin/activate
python scripts/finetune/build_dataset.py \
    --bars data/cache/btcusd_bitstamp_1min_2025_2026.csv --symbol BTC/USD \
    --horizon 5min --cost-bps 50 --lookback-bars 120 \
    --out data/cache/finetune/btcusd_5min_taker50bps
```

This writes `train.jsonl` / `calibration.jsonl` / `dev.jsonl` / `test.jsonl` in the
labelled-request shape both Kev and Laya's `/v1/systemone` server speak (`{"state": ...,
"questions": {id: {type, instructions, criteria, "label": ...}}}` -- see
`jevtrader/jev/finetune/dataset.py`'s docstring). **What was verified here, and what wasn't:**
`laya/serve.py`'s own module docstring says it exposes "Laya System-1 decisions over the
TypeSafe Jev `/v1/systemone` protocol" and its request/response models mirror
`typesafe_sdk`'s Choice/Score/Noul shapes exactly (confirmed by reading the installed `laya`
wheel's `serve.py`/`structured.py` in this environment) -- so the WIRE format this dataset
produces is the right one to *serve* a fine-tuned Laya checkpoint at inference time, the same
way it serves a fine-tuned Kev checkpoint. What was **not** verified here (no GPU, and the
RLCD notebook itself wasn't available to inspect in this environment) is whether that Kaggle
notebook's *training* input pipeline consumes this exact JSONL shape unmodified, or expects a
preprocessed/reformatted variant (RLCD training in particular may want preference PAIRS rather
than single labelled examples). **Before a real run**: open the notebook on Kaggle, check its
first data-loading cell against a few lines of `train.jsonl`, and adjust the loader (or add a
small conversion step) if its expected shape differs. If it needs pairs, a cheap way to get them
from this dataset is contrasting two examples with different `direction` labels at the same
`cost_bps`, using state A's label as "preferred" and B's as "dispreferred" for a noul/choice
target -- but confirm against the notebook's actual loader before investing training time.

## 2. Fine-tune on Kaggle (2x T4)

1. Upload `train.jsonl` (and `calibration.jsonl` for validation, if the notebook supports a
   held-out eval split) as a Kaggle dataset.
2. Open `laya_finetune_typed_decisions_2xT4_kaggle.ipynb`, point its data cell at your uploaded
   file(s), and set the base checkpoint to `convaiinnovations/laya` (general-purpose; use
   `-multilingual` only if the state/instructions include non-English text -- they don't, for
   JevTrader's numeric states).
3. Run all cells with a 2x T4 accelerator. Budget 4-5 hours for ~30k questions; this dataset's
   `train.jsonl` line count is in `manifest.json` next to it -- scale time roughly linearly.
4. Download the resulting checkpoint.

## 3. Serve it locally

```bash
pip install "laya[serve]"
LAYA_DEVICE=cuda LAYA_PRELOAD=1 LAYA_MODELS=/path/to/your/finetuned/checkpoint laya-serve
# defaults to 0.0.0.0:8000; override with LAYA_HOST / LAYA_PORT
```

Point JevTrader at it:

```bash
export DECISION_BACKEND=laya
export LAYA_BASE_URL=http://127.0.0.1:8000
# LOCAL_MODEL_API_KEY only needed if you set LAYA_API_KEY on the server
```

## 4. Score it before promoting it

```bash
python scripts/finetune/scoreboard.py \
    --test data/cache/finetune/btcusd_5min_taker50bps/test.jsonl \
    --offline --baseline runs/baseline.json --laya http://127.0.0.1:8000 --jev \
    --out runs/laya-btcusd-5min-scoreboard
```

Promote only if `beats()` says yes against every incumbent you actually run (see
`docs/DECISION_MODELS.md` for the full workflow and the rule itself). Laya's measured latency
(33-40ms/question on a T4, 193-464ms on an Apple GPU) is competitive with Kev's smaller
checkpoints but its accuracy needs to be established empirically on YOUR data via this
scoreboard -- there is no public number for "typed trading decisions" to cite.
