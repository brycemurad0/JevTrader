# Fine-tuning Kev on a rented GPU (Modal)

If you don't have a local NVIDIA GPU (or a large-enough Apple Silicon Mac) handy, Kev's
fine-tune can run on a rented cloud GPU for roughly $1/run on an H100 via the packaged Modal
skill, while **trading itself still runs entirely locally** -- the cloud GPU is only used to
produce a checkpoint file, which you then download and serve on your own machine (or wherever
`jevtrader/live` runs) with `kev.serve`. Nothing about JevTrader's live/paper loop depends on
Modal or any cloud GPU; it's purely a training-time convenience.

## 1. Build the dataset locally (no GPU needed)

```bash
. .venv/bin/activate
python scripts/finetune/build_dataset.py \
    --bars data/cache/btcusd_bitstamp_1min_2025_2026.csv --symbol BTC/USD \
    --horizon 5min --cost-bps 50 --lookback-bars 120 \
    --out data/cache/finetune/btcusd_5min_taker50bps
```

## 2. Install the Modal skill and fine-tune remotely

```bash
npx skills add jaredpalmer/kev@kev-finetune
```

Follow that skill's own instructions to point it at:
- `data/cache/finetune/btcusd_5min_taker50bps/train.jsonl` (training data)
- a base model + `--init_from` pair, e.g. `Qwen/Qwen3.5-4B-Base` + `jaredpalmer/kev-4b` (always
  use `--init_from`; see `scripts/finetune/kev_finetune.sh`'s comments for why)
- an H100 instance (the skill's documented ~$1/run pricing assumes this)

The skill trains via the same `kev.train` entry point `kev_finetune.sh` calls locally, just
provisioned and billed by Modal instead of running on your own hardware.

## 3. Download the checkpoint and evaluate

Once training completes, download the resulting `runs/<name>` directory (or wherever the skill
places it) to your local machine, then benchmark and score it exactly as
`scripts/finetune/kev_finetune.sh` does after a local run:

```bash
cd kev
uv run python -m kev.benchmark \
    --run /path/to/downloaded/run \
    --data ../data/cache/finetune/btcusd_5min_taker50bps/test.jsonl \
    --out ../runs/modal-eval

cd ..
uv run --extra serve python -m kev.serve --run /path/to/downloaded/run --port 8009 &
export DECISION_BACKEND=kev KEV_BASE_URL=http://127.0.0.1:8009
python scripts/finetune/scoreboard.py \
    --test data/cache/finetune/btcusd_5min_taker50bps/test.jsonl \
    --offline --baseline runs/baseline.json --kev http://127.0.0.1:8009 --jev \
    --out runs/modal-kev-scoreboard
```

## 4. Ongoing serving

After promotion, `kev.serve` runs locally (or on whatever machine hosts JevTrader's live
runner) against the downloaded checkpoint -- there is no ongoing Modal dependency once you have
the weights. Re-running the Modal skill is only needed the next time you want to fine-tune
again (new data, a bigger base model, a different horizon/cost variant, ...).
