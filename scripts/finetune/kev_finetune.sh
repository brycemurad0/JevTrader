#!/usr/bin/env bash
# Fine-tune Kev (github.com/jaredpalmer/kev) on a JevTrader dataset and serve it locally.
#
# Run this on YOUR machine (this sandbox has no GPU/network to do it) -- Apple Silicon (MLX,
# automatic) or an NVIDIA GPU (CUDA) both work; ROCm too. See docs/DECISION_MODELS.md for
# hardware guidance and measured latencies.
#
# Usage:
#   scripts/finetune/kev_finetune.sh <dataset_dir> [base_model] [device]
#
#   dataset_dir  Directory `scripts/finetune/build_dataset.py --out` wrote (train.jsonl,
#                calibration.jsonl, dev.jsonl, test.jsonl, manifest.json).
#   base_model   HF id to fine-tune from, matched to --init_from below. Default: Qwen3.5-4B,
#                a good default for a single consumer GPU or an M-series Mac with ~16GB+ unified
#                memory; use the 0.8B variant for lower latency / smaller boxes, 9B or the
#                Qwen3.8-27B checkpoint if you have the VRAM and want more headroom.
#   device       cuda | mps | rocm. Default: auto-detected by kev itself (MLX on Apple Silicon,
#                torch/bf16 elsewhere) -- set explicitly to force one.
#
# ALWAYS trains with --init_from (never from the base model): starting from Kev's own
# already-tuned checkpoint keeps its general System One skill and only specializes it further,
# per the Kev README (from-base loses that entirely).
set -euo pipefail

DATASET_DIR="${1:?usage: kev_finetune.sh <dataset_dir> [base_model] [device]}"
BASE_MODEL="${2:-Qwen/Qwen3.5-4B-Base}"
INIT_FROM="${3:-jaredpalmer/kev-4b}"
DEVICE="${4:-}"
RUN_NAME="$(basename "$DATASET_DIR")-$(date +%Y%m%d-%H%M%S)"
RUNS_DIR="${KEV_RUNS_DIR:-runs/kev}"
PORT="${KEV_PORT:-8009}"

if [ ! -f "$DATASET_DIR/train.jsonl" ]; then
  echo "error: $DATASET_DIR/train.jsonl not found -- run build_dataset.py first" >&2
  exit 1
fi

if [ ! -d kev ]; then
  echo "==> cloning jaredpalmer/kev"
  git clone --depth 1 https://github.com/jaredpalmer/kev.git kev
fi

cd kev
echo "==> uv sync (base + serve + train extras)"
uv sync --extra serve

DEVICE_FLAG=()
if [ -n "$DEVICE" ]; then
  DEVICE_FLAG=(--device "$DEVICE")
fi

echo "==> training: $BASE_MODEL, init_from=$INIT_FROM, run=$RUN_NAME"
uv run python -m kev.train \
  --data "../$DATASET_DIR/train.jsonl" \
  --base "$BASE_MODEL" \
  --init_from "$INIT_FROM" \
  --epochs 2 \
  --lr 2e-5 \
  --batch 1 \
  --accum 8 \
  --dtype bf16 \
  --checkpointing 1 \
  "${DEVICE_FLAG[@]}" \
  --out "../$RUNS_DIR/$RUN_NAME"

echo "==> benchmarking on held-out test split"
uv run python -m kev.benchmark \
  --run "../$RUNS_DIR/$RUN_NAME" \
  --data "../$DATASET_DIR/test.jsonl" \
  --out "../$RUNS_DIR/$RUN_NAME-eval"

echo
echo "==> eval report: $RUNS_DIR/$RUN_NAME-eval"
echo "==> compare this run's numbers against jevtrader's own scoreboard.py output for the SAME"
echo "    test.jsonl (kev.benchmark's metrics are its own; scoreboard.py's are what decide"
echo "    promotion here) -- see docs/DECISION_MODELS.md."
echo
echo "==> to serve this checkpoint for JevTrader (DECISION_BACKEND=kev):"
echo "    cd kev && uv run --extra serve python -m kev.serve --run ../$RUNS_DIR/$RUN_NAME --port $PORT"
echo "    export DECISION_BACKEND=kev KEV_BASE_URL=http://127.0.0.1:$PORT"
echo
echo "==> to score this SERVER against Jev/Laya/baseline on the same test set:"
echo "    python scripts/finetune/scoreboard.py --test $DATASET_DIR/test.jsonl \\"
echo "        --kev http://127.0.0.1:$PORT --jev --offline --baseline runs/baseline.json \\"
echo "        --out $RUNS_DIR/$RUN_NAME-scoreboard"
