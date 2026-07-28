#!/usr/bin/env bash
set -euo pipefail

# Universal fullK experiment. No MAGNN files are required.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR/../../../../../data}"
VARIANTS_ROOT="${VARIANTS_ROOT:-$DATA_ROOT/dataset_variant_3hops_filter}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-$DATA_ROOT/preprocessed/sehgnn_freebase_magnn}"
RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR/results/sehgnn_freebase_magnn}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$SCRIPT_DIR/checkpoint/sehgnn_freebase_magnn}"
K="${K:-2}"
SPLIT_SEED="${SPLIT_SEED:-1566911444}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-1566911444,20241017,20251017,20261017}"
CHANNEL_IDENTITY="${CHANNEL_IDENTITY:-type}"

python preprocess_freebase_magnn_channels.py \
  --variants-root "$VARIANTS_ROOT" \
  --output-root "$PREPROCESSED_ROOT" \
  --pipeline up_to_exact_2 \
  --flavor full_k \
  --k "$K" \
  --channel-identity "$CHANNEL_IDENTITY" \
  --split-seed "$SPLIT_SEED" \
  --require-endpoint-equals-union \
  --overwrite

DATA="$PREPROCESSED_ROOT/up_to_exact_2/k${K}/${CHANNEL_IDENTITY}_channels/full_k"
OUT="$RESULTS_ROOT/up_to_exact_2/k${K}/${CHANNEL_IDENTITY}_channels/full_k"
CKPT="$CHECKPOINT_ROOT/up_to_exact_2/k${K}/${CHANNEL_IDENTITY}_channels/full_k"
mkdir -p "$OUT" "$CKPT"

python run_freebase_magnn_channels.py \
  --data-dir "$DATA" \
  --output-json "$OUT/sehgnn_results.json" \
  --checkpoint-dir "$CKPT" \
  --seeds "$SEEDS" \
  --gpu "$GPU"

python aggregate_freebase_nc_metrics.py \
  --group fullK \
  --variants unchanged exact_2 \
  --result "universal=$OUT/sehgnn_results.json" \
  --output-dir "$(dirname "$OUT")/aggregate_full_k"
