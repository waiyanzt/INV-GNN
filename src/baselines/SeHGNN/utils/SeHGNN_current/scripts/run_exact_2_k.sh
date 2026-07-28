#!/usr/bin/env bash
set -euo pipefail

# Per-variant K experiment. No MAGNN files are required.
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
  --flavor k \
  --k "$K" \
  --channel-identity "$CHANNEL_IDENTITY" \
  --split-seed "$SPLIT_SEED" \
  --overwrite

DATA="$PREPROCESSED_ROOT/up_to_exact_2/k${K}/${CHANNEL_IDENTITY}_channels"
OUT="$RESULTS_ROOT/up_to_exact_2/k${K}/${CHANNEL_IDENTITY}_channels"
CKPT="$CHECKPOINT_ROOT/up_to_exact_2/k${K}/${CHANNEL_IDENTITY}_channels"

for V in unchanged exact_2; do
  mkdir -p "$OUT/k/$V" "$CKPT/k/$V"
  python run_freebase_magnn_channels.py \
    --data-dir "$DATA/k/$V" \
    --output-json "$OUT/k/$V/sehgnn_results.json" \
    --checkpoint-dir "$CKPT/k/$V" \
    --seeds "$SEEDS" \
    --gpu "$GPU"
done

python aggregate_freebase_nc_metrics.py \
  --group k \
  --variants unchanged exact_2 \
  --result "unchanged=$OUT/k/unchanged/sehgnn_results.json" \
  --result "exact_2=$OUT/k/exact_2/sehgnn_results.json" \
  --output-dir "$OUT/aggregate_k"
