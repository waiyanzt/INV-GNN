#!/usr/bin/env bash
set -euo pipefail

# Universal restrictedK experiment. This is the only standalone exact_2 script
# that needs the generated MAGNN metapath-definition scripts.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR/../../../../../data}"
VARIANTS_ROOT="${VARIANTS_ROOT:-$DATA_ROOT/dataset_variant_3hops_filter}"
MAGNN_ROOT="${MAGNN_ROOT:-$SCRIPT_DIR/../MAGNN/preprocess_scripts/freebase/full_magnn_preprocess_scripts}"
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
  --magnn-preprocess-root "$MAGNN_ROOT" \
  --output-root "$PREPROCESSED_ROOT" \
  --pipeline up_to_exact_2 \
  --flavor restricted_k \
  --k "$K" \
  --channel-identity "$CHANNEL_IDENTITY" \
  --split-seed "$SPLIT_SEED" \
  --require-endpoint-equals-union \
  --overwrite

DATA="$PREPROCESSED_ROOT/up_to_exact_2/k${K}/${CHANNEL_IDENTITY}_channels/restricted_k"
OUT="$RESULTS_ROOT/up_to_exact_2/k${K}/${CHANNEL_IDENTITY}_channels/restricted_k"
CKPT="$CHECKPOINT_ROOT/up_to_exact_2/k${K}/${CHANNEL_IDENTITY}_channels/restricted_k"
mkdir -p "$OUT" "$CKPT"

python run_freebase_magnn_channels.py \
  --data-dir "$DATA" \
  --output-json "$OUT/sehgnn_results.json" \
  --checkpoint-dir "$CKPT" \
  --seeds "$SEEDS" \
  --gpu "$GPU"

python aggregate_freebase_nc_metrics.py \
  --group restrictedK \
  --variants unchanged exact_2 \
  --result "universal=$OUT/sehgnn_results.json" \
  --output-dir "$(dirname "$OUT")/aggregate_restricted_k"
