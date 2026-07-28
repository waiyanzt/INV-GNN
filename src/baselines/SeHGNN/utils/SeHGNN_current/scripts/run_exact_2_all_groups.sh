#!/usr/bin/env bash
set -euo pipefail

# Complete unchanged + exact_2 pipeline:
#   K           : one independently preprocessed/trained model per variant
#   fullK       : all channels through K on the explicit universal graph
#   restrictedK : only MAGNN-union channels through K on that graph
#
# Run from src/baselines/SeHGNN (or invoke this script from anywhere).  The
# repository data root and MAGNN generated-script root below match the paths
# requested for the SlotGAT/SeHGNN baseline location.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

DATA_ROOT="${DATA_ROOT:-/nfs/stak/users/mousavij/hpc-share/gnn/data}"
VARIANTS_ROOT="${VARIANTS_ROOT:-$DATA_ROOT/dataset_variant_3hops_filtered}"
MAGNN_ROOT="${MAGNN_ROOT:-/nfs/stak/users/mousavij/hpc-share/gnn/new_code/INV-GNN/src/baselines/MAGNN/preprocess_scripts/freebase/full_magnn_preprocess_scripts}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-$DATA_ROOT/preprocessed/sehgnn_freebase_magnn}"
RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR/results/sehgnn_freebase_magnn}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$SCRIPT_DIR/checkpoint/sehgnn_freebase_magnn}"
K="${K:-2}"
SPLIT_SEED="${SPLIT_SEED:-1566911444}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-1566911444,20241017,20251017}"
CHANNEL_IDENTITY="${CHANNEL_IDENTITY:-type}"

COMMON_DATA="$PREPROCESSED_ROOT/up_to_exact_2/k${K}/${CHANNEL_IDENTITY}_channels"
COMMON_RESULTS="$RESULTS_ROOT/up_to_exact_2/k${K}/${CHANNEL_IDENTITY}_channels"
COMMON_CKPT="$CHECKPOINT_ROOT/up_to_exact_2/k${K}/${CHANNEL_IDENTITY}_channels"

# MAGNN_ROOT is required only because this combined command includes
# restricted_k.  The K and full_k branches do not read MAGNN files.
#python preprocess_freebase_magnn_channels.py \
#  --variants-root "$VARIANTS_ROOT" \
#  --magnn-preprocess-root "$MAGNN_ROOT" \
#  --output-root "$PREPROCESSED_ROOT" \
#  --pipeline up_to_exact_2 \
#  --flavor all \
#  --k "$K" \
#  --channel-identity "$CHANNEL_IDENTITY" \
#  --split-seed "$SPLIT_SEED" \
#  --require-endpoint-equals-union \
#  --overwrite

for VARIANT in unchanged exact_2; do
  mkdir -p "$COMMON_RESULTS/k/$VARIANT" "$COMMON_CKPT/k/$VARIANT"
  python run_freebase_magnn_channels.py \
    --data-dir "$COMMON_DATA/k/$VARIANT" \
    --output-json "$COMMON_RESULTS/k/$VARIANT/sehgnn_results.json" \
    --checkpoint-dir "$COMMON_CKPT/k/$VARIANT" \
    --seeds "$SEEDS" \
    --gpu "$GPU"
done

for GROUP in full_k restricted_k; do
  mkdir -p "$COMMON_RESULTS/$GROUP" "$COMMON_CKPT/$GROUP"
  python run_freebase_magnn_channels.py \
    --data-dir "$COMMON_DATA/$GROUP" \
    --output-json "$COMMON_RESULTS/$GROUP/sehgnn_results.json" \
    --checkpoint-dir "$COMMON_CKPT/$GROUP" \
    --seeds "$SEEDS" \
    --gpu "$GPU"
done

python aggregate_freebase_nc_metrics.py \
  --group k \
  --variants unchanged exact_2 \
  --result "unchanged=$COMMON_RESULTS/k/unchanged/sehgnn_results.json" \
  --result "exact_2=$COMMON_RESULTS/k/exact_2/sehgnn_results.json" \
  --output-dir "$COMMON_RESULTS/aggregate_k"

python aggregate_freebase_nc_metrics.py \
  --group fullK \
  --variants unchanged exact_2 \
  --result "universal=$COMMON_RESULTS/full_k/sehgnn_results.json" \
  --output-dir "$COMMON_RESULTS/aggregate_full_k"

python aggregate_freebase_nc_metrics.py \
  --group restrictedK \
  --variants unchanged exact_2 \
  --result "universal=$COMMON_RESULTS/restricted_k/sehgnn_results.json" \
  --output-dir "$COMMON_RESULTS/aggregate_restricted_k"

printf '\nCompleted exact_2 K/fullK/restrictedK pipeline.\nResults: %s\n' "$COMMON_RESULTS"
