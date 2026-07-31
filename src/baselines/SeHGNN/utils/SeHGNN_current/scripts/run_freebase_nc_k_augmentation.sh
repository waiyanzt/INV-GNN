#!/usr/bin/env bash
set -euo pipefail

# Shared-checkpoint SeHGNN data augmentation over original/native per-variant
# K channels. The canonical union aligns model parameters only. This script
# never creates or trains restricted_k/full_k representations or a union graph.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

DATA_ROOT_BASE="${DATA_ROOT_BASE:-$SCRIPT_DIR/../../../../../data}"
DEFAULT_VARIANTS_ROOT="$DATA_ROOT_BASE/raw/dataset_variant_3hops_filter"
if [[ ! -d "$DEFAULT_VARIANTS_ROOT" && -d "$DATA_ROOT_BASE/dataset_variant_3hops_filter" ]]; then
  DEFAULT_VARIANTS_ROOT="$DATA_ROOT_BASE/dataset_variant_3hops_filter"
fi
VARIANTS_ROOT="${VARIANTS_ROOT:-$DEFAULT_VARIANTS_ROOT}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-$DATA_ROOT_BASE/preprocessed/sehgnn_freebase_magnn}"
RESULTS_ROOT_BASE="${RESULTS_ROOT_BASE:-$SCRIPT_DIR/results/sehgnn_augmentation/FREEBASE_NC}"
PIPELINE="${PIPELINE:-up_to_exact_2}"
VARIANTS="${VARIANTS:-unchanged,exact_2}"
K="${K:-2}"
CHANNEL_IDENTITY="${CHANNEL_IDENTITY:-type}"
SPLIT_SEED="${SPLIT_SEED:-1566911444}"
SEEDS="${SEEDS:-1566911444,20241017,20251017}"
EXPECTED_RUNS="${EXPECTED_RUNS:-$(awk -F, '{print NF}' <<< "$SEEDS")}"
GPU="${GPU:-0}"
RUN_PREPROCESS="${RUN_PREPROCESS:-0}"
OVERWRITE_PREPROCESS="${OVERWRITE_PREPROCESS:-0}"
RESUME="${RESUME:-0}"
CPU="${CPU:-0}"

IFS=',' read -r -a variant_array <<< "$VARIANTS"

if [[ "$RUN_PREPROCESS" == "1" ]]; then
  preprocess_args=(
    --variants-root "$VARIANTS_ROOT"
    --output-root "$PREPROCESSED_ROOT"
    --pipeline "$PIPELINE"
    --variants "${variant_array[@]}"
    --flavor k
    --k "$K"
    --channel-identity "$CHANNEL_IDENTITY"
    --split-seed "$SPLIT_SEED"
  )
  [[ "$OVERWRITE_PREPROCESS" == "1" ]] && preprocess_args+=(--overwrite)
  python preprocess_freebase_magnn_channels.py "${preprocess_args[@]}"
fi

DATA_ROOT="$PREPROCESSED_ROOT/$PIPELINE/k${K}/${CHANNEL_IDENTITY}_channels"
RESULTS_ROOT="$RESULTS_ROOT_BASE/$PIPELINE/k${K}/${CHANNEL_IDENTITY}_channels"

run_args=(
  --data-root "$DATA_ROOT"
  --variants "$VARIANTS"
  --seeds "$SEEDS"
  --gpu "$GPU"
  --output-dir "$RESULTS_ROOT"
)
[[ "$RESUME" == "1" ]] && run_args+=(--resume)
[[ "$CPU" == "1" ]] && run_args+=(--cpu)

# Extra SeHGNN options can be supplied verbatim, for example:
# EXTRA_ARGS="--super-epochs 300 --patience 50 --allow-large-model"
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=( ${EXTRA_ARGS} )
  run_args+=("${extra_args[@]}")
fi

python run_freebase_magnn_channels_augmentation.py "${run_args[@]}"
python aggregate_sehgnn_augmentation_metrics.py \
  --input-dir "$RESULTS_ROOT" \
  --output-dir "$RESULTS_ROOT/aggregate" \
  --variants "$VARIANTS" \
  --seeds "$SEEDS" \
  --expected-runs "$EXPECTED_RUNS"

echo "[OK] Freebase NC K-channel shared augmentation: $RESULTS_ROOT"
