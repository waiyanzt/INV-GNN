#!/usr/bin/env bash
set -euo pipefail

# Shared-checkpoint SeHGNN data augmentation over the four original IMDB NC
# graphs. Restricted/full skip representations are intentionally not used.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

VARIANTS="${VARIANTS:-1,2,3,4}"
SEEDS="${SEEDS:-1566911444,20241017,20251017}"
GPU="${GPU:-0}"
RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR/results/sehgnn_augmentation/IMDB_NC_original}"
DATA_ROOT="${DATA_ROOT:-}"
EXPECTED_RUNS="${EXPECTED_RUNS:-$(awk -F, '{print NF}' <<< "$SEEDS")}"
RESUME="${RESUME:-0}"
CPU="${CPU:-0}"

run_args=(
  --variants "$VARIANTS"
  --seeds "$SEEDS"
  --gpu "$GPU"
  --output-dir "$RESULTS_ROOT"
)
[[ -n "$DATA_ROOT" ]] && run_args+=(--data-root "$DATA_ROOT")
[[ "$RESUME" == "1" ]] && run_args+=(--resume)
[[ "$CPU" == "1" ]] && run_args+=(--cpu)

# Extra SeHGNN options can be supplied verbatim, for example:
# EXTRA_ARGS="--super-epochs 300 --patience 50"
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=( ${EXTRA_ARGS} )
  run_args+=("${extra_args[@]}")
fi

python run_IMDB_nc_augmentation.py "${run_args[@]}"
python aggregate_sehgnn_augmentation_metrics.py \
  --input-dir "$RESULTS_ROOT" \
  --output-dir "$RESULTS_ROOT/aggregate" \
  --variants "$VARIANTS" \
  --seeds "$SEEDS" \
  --expected-runs "$EXPECTED_RUNS"

echo "[OK] IMDB NC shared augmentation: $RESULTS_ROOT"
