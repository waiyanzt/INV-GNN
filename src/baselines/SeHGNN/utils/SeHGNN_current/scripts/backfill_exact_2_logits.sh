#!/usr/bin/env bash
set -euo pipefail

# Recover raw logits and memory metadata from already-trained exact_2 checkpoints.
# Static memory is exact; GPU peaks are replayed on the current GPU without full retraining.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

RESULTS_ROOT="${RESULTS_ROOT:-$SCRIPT_DIR/results/sehgnn_freebase_magnn}"
K="${K:-2}"
GPU="${GPU:-0}"
CHANNEL_IDENTITY="${CHANNEL_IDENTITY:-type}"
COMMON_RESULTS="$RESULTS_ROOT/up_to_exact_2/k${K}/${CHANNEL_IDENTITY}_channels"

RESULT_FILES=(
  "$COMMON_RESULTS/k/unchanged/sehgnn_results.json"
  "$COMMON_RESULTS/k/exact_2/sehgnn_results.json"
  "$COMMON_RESULTS/full_k/sehgnn_results.json"
  "$COMMON_RESULTS/restricted_k/sehgnn_results.json"
)

EXISTING=()
for path in "${RESULT_FILES[@]}"; do
  if [[ -f "$path" ]]; then
    EXISTING+=("$path")
  else
    printf 'Skipping missing result: %s\n' "$path"
  fi
done

if [[ ${#EXISTING[@]} -eq 0 ]]; then
  echo "No result JSON files were found under $COMMON_RESULTS" >&2
  exit 1
fi

python backfill_freebase_nc_logits.py \
  --result-json "${EXISTING[@]}" \
  --gpu "$GPU" \
  --overwrite-memory

if [[ -f "${RESULT_FILES[0]}" && -f "${RESULT_FILES[1]}" ]]; then
  python aggregate_freebase_nc_metrics.py \
    --group k \
    --variants unchanged exact_2 \
    --result "unchanged=${RESULT_FILES[0]}" \
    --result "exact_2=${RESULT_FILES[1]}" \
    --output-dir "$COMMON_RESULTS/aggregate_k"
fi

if [[ -f "${RESULT_FILES[2]}" ]]; then
  python aggregate_freebase_nc_metrics.py \
    --group fullK \
    --variants unchanged exact_2 \
    --result "universal=${RESULT_FILES[2]}" \
    --output-dir "$COMMON_RESULTS/aggregate_full_k"
fi

if [[ -f "${RESULT_FILES[3]}" ]]; then
  python aggregate_freebase_nc_metrics.py \
    --group restrictedK \
    --variants unchanged exact_2 \
    --result "universal=${RESULT_FILES[3]}" \
    --output-dir "$COMMON_RESULTS/aggregate_restricted_k"
fi