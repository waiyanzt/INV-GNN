#!/usr/bin/env bash
set -euo pipefail

# Convenience launcher for both supported SeHGNN data-augmentation datasets.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_IMDB="${RUN_IMDB:-1}"
RUN_FREEBASE="${RUN_FREEBASE:-1}"

if [[ "$RUN_IMDB" == "1" ]]; then
  (
    export VARIANTS="${IMDB_VARIANTS:-1,2,3,4}"
    export SEEDS="${IMDB_SEEDS:-1566911444,20241017,20251017}"
    export EXPECTED_RUNS="${IMDB_EXPECTED_RUNS:-$(awk -F, '{print NF}' <<< "$SEEDS")}"
    export DATA_ROOT="${IMDB_DATA_ROOT:-}"
    export RESULTS_ROOT="${IMDB_RESULTS_ROOT:-}"
    if [[ -z "$RESULTS_ROOT" ]]; then unset RESULTS_ROOT; fi
    bash "$SCRIPT_DIR/run_IMDB_nc_augmentation.sh"
  )
fi

if [[ "$RUN_FREEBASE" == "1" ]]; then
  (
    export VARIANTS="${FREEBASE_VARIANTS:-unchanged,exact_2,exact_3,range_2_3}"
    export SEEDS="${FREEBASE_SEEDS:-1566911444,20241017,20251017,20261017}"
    export EXPECTED_RUNS="${FREEBASE_EXPECTED_RUNS:-$(awk -F, '{print NF}' <<< "$SEEDS")}"
    export VARIANTS_ROOT="${FREEBASE_VARIANTS_ROOT:-${VARIANTS_ROOT:-}}"
    export PREPROCESSED_ROOT="${FREEBASE_PREPROCESSED_ROOT:-${PREPROCESSED_ROOT:-}}"
    export RESULTS_ROOT_BASE="${FREEBASE_RESULTS_ROOT_BASE:-${RESULTS_ROOT_BASE:-}}"
    [[ -z "$VARIANTS_ROOT" ]] && unset VARIANTS_ROOT
    [[ -z "$PREPROCESSED_ROOT" ]] && unset PREPROCESSED_ROOT
    [[ -z "$RESULTS_ROOT_BASE" ]] && unset RESULTS_ROOT_BASE
    bash "$SCRIPT_DIR/run_freebase_nc_k_augmentation.sh"
  )
fi
