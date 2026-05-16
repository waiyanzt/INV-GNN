#!/usr/bin/env bash
# Master orchestrator for all baseline experiments.
#
# Usage:
#   ./run_all.sh                                 # everything (long-running)
#   ./run_all.sh --task imdb_nc                  # only IMDB node classification
#   ./run_all.sh --task dblp_lp --baseline magnn # only MAGNN DBLP LP
#   ./run_all.sh --preprocess-only               # all preprocess, no training
#   ./run_all.sh --train-only                    # assumes preprocessed data exists
#
# Each baseline directory has a ``data`` symlink to ``../../../data`` so the
# hardcoded ``data/raw/...`` paths in the scripts work when invoked from
# inside the baseline folder.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK="all"
BASELINE="all"
PREPROCESS=1
TRAIN=1
SEEDS="1566911444,20241017,20251017"

usage() {
  cat <<EOF
Usage: $0 [--task TASK] [--baseline BASE] [--preprocess-only|--train-only] [--seeds CSV]

  --task        all | imdb_nc | dblp_lp | imdb_lp   (default: all)
  --baseline    all | magnn   | rgcn    | cmpnn   | sehgnn | slotgat  (default: all)
  --preprocess-only / --train-only
  --seeds CSV   comma-separated seeds (default: $SEEDS)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --baseline) BASELINE="$2"; shift 2 ;;
    --preprocess-only) TRAIN=0; shift ;;
    --train-only) PREPROCESS=0; shift ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown flag: $1" >&2; usage; exit 1 ;;
  esac
done

want_task()     { [[ "$TASK" == "all" || "$TASK" == "$1" ]]; }
want_baseline() { [[ "$BASELINE" == "all" || "$BASELINE" == "$1" ]]; }


step() { echo; echo ">>> [$(date +%H:%M:%S)] $*"; }

run_in() {
  local dir="$1"; shift
  step "(cd $dir && $*)"
  (cd "$dir" && "$@")
}

############################################################
# IMDB node classification
############################################################
if want_task imdb_nc; then
  if want_baseline magnn; then
    if [[ "$PREPROCESS" -eq 1 ]]; then
      run_in src/baselines/MAGNN python preprocess_IMDB_star.py
      run_in src/baselines/MAGNN python preprocess_IMDB_star_t.py
      run_in src/baselines/MAGNN python preprocess_IMDB_star_t_2.py
      run_in src/baselines/MAGNN python preprocess_IMDB_star_t_3.py
      run_in src/baselines/MAGNN python preprocess_IMDB_skip.py
    fi
    if [[ "$TRAIN" -eq 1 ]]; then
      run_in src/baselines/MAGNN python run_IMDB.py
      run_in src/baselines/MAGNN python run_IMDB_skip.py
    fi
  fi
  if want_baseline rgcn; then
    if [[ "$PREPROCESS" -eq 1 ]]; then
      run_in src/baselines/RGCN python preprocess_IMDB_rgcn.py --variant v1,v2,v3,v4
      run_in src/baselines/RGCN python preprocess_IMDB_rgcn_skip.py --variant v1,v2,v3,v4
    fi
    if [[ "$TRAIN" -eq 1 ]]; then
      run_in src/baselines/RGCN python run_IMDB_rgcn.py --variants v1,v2,v3,v4 --seeds "$SEEDS"
      run_in src/baselines/RGCN python run_IMDB_rgcn_skip.py --variants v1,v2,v3,v4 --seeds "$SEEDS"
    fi
  fi
  if want_baseline sehgnn; then
    if [[ "$PREPROCESS" -eq 1 ]]; then
      run_in src/baselines/SeHGNN python preprocess_IMDB.py
    fi
    if [[ "$TRAIN" -eq 1 ]]; then
      run_in src/baselines/SeHGNN python run_IMDB_nc.py --seeds "$SEEDS"
    fi
  fi
  if want_baseline slotgat; then
    if [[ "$PREPROCESS" -eq 1 ]]; then
      run_in src/baselines/SlotGAT python preprocess_IMDB.py
    fi
    if [[ "$TRAIN" -eq 1 ]]; then
      run_in src/baselines/SlotGAT python run_IMDB_nc.py --seeds "$SEEDS"
    fi
  fi
fi

############################################################
# DBLP link prediction
############################################################
if want_task dblp_lp; then
  # Shared splits are needed by RGCN and CMPNN DBLP preprocess.
  if [[ "$PREPROCESS" -eq 1 ]]; then
    run_in src/baselines/MAGNN python paper-venue_shared_splits.py
  fi
  if want_baseline magnn; then
    if [[ "$PREPROCESS" -eq 1 ]]; then
      run_in src/baselines/MAGNN python preprocess_DBLP_pc_trainpc.py --variants v1,v2,v3
      run_in src/baselines/MAGNN python preprocess_DBLP_skip.py --variant all
    fi
    if [[ "$TRAIN" -eq 1 ]]; then
      run_in src/baselines/MAGNN python run_DBLP_pc_trainpc.py --variants v1,v2,v3 --seeds "$SEEDS"
      run_in src/baselines/MAGNN python run_DBLP_skip.py --variants v1,v2,v3 --seeds "$SEEDS"
    fi
  fi
  if want_baseline rgcn; then
    if [[ "$PREPROCESS" -eq 1 ]]; then
      run_in src/baselines/RGCN python preprocess_DBLP_rgcn.py --variant v1,v2,v3
      run_in src/baselines/RGCN python preprocess_DBLP_rgcn_skip.py --variant v1,v2,v3
    fi
    if [[ "$TRAIN" -eq 1 ]]; then
      run_in src/baselines/RGCN python run_DBLP_rgcn.py --variants v1,v2,v3 --seeds "$SEEDS"
      run_in src/baselines/RGCN python run_DBLP_rgcn_skip.py --variants v1,v2,v3 --seeds "$SEEDS"
    fi
  fi
  if want_baseline cmpnn; then
    if [[ "$PREPROCESS" -eq 1 ]]; then
      run_in src/baselines/CMPNN python preprocess_DBLP_cmpnn_pc.py --variant v1,v2,v3
      run_in src/baselines/CMPNN python preprocess_DBLP_cmpnn_skip.py --variant v1,v2,v3
    fi
    if [[ "$TRAIN" -eq 1 ]]; then
      run_in src/baselines/CMPNN python run_DBLP_cmpnn_pc.py --variants v1,v2,v3 --seeds "$SEEDS"
      run_in src/baselines/CMPNN python run_DBLP_cmpnn_skip.py --variants v1,v2,v3 --seeds "$SEEDS"
    fi
  fi
fi

############################################################
# IMDB link prediction
############################################################
if want_task imdb_lp; then
  if [[ "$PREPROCESS" -eq 1 ]]; then
    # CMPNN builds the shared IMDB md/ml splits npz that RGCN + MAGNN consume.
    run_in src/baselines/CMPNN python build_IMDB_md_shared_splits.py
    run_in src/baselines/CMPNN python build_IMDB_ml_shared_splits.py
  fi
  if want_baseline magnn; then
    if [[ "$PREPROCESS" -eq 1 ]]; then
      run_in src/baselines/MAGNN python preprocess_IMDB_magnn_lp.py --task md --variant v1,v3
      run_in src/baselines/MAGNN python preprocess_IMDB_magnn_lp.py --task ml --variant v1,v2,v3,v4
      run_in src/baselines/MAGNN python preprocess_IMDB_magnn_lp_skip.py --task md --variant v1,v3
      run_in src/baselines/MAGNN python preprocess_IMDB_magnn_lp_skip.py --task ml --variant v1,v2,v3,v4
    fi
    if [[ "$TRAIN" -eq 1 ]]; then
      run_in src/baselines/MAGNN python run_IMDB_magnn_lp.py --task md --variants v1,v3 --seeds "$SEEDS"
      run_in src/baselines/MAGNN python run_IMDB_magnn_lp.py --task ml --variants v1,v2,v3,v4 --seeds "$SEEDS"
      run_in src/baselines/MAGNN python run_IMDB_magnn_lp_skip.py --task md --variants v1,v3 --seeds "$SEEDS"
      run_in src/baselines/MAGNN python run_IMDB_magnn_lp_skip.py --task ml --variants v1,v2,v3,v4 --seeds "$SEEDS"
    fi
  fi
  if want_baseline rgcn; then
    if [[ "$PREPROCESS" -eq 1 ]]; then
      run_in src/baselines/RGCN python preprocess_IMDB_rgcn_lp.py --task md --variant v1,v3 \
        --shared-npz ../CMPNN/IMDB_md_shared_splits.npz
      run_in src/baselines/RGCN python preprocess_IMDB_rgcn_lp.py --task ml --variant v1,v2,v3,v4 \
        --shared-npz ../CMPNN/IMDB_ml_shared_splits.npz
      run_in src/baselines/RGCN python preprocess_IMDB_rgcn_lp_skip.py --task md --variant v1,v3 \
        --shared-npz ../CMPNN/IMDB_md_shared_splits.npz
      run_in src/baselines/RGCN python preprocess_IMDB_rgcn_lp_skip.py --task ml --variant v1,v2,v3,v4 \
        --shared-npz ../CMPNN/IMDB_ml_shared_splits.npz
    fi
    if [[ "$TRAIN" -eq 1 ]]; then
      run_in src/baselines/RGCN python run_IMDB_rgcn_lp.py --task md --variants v1,v3 --seeds "$SEEDS"
      run_in src/baselines/RGCN python run_IMDB_rgcn_lp.py --task ml --variants v1,v2,v3,v4 --seeds "$SEEDS"
      run_in src/baselines/RGCN python run_IMDB_rgcn_lp_skip.py --task md --variants v1,v3 --seeds "$SEEDS"
      run_in src/baselines/RGCN python run_IMDB_rgcn_lp_skip.py --task ml --variants v1,v2,v3,v4 --seeds "$SEEDS"
    fi
  fi
  if want_baseline cmpnn; then
    if [[ "$PREPROCESS" -eq 1 ]]; then
      run_in src/baselines/CMPNN python preprocess_IMDB_cmpnn_lp_skip.py --task md --variant v1,v3 \
        --shared-npz IMDB_md_shared_splits.npz
      run_in src/baselines/CMPNN python preprocess_IMDB_cmpnn_lp_skip.py --task ml --variant v1,v2,v3,v4 \
        --shared-npz IMDB_ml_shared_splits.npz
    fi
    if [[ "$TRAIN" -eq 1 ]]; then
      for v in v1 v3; do
        run_in src/baselines/CMPNN python run_CMPNN_IMDB_md.py --variant "$v" --seeds "$SEEDS"
      done
      for v in v1 v2 v3 v4; do
        run_in src/baselines/CMPNN python run_CMPNN_IMDB_ml.py --variant "$v" --seeds "$SEEDS"
      done
      run_in src/baselines/CMPNN python run_CMPNN_IMDB_md_skip.py --variants v1,v3 --seeds "$SEEDS"
      run_in src/baselines/CMPNN python run_CMPNN_IMDB_ml_skip.py --variants v1,v2,v3,v4 --seeds "$SEEDS"
    fi
  fi
fi

echo
echo "All requested steps complete."
