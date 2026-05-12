#!/bin/bash
# Final rerun with 3 seeds on all 6 IMDB variants using the best config
# selected by pick_best.py. Fill in the hyperparameter values on the
# `python run_IMDB_nc.py` line below (copy from pick_best.py output).

#SBATCH --job-name=sehgnn_final
#SBATCH --partition=ampere               
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/SeHGNN_test-%j.out
#SBATCH --error=logs/SeHGNN_test-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sullivk3@oregonstate.edu

set -euo pipefail

# Inside sbatch the script is staged to /var/spool/slurm, so BASH_SOURCE is
# useless here. Use SLURM_SUBMIT_DIR (set by sbatch to the submit CWD) and
# fall back to BASH_SOURCE only when running the script directly.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
fi
cd "$SCRIPT_DIR"

mkdir -p logs

module purge || true
module load cuda/11.7 || true

if [[ -f venv/bin/activate ]]; then
  source venv/bin/activate
else
  echo "ERROR: venv/bin/activate not found in $SCRIPT_DIR" >&2
  exit 1
fi

echo "=============================================="
echo "Job:        ${SLURM_JOB_ID:-<interactive>}"
echo "Node:       $(hostname)"
echo "CWD:        $(pwd)"
echo "Python:     $(which python)"
echo "Start:      $(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo "=============================================="

python -u run_IMDB_nc.py \
    --variant 1 2 3 4 5 6\
    --seeds 1566911444 20241017 20251017 \
    --num-hops 4 \
    --label-num-hops 0 \
    --n-task-layers 1 \
    --input-drop 0.1 \
    --lr 0.005 \
    --weight-decay 0 \

echo "End: $(date)"
