#!/bin/bash
# Run SeHGNN IMDB Kendall tau on all variant pairs across 3 seeds.
# Submit from src/baselines/SeHGNN:
#   sbatch run_kendall_tau_IMDB_nc.sbatch

#SBATCH --job-name=sehgnn_kendall
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/kendall_tau_imdb-%j.out
#SBATCH --error=logs/kendall_tau_imdb-%j.err
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

SEEDS=(1566911444 20241017 20251017)
PAIRS=(
  "1 2"
  "1 3"
  "1 4"
  "2 3"
  "2 4"
  "3 4"
)

for pair in "${PAIRS[@]}"; do
  read -r V1 V2 <<< "$pair"
  echo
  echo "----------------------------------------------"
  echo "Running Kendall tau for variants ${V1} vs ${V2}"
  echo "Seeds: ${SEEDS[*]}"
  echo "----------------------------------------------"

  python -u kendall_tau_IMDB_nc.py \
    --variant "$V1" "$V2" \
    --seeds "${SEEDS[@]}"
done

echo "End: $(date)"
