#!/bin/bash
#SBATCH --job-name=slotgat-fb
#SBATCH --account=eecs
#SBATCH --partition=dgxh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=slotgat_freebase_%j.out
#SBATCH --error=slotgat_freebase_%j.err

set -euo pipefail

REPOSITORY_ROOT="$HOME/hpc-share/idealab/INV-GNN"
SLOTGAT_ROOT="$REPOSITORY_ROOT/src/baselines/SlotGAT"
PYTHON_EXECUTABLE="$REPOSITORY_ROOT/data/.venv/bin/python"

cd "$SLOTGAT_ROOT"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Python: $PYTHON_EXECUTABLE"
nvidia-smi
"$PYTHON_EXECUTABLE" --version

export PYTHONUNBUFFERED=1

"$PYTHON_EXECUTABLE" \
  slotgat_data_augmentation/run_FREEBASE_slotgat_augmentation.py \
  --variants unchanged,exact_2 \
  --seeds 1566911444,20241017,20251017 \
  --super-epochs 100 \
  --patience 30 \
  --label-batch-size 0 \
  --hidden-dim 64 \
  --num-layers 2 \
  --num-heads 8 \
  --edge-feats 64 \
  --dropout-feat 0.5 \
  --dropout-attn 0.2 \
  --slope 0.05 \
  --alpha 0.05 \
  --aggregator SA \
  --sa-att-dim 3 \
  --slotgat-edge-chunk-size 0 \
  --lr 0.005 \
  --weight-decay 0.001 \
  --grad-clip 0 \
  --device cuda \
  --output-dir results/slotgat_augmentation/FREEBASE_original_dgl \
  --resume
