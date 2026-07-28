# SlotGAT joint graph-variant augmentation: IMDb link prediction

This experiment uses one shared SlotGAT encoder for each of the repository's
two IMDb link-prediction tasks:

```text
md: movie-director prediction on v1,v3
ml: movie-link prediction on v1,v2,v3,v4
```

The task-specific model is jointly trained across all selected graph variants.
Every variant uses the same CMPNN-owned positive rows and fixed negative-tail
matrices. The best shared checkpoint is selected by mean validation BCE across
the variants and then evaluated separately on each graph.

The training/evaluation protocol is shared with the RGCN augmentation runner.
This deliberately keeps the graph construction, global relation vocabulary,
splits, negative candidates, decoder, threshold, metrics, resume state, and
score-table schema identical between the two baselines.

## One-time shared-split generation

Run from `src/baselines/SlotGAT`:

```bash
cd ../CMPNN
python build_IMDB_md_shared_splits.py
python build_IMDB_ml_shared_splits.py
cd ../SlotGAT
```

## One-time graph preprocessing

The wrapper writes the shared graph artifacts under the existing RGCN
`data/preprocessed` directory, so RGCN and SlotGAT do not create divergent
copies:

```bash
python slotgat_data_augmentation/preprocess_IMDB_slotgat_lp_augmentation.py \
  --task md

python slotgat_data_augmentation/preprocess_IMDB_slotgat_lp_augmentation.py \
  --task ml
```

The expected outputs are:

```text
../RGCN/data/preprocessed/IMDB_rgcn_lp_md_v1/
../RGCN/data/preprocessed/IMDB_rgcn_lp_md_v3/
../RGCN/data/preprocessed/IMDB_rgcn_lp_ml_v1/
../RGCN/data/preprocessed/IMDB_rgcn_lp_ml_v2/
../RGCN/data/preprocessed/IMDB_rgcn_lp_ml_v3/
../RGCN/data/preprocessed/IMDB_rgcn_lp_ml_v4/
```

## Production: movie-director (`md`)

```bash
python slotgat_data_augmentation/run_IMDB_slotgat_lp_augmentation.py \
  --task md \
  --variants v1,v3 \
  --seeds 1566911444,20241017,20251017 \
  --super-epochs 300 \
  --patience 40 \
  --batch-size 0 \
  --threshold 0.5 \
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
  --emb-reg 0.000001 \
  --grad-clip 0 \
  --device cuda \
  --output-dir results/slotgat_augmentation/IMDB_LP \
  --resume
```

With `--batch-size 0`, `md` makes two optimizer updates per super-epoch: one
full positive-row update on v1 and one on v3, in randomized order.

## Production: movie-link (`ml`)

```bash
python slotgat_data_augmentation/run_IMDB_slotgat_lp_augmentation.py \
  --task ml \
  --variants v1,v2,v3,v4 \
  --seeds 1566911444,20241017,20251017 \
  --super-epochs 300 \
  --patience 40 \
  --batch-size 0 \
  --threshold 0.5 \
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
  --emb-reg 0.000001 \
  --grad-clip 0 \
  --device cuda \
  --output-dir results/slotgat_augmentation/IMDB_LP \
  --resume
```

With `--batch-size 0`, `ml` makes four optimizer updates per super-epoch, one
per graph. Patience counts completed balanced super-epochs because validation
happens once after every selected graph has contributed its update.

The task name is appended automatically, producing separate `md/` and `ml/`
directories beneath the common output directory.

## Output contract

Each `<output-dir>/<task>/seed_<seed>/` contains:

```text
shared_checkpoint.pt
latest_training_state.pt
training_history.csv
test_scores_<variant>.csv
IMDB_slotgat_lp_augmentation_<task>_<variant>_seed<seed>_scores.csv
pairwise_invariance.csv
test_metrics_by_variant.csv
summary.json
```

The task output directory additionally contains `seed_summary.csv` and
`all_seed_summaries.json`. Metrics include AUC, AP, precision, recall, F1,
accuracy, Hits@1/3/5, and MRR. Every per-variant score table for a seed comes
from the same best shared checkpoint.
