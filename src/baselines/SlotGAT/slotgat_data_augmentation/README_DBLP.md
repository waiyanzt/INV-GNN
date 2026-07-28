# SlotGAT joint graph-variant augmentation: DBLP link prediction

This experiment trains one shared SlotGAT encoder on DBLP1--3 for
paper-conference link prediction:

```text
v1: shared author-paper, paper-conference, and paper-term edges + paper-area
v2: shared author-paper, paper-conference, and paper-term edges + conference-area
v3: shared author-paper, paper-conference, and paper-term edges + author-area
```

All variants use the same paper-disjoint positive splits and fixed negative
paper-conference candidates. Only training-positive paper-conference edges are
present in the message-passing graph.

The training/evaluation protocol is shared with RGCN. The SlotGAT entry point
therefore uses the same graph construction, globally aligned relation
vocabulary, split rows, negative subsampling, dot-product decoder, threshold,
metrics, checkpoint selection, exact resume state, and score-table schema.

## One-time shared-split generation

The existing shared-split script uses paths relative to the MAGNN directory:

```bash
cd ../MAGNN
python paper-venue_shared_splits.py
cd ../SlotGAT
```

This normally creates:

```text
../MAGNN/data/preprocessed/DBLP_shared_splits/DBLP_pc_shared_splits.npz
```

## One-time graph preprocessing

Run from `src/baselines/SlotGAT`:

```bash
python slotgat_data_augmentation/preprocess_DBLP_slotgat_augmentation.py
```

The wrapper finds the shared split under either the RGCN or MAGNN baseline and
writes the reusable graph artifacts to:

```text
../RGCN/data/preprocessed/DBLP_rgcn_v1/
../RGCN/data/preprocessed/DBLP_rgcn_v2/
../RGCN/data/preprocessed/DBLP_rgcn_v3/
```

## Production command

```bash
python slotgat_data_augmentation/run_DBLP_slotgat_augmentation.py \
  --variants v1,v2,v3 \
  --seeds 1566911444,20241017,20251017 \
  --super-epochs 300 \
  --patience 40 \
  --batch-size 0 \
  --neg-per-paper 3 \
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
  --output-dir results/slotgat_augmentation/DBLP \
  --resume
```

`--batch-size 0` selects one full positive-edge update per variant. Therefore:

```text
variant_epochs = optimizer_steps = 3 * completed_super_epochs
```

Validation occurs after all three variants have contributed their update.
Patience 40 consequently means 40 consecutive balanced DBLP1--3 super-epochs
without a lower mean validation BCE.

## Output contract

Each `seed_<seed>/` directory contains:

```text
shared_checkpoint.pt
latest_training_state.pt
training_history.csv
test_scores_v1.csv
test_scores_v2.csv
test_scores_v3.csv
pairwise_invariance.csv
test_metrics_by_variant.csv
summary.json
```

The output root additionally contains `seed_summary.csv` and
`all_seed_summaries.json`. Metrics include AUC, AP, precision, recall, F1,
accuracy, Hits@1/3/5, and MRR. All three score tables for a seed are generated
by the same best shared checkpoint.
