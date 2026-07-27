# SlotGAT joint graph-variant augmentation: WordNet link prediction

This first experiment trains one shared SlotGAT encoder and DistMult decoder
over three selected WordNet training-graph variants:

```text
no_changes
universal_edges
all_inverse_edges
```

`transitive_edges` remains available in the prepared four-variant NPZ but is
intentionally excluded from this run. It does not contribute training updates,
checkpoint selection, test metrics, or fixed-candidate negative exclusions.

A super-epoch visits every selected variant once in randomized order. With the
default full-triple batch, this is three optimizer updates per super-epoch.
Validation checkpointing maximizes mean filtered MRR across the three selected
variants.

WordNet contains one node type, so SlotGAT has one semantic slot. Its learned
edge-type attention remains active, but multi-node-type slot aggregation is
necessarily degenerate on this dataset. Directed variant edges are preserved;
the runner adds only a dedicated structural self-loop edge type.

## Environment and data

Run from `src/baselines/SlotGAT`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export DGLBACKEND=pytorch
```

The expected source directory is either:

```text
data/wordnet_3hops_augmented_full/
data/raw/wordnet_3hops_augmented_full/
```

It must contain `original_splits/`, `shared_relations.dict`, and the four
variant directories. The wrappers discover either repository-level layout.

## Preprocess once

```bash
python slotgat_data_augmentation/preprocess_WORDNET_slotgat_augmentation.py
```

This writes `wordnet_splits.npz` beside the source data. It preserves the
official train/validation/test split and checks direct and inverse held-out
leakage. The NPZ is intentionally shared with the RGCN comparison.

## One-super-epoch HPC smoke test

First verify the encoder and DGL build on the allocated GPU:

```bash
python slotgat_data_augmentation/verify_WORDNET_slotgat.py --device cuda
```

Then use a disposable output directory for one complete joint super-epoch:

```bash
python slotgat_data_augmentation/run_WORDNET_slotgat_augmentation.py \
  --variants no_changes,universal_edges,all_inverse_edges \
  --seeds 1566911444 \
  --super-epochs 1 \
  --eval-interval 1 \
  --patience 1 \
  --batch-size 0 \
  --device cuda \
  --output-dir results/smoke/slotgat_augmentation/WORDNET
```

## Production command

```bash
python slotgat_data_augmentation/run_WORDNET_slotgat_augmentation.py \
  --variants no_changes,universal_edges,all_inverse_edges \
  --seeds 1566911444,20241017,20251017 \
  --super-epochs 3000 \
  --eval-interval 1 \
  --patience 30 \
  --batch-size 0 \
  --neg-per-pos 1 \
  --hidden-dim 64 \
  --num-layers 2 \
  --num-heads 8 \
  --edge-feats 64 \
  --dropout-feat 0.5 \
  --dropout-attn 0.2 \
  --slope 0.05 \
  --alpha 0.05 \
  --lr 0.005 \
  --weight-decay 0.001 \
  --grad-clip 0 \
  --eval-batch-size 512 \
  --binary-k 50 \
  --score-batch-size 65536 \
  --fixed-negatives 50 \
  --candidate-seed 1566911444 \
  --candidate-known-scope selected \
  --device cuda \
  --resume
```

`--resume` restores the model, optimizer, all RNG states, early stopping,
history, counters, elapsed runtime, and memory telemetry from the latest
completed super-epoch. Use one stable output directory for production.

## Outputs

Each `seed_<seed>/` directory contains:

```text
shared_checkpoint.pt
latest_training_state.pt
training_history.csv
legacy_test_metrics_by_variant.csv
shared_candidate_metrics_by_variant.csv
shared_candidate_test_scores_<variant>.csv
pairwise_invariance.csv
summary.json
```

The legacy metric tables contain raw/filtered MRR, Hits@1/3/10, and binary
metrics. Shared-candidate files use identical candidate triples across graph
variants and are used only for invariance analysis.

For the first cluster run, retain the Slurm `MaxRSS`, GPU model, package
versions, Git commit, and exact command. The universal graph is the peak-memory
case (893,629 directed training edges in the current data, plus one structural
self-loop per entity). Prefer an 80 GB accelerator for the first production
measurement; the smoke run will establish whether a 40 GB device is sufficient
for this environment. If full-batch supervision does not fit,
`--batch-size` can be reduced, but every minibatch recomputes a complete
SlotGAT graph encoding and will be substantially slower.
