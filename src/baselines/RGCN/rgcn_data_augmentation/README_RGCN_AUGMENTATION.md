# RGCN joint graph-variant data augmentation

This package trains one shared model sequentially over separate graph variants.
A **super-epoch** visits every selected variant once in randomized order. It
supports exact resume from the most recently completed super-epoch.

## Model compatibility

- **IMDb node classification, IMDb link prediction, and DBLP:** current repository DGL RGCN implementations and defaults.
- **Freebase:** attached legacy PyG `RGCNFeatureless` architecture and legacy
  validation-loss selection.
- **WordNet:** attached legacy custom RGCN/DistMult architecture, normalization,
  edge/root dropout, optimizer grouping, negative sampling, and filtered-MRR
  checkpoint selection.

See `LEGACY_COMPATIBILITY_NOTES.md` for the remaining unavoidable Freebase
relation-layout caveat.

## Dependencies

IMDb node classification, IMDb link prediction, and DBLP require DGL. Freebase uses the original PyG `RGCNConv`, so install
the same `torch-geometric` version used by the old Freebase experiments.
WordNet now uses the exact updated single-variant model implementation and
therefore requires `torch-geometric` for `torch_geometric.utils.scatter`.


## Standard output layout

Use one stable directory per dataset:

```text
data/preprocessed/                         # existing IMDb/IMDb-LP/DBLP preprocessing
data/rgcn_augmentation/freebase/           # Freebase joint-training data
data/wordnet_3hops_augmented_full/wordnet_splits.npz  # leakage-free WordNet splits

results/rgcn_augmentation/IMDB/
results/rgcn_augmentation/IMDB_LP/md/
results/rgcn_augmentation/IMDB_LP/ml/
results/rgcn_augmentation/DBLP/
results/rgcn_augmentation/FREEBASE/
results/rgcn_augmentation/WORDNET/
```

Each results directory contains one `seed_<seed>/` subdirectory, so checkpoints
and histories for different seeds cannot overwrite each other. `--resume` uses
the state inside that seed directory.

## Kendall-tau postprocessing

After the production runs have written their test-score CSVs, recompute the
per-instance/query Kendall statistics for every available augmentation
benchmark with:

```bash
python rgcn_data_augmentation/calculate_kendall_tau.py
```

To process selected benchmarks only:

```bash
python rgcn_data_augmentation/calculate_kendall_tau.py \
  --datasets imdb_nc imdb_lp_ml imdb_lp_md
```

Each selected result directory receives:

```text
kendall_tau_per_seed.csv
kendall_tau_summary.csv
```

The training-regime label (for example, `IMDb1--4`) identifies the variants
visited by the shared model during training. Kendall comparisons remain between
the model's per-variant test rankings, such as `IMDb1 vs IMDb2`; there is no
separate `IMDb1--4` test graph or prediction vector.

## Epoch accounting

### IMDb

```text
updates_per_super_epoch = number_of_variants
```

### IMDb link prediction

```text
batches_per_variant = ceil(num_training_positive_rows / batch_size)
updates_per_super_epoch = number_of_variants * batches_per_variant
```

The `md` task normally uses `v1,v3`; the `ml` task normally uses
`v1,v2,v3,v4`. Each minibatch recomputes full-graph embeddings, matching the
legacy runner.

### DBLP

```text
batches_per_variant = ceil(num_training_edges / batch_size)
updates_per_super_epoch = number_of_variants * batches_per_variant
```

### Freebase

```text
batches_per_variant = 1                              # default full batch
updates_per_super_epoch = variants * batches_per_variant
```

`--label-batch-size > 0` batches only labeled nodes; each update still performs
full-graph propagation.

### WordNet

```text
batches_v = ceil(num_training_triples_v / batch_size_v)
updates_per_super_epoch = sum_v batches_v
```

`--batch-size 0` gives one supervised full-triple batch per variant. With all four variants, this gives four optimizer updates per super-epoch.

## Freebase

Preprocess:

```bash
python rgcn_data_augmentation/preprocess_FREEBASE_rgcn_augmentation.py \
  --variants unchanged exact_2 \
  --data-root data/raw/dataset_variant_3hops_filter \
  --output-root data/rgcn_augmentation/freebase \
  --split-seed 1566911444
```

Run:

```bash
python rgcn_data_augmentation/run_FREEBASE_rgcn_augmentation.py \
  --variants unchanged,exact_2 \
  --seeds 1566911444,20241017,20251017 \
  --data-root data/rgcn_augmentation/freebase \
  --output-dir results/rgcn_augmentation/FREEBASE_chunked_recompute \
  --super-epochs 100 \
  --patience 30 \
  --label-batch-size 0 \
  --edge-chunk-size 250000 \
  --hidden-dim 64 \
  --num-bases 30 \
  --dropout 0.0 \
  --lr 0.001 \
  --weight-decay 0.001 \
  --grad-clip 0 \
  --device cuda \
  --resume
```

The best checkpoint minimizes mean validation NLL across variants, matching the
old single-variant selection criterion.

`--edge-chunk-size 0` uses the original PyG `RGCNConv`. A positive value uses
the same parameters and complete graph but bounds the number of relation edges
whose source features are gathered at once. Its custom backward recomputes
relation aggregates instead of retaining each chunk's autograd bookkeeping.
This is exact full-graph training, not edge or neighborhood sampling, and
trades additional runtime for lower peak memory. Before the first chunked run
in a new environment, run:

```bash
python rgcn_data_augmentation/verify_FREEBASE_chunked_rgcn.py --device cpu
```

## WordNet

The leakage-free preprocessing uses the official original train/valid/test
splits and four graph variants:

```text
no_changes
all_inverse_edges
transitive_edges
universal_edges
```

Preprocess:

```bash
python rgcn_data_augmentation/preprocess_WORDNET_rgcn_augmentation.py \
  --data-dir data/wordnet_3hops_augmented_full \
  --source-splits-dir data/wordnet_3hops_augmented_full/original_splits \
  --output data/wordnet_3hops_augmented_full/wordnet_splits.npz
```

Run:

```bash
python rgcn_data_augmentation/run_WORDNET_rgcn_augmentation.py \
  --variants no_changes,all_inverse_edges,transitive_edges,universal_edges \
  --seeds 1566911444,20241017,20251017 \
  --splits-npz data/wordnet_3hops_augmented_full/wordnet_splits.npz \
  --output-dir results/rgcn_augmentation/WORDNET \
  --super-epochs 3000 \
  --eval-interval 1 \
  --patience 30 \
  --batch-size 0 \
  --neg-per-pos 1 \
  --hidden-dim 200 \
  --num-bases 30 \
  --edge-dropout-other 0.4 \
  --root-dropout-loop 0.2 \
  --lr 0.01 \
  --weight-decay 0.01 \
  --grad-clip 0 \
  --eval-batch-size 512 \
  --binary-k 50 \
  --fixed-negatives 50 \
  --candidate-seed 1566911444 \
  --device cuda \
  --resume
```

The runner loads every graph through the same `WordNetLPDataset` class and uses
the exact `WordNetRGCNLinkPredictor` used by the updated single-variant run. It
repeats the direct held-out and held-out-inverse leakage checks before training.
The best checkpoint maximizes mean **legacy filtered MRR** across all four graph
variants. Validation occurs at every super-epoch with the new default `--eval-interval 1`, matching the single-variant runner when it uses `--eval_interval 1`. Fixed negatives are used only for shared-candidate
invariance analysis and do not affect checkpoint selection or legacy metrics.

### WordNet outputs

`legacy_test_metrics_by_variant.csv` contains the legacy metric implementation. Compare it to per-variant runs produced from the same leakage-free NPZ:
raw/filtered MRR, Hits@1/3/10, and the old deterministic binary metrics.

`shared_candidate_metrics_by_variant.csv` and `pairwise_invariance.csv` use the
identical fixed candidate triples across variants. These are additional
invariance results and do not alter the legacy metrics.

## IMDb node classification and DBLP

The existing package commands remain unchanged:

```bash
python rgcn_data_augmentation/run_IMDB_rgcn_augmentation.py \
  --variants v1,v2,v3,v4 \
  --super-epochs 200 \
  --output-dir results/rgcn_augmentation/IMDB \
  --resume

python rgcn_data_augmentation/run_DBLP_rgcn_augmentation.py \
  --variants v1,v2,v3 \
  --super-epochs 200 \
  --batch-size 1024 \
  --output-dir results/rgcn_augmentation/DBLP \
  --resume
```

## Resume files

Each seed directory contains:

```text
shared_checkpoint.pt          # best validation model
latest_training_state.pt      # latest completed super-epoch
training_history.csv
summary.json
```

If Slurm terminates midway through a super-epoch, rerunning the same command
with `--resume` restarts from the preceding completed super-epoch. Optimizer,
RNG, early-stopping, history, epoch counters, runtime, and peak-memory state are
restored.


## IMDb link prediction augmentation

The package also supports the repository's IMDb link-prediction tasks using
`run_IMDB_rgcn_lp_augmentation.py`:

- `md`: movie-director prediction over `v1,v3`;
- `ml`: movie-link prediction over `v1,v2,v3,v4`.

Use the repository's `preprocess_IMDB_rgcn_lp.py` with the CMPNN-owned shared
split files, then run `prepare_rgcn_augmentation.py --dataset IMDB_LP --task
<md|ml>` before training. The shared runner preserves the legacy RGCN encoder,
pairwise loss, fixed negative matrices, validation BCE, binary/ranking metrics,
and default hyperparameters. It adds balanced graph-variant super-epochs, a
global semantic relation vocabulary, invariance metrics, and exact resumption.

Results are written under:

```text
results/rgcn_augmentation/IMDB_LP/md/seed_<seed>/
results/rgcn_augmentation/IMDB_LP/ml/seed_<seed>/
```

See `RGCN_AUGMENTATION_RUNBOOK.md` for complete commands.
