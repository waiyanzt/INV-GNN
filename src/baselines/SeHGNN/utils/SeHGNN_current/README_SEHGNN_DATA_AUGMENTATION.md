# SeHGNN shared-checkpoint data augmentation

This implementation adds graph-variant data augmentation for:

- **IMDB node classification:** original graph variants `1,2,3,4`.
- **Freebase node classification:** native per-variant **K** channels (`k`) for
  `unchanged` and `exact_2` by default, matching the RGCN and SlotGAT joint
  augmentation experiment.

It intentionally does **not** train IMDB skip/universal variants or Freebase `restricted_k` / `full_k` representations.

For Freebase, **original/native representation** means flavor `k`: channels
are generated independently from each variant's own graph. The canonical
channel union only fixes one shared model parameter layout; it does not merge
the variant graphs or replace their inputs with `full_k`/`restricted_k`.

## Training protocol

One **super-epoch** is:

```text
shuffle variants
for each variant:
    shuffle that variant's training nodes
    for each minibatch:
        load that variant's semantic inputs
        run the shared SeHGNN model
        update the shared optimizer once
validate the same shared checkpoint on every variant
update early stopping from the mean validation cross-entropy
```

The best checkpoint is selected only after every variant has been visited. Resume state is also saved only at a completed super-epoch boundary, so a resumed run does not create an unbalanced partial cycle.

## Parameter compatibility across variants

SeHGNN's parameter shapes depend on its semantic-channel set. The augmentation runners therefore build one deterministic **canonical union of channel identities** before constructing the model:

- IMDB uses the union of original propagated feature and label-feature keys.
- Freebase uses the union of K-channel semantic identities from the per-variant manifests. `type` identity remains the default, while `relation` identity is also supported when preprocessing used it.

Every variant uses this same model, optimizer, and checkpoint. A channel absent from a variant is supplied as a structural zero and has a zero channel mask. The model changes in `model/sehgnn.py` make the mask effective at every relevant stage:

1. per-channel projection biases are removed for absent channels;
2. feature projection uses mask-aware LayerNorm, excluding absent channels from mean/variance statistics;
3. semantic attention excludes absent key/value channels;
4. absent channels are zeroed again before concatenation.

With an all-one mask, the projection and normalization path matches the original SeHGNN computation. Thus parameter count and checkpoint shape are identical for every input variant, while each variant still uses its own original/K graph-derived input channels. This is not a `full_k` input graph.

## Preserved defaults

### IMDB NC

The augmentation runner preserves the current IMDB SeHGNN defaults:

- maximum 200 super-epochs;
- embedding/hidden size 512;
- 4 feature-propagation hops and 4 label-propagation hops;
- 2 feature-projection layers and 4 task layers;
- dropout 0.5;
- Adam, learning rate 0.005, weight decay 0.0001;
- batch size 10,000;
- patience 50;
- selection by minimum validation cross-entropy.

### Freebase NC

The augmentation runner preserves the current Freebase SeHGNN defaults:

- maximum 200 super-epochs;
- embedding/hidden size 512;
- 2 feature-projection layers and 4 task layers;
- 1 semantic-attention head;
- dropout 0.5;
- Adam, learning rate 0.005, weight decay 0.0001;
- batch size 10,000 and evaluation batch size 20,000;
- patience 50 with minimum improvement `1e-10`;
- selection by minimum validation cross-entropy.

A super-epoch has more optimizer updates than one ordinary single-variant epoch because every selected graph is visited once, as required by the augmentation protocol.

## Metrics and artifacts

For train, validation, and test splits of every variant, each seed stores:

- accuracy and balanced accuracy;
- macro/micro precision and recall;
- micro, macro, and weighted F1;
- Hit@1, Hit@3, and MRR;
- cross-entropy loss;
- per-class precision, recall, F1, and support;
- confusion matrix.

Pairwise test-output invariance includes full Kendall tau, tau@1, tau@3, prediction agreement, and logit/probability distances. It also stores logits, test-score CSVs, epoch history, exact epoch/update accounting, parameter/checkpoint memory, process peak RSS, and CUDA training/inference peaks.

Resource aggregation counts parameter size and checkpoint size **once per seed**, because all variants share one model/checkpoint.

## Run IMDB original variants

```bash
bash scripts/run_IMDB_nc_augmentation.sh
```

Useful environment overrides:

```bash
DATA_ROOT=/path/containing/IMDB_var1...IMDB_var4 \
SEEDS=1566911444,20241017,20251017 \
GPU=0 \
RESUME=1 \
bash scripts/run_IMDB_nc_augmentation.sh
```

## Run Freebase K variants

Use already-preprocessed K data:

```bash
bash scripts/run_freebase_nc_k_augmentation.sh
```

The production defaults are:

```text
pipeline: up_to_exact_2
variants: unchanged,exact_2
K: 2
channel identity: type
seeds: 1566911444,20241017,20251017
```

Run K preprocessing first, then train and aggregate:

```bash
RUN_PREPROCESS=1 \
bash scripts/run_freebase_nc_k_augmentation.sh
```

By default, the launcher reads raw variants from
`data/raw/dataset_variant_3hops_filter` and writes preprocessed channels under
`data/preprocessed/sehgnn_freebase_magnn`. Set `VARIANTS_ROOT` or
`PREPROCESSED_ROOT` only when using a different layout.

Existing four-variant studies remain available as an explicit override:

```bash
PIPELINE=full \
VARIANTS=unchanged,exact_2,exact_3,range_2_3 \
bash scripts/run_freebase_nc_k_augmentation.sh
```

The expected preprocessed layout is:

```text
<PREPROCESSED_ROOT>/<PIPELINE>/k<K>/<CHANNEL_IDENTITY>_channels/k/<variant>/
```

## Direct runner examples

```bash
python run_IMDB_nc_augmentation.py \
  --variants 1,2,3,4 \
  --seeds 1566911444,20241017,20251017 \
  --output-dir results/sehgnn_augmentation/IMDB_NC_original
```

```bash
python run_freebase_magnn_channels_augmentation.py \
  --data-root ../../../../../data/preprocessed/sehgnn_freebase_magnn/up_to_exact_2/k2/type_channels \
  --variants unchanged,exact_2 \
  --seeds 1566911444,20241017,20251017 \
  --output-dir results/sehgnn_augmentation/FREEBASE_NC/up_to_exact_2/k2/type_channels \
  --resume
```

Each seed directory contains one shared checkpoint and exact resume state,
training history, `<variant>_logits.npz`, both
`<variant>_test_scores.csv` and RGCN-compatible
`test_scores_<variant>.csv`, pairwise invariance metrics,
`test_metrics_by_variant.csv`, and `summary.json`.

## Aggregate an existing run

```bash
python aggregate_sehgnn_augmentation_metrics.py \
  --input-dir results/sehgnn_augmentation/IMDB_NC_original \
  --expected-runs 3
```

The aggregate directory contains:

- `aggregated_metrics.json`;
- `node_metrics.csv`;
- `invariance_metrics.csv`;
- `training_memory_metrics.csv`;
- `latex_rows.txt`.

## Run both datasets

```bash
bash scripts/run_all_nc_augmentation.sh
```

Set `RUN_IMDB=0` or `RUN_FREEBASE=0` to skip one dataset. Dataset-specific overrides use the `IMDB_...` and `FREEBASE_...` prefixes shown in the script.
