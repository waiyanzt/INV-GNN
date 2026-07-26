# RGCN Joint Graph-Variant Augmentation Runbook

This runbook is for the current resumable RGCN graph-variant augmentation package.

The commands assume the package directory is copied to:

```text
INV-GNN/src/baselines/RGCN/rgcn_data_augmentation/
```

Run all commands from:

```bash
cd /path/to/INV-GNN/src/baselines/RGCN
```


## Standard directory layout

```text
data/preprocessed/                         # existing IMDb/DBLP preprocessing
data/rgcn_augmentation/freebase/           # Freebase augmentation preprocessing
data/wordnet_3hops_augmented_full/         # WordNet variants + official splits + NPZ

results/rgcn_augmentation/IMDB/
results/rgcn_augmentation/IMDB_LP/md/
results/rgcn_augmentation/IMDB_LP/ml/
results/rgcn_augmentation/DBLP/
results/rgcn_augmentation/FREEBASE/
results/rgcn_augmentation/WORDNET/
```

Do not add `_resume`, `_legacy_aligned`, or version suffixes for the initial
runs. Resume state and best checkpoints are already separated by dataset and
`seed_<seed>/`. Use a different output directory only when intentionally running
a different incompatible experiment configuration.

---

## 1. Environment check

The IMDb and DBLP runners use DGL. Freebase uses PyTorch Geometric's
`RGCNConv`. WordNet uses the exact updated single-variant model and requires
`torch_geometric.utils.scatter`.

For a PyTorch 2.4 / CUDA 12.4 environment:

```bash
python -m pip install \
  torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu124

python -m pip install \
  dgl==2.4.0 \
  -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html

python -m pip install torch-geometric numpy pandas scipy scikit-learn
```

Verify:

```bash
python - <<'PY'
import torch
import dgl
import torch_geometric

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("DGL:", dgl.__version__)
print("PyG:", torch_geometric.__version__)
PY
```

---

# 2. Early stopping

## General mechanism

After a validation check, the runner compares the current selection metric to
the best value seen so far. IMDb, IMDb LP, DBLP, and Freebase use a `1e-6`
improvement tolerance. WordNet intentionally uses a strict zero-tolerance
comparison (`current > best`) to match the updated single-variant runner.

- For a minimized metric, improvement means:

  ```text
  current < best - 1e-6
  ```

- For a maximized metric in the general runners, improvement means:

  ```text
  current > best + 1e-6
  ```

- For WordNet specifically, improvement means:

  ```text
  current > best
  ```

When the metric improves:

1. The best metric is updated.
2. The non-improvement counter is reset to zero.
3. `shared_checkpoint.pt` is overwritten with the new best model.

When the metric does not improve, the counter increases by one. Training stops
when the counter reaches the configured patience.

The exact resumable state is saved independently to
`latest_training_state.pt` after each completed super-epoch.

## Current stopping rules

| Dataset | Selection metric | Direction | Validation frequency | Default patience |
|---|---|---:|---:|---:|
| IMDb | Mean validation Macro-F1 across v1-v4 | Maximize | Every super-epoch | 20 checks |
| IMDb LP (`md`, `ml`) | Mean validation BCE across selected variants | Minimize | Every super-epoch | 15 checks |
| DBLP | Mean validation BCE/loss across v1-v3 | Minimize | Every super-epoch | 15 checks |
| Freebase | Mean validation NLL across variants | Minimize | Every super-epoch | 30 checks |
| WordNet | Mean legacy filtered MRR across variants | Maximize | Every super-epoch | 30 checks |

For WordNet, the default command now uses:

```text
--patience 30
--eval-interval 1
```

This gives 30 consecutive failed validation checks before early stopping.

You can override the number of failed validation checks directly with
`--patience-evals`.

## Historical note: earlier draft stopping rules

IMDb and DBLP were unchanged by the legacy-alignment patch.

| Dataset | Before alignment | Current behavior |
|---|---|---|
| IMDb | Maximize mean validation Macro-F1; patience 20 | Same |
| DBLP | Minimize mean validation BCE/loss; patience 15 | Same |
| Freebase | **Maximize mean validation Macro-F1**; patience 30 | **Minimize mean validation NLL**; patience 30 |
| WordNet | **Minimize mean fixed-candidate validation BCE**; validate at super-epoch 1, every 10 super-epochs, and final; `--patience-evals 300` | **Maximize mean legacy filtered MRR**; validate every super-epoch; patience 30 checks |

The fixed WordNet candidate sets are still reported for invariance analysis,
but they no longer determine checkpoint selection.

---

# 3. IMDb node classification

## 3.1 Preprocess all four native variants

```bash
python preprocess_IMDB_rgcn.py \
  --variant v1,v2,v3,v4 \
  --raw-dir data/raw/IMDB/ \
  --movie-metadata-file movie_metadata.csv \
  --out-dir data/preprocessed \
  --seed 1566911444
```

Optional: reuse an existing split file by adding:

```bash
--split-npz /path/to/shared_imdb_split.npz
```

Expected outputs:

```text
data/preprocessed/IMDB_rgcn_v1/
data/preprocessed/IMDB_rgcn_v2/
data/preprocessed/IMDB_rgcn_v3/
data/preprocessed/IMDB_rgcn_v4/
```

## 3.2 Audit alignment

```bash
mkdir -p results/rgcn_augmentation/IMDB

python rgcn_data_augmentation/prepare_rgcn_augmentation.py \
  --dataset IMDB \
  --output results/rgcn_augmentation/IMDB/preprocess_manifest.json
```

## 3.3 Run joint augmentation

```bash
python rgcn_data_augmentation/run_IMDB_rgcn_augmentation.py \
  --variants v1,v2,v3,v4 \
  --seeds 1566911444,20241017,20251017 \
  --super-epochs 200 \
  --patience 20 \
  --in-dim 128 \
  --hid-dim 128 \
  --out-dim 128 \
  --layers 2 \
  --num-bases 8 \
  --dropout 0.3 \
  --lr 0.002 \
  --weight-decay 0.00001 \
  --grad-clip 2.0 \
  --device cuda \
  --output-dir results/rgcn_augmentation/IMDB \
  --resume
```

### IMDb epoch accounting

IMDb is full-batch. With four variants:

```text
updates_per_super_epoch = 4
optimizer_steps = completed_super_epochs * 4
```

---


# 4. IMDb link prediction

The repository documents two shared-split tasks:

- `md`: movie-director prediction with variants `v1,v3`;
- `ml`: movie-link prediction with variants `v1,v2,v3,v4`.

The augmentation runner preserves the standalone RGCN defaults: 128 input
dimensions, 256 hidden/output dimensions, three RGCN layers, 16 bases, 0.1
dropout, learning rate 0.002, weight decay `1e-5`, embedding penalty `1e-6`,
batch size 1024, threshold 0.5, gradient clipping 2.0, and patience 15.

## 4.1 Build the shared IMDb link-prediction splits

Run from the CMPNN directory:

```bash
cd /path/to/INV-GNN/src/baselines/CMPNN

python build_IMDB_md_shared_splits.py
python build_IMDB_ml_shared_splits.py
```

Expected files:

```text
IMDB_md_shared_splits.npz
IMDB_ml_shared_splits.npz
```

## 4.2 Preprocess the `md` variants

```bash
cd ../RGCN

python preprocess_IMDB_rgcn_lp.py \
  --task md \
  --variant v1,v3 \
  --shared-npz ../CMPNN/IMDB_md_shared_splits.npz
```

Expected outputs:

```text
data/preprocessed/IMDB_rgcn_lp_md_v1/
data/preprocessed/IMDB_rgcn_lp_md_v3/
```

## 4.3 Audit the `md` preprocessing

```bash
mkdir -p results/rgcn_augmentation/IMDB_LP/md

python rgcn_data_augmentation/prepare_rgcn_augmentation.py \
  --dataset IMDB_LP \
  --task md \
  --output results/rgcn_augmentation/IMDB_LP/md/preprocess_manifest.json
```

## 4.4 Run joint `md` augmentation

```bash
python rgcn_data_augmentation/run_IMDB_rgcn_lp_augmentation.py \
  --task md \
  --variants v1,v3 \
  --seeds 1566911444,20241017,20251017 \
  --data-root data/preprocessed \
  --super-epochs 200 \
  --patience 15 \
  --batch-size 1024 \
  --threshold 0.5 \
  --in-dim 128 \
  --hid-dim 256 \
  --out-dim 256 \
  --layers 3 \
  --num-bases 16 \
  --dropout 0.1 \
  --lr 0.002 \
  --weight-decay 0.00001 \
  --emb-reg 0.000001 \
  --grad-clip 2.0 \
  --device cuda \
  --output-dir results/rgcn_augmentation/IMDB_LP \
  --resume
```

## 4.5 Preprocess the `ml` variants

```bash
python preprocess_IMDB_rgcn_lp.py \
  --task ml \
  --variant v1,v2,v3,v4 \
  --shared-npz ../CMPNN/IMDB_ml_shared_splits.npz
```

Expected outputs:

```text
data/preprocessed/IMDB_rgcn_lp_ml_v1/
data/preprocessed/IMDB_rgcn_lp_ml_v2/
data/preprocessed/IMDB_rgcn_lp_ml_v3/
data/preprocessed/IMDB_rgcn_lp_ml_v4/
```

## 4.6 Audit the `ml` preprocessing

```bash
mkdir -p results/rgcn_augmentation/IMDB_LP/ml

python rgcn_data_augmentation/prepare_rgcn_augmentation.py \
  --dataset IMDB_LP \
  --task ml \
  --output results/rgcn_augmentation/IMDB_LP/ml/preprocess_manifest.json
```

## 4.7 Run joint `ml` augmentation

```bash
python rgcn_data_augmentation/run_IMDB_rgcn_lp_augmentation.py \
  --task ml \
  --variants v1,v2,v3,v4 \
  --seeds 1566911444,20241017,20251017 \
  --data-root data/preprocessed \
  --super-epochs 200 \
  --patience 15 \
  --batch-size 1024 \
  --threshold 0.5 \
  --in-dim 128 \
  --hid-dim 256 \
  --out-dim 256 \
  --layers 3 \
  --num-bases 16 \
  --dropout 0.1 \
  --lr 0.002 \
  --weight-decay 0.00001 \
  --emb-reg 0.000001 \
  --grad-clip 2.0 \
  --device cuda \
  --output-dir results/rgcn_augmentation/IMDB_LP \
  --resume
```

### IMDb-LP early stopping

After all selected variants have completed a super-epoch, the shared encoder is
evaluated on the same validation positives and fixed negative-tail matrix for
each graph. The runner averages the per-variant BCE values. It saves the best
checkpoint when:

```text
mean_validation_BCE < best_mean_validation_BCE - 1e-6
```

Training stops after 15 consecutive completed super-epochs without an
improvement.

### IMDb-LP epoch accounting

Let:

```text
B = ceil(number_of_positive_training_rows / 1024)
V = number_of selected variants
```

Then:

```text
updates_per_super_epoch = V * B
optimizer_steps = completed_super_epochs * V * B
```

Each minibatch recomputes full-graph embeddings, matching the standalone
repository runner.

---

# 5. DBLP link prediction

## 5.1 Create the shared paper-conference splits

Run the repository's shared-split script from the MAGNN directory:

```bash
cd /path/to/INV-GNN/src/baselines/MAGNN
python paper-venue_shared_splits.py
cd ../RGCN
```

Expected split file:

```text
data/preprocessed/DBLP_shared_splits/DBLP_pc_shared_splits.npz
```

If your MAGNN and RGCN directories have separate `data/` folders, copy or link
the generated shared split into the path expected by the RGCN preprocessor.

## 5.2 Preprocess all three variants

```bash
python preprocess_DBLP_rgcn.py \
  --variant v1,v2,v3 \
  --raw-dir data/raw/DBLP/ \
  --shared-npz data/preprocessed/DBLP_shared_splits/DBLP_pc_shared_splits.npz \
  --min-conf 0 \
  --out-dir data/preprocessed
```

Expected outputs:

```text
data/preprocessed/DBLP_rgcn_v1/
data/preprocessed/DBLP_rgcn_v2/
data/preprocessed/DBLP_rgcn_v3/
```

## 5.3 Audit alignment

```bash
mkdir -p results/rgcn_augmentation/DBLP

python rgcn_data_augmentation/prepare_rgcn_augmentation.py \
  --dataset DBLP \
  --output results/rgcn_augmentation/DBLP/preprocess_manifest.json
```

## 5.4 Run joint augmentation

```bash
python rgcn_data_augmentation/run_DBLP_rgcn_augmentation.py \
  --variants v1,v2,v3 \
  --seeds 1566911444,20241017,20251017 \
  --super-epochs 200 \
  --patience 15 \
  --batch-size 1024 \
  --neg-per-paper 3 \
  --threshold 0.5 \
  --in-dim 128 \
  --hid-dim 256 \
  --out-dim 256 \
  --layers 3 \
  --num-bases 16 \
  --dropout 0.1 \
  --lr 0.002 \
  --weight-decay 0.00001 \
  --emb-reg 0.000001 \
  --grad-clip 2.0 \
  --device cuda \
  --output-dir results/rgcn_augmentation/DBLP \
  --resume
```

### DBLP epoch accounting

Let:

```text
B = ceil(number_of_positive_training_edges / 1024)
```

With three variants:

```text
updates_per_super_epoch = 3 * B
optimizer_steps = completed_super_epochs * 3 * B
```

The supervised edges are minibatched, but each batch recomputes full-graph
RGCN embeddings.

---

# 6. Freebase node classification

## 6.1 Preprocess and align relation IDs

```bash
python rgcn_data_augmentation/preprocess_FREEBASE_rgcn_augmentation.py \
  --variants unchanged exact_2 \
  --data-root data/raw/dataset_variant_3hops_filter \
  --output-root data/rgcn_augmentation/freebase \
  --split-seed 1566911444
```

Expected outputs include:

```text
data/rgcn_augmentation/freebase/unchanged/rgcn_data.pt
data/rgcn_augmentation/freebase/exact_2/rgcn_data.pt
data/rgcn_augmentation/freebase/manifest.json
```

The preprocessor creates one shared BOOK-node split and a globally stable
forward/reverse relation layout for the shared checkpoint.

## 6.2 Run joint augmentation

```bash
python rgcn_data_augmentation/run_FREEBASE_rgcn_augmentation.py \
  --variants unchanged,exact_2 \
  --seeds 1566911444,20241017,20251017 \
  --data-root data/rgcn_augmentation/freebase \
  --output-dir results/rgcn_augmentation/FREEBASE \
  --super-epochs 100 \
  --patience 30 \
  --label-batch-size 0 \
  --hidden-dim 64 \
  --num-bases 30 \
  --dropout 0.0 \
  --lr 0.001 \
  --weight-decay 0.001 \
  --grad-clip 0 \
  --device cuda \
  --resume
```

### Freebase epoch accounting

With the default full labeled-node batch and two variants:

```text
updates_per_super_epoch = 2
optimizer_steps = completed_super_epochs * 2
```

If `--label-batch-size` is positive, each label minibatch causes one optimizer
update, although each update still runs full-graph propagation.

---

# 7. WordNet link prediction

The updated WordNet preprocessing is leakage-free and uses the official
`train.txt`, `valid.txt`, and `test.txt` files from `original_splits`. It builds
four training graphs using one shared entity and relation vocabulary:

```text
no_changes
all_inverse_edges
transitive_edges
universal_edges
```

The inverse and universal variants are checked to ensure that neither contains
an inverse of an official validation or test triple.

## 7.1 Generate the leakage-free four-variant NPZ

```bash
python rgcn_data_augmentation/preprocess_WORDNET_rgcn_augmentation.py \
  --data-dir data/wordnet_3hops_augmented_full \
  --source-splits-dir data/wordnet_3hops_augmented_full/original_splits \
  --output data/wordnet_3hops_augmented_full/wordnet_splits.npz
```

If `original_splits` is already under `--data-dir`, the shorter equivalent is:

```bash
python rgcn_data_augmentation/preprocess_WORDNET_rgcn_augmentation.py \
  --data-dir data/wordnet_3hops_augmented_full
```

Expected outputs:

```text
data/wordnet_3hops_augmented_full/wordnet_splits.npz
data/wordnet_3hops_augmented_full/wordnet_split_stats.json
```

The NPZ contains:

```text
train_pos_no_changes
train_pos_all_inverse_edges
train_pos_transitive_edges
train_pos_universal_edges
val_pos
test_pos
entity_vocab
relation_vocab
num_entities
num_relations
num_base_relations
base_relation_ids
variant_names
format_version
```

Unlike the previous augmentation preprocessor, this stage does not resplit a
graph intersection. It preserves the official validation and test positives and
uses each leakage-free `data.txt` as that variant's complete training graph.

The packaged preprocessor is the same schema as the updated single-variant
preprocessor. It validates dense IDs and writes
`format_version=wordnet_lp_four_variants_v1` plus the canonical `variant_names`.
The augmentation package also includes the same `wordnet_lp.py` loader used by
the single-variant runner.

> **Updated stopping schedule:** both the standalone comparison and joint augmentation commands below use `--patience 30` and `--eval_interval 1`/`--eval-interval 1`. This preserves comparison fairness but makes validation substantially more expensive because full filtered ranking runs every epoch or super-epoch.

## 7.2 Optional single-variant comparison command

Run the updated standalone code from the RGCN directory using the same NPZ:

```bash
python run_wordnet_lp.py \
  --variant all \
  --splits-path data/wordnet_3hops_augmented_full/wordnet_splits.npz \
  --checkpoint-dir checkpoint/wordnet \
  --epochs 3000 \
  --seeds 1566911444 20241017 20251017 \
  --hidden_dim 200 \
  --num_bases 30 \
  --edge_dropout_other 0.4 \
  --root_dropout_loop 0.2 \
  --lr 0.01 \
  --weight_decay 0.01 \
  --neg_per_pos 1 \
  --batch_size 0 \
  --patience 30 \
  --eval_interval 1 \
  --device cuda
```

The standalone runner trains four independent checkpoints. The augmentation
runner below trains one shared checkpoint across the same four graph inputs.

## 7.3 Run joint augmentation over all four variants

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
  --score-batch-size 65536 \
  --fixed-negatives 50 \
  --candidate-seed 1566911444 \
  --device cuda \
  --resume
```

`--splits-npz` may be omitted when the file is located at
`data/wordnet_3hops_augmented_full/wordnet_splits.npz`, because that path is the
default derived from `--data-root`.

Optional direct patience override:

```bash
--patience-evals 30
```

### Leakage checks repeated by the runner

Before creating the model, the runner verifies:

- validation and test positives do not overlap;
- no variant directly contains a held-out positive;
- relation and entity IDs are within the shared vocabulary;
- `all_inverse_edges` and `universal_edges` do not contain inverses of held-out
  base-relation triples;
- all four required training arrays are present.

### Evaluation behavior

The leakage-free official validation/test sets replace the old
intersection-derived evaluation sets. Evaluation still has two roles:

1. **Legacy-style filtered ranking and binary metrics:** each graph uses its own
   `train + validation + test` filtering dictionary. Mean validation filtered
   MRR across all four variants selects the shared checkpoint.
2. **Shared-candidate invariance:** the runner deterministically creates the
   same candidate triples for every graph, excluding positives from the union
   of all four training graphs and the official validation/test sets. These
   fixed candidates do not affect training or checkpoint selection.

### Compatibility with the updated single-variant run

The two paths now share all of the following:

- the same `wordnet_splits.npz` schema and format-version checks;
- the same canonical variant names and aliases;
- the same `WordNetLPDataset` graph construction;
- the exact same `WordNetRGCNLinkPredictor` implementation;
- 200-dimensional embeddings and 30 bases by default;
- neighbor-edge dropout 0.4 and root dropout 0.2;
- Adam parameter groups with weight decay only on `rel_emb`;
- 50/50 online head/tail corruption with one negative per positive;
- full-batch training by default;
- the same variant-specific filtered-ranking dictionaries;
- the same raw/filtered MRR, Hits@1/3/10, and deterministic 50-negative binary metrics;
- strict filtered-MRR improvement and validation only every 10 epochs/super-epochs.

The necessary protocol-level differences are that the standalone runner creates
one model and optimizer per variant, whereas Protocol A uses one shared model and
optimizer, shuffles variant order, performs one update per selected graph in a
full-batch super-epoch, and selects the shared checkpoint by mean validation
filtered MRR across variants.

The augmentation runner accepts both hyphenated options and the standalone
underscore spellings for the main WordNet settings. `--epochs` is also accepted
as an alias for `--super-epochs`, although its meaning remains total joint
super-epochs.

### WordNet epoch accounting

With `--batch-size 0`, each graph uses one supervised triple batch:

```text
updates_per_super_epoch = 4
optimizer_steps = completed_super_epochs * 4
```

For a positive batch size:

```text
batches_v = ceil(number_of_training_triples_v / batch_size)
updates_per_super_epoch = sum_v batches_v
```

The four training sets can differ in size, so their minibatch counts can differ.

---

# 8. Resume after a Slurm timeout

Use `--resume` on the initial launch and on every restarted launch. Rerun the
same command with the same training-relevant arguments.

Each seed directory contains:

```text
shared_checkpoint.pt          # best validation checkpoint
latest_training_state.pt      # most recent completed super-epoch
training_history.csv
summary.json
```

If Slurm terminates during a super-epoch, the next launch restores the end of
the previous completed super-epoch. The partially completed cycle is repeated
from its beginning.

`--super-epochs` is the total target. For example, if 73 super-epochs were
completed and the command still specifies `--super-epochs 200`, the resumed run
continues with super-epoch 74 and stops no later than 200.

Because these are fresh augmentation runs, use the default dataset folders shown above.
Do not mix unrelated experimental configurations in the same dataset/seed directory.

---

# 9. Suggested Slurm wrapper

```bash
#!/bin/bash
#SBATCH --job-name=rgcn_wordnet_aug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

source ~/.bashrc
conda activate rgcn_aug

cd /path/to/INV-GNN/src/baselines/RGCN
mkdir -p logs

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
  --score-batch-size 65536 \
  --fixed-negatives 50 \
  --candidate-seed 1566911444 \
  --device cuda \
  --resume
```

Submit with:

```bash
sbatch run_wordnet_aug.slurm
```

After a timeout, submit the same script again. The runner resumes from
`latest_training_state.pt`.
