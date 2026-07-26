# HPC quick start: RGCN node classification

This checklist covers only the joint graph-variant augmentation experiments for
IMDb and Freebase node classification.

Run repository commands from `src/baselines/RGCN`. The tracked `data` symlink
then resolves paths against the repository-level `data/` directory.

## 1. Clone and environment

Clone the fork on the HPC, check out the intended commit, and create the
repository environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/gnn.txt
export DGLBACKEND=pytorch
```

Confirm the environment before submitting long jobs:

```bash
python -c "import torch, dgl, torch_geometric; print(torch.__version__, dgl.__version__, torch_geometric.__version__, torch.cuda.is_available())"
```

Transfer the datasets separately from Git. The required inputs are:

```text
data/raw/IMDB/movie_metadata.csv
data/raw/dataset_variant_3hops_filter/unchanged/{node.dat,link.dat,label.dat}
data/raw/dataset_variant_3hops_filter/exact_2/{node.dat,link.dat,label.dat}
```

The `exact_3` Freebase variant is not used by these commands.

## 2. IMDb preprocessing and alignment audit

```bash
cd src/baselines/RGCN

python preprocess_IMDB_rgcn.py \
  --variant v1,v2,v3,v4 \
  --raw-dir data/raw/IMDB \
  --movie-metadata-file movie_metadata.csv \
  --out-dir data/preprocessed \
  --seed 1566911444

mkdir -p results/rgcn_augmentation/IMDB

python rgcn_data_augmentation/prepare_rgcn_augmentation.py \
  --dataset IMDB \
  --output results/rgcn_augmentation/IMDB/preprocess_manifest.json
```

## 3. IMDb smoke test

Use a disposable output directory so it cannot be confused with production
resume state:

```bash
python rgcn_data_augmentation/run_IMDB_rgcn_augmentation.py \
  --variants v1,v2,v3,v4 \
  --seeds 1566911444 \
  --super-epochs 1 \
  --patience 1 \
  --device cuda \
  --output-dir results/smoke/rgcn_augmentation/IMDB
```

## 4. IMDb production run

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

## 5. Freebase preprocessing

Freebase `exact_2` has 47,368,991 forward edges. Request a high-memory CPU node
for preprocessing. The current implementation holds large Python and NumPy
representations concurrently, so start with at least 128 GB of system RAM and
use the scheduler's peak-RSS report to adjust if necessary.

```bash
python rgcn_data_augmentation/preprocess_FREEBASE_rgcn_augmentation.py \
  --variants unchanged exact_2 \
  --data-root data/raw/dataset_variant_3hops_filter \
  --output-root data/rgcn_augmentation/freebase \
  --split-seed 1566911444
```

Expected outputs:

```text
data/rgcn_augmentation/freebase/unchanged/rgcn_data.pt
data/rgcn_augmentation/freebase/exact_2/rgcn_data.pt
data/rgcn_augmentation/freebase/manifest.json
```

## 6. Freebase smoke test

The joint test still performs full-graph propagation over `exact_2`. Verify the
chunked layer against the installed PyG implementation first:

```bash
python rgcn_data_augmentation/verify_FREEBASE_chunked_rgcn.py --device cpu
```

This backend retains all graph edges, bounds relation-message tensors, and
recomputes relation aggregates during backward instead of retaining every
chunk's autograd bookkeeping. It is exact full-graph training, with additional
compute time traded for substantially lower peak CUDA memory.

```bash
python rgcn_data_augmentation/run_FREEBASE_rgcn_augmentation.py \
  --variants unchanged,exact_2 \
  --seeds 1566911444 \
  --data-root data/rgcn_augmentation/freebase \
  --output-dir results/smoke/rgcn_augmentation/FREEBASE_chunked_recompute \
  --super-epochs 1 \
  --patience 1 \
  --label-batch-size 0 \
  --edge-chunk-size 250000 \
  --device cuda
```

## 7. Freebase production run

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

## 8. Results and scheduler metrics

Each production seed directory contains checkpoints, training history, test
metrics, score tables, pairwise invariance results, and `summary.json`.
`summary.json` reports runtime, process peak RSS, model/checkpoint size, and
CUDA peak allocated/reserved memory.

Also retain scheduler measurements. For Slurm:

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,AllocTRES,MaxRSS,MaxVMSize
```

Record the Git commit, environment/package versions, GPU model, CPU-memory
request, scheduler job ID, and exact command for every production run.
