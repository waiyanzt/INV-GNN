# GNN Information-based Invariance 
This repository contains the experiment code for **Information-based Invariance, Equivariance, and Expressivity for Graph Neural Networks**. The code evaluates whether graph neural network baselines remain stable across non-isomorphic graph representations that encode the same information, and includes invariant/skip adaptations of the evaluated models.

The paper evaluates the method on IMDb and DBLP variants using node classification and link prediction tasks. The included baselines cover local message passing, path-conditioned propagation, metapath-instance aggregation, metapath-feature propagation, and node-type-aware aggregation.

## Heterogeneous-graph GNN baselines on four tasks:

| Task                     | MAGNN | RGCN | CMPNN | SeHGNN | SlotGAT |
|--------------------------|:-----:|:----:|:-----:|:------:|:-------:|
| IMDB node classification |   ✓   |  ✓   |       |   ✓    |    ✓    |
| DBLP link prediction     |   ✓   |  ✓   |   ✓   |        |    ✓    |
| IMDB link prediction     |   ✓   |  ✓   |   ✓   |        |    ✓    |
| WordNet link prediction (joint augmentation) | | ✓ | | | ✓ |
| Freebase node classification (joint augmentation) | | ✓ | | | ✓ |

## Data
Download the experiment data from:

```
https://drive.google.com/file/d/1AQUhIOXVHdvuGcEsqDXcR_S3bAmQThnt/view?usp=sharing
```

After downloading, place or extract the data so that each baseline can find its expected `data/` and `data/raw/` directories

## Repository layout

```
.
├── data/
│   └── raw/
│       ├── DBLP/     # author/paper/conf/term text files (13 files)
│       └── IMDB/     # movie_metadata.csv
├── src/baselines/
│   ├── MAGNN/        # MAGNN preprocess + run scripts, model/, utils/
│   ├── RGCN/         # RGCN preprocess + run scripts (IMDB, DBLP)
│   ├── CMPNN/        # CMPNN preprocess + run scripts, cmpnn/ package
│   ├── SeHGNN/       # SeHGNN preprocess + run scripts (IMDB NC)
│   └── SlotGAT/      # SlotGAT scripts (IMDB NC, WordNet LP augmentation)
├── requirements/
│   ├── gnn.txt       # base stack (torch 2.4 + DGL + sklearn + ...)
│   ├── cmpnn.txt     # adds torchdrug 0.2.1
│   └── README.md
├── run_all.sh        # one-command orchestrator
└── README.md
```

Each baseline directory has a `data` symlink → `../../../data`, so the
existing hardcoded `data/raw/...` paths inside the scripts resolve correctly
when scripts are launched from the baseline folder.

## Environment

Python **3.11**, CUDA **12.4**, single virtualenv covers everything:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements/cmpnn.txt        # cmpnn.txt -r's gnn.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('punkt_tab')"
export DGLBACKEND=pytorch                    # also add to ~/.bashrc
```

See `requirements/README.md` for the conda alternative and SeHGNN extras.

## Run everything

```bash
./run_all.sh
```

Common subsets:

```bash
./run_all.sh --task imdb_nc                        # just IMDB node classification
./run_all.sh --task dblp_lp --baseline magnn       # just MAGNN DBLP LP
./run_all.sh --preprocess-only                     # build all preprocessed data
./run_all.sh --train-only                          # skip preprocess, train all
./run_all.sh --seeds 1566911444                    # single seed
```

Run `./run_all.sh --help` for all flags. Each `run_*` script also accepts
`--help` if you want to inspect or override hyperparameters.

## Run individual experiments

Always `cd` into the baseline folder first (the `data/` symlink and the
imports both assume CWD is the baseline directory).

### IMDB node classification

```bash
cd src/baselines/MAGNN
python preprocess_IMDB_star.py
python preprocess_IMDB_star_t.py
python preprocess_IMDB_star_t_2.py
python preprocess_IMDB_star_t_3.py
python preprocess_IMDB_skip.py
python run_IMDB.py
python run_IMDB_skip.py

cd ../RGCN
python preprocess_IMDB_rgcn.py --variant v1,v2,v3,v4
python preprocess_IMDB_rgcn_skip.py --variant v1,v2,v3,v4
python run_IMDB_rgcn.py --variants v1,v2,v3,v4
python run_IMDB_rgcn_skip.py --variants v1,v2,v3,v4

cd ../SeHGNN
python preprocess_IMDB.py
python run_IMDB_nc.py

cd ../SlotGAT
python preprocess_IMDB.py
python run_IMDB_nc.py
```

### DBLP link prediction

Build the shared paper-venue splits first (consumed by RGCN and CMPNN):

The joint SlotGAT experiment trains one shared model on `v1,v2,v3`. See
[`src/baselines/SlotGAT/slotgat_data_augmentation/README_DBLP.md`](src/baselines/SlotGAT/slotgat_data_augmentation/README_DBLP.md)
for shared preprocessing, the production command, balanced super-epoch
accounting, resume behavior, and outputs.

```bash
cd src/baselines/MAGNN
python paper-venue_shared_splits.py
```

Then per baseline:

```bash
# MAGNN
python preprocess_DBLP_pc_trainpc.py --variants v1,v2,v3
python preprocess_DBLP_skip.py --variant all
python run_DBLP_pc_trainpc.py --variants v1,v2,v3
python run_DBLP_skip.py --variants v1,v2,v3

cd ../RGCN
python preprocess_DBLP_rgcn.py --variant v1,v2,v3
python preprocess_DBLP_rgcn_skip.py --variant v1,v2,v3
python run_DBLP_rgcn.py --variants v1,v2,v3
python run_DBLP_rgcn_skip.py --variants v1,v2,v3

cd ../CMPNN
python preprocess_DBLP_cmpnn_pc.py --variant v1,v2,v3
python preprocess_DBLP_cmpnn_skip.py --variant v1,v2,v3
python run_DBLP_cmpnn_pc.py --variants v1,v2,v3
python run_DBLP_cmpnn_skip.py --variants v1,v2,v3
```

### WordNet joint-augmentation link prediction

The initial SlotGAT experiment jointly trains on `no_changes`,
`universal_edges`, and `all_inverse_edges`; it intentionally excludes the
available `transitive_edges` graph. See
[`src/baselines/SlotGAT/slotgat_data_augmentation/README_WORDNET.md`](src/baselines/SlotGAT/slotgat_data_augmentation/README_WORDNET.md)
for preprocessing, smoke-test, production, resume, and output commands.

### Freebase joint-augmentation node classification

The SlotGAT experiment jointly trains one shared model on `unchanged` and
`exact_2`. See
[`src/baselines/SlotGAT/slotgat_data_augmentation/README_FREEBASE.md`](src/baselines/SlotGAT/slotgat_data_augmentation/README_FREEBASE.md)
for preprocessing, the production command, memory requirements, resume
behavior, and outputs.

### IMDb joint-augmentation node classification

The SlotGAT experiment jointly trains one shared model on IMDb variants
`v1,v2,v3,v4`, with globally aligned semantic relation IDs. See
[`src/baselines/SlotGAT/slotgat_data_augmentation/README_IMDB.md`](src/baselines/SlotGAT/slotgat_data_augmentation/README_IMDB.md)
for preprocessing, the production command, balanced super-epoch accounting,
resume behavior, and outputs.

### IMDB link prediction

CMPNN owns the shared IMDB md/ml splits. Build them first:

The joint SlotGAT experiment trains one shared model on `v1,v3` for `md` and
on `v1,v2,v3,v4` for `ml`. See
[`src/baselines/SlotGAT/slotgat_data_augmentation/README_IMDB_LP.md`](src/baselines/SlotGAT/slotgat_data_augmentation/README_IMDB_LP.md)
for its shared preprocessing, production commands, resume behavior, and output
contract.

```bash
cd src/baselines/CMPNN
python build_IMDB_md_shared_splits.py
python build_IMDB_ml_shared_splits.py
```

Then per baseline (`md` allows variants `v1,v3`; `ml` allows `v1,v2,v3,v4`):

```bash
# MAGNN
cd ../MAGNN
python preprocess_IMDB_magnn_lp.py --task md --variant v1,v3
python preprocess_IMDB_magnn_lp.py --task ml --variant v1,v2,v3,v4
python preprocess_IMDB_magnn_lp_skip.py --task md --variant v1,v3
python preprocess_IMDB_magnn_lp_skip.py --task ml --variant v1,v2,v3,v4
python run_IMDB_magnn_lp.py --task md --variants v1,v3
python run_IMDB_magnn_lp.py --task ml --variants v1,v2,v3,v4
python run_IMDB_magnn_lp_skip.py --task md --variants v1,v3
python run_IMDB_magnn_lp_skip.py --task ml --variants v1,v2,v3,v4

# RGCN
cd ../RGCN
python preprocess_IMDB_rgcn_lp.py --task md --variant v1,v3 \
       --shared-npz ../CMPNN/IMDB_md_shared_splits.npz
python preprocess_IMDB_rgcn_lp.py --task ml --variant v1,v2,v3,v4 \
       --shared-npz ../CMPNN/IMDB_ml_shared_splits.npz
python preprocess_IMDB_rgcn_lp_skip.py --task md --variant v1,v3 \
       --shared-npz ../CMPNN/IMDB_md_shared_splits.npz
python preprocess_IMDB_rgcn_lp_skip.py --task ml --variant v1,v2,v3,v4 \
       --shared-npz ../CMPNN/IMDB_ml_shared_splits.npz
python run_IMDB_rgcn_lp.py --task md --variants v1,v3
python run_IMDB_rgcn_lp.py --task ml --variants v1,v2,v3,v4
python run_IMDB_rgcn_lp_skip.py --task md --variants v1,v3
python run_IMDB_rgcn_lp_skip.py --task ml --variants v1,v2,v3,v4

# CMPNN
cd ../CMPNN
python preprocess_IMDB_cmpnn_lp_skip.py --task md --variant v1,v3 \
       --shared-npz IMDB_md_shared_splits.npz
python preprocess_IMDB_cmpnn_lp_skip.py --task ml --variant v1,v2,v3,v4 \
       --shared-npz IMDB_ml_shared_splits.npz
python run_CMPNN_IMDB_md.py --variant v1
python run_CMPNN_IMDB_ml.py --variant v1
python run_CMPNN_IMDB_md_skip.py --variants v1,v3
python run_CMPNN_IMDB_ml_skip.py --variants v1,v2,v3,v4
```

## Outputs

- Preprocessed data → `data/preprocessed/...` (gitignored)
- Trained checkpoints → `src/baselines/<name>/checkpoint/` (gitignored)
- Score CSVs → `src/baselines/<name>/*_scores.csv` (gitignored)
