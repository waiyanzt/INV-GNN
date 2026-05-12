# INV-GNN Experiment Code

This repository contains the experiment code for **Information-based Invariance, Equivariance, and Expressivity for Graph Neural Networks**. The code evaluates whether graph neural network baselines remain stable across non-isomorphic graph representations that encode the same information, and includes invariant/skip adaptations of the evaluated models.

The paper evaluates the method on IMDb and DBLP variants using node classification and link prediction tasks. The included baselines cover local message passing, path-conditioned propagation, metapath-instance aggregation, metapath-feature propagation, and node-type-aware aggregation.

## Repository layout

```text
src/
└── baselines/
    ├── MAGNN/          # MAGNN baseline, MAGNN skip variants, and RGCN experiment scripts
    ├── CMPNN/          # CMPNN baseline and CMPNN skip variants
    ├── SlotGAT/        # SlotGAT IMDb node-classification baseline and skip/universal variant support
    ├── SeHGNN/         # SeHGNN IMDb node-classification baseline and skip/universal variant support
    └── rgcn-baseline/  # Additional standalone RGCN baseline utilities
```

In the scripts and notes, **skip** refers to the invariant algorithm proposed in the paper. Original scripts run the baseline model on individual dataset variants; skip scripts run the invariant/skip version designed to produce stable predictions across equivalent variants.

## Data

Download the experiment data from:

```text
https://drive.google.com/file/d/1AQUhIOXVHdvuGcEsqDXcR_S3bAmQThnt/view?usp=sharing
```

After downloading, place or extract the data so that each baseline can find its expected `data/` and `data/preprocessed/` directories. Most scripts assume they are run from the corresponding baseline directory, for example:

```bash
cd src/baselines/MAGNN
mkdir -p data/preprocessed checkpoint
```

The paper uses IMDb variants for node classification and link prediction, and DBLP variants for link prediction. The IMDb variants include movie, actor, director, and link nodes; the DBLP variants include author, paper, term, venue, and area nodes.

## Environments

Different baselines use different dependency stacks. Use a separate virtual environment for each dependency group.

### MAGNN and RGCN environment

MAGNN and the RGCN scripts in `src/baselines/MAGNN/` use the same environment.

```bash
cd src/baselines/MAGNN

# Recommended Python version from the project notes
python --version   # should be 3.11.4
python -m venv ./env
source ./env/bin/activate

pip install numpy
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
pip install scikit-learn
pip install nltk
```

If NLTK reports a missing WordNet resource:

```bash
python - <<'PY'
import nltk
nltk.download('wordnet')
PY
```

Set the DGL backend to PyTorch. For an interactive shell:

```bash
export DGLBACKEND=pytorch
```

To make this permanent, add the following line to your shell startup file such as `~/.bashrc`:

```bash
export DGLBACKEND=pytorch
```

> Note: `src/baselines/MAGNN/requirements.txt` is inherited from the original MAGNN code and pins older versions such as PyTorch 1.2.0 and DGL 0.3.1. For this repository's MAGNN/RGCN experiments, prefer the environment above unless you are intentionally reproducing the original MAGNN setup.

### CMPNN environment

CMPNN does not include a project-specific `requirements.txt`. The included CMPNN README from the original paper lists the following installation commands:

```bash
cd src/baselines/CMPNN
python -m venv ./env
source ./env/bin/activate

pip install torch
pip install torchdrug
pip install ogb easydict pyyaml
```

CMPNN scripts can also be run on CPU where supported by passing `--cpu` to this repository's runner scripts, or `--gpus null` for the original `script/run.py` workflow.

### SlotGAT environment

SlotGAT has its own requirements file:

```bash
cd src/baselines/SlotGAT
python -m venv ./env
source ./env/bin/activate
pip install -r requirements.txt
```

### SeHGNN environment

SeHGNN has a base requirements file plus PyG extension requirements. Install them in this order:

```bash
cd src/baselines/SeHGNN
python -m venv ./env
source ./env/bin/activate
pip install -r requirements.txt
pip install -r requirements-pyg.txt
```

## General workflow

For each experiment:

1. Activate the environment for the relevant baseline.
2. Change into the baseline directory.
3. Run the preprocessing script for the dataset/model/variant.
4. Run the corresponding training/evaluation script.

Example pattern:

```bash
cd src/baselines/MAGNN
source ./env/bin/activate
python preprocess_IMDB_star.py
python run_IMDB.py
```

Many runner scripts have useful options for selecting variants, seeds, epochs, and comparison mode. Check the available arguments with:

```bash
python <runner>.py --help
```

Default seeds used by many scripts are:

```text
1566911444,20241017,20251017
```

## Experiment scripts

### 1. IMDb node classification

#### MAGNN

Original preprocessing:

```bash
cd src/baselines/MAGNN
python preprocess_IMDB_star.py
python preprocess_IMDB_star_t.py
python preprocess_IMDB_star_t_2.py
python preprocess_IMDB_star_t_3.py
```

Skip/invariant preprocessing:

```bash
python preprocess_IMDB_skip.py
```

Original runner:

```bash
python run_IMDB.py
```

Skip/invariant runner:

```bash
python run_IMDB_skip.py
```

#### RGCN

Preprocessing:

```bash
cd src/baselines/MAGNN
python preprocess_IMDB_rgcn.py          # original
python preprocess_IMDB_rgcn_skip.py     # skip/invariant
```

Runners:

```bash
python run_IMDB_rgcn.py                 # original
python run_IMDB_rgcn_skip.py            # skip/invariant
```

#### SlotGAT

SlotGAT uses the same preprocessing and runner for original and skip/universal IMDb variants. `IMDB5` is the universal/skip variant.

```bash
cd src/baselines/SlotGAT
python preprocess_IMDB.py
python run_IMDB_nc.py
```

To select specific variants:

```bash
python run_IMDB_nc.py --variant 1 2 3 4 5
```

#### SeHGNN

SeHGNN also uses the same preprocessing and runner for original and skip/universal IMDb variants. `IMDB5` is the universal variant.

```bash
cd src/baselines/SeHGNN
python preprocess_IMDB.py
python run_IMDB_nc.py
```

To select specific variants:

```bash
python run_IMDB_nc.py --variant 1 2 3 4 5 
```

### 2. DBLP link prediction

#### MAGNN

Original preprocessing:

```bash
cd src/baselines/MAGNN
python preprocess_DBLP1.py
python preprocess_DBLP2.py
python preprocess_DBLP3.py
```

Combined original preprocessing/runner helper:

```bash
python preprocess_DBLP_pc_trainpc.py
python run_DBLP_pc_trainpc.py
```

Original runners for individual variants:

```bash
python run_DBLP_pc.py
python run_DBLP_pc_t.py
python run_DBLP_pc_a.py
```

Skip/invariant preprocessing and runner:

```bash
python preprocess_DBLP_skip.py
python run_IMDB_magnn_lp_skip.py
```

#### RGCN

Preprocessing:

```bash
cd src/baselines/MAGNN
python preprocess_DBLP_rgcn.py          # original
python preprocess_DBLP_rgcn_skip.py     # skip/invariant
```

Runners:

```bash
python run_DBLP_rgcn.py                 # original
python run_DBLP_rgcn_skip.py            # skip/invariant
```

#### CMPNN

Preprocessing:

```bash
cd src/baselines/CMPNN
python preprocess_DBLP_cmpnn_pc.py      # original
python preprocess_DBLP_cmpnn_skip.py    # skip/invariant
```

Original runners:

```bash
python run_CMPNN_DBLP_pc.py
python run_CMPNN_DBLP_pc_var2.py
python run_CMPNN_DBLP_pc_var3.py
```

Combined original runner:

```bash
python run_DBLP_cmpnn_pc.py
```

Skip/invariant runner:

```bash
python run_DBLP_cmpnn_skip.py
```

### 3. IMDb link prediction

#### MAGNN

Preprocessing:

```bash
cd src/baselines/MAGNN
python preprocess_IMDB_magnn_lp.py          # original
python preprocess_IMDB_magnn_lp_skip.py     # skip/invariant
```

Runners:

```bash
python run_IMDB_magnn_lp.py --task ml       # Movie-Link original
python run_IMDB_magnn_lp.py --task md       # Movie-Director original
python run_IMDB_magnn_lp_skip.py --task ml  # Movie-Link skip/invariant
python run_IMDB_magnn_lp_skip.py --task md  # Movie-Director skip/invariant
```

#### RGCN

Preprocessing:

```bash
cd src/baselines/MAGNN
python preprocess_IMDB_rgcn_lp.py           # original
python preprocess_IMDB_rgcn_lp_skip.py      # skip/invariant
```

Runners:

```bash
python run_IMDB_rgcn_lp.py                  # original
python run_IMDB_rgcn_lp_skip.py             # skip/invariant
```

#### CMPNN

Original preprocessing:

```bash
cd src/baselines/CMPNN
python build_IMDB_ml_shared_splits.py       # Movie-Link
python build_IMDB_md_shared_splits.py       # Movie-Director
```

Skip/invariant preprocessing:

```bash
python preprocess_IMDB_cmpnn_lp_skip.py
```

Original runners:

```bash
python run_CMPNN_IMDB_ml.py                 # Movie-Link original
python run_CMPNN_IMDB_md.py                 # Movie-Director original
```

Skip/invariant runners:

```bash
python run_CMPNN_IMDB_ml_skip.py            # Movie-Link skip/invariant
python run_CMPNN_IMDB_md_skip.py            # Movie-Director skip/invariant
```

## Metrics and outputs

The paper reports:

- IMDb node classification: Micro-F1 and Macro-F1.
- Link prediction: MRR and Hits@3.
- Invariance: average pairwise Kendall-τ over output-score rankings across equivalent variants.
- Efficiency: average training time.

Runner scripts typically create or use a `checkpoint/` directory and may write comparison files, score CSVs, or logs depending on arguments such as `--compare`, `--compare-only`, `--score-csv-a`, and `--score-csv-b`.

## Reproducibility notes

- Run scripts from their baseline directory unless you update relative paths manually.
- Create `checkpoint/` before running older scripts if the script does not create it automatically.
- Keep separate environments for MAGNN/RGCN, CMPNN, SlotGAT, and SeHGNN to avoid dependency conflicts.
- Use the same seeds across original and skip/invariant variants when comparing invariance.
- For GPU runs, ensure the installed PyTorch, CUDA, DGL, and PyG wheels match your system.
- For CPU-only smoke tests, prefer small epoch counts, for example `--epoch 1` or `--epochs 1`, where supported.

## References

The repository also builds on the original implementations or ideas for MAGNN, CMPNN, RGCN, SlotGAT, and SeHGNN. See the baseline subdirectories for original README files and citation information where available.
