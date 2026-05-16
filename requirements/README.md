# Environments

Two environments cover everything in this repo. They can share a single
Python 3.11 environment because CMPNN's extras (`torchdrug`) coexist with
the MAGNN/RGCN/SeHGNN/SlotGAT base.

## Option A — single combined environment (recommended)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements/cmpnn.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('punkt_tab')"
export DGLBACKEND=pytorch
```

Add `export DGLBACKEND=pytorch` to your shell rc file so DGL doesn't fall
back to MXNet.

## Option B — conda environment named `gnn`

```bash
conda create -n gnn python=3.11 -y
conda activate gnn
pip install -r requirements/cmpnn.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('punkt_tab')"
export DGLBACKEND=pytorch
```

## SeHGNN extras

SeHGNN additionally needs PyG extensions (`torch-sparse`, `torch-scatter`).
Install them after the base env is set up:

```bash
pip install -r src/baselines/SeHGNN/requirements-pyg.txt
```
