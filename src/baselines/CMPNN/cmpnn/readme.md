Run commands for DBLP.

**All-in-one runner** (in-memory preprocessing + CMPNN; default `--neg-k 19` sampled negatives per query):

```bash
cd ..   # CMPNN repo root
PYTHONPATH=. python run_CMPNN_DBLP_pc.py \
  --raw-dir ../MAGNN/data/raw/DBLP \
  --shared-npz ../MAGNN/data/preprocessed/DBLP_shared_splits/DBLP_pc_shared_splits.npz \
  --neg-k 19 \
  --batch-size 8 --epoch 20
```

---

1. First run the preprocessing (optional; for `script/run_cmpnn_dblp_var1.py` only):
python cmpnn/preprocess_dblp_var1.py \
  --raw_dir ../MAGNN/data/raw/DBLP \
  --shared_npz ../MAGNN/data/preprocessed/DBLP_shared_splits/DBLP_pc_shared_splits.npz \
  --out_dir ./cmpnn_data/DBLP_pc_var1 \
  --seed 1566911444 \
  --neg_k 50

2. Run the run files: 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python script/run_cmpnn_dblp_var1.py \
  --data_dir ./cmpnn_data/DBLP_pc_var1 \
  --seed 1566911444 \
  --batch_size 8 \
  --epochs 20 \
  --input_dim 32 \
  --hidden_dim 32 \
  --layers 3
