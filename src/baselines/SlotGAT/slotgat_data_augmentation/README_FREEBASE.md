# SlotGAT joint graph-variant augmentation: Freebase node classification

This experiment trains one shared SlotGAT model jointly on:

```text
unchanged
exact_2
```

The target is BOOK (node type 0) with the canonical eight Freebase classes. A
full-batch super-epoch performs two optimizer updates, one per graph in
randomized order. Mean validation NLL across both variants selects the shared
checkpoint.

## One-time preprocessing

Run from `src/baselines/SlotGAT`:

```bash
python slotgat_data_augmentation/preprocess_FREEBASE_slotgat_augmentation.py
```

The preprocessor creates one shared stratified BOOK split and a global
forward/reverse relation vocabulary. `exact_2` contains 47,368,991 raw edges,
so preprocessing should run on a high-memory CPU node with at least 128 GB of
RAM.

## Production command

```bash
python slotgat_data_augmentation/run_FREEBASE_slotgat_augmentation.py \
  --variants unchanged,exact_2 \
  --seeds 1566911444,20241017,20251017 \
  --super-epochs 100 \
  --patience 30 \
  --label-batch-size 0 \
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
  --lr 0.005 \
  --weight-decay 0.001 \
  --grad-clip 0 \
  --device cuda \
  --output-dir results/slotgat_augmentation/FREEBASE \
  --resume
```

The convolution computes edge-attention contributions per relation before
gathering them per edge. This is algebraically equivalent to the reference
SlotGAT calculation but avoids its roughly 181 GiB
`E x heads x edge_features` intermediate on `exact_2`.

Request an 80 GB accelerator for the initial production run. Even after the
edge-score fusion, `exact_2` has about 94.7 million directed edges and requires
multi-gigabyte graph, attention, and autograd tensors. A 40 GB accelerator is
not recommended without a measured successful run.

Each `seed_<seed>/` directory receives the best checkpoint, exact resume state,
training history, per-variant metrics and score tables, pairwise invariance
metrics, runtime, and CPU/GPU memory telemetry.
