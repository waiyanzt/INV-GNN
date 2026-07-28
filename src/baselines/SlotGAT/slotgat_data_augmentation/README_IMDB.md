# SlotGAT joint graph-variant augmentation: IMDb node classification

This experiment trains one shared SlotGAT model jointly on IMDb1--4:

```text
v1: actor-movie, director-movie
v2: actor-link,  director-link
v3: actor-link,  director-movie
v4: actor-movie, director-link
```

All four graphs share the same movie, director, actor, and link nodes, movie
features, labels, and train/validation/test split. The target is movie genre
classification with three classes: Action, Comedy, and Drama.

A full-batch super-epoch makes four optimizer updates, one per graph in
randomized order. Validation happens once after all four updates. Mean
validation NLL across the selected variants chooses the single shared
checkpoint. Thus, with all four variants:

```text
variant_epochs = optimizer_steps = 4 * super_epochs
```

The preprocessed HGB files use compact relation IDs whose meanings differ
between variants. The runner remaps them to one global directed semantic
vocabulary before training, so a shared SlotGAT edge embedding always represents
the same relation.

## One-time preprocessing

Run from `src/baselines/SlotGAT`:

```bash
python preprocess_IMDB.py
```

This generates `data/IMDB_var1` through `data/IMDB_var5`. The augmentation
experiment intentionally uses only variants 1--4.

## Production command

Run from `src/baselines/SlotGAT`:

```bash
python slotgat_data_augmentation/run_IMDB_slotgat_augmentation.py \
  --variants v1,v2,v3,v4 \
  --seeds 1566911444,20241017,20251017 \
  --super-epochs 300 \
  --patience 40 \
  --feats-type 0 \
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
  --output-dir results/slotgat_augmentation/IMDB \
  --resume
```

Patience is counted in completed super-epochs. Because this runner validates
after every super-epoch, `--patience 40` means 40 consecutive balanced
IMDb1--4 cycles without a lower mean validation loss.

## Output contract

Each `seed_<seed>/` directory contains:

```text
shared_checkpoint.pt
latest_training_state.pt
training_history.csv
test_scores_v1.csv
test_scores_v2.csv
test_scores_v3.csv
test_scores_v4.csv
pairwise_invariance.csv
test_metrics_by_variant.csv
summary.json
```

The output root additionally contains `seed_summary.csv` and
`all_seed_summaries.json`. All four test score files come from the same best
shared checkpoint; they are evaluations of that model on the four graph
variants, not four separately trained models.
