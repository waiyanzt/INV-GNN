# Legacy compatibility notes

## Freebase

The augmentation runner uses the attached `RGCNFeatureless` model parameters
and computation:

- learned `num_nodes x hidden_dim` node parameter matrix;
- two PyG `RGCNConv` layers;
- ReLU after the first layer;
- optional dropout between layers;
- `log_softmax` output;
- `NLLLoss`;
- Adam over all model parameters with the legacy learning rate and weight decay;
- checkpoint selection by validation NLL.

The optional `--edge-chunk-size` backend subclasses PyG `RGCNConv` and retains
the same parameter/state-dict layout. It evaluates each relation's complete
mean aggregation in bounded edge chunks and uses a custom backward that
recomputes those aggregates instead of retaining per-chunk autograd state. No
edge is sampled or omitted. This exchanges additional compute time for lower
peak CUDA memory. Because floating-point additions are grouped into chunks,
results can differ at normal floating-point roundoff scale. The included
`verify_FREEBASE_chunked_rgcn.py` script checks forward and gradient agreement
against the installed PyG implementation.

The defaults match the old runner: hidden dimension 64, 30 bases, dropout 0,
learning rate 0.001, weight decay 0.001, patience 30, and no gradient clipping.

The one unavoidable joint-model difference is relation layout. The old
preprocessor used a variant-local reverse offset. Joint training must use one
global offset so a shared relation slot has one meaning in every graph. Forward
relation IDs are unchanged. If all variants already have the same forward
relation count, the produced IDs are identical to the legacy preprocessing.
Otherwise, unused relation rows are padded in smaller variants.

Freebase therefore requires the same `torch-geometric` model dependency used by
the old experiment, in addition to DGL used by the IMDb/DBLP runners.

## WordNet

The augmentation runner now uses the exact updated single-variant model file
and loads each graph through the exact updated `WordNetLPDataset` loader:

- one 200-dimensional RGCN encoding layer;
- basis decomposition (runner default: 30 bases, matching the old run script);
- node-wise incoming-edge normalization `c_i`;
- neighbor-edge dropout 0.4;
- separate root/self-loop dropout 0.2;
- DistMult decoder;
- weight decay 0.01 on relation embeddings only;
- one training negative per positive;
- no gradient clipping by default;
- checkpoint selection by mean legacy-style validation filtered MRR across all four variants.

The preprocessing source has now changed to the leakage-free official-split
protocol. It uses the original `train.txt`, `valid.txt`, and `test.txt` files,
one shared relation dictionary, and four training arrays: no-change, inverse,
transitive, and universal. The inverse and universal graphs are explicitly
checked for inverse edges derived from held-out validation/test triples. This is
a deliberate data-correction change, so the evaluation split is no longer the
old graph-intersection split.

Two evaluation protocols are intentionally reported:

1. **Legacy-compatible evaluation** uses each variant's own
   `train + validation + test` filter dictionary and the old deterministic
   50-negative binary procedure. These metrics are directly comparable with per-variant runs that use the same
   corrected leakage-free NPZ. They should not be compared as if the earlier leaked
   split were unchanged.
2. **Shared-candidate evaluation** uses one fixed candidate table for every
   graph. This is an additional invariance measurement and does not replace the
   legacy metrics or determine the checkpoint.

The current WordNet package intentionally uses patience 30 and evaluation every 1 epoch/super-epoch for both standalone and augmentation commands. This no longer preserves the older 3000/10 scheduling default, but it keeps the two comparison paths aligned with each other. The model, data, loss, ranking metrics, and checkpoint criterion remain legacy-compatible.

## Resume compatibility

Because model structure, selection metrics, and run configurations changed,
resume states from an earlier augmentation package must not be reused. Start in
a new output directory. Resume states created by this package restore the model,
optimizer, RNG, history, counters, and early-stopping state with `--resume`.

## IMDb link prediction

`run_IMDB_rgcn_lp_augmentation.py` follows the repository's standalone IMDb-LP
model and evaluation protocol: the same learned homogeneous embedding table,
three-layer DGL RGCN encoder, dot-product decoder, pairwise training loss,
embedding regularization, fixed preprocessed negative-tail matrices,
validation BCE, threshold metrics, and row-wise Hits@1/3/5 and MRR.

The required augmentation-specific changes are:

1. one encoder/optimizer/checkpoint is shared across graph variants;
2. one super-epoch processes every selected variant before validation;
3. checkpoint selection minimizes mean validation BCE across variants;
4. relation names use a global union vocabulary so a parameter row retains the
   same meaning when a variant omits a relation;
5. pairwise score/rank invariance and exact resume state are additionally saved.

The shared positive and negative splits are asserted byte-identical across
variants before training.
