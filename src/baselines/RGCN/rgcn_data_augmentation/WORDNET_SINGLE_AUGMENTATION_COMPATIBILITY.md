# WordNet single-variant and augmentation compatibility

## Compatible components

The package now copies the updated WordNet preprocessor, model, and dataset
loader directly:

- `preprocess_WORDNET_rgcn_augmentation.py`
- `model_RGCN_lp_wordnet.py`
- `wordnet_lp.py`

The augmentation runner uses the same NPZ arrays, graph construction, model,
optimizer parameter groups, online negatives, filtered-ranking dictionaries,
ranking tie rule, binary negative sampler, metrics, and default hyperparameters
as the updated standalone runner.

## Early stopping alignment

The standalone runner validates only when `epoch % eval_interval == 0`, treats
any strict filtered-MRR increase as improvement, resets patience on improvement,
and increments patience by `eval_interval` after each failed validation.

The augmentation runner now applies the direct shared-checkpoint analogue:

- validate only when `super_epoch % eval_interval == 0`;
- compare `mean_filtered_MRR > best_mean_filtered_MRR` with no epsilon;
- reset patience after improvement;
- stop after `ceil(patience / eval_interval)` failed validation checks.

No extra final validation is forced when the requested super-epoch count is not
divisible by the evaluation interval. If a short diagnostic run never reaches a
validation boundary, final weights are evaluated, matching the standalone
runner's behavior.

## Necessary Protocol A differences

The standalone code trains one independent model per variant. Protocol A trains
one shared model and optimizer by rotating all four graph variants. Therefore,
training trajectories and final values are not expected to match the independent
models. The purpose of compatibility is to keep the model, data, loss, negative
sampling, evaluation, and defaults controlled so the comparison isolates the
augmentation protocol.

## Validation performed

On a synthetic leakage-free NPZ, the augmentation runner restricted to
`--variants no_changes` produced a checkpoint with every tensor exactly equal to
the standalone runner under the same seed and settings. The complete metric
dictionaries were also exactly equal. A four-variant interrupted/resumed run
matched an uninterrupted run exactly in model tensors and cumulative counters in
the CPU validation environment.


## Updated early-stopping defaults

Both packaged WordNet runners now default to:

```text
patience = 30
eval_interval = 1
```

Thus the standalone runner validates every epoch and the augmentation runner
validates every super-epoch. Both stop after 30 consecutive failed validation
checks. This intentionally differs from the uploaded standalone file's former
defaults of patience 3000 and evaluation every 10 epochs. Exact one-variant
equivalence still holds when both paths use the same two values.
