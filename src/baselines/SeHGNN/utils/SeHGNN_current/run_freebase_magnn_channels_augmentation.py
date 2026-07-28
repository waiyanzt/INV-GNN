#!/usr/bin/env python3
"""Shared-checkpoint SeHGNN data augmentation for Freebase NC per-variant K.

This runner intentionally supports only the ``k`` flavor: each graph retains
its own native channels through K.  It does not train ``full_k`` or
``restricted_k`` universal inputs.

A canonical semantic channel identity is recovered from each K manifest
(node-type path for ``channel_identity=type`` or relation signature for
``channel_identity=relation``).  The union of those identities defines one
SeHGNN architecture.  Missing channels are structural-zero SparseTensors and
are explicitly masked, so one optimizer/checkpoint can safely visit every
variant without parameter-shape changes or absent-channel bias leakage.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torch_sparse import SparseTensor
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Freebase SeHGNN augmentation requires torch_sparse matching the installed PyTorch/CUDA build."
    ) from exc

from model import SeHGNN
from sehgnn_augmentation_common import (
    EarlyStopper,
    array_sha256,
    atomic_torch_save,
    atomic_write_csv,
    atomic_write_json,
    classification_metrics,
    cuda_memory_stats,
    load_latest_training_state,
    mean_scalar_dict,
    merge_peak_memory,
    model_memory_stats,
    pairwise_invariance,
    process_peak_rss_bytes,
    reset_cuda_peak,
    resolve_device,
    save_latest_training_state,
    set_determinism,
    state_shape_signature,
    torch_load,
)

DEFAULT_VARIANTS = ("unchanged", "exact_2", "exact_3", "range_2_3")


def parse_csv(spec: str) -> List[str]:
    values = [item.strip() for item in spec.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("Expected a non-empty comma-separated list of distinct values")
    return values


def scipy_to_sparse_tensor(matrix: sp.csr_matrix) -> SparseTensor:
    coo = matrix.tocoo()
    return SparseTensor(
        row=torch.from_numpy(coo.row.astype(np.int64, copy=False)),
        col=torch.from_numpy(coo.col.astype(np.int64, copy=False)),
        value=torch.from_numpy(coo.data.astype(np.float32, copy=False)),
        sparse_sizes=(int(matrix.shape[0]), int(matrix.shape[1])),
        is_sorted=False,
    ).coalesce()


def load_preprocessed(data_dir: Path) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, SparseTensor]]:
    manifest_path = data_dir / "channels_manifest.json"
    dataset_path = data_dir / "dataset.npz"
    if not manifest_path.is_file() or not dataset_path.is_file():
        raise FileNotFoundError(f"Expected channels_manifest.json and dataset.npz under {data_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("flavor")) != "k":
        raise ValueError(f"{data_dir} has flavor={manifest.get('flavor')!r}; augmentation supports k only")
    loaded = np.load(dataset_path)
    dataset = {key: loaded[key] for key in loaded.files}
    features: Dict[str, SparseTensor] = {}
    for channel in manifest["channels"]:
        local_key = str(channel["model_key"])
        matrix = sp.load_npz(data_dir / channel["matrix_file"]).tocsr().astype(np.float32)
        expected = tuple(int(x) for x in channel["shape"])
        if matrix.shape != expected:
            raise ValueError(f"Channel shape mismatch for {local_key}: {matrix.shape} vs {expected}")
        features[local_key] = scipy_to_sparse_tensor(matrix)
    return manifest, dataset, features


def resolve_variant_dir(data_root: Path, variant: str) -> Path:
    candidates = [
        data_root / "k" / variant,
        data_root / variant,
    ]
    for candidate in candidates:
        if (candidate / "channels_manifest.json").is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def semantic_channel_id(channel: Mapping[str, Any], identity_mode: str, target_type: int) -> str:
    hop_count = int(channel["hop_count"])
    source_type = int(channel["source_type"])
    if hop_count == 0:
        if source_type != target_type:
            raise ValueError("The identity channel source type must equal the target type")
        return f"identity:{target_type}"
    if identity_mode == "type":
        path = tuple(int(x) for x in channel["node_type_path"])
        if len(path) != hop_count + 1 or path[-1] != source_type:
            raise ValueError(f"Malformed type-path channel: {channel}")
        return "type:" + ">".join(str(x) for x in path)
    if identity_mode == "relation":
        signatures = channel.get("semantic_signatures") or []
        if len(signatures) != 1:
            raise ValueError(
                "Per-variant relation-aware K channels must contain exactly one semantic signature"
            )
        signature = tuple(str(x) for x in signatures[0])
        if len(signature) != hop_count:
            raise ValueError(f"Malformed relation signature: {signature}")
        return "relation:" + "|".join(signature)
    raise ValueError(f"Unsupported channel_identity={identity_mode!r}")


def canonical_model_key(index: int, source_type: int, identity: bool) -> str:
    if identity:
        return str(source_type)
    if not 0 <= int(source_type) <= 9:
        raise ValueError("Repository SeHGNN Freebase keys require single-digit source types")
    return f"C{index:06d}_T{source_type}"


def _assert_same_array(name: str, arrays: Mapping[str, np.ndarray]) -> None:
    names = list(arrays)
    reference = np.asarray(arrays[names[0]])
    for other_name in names[1:]:
        if not np.array_equal(reference, np.asarray(arrays[other_name])):
            raise ValueError(f"{name} differs between {names[0]} and {other_name}")


def prepare_bundles(
    data_root: Path,
    variants: Sequence[str],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    raw: Dict[str, Dict[str, Any]] = {}
    identity_mode: str | None = None
    global_specs: Dict[str, Dict[str, Any]] = {}
    common_fields: Dict[str, Any] | None = None

    for variant in variants:
        data_dir = resolve_variant_dir(data_root, variant)
        manifest, dataset, local_features = load_preprocessed(data_dir)
        actual_variant = str(manifest.get("variant"))
        if actual_variant != variant:
            raise ValueError(f"Expected variant {variant!r}, got manifest.variant={actual_variant!r}")
        current_mode = str(manifest.get("channel_identity"))
        if identity_mode is None:
            identity_mode = current_mode
        elif current_mode != identity_mode:
            raise ValueError("All variants must use the same channel_identity")

        fields = {
            "pipeline": manifest.get("pipeline"),
            "k": int(manifest["k"]),
            "channel_identity": current_mode,
            "target_type": int(manifest["target_type"]),
            "num_classes": int(manifest["num_classes"]),
            "split_seed": int(manifest.get("split_seed", -1)),
            "node_counts": {str(k): int(v) for k, v in manifest["node_counts"].items()},
        }
        if common_fields is None:
            common_fields = fields
        elif fields != common_fields:
            raise ValueError(f"Manifest configuration differs for {variant}: {fields} vs {common_fields}")

        target_type = int(manifest["target_type"])
        active_by_semantic: Dict[str, SparseTensor] = {}
        channel_audit: List[Dict[str, Any]] = []
        channels_by_key = {str(item["model_key"]): item for item in manifest["channels"]}
        for local_key, feature in local_features.items():
            record = channels_by_key[local_key]
            semantic_id = semantic_channel_id(record, current_mode, target_type)
            if semantic_id in active_by_semantic:
                raise ValueError(f"Duplicate semantic channel {semantic_id} in {variant}")
            source_type = int(record["source_type"])
            shape = tuple(int(x) for x in record["shape"])
            spec = {
                "semantic_id": semantic_id,
                "source_type": source_type,
                "shape": shape,
                "hop_count": int(record["hop_count"]),
                "node_type_path": [int(x) for x in record["node_type_path"]],
                "semantic_signatures": record.get("semantic_signatures") or [],
            }
            if semantic_id in global_specs:
                prior = global_specs[semantic_id]
                if prior["source_type"] != source_type or tuple(prior["shape"]) != shape:
                    raise ValueError(f"Channel {semantic_id} is incompatible across variants")
            else:
                global_specs[semantic_id] = spec
            active_by_semantic[semantic_id] = feature
            channel_audit.append({"local_model_key": local_key, **spec})

        raw[variant] = {
            "name": variant,
            "data_dir": str(data_dir),
            "manifest": manifest,
            "dataset": dataset,
            "active_by_semantic": active_by_semantic,
            "channel_audit": channel_audit,
            "edge_count": int(manifest.get("graph_audit", {}).get("total_directed_edges", 0)),
        }

    assert common_fields is not None and identity_mode is not None
    for key in ("labels", "train_idx", "val_idx", "test_idx"):
        _assert_same_array(key, {name: bundle["dataset"][key] for name, bundle in raw.items()})
    for optional_key in ("target_global_ids", "primary_target_nodes", "target_idx"):
        present = {name: bundle["dataset"][optional_key] for name, bundle in raw.items() if optional_key in bundle["dataset"]}
        if present:
            if len(present) != len(raw):
                raise ValueError(f"{optional_key} is not present for every variant")
            _assert_same_array(optional_key, present)

    identity_id = f"identity:{common_fields['target_type']}"
    if identity_id not in global_specs:
        raise ValueError("The target identity channel is missing")
    ordered_ids = [identity_id] + sorted(key for key in global_specs if key != identity_id)
    semantic_to_model_key: Dict[str, str] = {}
    global_channels: List[Dict[str, Any]] = []
    for index, semantic_id in enumerate(ordered_ids):
        spec = global_specs[semantic_id]
        model_key = canonical_model_key(index, int(spec["source_type"]), identity=(index == 0))
        semantic_to_model_key[semantic_id] = model_key
        global_channels.append({"model_key": model_key, **spec})

    for variant, bundle in raw.items():
        bundle["features"] = {
            semantic_to_model_key[semantic_id]: feature
            for semantic_id, feature in bundle.pop("active_by_semantic").items()
        }
        bundle["active_model_keys"] = sorted(bundle["features"])

    architecture = {
        **common_fields,
        "strategy": "canonical_union_channels_with_explicit_absence_mask",
        "num_channels": len(global_channels),
        "feature_keys": [record["model_key"] for record in global_channels],
        "channels": global_channels,
        "semantic_to_model_key": semantic_to_model_key,
    }
    return raw, architecture


def empty_sparse_batch(batch_size: int, source_count: int, device: torch.device) -> SparseTensor:
    zero = SparseTensor(
        row=torch.empty(0, dtype=torch.long),
        col=torch.empty(0, dtype=torch.long),
        value=torch.empty(0, dtype=torch.float32),
        sparse_sizes=(int(batch_size), int(source_count)),
        is_sorted=True,
    )
    return zero.to(device)


def batch_sparse_channels(
    model: nn.Module,
    bundle: Mapping[str, Any],
    architecture: Mapping[str, Any],
    batch_cpu: torch.Tensor,
    device: torch.device,
) -> Dict[str, SparseTensor]:
    source_by_key = {record["model_key"]: int(record["source_type"]) for record in architecture["channels"]}
    node_counts = {str(k): int(v) for k, v in architecture["node_counts"].items()}
    output: Dict[str, SparseTensor] = {}
    for key in model.feat_keys:
        if key in bundle["features"]:
            output[key] = bundle["features"][key][batch_cpu].to(device)
        else:
            source_type = source_by_key[key]
            output[key] = empty_sparse_batch(len(batch_cpu), node_counts[str(source_type)], device)
    return output


def channel_mask(model: nn.Module, bundle: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    active = set(bundle["features"])
    mask = torch.tensor(
        [1.0 if key in active else 0.0 for key in model.feat_keys],
        dtype=torch.float32,
        device=device,
    )
    if mask.numel() != model.num_channels:
        raise AssertionError("Channel mask does not match the shared model")
    target_key = str(model.tgt_type)
    if target_key not in active:
        raise ValueError("Every variant must contain the target identity channel")
    return mask


def make_model(args: argparse.Namespace, architecture: Mapping[str, Any], device: torch.device) -> nn.Module:
    model = SeHGNN(
        dataset="Freebase",
        nfeat=args.embed_size,
        hidden=args.hidden,
        nclass=int(architecture["num_classes"]),
        feat_keys=architecture["feature_keys"],
        label_feat_keys=[],
        tgt_type=str(architecture["target_type"]),
        dropout=args.dropout,
        input_drop=args.input_drop,
        att_drop=args.att_drop,
        n_fp_layers=args.n_fp_layers,
        n_task_layers=args.n_task_layers,
        act=args.act,
        residual=args.residual,
        data_size={int(k): int(v) for k, v in architecture["node_counts"].items()},
        num_heads=args.num_heads,
    )
    return model.to(device)


def parameter_estimate(args: argparse.Namespace, architecture: Mapping[str, Any]) -> int:
    """Exact parameter-count formula for this Freebase SeHGNN configuration."""
    channels = int(architecture["num_channels"])
    embed = int(args.embed_size)
    hidden = int(args.hidden)
    num_classes = int(architecture["num_classes"])

    # One dense source-node embedding table per node type.
    value = sum(int(count) * embed for count in architecture["node_counts"].values())

    # Feature projection: LinearPerMetapath + LayerNorm + PReLU per layer.
    for layer_index in range(int(args.n_fp_layers)):
        cin = embed if layer_index == 0 else hidden
        value += channels * cin * hidden                 # per-channel weights
        value += channels * hidden                       # per-channel bias
        value += 2 * channels * hidden                   # LayerNorm weight/bias
        value += 1                                       # PReLU

    quarter = hidden // 4
    value += hidden * quarter + quarter                  # query
    value += hidden * quarter + quarter                  # key
    value += hidden * hidden + hidden                    # value
    value += 1                                           # gamma
    value += channels * hidden * hidden + hidden         # fc_after_concat
    if args.residual:
        value += embed * hidden + hidden

    value += 1                                           # first task PReLU
    for _ in range(max(0, int(args.n_task_layers) - 1)):
        value += hidden * hidden + hidden + 1             # Linear + PReLU
    value += hidden * num_classes + num_classes          # final classifier
    return int(value)


def predict_indices(
    model: nn.Module,
    bundle: Mapping[str, Any],
    architecture: Mapping[str, Any],
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    mask: torch.Tensor,
) -> torch.Tensor:
    model.eval()
    outputs: List[torch.Tensor] = []
    loader = DataLoader(torch.as_tensor(indices, dtype=torch.long), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_cpu in loader:
            feats = batch_sparse_channels(model, bundle, architecture, batch_cpu, device)
            outputs.append(model(batch_cpu.to(device), feats, {}, mask).cpu())
    if not outputs:
        return torch.empty((0, int(architecture["num_classes"])), dtype=torch.float32)
    return torch.cat(outputs, dim=0)


def train_variant_epoch(
    args: argparse.Namespace,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    bundle: Mapping[str, Any],
    architecture: Mapping[str, Any],
    rng: np.random.RandomState,
    device: torch.device,
    mask: torch.Tensor,
) -> Tuple[float, int]:
    labels = torch.from_numpy(bundle["dataset"]["labels"].astype(np.int64, copy=False))
    permutation = rng.permutation(bundle["dataset"]["train_idx"].astype(np.int64, copy=False))
    total_loss = 0.0
    total_count = 0
    updates = 0
    model.train()
    for start in range(0, len(permutation), args.batch_size):
        batch_cpu = torch.as_tensor(permutation[start : start + args.batch_size], dtype=torch.long)
        feats = batch_sparse_channels(model, bundle, architecture, batch_cpu, device)
        targets = labels[batch_cpu].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_cpu.to(device), feats, {}, mask)
        loss = criterion(logits, targets)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        count = int(batch_cpu.numel())
        total_loss += float(loss.detach().cpu()) * count
        total_count += count
        updates += 1
    return total_loss / max(total_count, 1), updates


def evaluate_variant(
    args: argparse.Namespace,
    model: nn.Module,
    bundle: Mapping[str, Any],
    architecture: Mapping[str, Any],
    split: str,
    device: torch.device,
    mask: torch.Tensor,
) -> Tuple[Dict[str, Any], torch.Tensor]:
    indices = bundle["dataset"][f"{split}_idx"].astype(np.int64, copy=False)
    labels = torch.from_numpy(bundle["dataset"]["labels"].astype(np.int64, copy=False))[torch.as_tensor(indices)]
    logits = predict_indices(model, bundle, architecture, indices, args.eval_batch_size, device, mask)
    return classification_metrics(logits, labels), logits


def save_variant_outputs(
    seed_dir: Path,
    variant: str,
    bundle: Mapping[str, Any],
    split_logits: Mapping[str, torch.Tensor],
    seed: int,
    split_seed: int,
) -> Dict[str, str]:
    dataset = bundle["dataset"]
    labels = dataset["labels"].astype(np.int64, copy=False)
    arrays: Dict[str, np.ndarray] = {
        "training_seed": np.asarray(seed, dtype=np.int64),
        "split_seed": np.asarray(split_seed, dtype=np.int64),
    }
    for split in ("train", "val", "test"):
        idx = dataset[f"{split}_idx"].astype(np.int64, copy=False)
        arrays[f"{split}_idx"] = idx
        arrays[f"{split}_labels"] = labels[idx]
        arrays[f"{split}_logits"] = split_logits[split].numpy().astype(np.float32, copy=False)
    for global_key in ("target_global_ids", "primary_target_nodes", "target_idx"):
        if global_key in dataset:
            global_ids = np.asarray(dataset[global_key], dtype=np.int64)
            arrays["target_global_ids"] = global_ids
            for split in ("train", "val", "test"):
                arrays[f"{split}_global_ids"] = global_ids[arrays[f"{split}_idx"]]
            break
    logits_path = seed_dir / f"{variant}_logits.npz"
    np.savez_compressed(logits_path, **arrays)

    test_logits = arrays["test_logits"]
    probabilities = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
    frame = pd.DataFrame(
        {
            "node_id": arrays.get("test_global_ids", arrays["test_idx"]),
            "local_target_id": arrays["test_idx"],
            "label": arrays["test_labels"],
            "prediction": probabilities.argmax(axis=1),
            "confidence": probabilities.max(axis=1),
        }
    )
    for class_id in range(probabilities.shape[1]):
        frame[f"prob_class_{class_id}"] = probabilities[:, class_id]
        frame[f"logit_class_{class_id}"] = test_logits[:, class_id]
    score_path = seed_dir / f"{variant}_test_scores.csv"
    frame.to_csv(score_path, index=False)
    return {"logits_file": str(logits_path), "test_scores_file": str(score_path)}


def run_seed(
    args: argparse.Namespace,
    bundles: Mapping[str, Dict[str, Any]],
    architecture: Mapping[str, Any],
    seed: int,
    output_root: Path,
) -> Dict[str, Any]:
    set_determinism(seed)
    device = resolve_device(args.cpu, args.gpu)
    estimate = parameter_estimate(args, architecture)
    if estimate > args.max_estimated_parameters and not args.allow_large_model:
        raise RuntimeError(
            f"Estimated shared model parameters {estimate:,} exceed --max-estimated-parameters="
            f"{args.max_estimated_parameters:,}. Pass --allow-large-model after checking memory."
        )
    model = make_model(args, architecture, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta)
    rng = np.random.RandomState(seed)
    masks = {name: channel_mask(model, bundle, device) for name, bundle in bundles.items()}
    parameter_signature = state_shape_signature(model)
    parameter_stats = model_memory_stats(model)
    if int(parameter_stats["parameter_count"]) != int(estimate):
        raise AssertionError(
            f"Parameter-count formula mismatch: estimated={estimate:,}, "
            f"actual={parameter_stats['parameter_count']:,}"
        )

    seed_dir = output_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = seed_dir / "shared_checkpoint.pt"
    latest_path = seed_dir / "latest_training_state.pt"
    run_config = {
        "dataset": "Freebase NC",
        "flavor": "k",
        "variants": list(bundles),
        "seed": seed,
        "architecture": architecture,
        "model": {
            "embed_size": args.embed_size,
            "hidden": args.hidden,
            "n_fp_layers": args.n_fp_layers,
            "n_task_layers": args.n_task_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "input_drop": args.input_drop,
            "att_drop": args.att_drop,
            "act": args.act,
            "residual": args.residual,
        },
        "training": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "grad_clip": args.grad_clip,
            "selection_metric": "mean_validation_cross_entropy",
        },
        "data_dirs": {name: bundle["data_dir"] for name, bundle in bundles.items()},
    }

    history: List[Dict[str, Any]] = []
    super_epochs_ran = 0
    variant_epochs = 0
    optimizer_steps = 0
    prior_training_seconds = 0.0
    prior_peak_rss = 0
    prior_gpu: Dict[str, Any] = {}
    resume_state = load_latest_training_state(
        latest_path,
        resume=args.resume,
        model=model,
        optimizer=optimizer,
        early_stopper=stopper,
        local_rng=rng,
        run_config=run_config,
        device=device,
    )
    if resume_state is not None:
        history = list(resume_state["history"])
        counters = resume_state["counters"]
        super_epochs_ran = int(resume_state["completed_super_epoch"])
        variant_epochs = int(counters["variant_epochs"])
        optimizer_steps = int(counters["optimizer_steps"])
        prior_training_seconds = float(resume_state["training_seconds_elapsed"])
        prior_peak_rss = int(resume_state.get("peak_rss_bytes", 0))
        prior_gpu = dict(resume_state.get("training_gpu", {}))
        print(f"[resume] seed={seed} after super-epoch {super_epochs_ran}", flush=True)

    variant_names = list(bundles)
    batches_per_variant = {
        name: int(math.ceil(len(bundle["dataset"]["train_idx"]) / args.batch_size))
        for name, bundle in bundles.items()
    }
    updates_per_super_epoch = sum(batches_per_variant.values())
    reset_cuda_peak(device)
    segment_start = time.perf_counter()

    for super_epoch_index in range(super_epochs_ran, args.super_epochs):
        if stopper.should_stop:
            break
        cycle_start = time.perf_counter()
        order = [variant_names[i] for i in rng.permutation(len(variant_names))]
        train_losses: Dict[str, float] = {}
        update_counts: Dict[str, int] = {}
        for name in order:
            loss, updates = train_variant_epoch(
                args, model, optimizer, criterion, bundles[name], architecture, rng, device, masks[name]
            )
            train_losses[name] = loss
            update_counts[name] = updates
            optimizer_steps += updates
            variant_epochs += 1

        val_metrics: Dict[str, Dict[str, Any]] = {}
        for name in variant_names:
            metrics, _ = evaluate_variant(
                args, model, bundles[name], architecture, "val", device, masks[name]
            )
            val_metrics[name] = metrics
        mean_val_loss = float(np.mean([val_metrics[name]["loss"] for name in variant_names]))
        mean_val_macro_f1 = float(np.mean([val_metrics[name]["macro_f1"] for name in variant_names]))
        super_epochs_ran = super_epoch_index + 1
        improved = stopper.update(mean_val_loss, super_epochs_ran)
        if improved:
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "metadata": {
                        "dataset": "Freebase NC",
                        "flavor": "k",
                        "seed": seed,
                        "variants": variant_names,
                        "best_super_epoch": super_epochs_ran,
                        "selection_metric": "mean_validation_cross_entropy",
                        "best_mean_val_loss": stopper.best,
                        "parameter_shape_signature": parameter_signature,
                        "architecture": architecture,
                    },
                },
                checkpoint_path,
            )

        row: Dict[str, Any] = {
            "super_epoch": super_epochs_ran,
            "variant_order": ",".join(order),
            "variant_epochs_cumulative": variant_epochs,
            "optimizer_steps_cumulative": optimizer_steps,
            "updates_per_super_epoch": updates_per_super_epoch,
            "mean_train_loss": float(np.mean(list(train_losses.values()))),
            "mean_val_loss": mean_val_loss,
            "mean_val_macro_f1": mean_val_macro_f1,
            "best_mean_val_loss": stopper.best,
            "bad_super_epochs": stopper.bad_super_epochs,
            "cycle_seconds": time.perf_counter() - cycle_start,
        }
        for name in variant_names:
            row[f"train_loss_{name}"] = train_losses[name]
            row[f"updates_{name}"] = update_counts[name]
            row[f"val_loss_{name}"] = val_metrics[name]["loss"]
            row[f"val_accuracy_{name}"] = val_metrics[name]["accuracy"]
            row[f"val_macro_f1_{name}"] = val_metrics[name]["macro_f1"]
        history.append(row)
        atomic_write_csv(seed_dir / "training_history.csv", history)

        elapsed = prior_training_seconds + (time.perf_counter() - segment_start)
        current_gpu = merge_peak_memory(prior_gpu, cuda_memory_stats(device))
        current_rss = max(prior_peak_rss, process_peak_rss_bytes())
        save_latest_training_state(
            latest_path,
            model=model,
            optimizer=optimizer,
            early_stopper=stopper,
            local_rng=rng,
            run_config=run_config,
            completed_super_epoch=super_epochs_ran,
            counters={"variant_epochs": variant_epochs, "optimizer_steps": optimizer_steps},
            history=history,
            training_seconds_elapsed=elapsed,
            peak_rss_bytes=current_rss,
            training_gpu=current_gpu,
        )
        if super_epochs_ran == 1 or super_epochs_ran % args.log_every == 0 or stopper.should_stop:
            print(
                f"seed={seed} super_epoch={super_epochs_ran:03d} steps={optimizer_steps} "
                f"mean_val_loss={mean_val_loss:.6f} mean_val_macro_f1={mean_val_macro_f1:.4f} "
                f"order={order}",
                flush=True,
            )
        if stopper.should_stop:
            print("Early stopping after a complete balanced super-epoch.", flush=True)
            break

    training_seconds = prior_training_seconds + (time.perf_counter() - segment_start)
    training_gpu = merge_peak_memory(prior_gpu, cuda_memory_stats(device))
    peak_rss = max(prior_peak_rss, process_peak_rss_bytes())
    if not checkpoint_path.exists():
        raise RuntimeError("No best shared checkpoint was saved")
    checkpoint = torch_load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    if state_shape_signature(model) != parameter_signature:
        raise AssertionError("Checkpoint parameter shapes differ from the canonical architecture")
    del checkpoint

    reset_cuda_peak(device)
    inference_start = time.perf_counter()
    per_variant_split_metrics: Dict[str, Dict[str, Any]] = {}
    per_variant_artifacts: Dict[str, Dict[str, str]] = {}
    test_outputs: Dict[str, Dict[str, np.ndarray]] = {}
    for name, bundle in bundles.items():
        split_metrics: Dict[str, Any] = {}
        split_logits: Dict[str, torch.Tensor] = {}
        for split in ("train", "val", "test"):
            metrics, logits = evaluate_variant(
                args, model, bundle, architecture, split, device, masks[name]
            )
            split_metrics[split] = metrics
            split_logits[split] = logits
        split_metrics["graph"] = {
            "edge_count": bundle["edge_count"],
            "active_channels": int(masks[name].sum().item()),
            "global_channels": int(model.num_channels),
            "total_channel_nnz": int(sum(int(record.get("nnz", 0)) for record in bundle["manifest"]["channels"])),
        }
        per_variant_split_metrics[name] = split_metrics
        per_variant_artifacts[name] = save_variant_outputs(
            seed_dir,
            name,
            bundle,
            split_logits,
            seed,
            int(architecture["split_seed"]),
        )
        test_ids = bundle["dataset"]["test_idx"].astype(np.int64, copy=False)
        for global_key in ("target_global_ids", "primary_target_nodes", "target_idx"):
            if global_key in bundle["dataset"]:
                test_ids = np.asarray(bundle["dataset"][global_key], dtype=np.int64)[test_ids]
                break
        test_outputs[name] = {"item_ids": test_ids, "logits": split_logits["test"].numpy()}

    invariance = pairwise_invariance(test_outputs)
    pd.DataFrame(invariance).to_csv(seed_dir / "pairwise_invariance.csv", index=False)
    pd.DataFrame(
        [{"variant": name, **mean_scalar_dict([metrics["test"]])} for name, metrics in per_variant_split_metrics.items()]
    ).to_csv(seed_dir / "test_metrics_by_variant.csv", index=False)
    inference_seconds = time.perf_counter() - inference_start
    inference_gpu = cuda_memory_stats(device)

    expected_steps = super_epochs_ran * updates_per_super_epoch
    expected_variant_epochs = super_epochs_ran * len(variant_names)
    if optimizer_steps != expected_steps or variant_epochs != expected_variant_epochs:
        raise AssertionError(
            f"Epoch accounting mismatch: optimizer steps {optimizer_steps}/{expected_steps}; "
            f"variant epochs {variant_epochs}/{expected_variant_epochs}"
        )
    mean_test_metrics = mean_scalar_dict(
        [per_variant_split_metrics[name]["test"] for name in variant_names]
    )
    reference_dataset = next(iter(bundles.values()))["dataset"]
    summary = {
        "format_version": 1,
        "dataset": "Freebase NC",
        "model": "SeHGNN shared graph-variant augmentation",
        "flavor": "k",
        "seed": seed,
        "variants": variant_names,
        "data_dirs": {name: bundle["data_dir"] for name, bundle in bundles.items()},
        "parameter_compatibility": {
            "strategy": architecture["strategy"],
            "parameter_shape_signature": parameter_signature,
            "same_checkpoint_for_all_variants": True,
            "global_channels": int(architecture["num_channels"]),
            "per_variant_active_channels": {name: int(masks[name].sum().item()) for name in variant_names},
        },
        "architecture": architecture,
        "estimated_parameters": estimate,
        "hyperparameters": {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (str, int, float, bool)) and key not in {"resume"}
        },
        "selection_metric": "mean_validation_cross_entropy",
        "best_mean_val_loss": stopper.best,
        "best_super_epoch": stopper.best_super_epoch,
        "epoch_accounting": {
            "definition": "one super-epoch visits every K-graph variant once; every label minibatch performs one optimizer update",
            "super_epochs_ran": super_epochs_ran,
            "variant_epochs_ran": variant_epochs,
            "batches_per_variant": batches_per_variant,
            "updates_per_super_epoch": updates_per_super_epoch,
            "optimizer_steps": optimizer_steps,
            "expected_optimizer_steps": expected_steps,
        },
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "mean_test_metrics": mean_test_metrics,
        "per_variant_split_metrics": per_variant_split_metrics,
        "pairwise_invariance": invariance,
        "artifacts": {
            "shared_checkpoint": str(checkpoint_path),
            "latest_training_state": str(latest_path),
            "training_history": str(seed_dir / "training_history.csv"),
            "per_variant": per_variant_artifacts,
        },
        "split_fingerprints": {
            "labels_sha256": array_sha256(reference_dataset["labels"]),
            **{
                f"{split}_idx_sha256": array_sha256(reference_dataset[f"{split}_idx"])
                for split in ("train", "val", "test")
            },
        },
        "memory": {
            **parameter_stats,
            "checkpoint_bytes": int(checkpoint_path.stat().st_size),
            "checkpoint_mib": float(checkpoint_path.stat().st_size) / (1024.0 ** 2),
            "process_peak_rss_bytes": peak_rss,
            "process_peak_rss_mib": float(peak_rss) / (1024.0 ** 2),
            "training_gpu": training_gpu,
            "inference_gpu": inference_gpu,
        },
    }
    atomic_write_json(seed_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shared-checkpoint SeHGNN augmentation for Freebase NC K channels")
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing k/<variant>")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--output-dir", type=Path, default=Path("results/sehgnn_augmentation/FREEBASE_NC_k"))
    parser.add_argument("--seeds", default="1566911444,20241017,20251017,20261017")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--super-epochs", type=int, default=200)
    parser.add_argument("--embed-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--n-fp-layers", type=int, default=2)
    parser.add_argument("--n-task-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--input-drop", type=float, default=0.0)
    parser.add_argument("--att-drop", type=float, default=0.0)
    parser.add_argument("--act", choices=["none", "relu", "leaky_relu", "sigmoid"], default="none")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--eval-batch-size", type=int, default=20000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min-delta", type=float, default=1e-10)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--max-estimated-parameters", type=int, default=500_000_000)
    parser.add_argument("--allow-large-model", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = parse_csv(args.variants)
    seeds = [int(value) for value in parse_csv(args.seeds)]
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing Freebase K variants {variants} and canonical channels...", flush=True)
    bundles, architecture = prepare_bundles(args.data_root, variants)
    atomic_write_json(args.output_dir / "architecture_manifest.json", architecture)
    for name, bundle in bundles.items():
        print(
            f"  {name}: native_channels={len(bundle['features'])}, "
            f"channel_nnz={sum(int(x.get('nnz', 0)) for x in bundle['manifest']['channels']):,}",
            flush=True,
        )
    print(f"  shared canonical channels={architecture['num_channels']}", flush=True)

    summaries = [run_seed(args, bundles, architecture, seed, args.output_dir) for seed in seeds]
    rows = []
    for summary in summaries:
        rows.append(
            {
                "seed": summary["seed"],
                "super_epochs_ran": summary["epoch_accounting"]["super_epochs_ran"],
                "variant_epochs_ran": summary["epoch_accounting"]["variant_epochs_ran"],
                "optimizer_steps": summary["epoch_accounting"]["optimizer_steps"],
                "training_seconds": summary["training_seconds"],
                "best_mean_val_loss": summary["best_mean_val_loss"],
                **{f"mean_test_{key}": value for key, value in summary["mean_test_metrics"].items()},
            }
        )
    pd.DataFrame(rows).to_csv(args.output_dir / "seed_summary.csv", index=False)
    atomic_write_json(
        args.output_dir / "all_seed_summaries.json",
        {
            "format_version": 1,
            "dataset": "Freebase NC",
            "model": "SeHGNN shared graph-variant augmentation",
            "flavor": "k",
            "variants": variants,
            "seeds": seeds,
            "architecture": architecture,
            "metric_definitions": {
                "classification": "accuracy, balanced accuracy, macro/micro precision and recall, micro/macro/weighted F1, Hit@1, Hit@3, MRR, cross-entropy, per-class metrics, and confusion matrix",
                "selection": "minimum mean validation cross-entropy across variants after each complete super-epoch",
                "kendall_tau": "test-node macro Kendall tau-b over all class logits",
                "kendall_tau_at_1": "test-node top-1 agreement",
                "kendall_tau_at_3": "test-node macro Kendall tau-b on the union of top-3 classes",
            },
            "runs": summaries,
        },
    )
    print(f"[OK] Freebase K augmentation results: {args.output_dir}")


if __name__ == "__main__":
    main()
