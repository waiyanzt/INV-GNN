#!/usr/bin/env python3
"""Shared-checkpoint SeHGNN data augmentation for IMDB node classification.

Only the original per-graph SeHGNN representation is used (variants 1--4).
A super-epoch shuffles the selected graph variants, trains one normal SeHGNN
minibatch epoch on each graph using the same model/optimizer, then validates the
same checkpoint on every graph.  Early stopping minimizes mean validation
cross-entropy after a complete super-epoch, preserving the existing IMDB
SeHGNN training and stopping criterion.

Parameter compatibility
-----------------------
SeHGNN's projection/concatenation parameters depend on semantic-channel count.
This runner constructs one canonical union of feature and label channels across
all selected variants.  Every graph uses that architecture; channels absent
from a graph are supplied as structural zeros and explicitly masked inside the
model.  Therefore every variant has exactly the same parameter shapes while
absent channels make no contribution.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
from utils.data import load_imdb_graph, load_imdb_nc_labels
from utils.imdb_variant_dir import resolve_imdb_nc_dir
from utils.tools import hg_propagate_feat_dgl

TGT_TYPE = "M"
ORIGINAL_VARIANTS = (1, 2, 3, 4)


def parse_csv_ints(spec: str) -> List[int]:
    values = [int(item.strip()) for item in spec.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("Expected a non-empty comma-separated list of distinct integers")
    return values


def resolve_variant_dir(variant: int, data_root: Path | None) -> Path:
    if data_root is None:
        return Path(resolve_imdb_nc_dir(variant, imdb_skip_data=False)).resolve()
    candidates = [
        data_root / f"IMDB_var{variant}",
        data_root / str(variant),
        data_root / f"v{variant}",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return candidates[0].resolve()


def _assert_same_array(name: str, arrays: Mapping[str, np.ndarray]) -> None:
    names = list(arrays)
    reference = arrays[names[0]]
    for other_name in names[1:]:
        other = arrays[other_name]
        if not np.array_equal(reference, other):
            raise ValueError(f"{name} differs between {names[0]} and {other_name}")


def prepare_variant_raw(args: argparse.Namespace, variant: int) -> Dict[str, Any]:
    base_dir = resolve_variant_dir(variant, args.data_root)
    if not base_dir.is_dir():
        raise FileNotFoundError(base_dir)
    graph, node_counts, _ = load_imdb_graph(str(base_dir))
    labels, train_idx, val_idx, test_idx, num_classes = load_imdb_nc_labels(str(base_dir))

    graph_features, _, _ = load_imdb_graph(str(base_dir))
    graph_features = hg_propagate_feat_dgl(
        graph_features,
        TGT_TYPE,
        args.num_hops,
        args.num_hops + 1,
        [],
        echo=False,
    )
    features = {
        key: graph_features.nodes[TGT_TYPE].data.pop(key).cpu()
        for key in list(graph_features.nodes[TGT_TYPE].data.keys())
    }

    label_features: Dict[str, torch.Tensor] = {}
    if args.label_num_hops > 0:
        graph_labels, _, _ = load_imdb_graph(str(base_dir))
        for ntype in graph_labels.ntypes:
            for key in list(graph_labels.nodes[ntype].data.keys()):
                graph_labels.nodes[ntype].data.pop(key)
        seed = torch.zeros((node_counts[TGT_TYPE], num_classes), dtype=torch.float32)
        seed[train_idx, labels[train_idx].long()] = 1.0
        graph_labels.nodes[TGT_TYPE].data["Y"] = seed
        graph_labels = hg_propagate_feat_dgl(
            graph_labels,
            TGT_TYPE,
            args.label_num_hops,
            args.label_num_hops + 1,
            [],
            echo=False,
        )
        label_features = {
            key: graph_labels.nodes[TGT_TYPE].data[key].cpu()
            for key in list(graph_labels.nodes[TGT_TYPE].data.keys())
        }
        del graph_labels

    result = {
        "name": f"v{variant}",
        "variant": variant,
        "base_dir": str(base_dir),
        "features": features,
        "label_features": label_features,
        "labels": labels.long().cpu(),
        "train_idx": np.asarray(train_idx, dtype=np.int64),
        "val_idx": np.asarray(val_idx, dtype=np.int64),
        "test_idx": np.asarray(test_idx, dtype=np.int64),
        "num_classes": int(num_classes),
        "num_target_nodes": int(node_counts[TGT_TYPE]),
        "node_counts": {str(k): int(v) for k, v in node_counts.items()},
        "edge_count": int(graph.num_edges()),
        "active_feature_keys": sorted(features),
        "active_label_feature_keys": sorted(label_features),
    }
    del graph, graph_features
    gc.collect()
    return result


def prepare_bundles(args: argparse.Namespace, variants: Sequence[int]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    raw = {f"v{variant}": prepare_variant_raw(args, variant) for variant in variants}
    names = list(raw)
    reference = raw[names[0]]
    for name in names[1:]:
        current = raw[name]
        for key in ("num_classes", "num_target_nodes", "node_counts"):
            if current[key] != reference[key]:
                raise ValueError(f"{key} differs between {names[0]} and {name}")

    _assert_same_array("labels", {name: bundle["labels"].numpy() for name, bundle in raw.items()})
    for split in ("train_idx", "val_idx", "test_idx"):
        _assert_same_array(split, {name: bundle[split] for name, bundle in raw.items()})

    feature_dims: Dict[str, int] = {}
    label_dims: Dict[str, int] = {}
    for name, bundle in raw.items():
        for key, tensor in bundle["features"].items():
            dim = int(tensor.shape[1])
            if key in feature_dims and feature_dims[key] != dim:
                raise ValueError(f"Feature channel {key} has inconsistent dimensions")
            feature_dims[key] = dim
        for key, tensor in bundle["label_features"].items():
            dim = int(tensor.shape[1])
            if key in label_dims and label_dims[key] != dim:
                raise ValueError(f"Label channel {key} has inconsistent dimensions")
            label_dims[key] = dim

    global_feature_keys = sorted(feature_dims)
    global_label_keys = sorted(label_dims)
    if TGT_TYPE not in global_feature_keys:
        raise ValueError("The target identity feature channel M is missing")

    architecture = {
        "strategy": "canonical_union_channels_with_explicit_absence_mask",
        "feature_keys": global_feature_keys,
        "label_feature_keys": global_label_keys,
        "feature_dims": feature_dims,
        "label_dims": label_dims,
        "num_feature_channels": len(global_feature_keys),
        "num_label_channels": len(global_label_keys),
        "num_total_channels": len(global_feature_keys) + len(global_label_keys),
        "num_classes": reference["num_classes"],
        "num_target_nodes": reference["num_target_nodes"],
        "node_counts": reference["node_counts"],
    }
    return raw, architecture


def batch_dense_channels(
    active: Mapping[str, torch.Tensor],
    keys: Sequence[str],
    dims: Mapping[str, int],
    batch_cpu: torch.Tensor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    batch_size = int(batch_cpu.numel())
    output: Dict[str, torch.Tensor] = {}
    for key in keys:
        if key in active:
            output[key] = active[key][batch_cpu].to(device)
        else:
            output[key] = torch.zeros((batch_size, int(dims[key])), dtype=torch.float32, device=device)
    return output


def channel_mask(model: nn.Module, bundle: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    active_features = set(bundle["features"])
    active_labels = set(bundle["label_features"])
    values = [1.0 if key in active_features else 0.0 for key in model.feat_keys]
    values += [1.0 if key in active_labels else 0.0 for key in model.label_feat_keys]
    mask = torch.tensor(values, dtype=torch.float32, device=device)
    if mask.numel() != model.num_channels:
        raise AssertionError("Channel-mask length does not match the shared model")
    if TGT_TYPE not in active_features:
        raise ValueError("Every IMDB variant must contain the target movie identity channel M")
    if float(mask.sum().item()) <= 0:
        raise ValueError("A variant has no active channels")
    return mask


def predict_indices(
    model: nn.Module,
    bundle: Mapping[str, Any],
    indices: np.ndarray,
    architecture: Mapping[str, Any],
    batch_size: int,
    device: torch.device,
    mask: torch.Tensor,
) -> torch.Tensor:
    model.eval()
    outputs: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_cpu = torch.as_tensor(indices[start : start + batch_size], dtype=torch.long)
            feats = batch_dense_channels(
                bundle["features"], model.feat_keys, architecture["feature_dims"], batch_cpu, device
            )
            label_feats = batch_dense_channels(
                bundle["label_features"], model.label_feat_keys, architecture["label_dims"], batch_cpu, device
            )
            outputs.append(model(batch_cpu.to(device), feats, label_feats, mask).cpu())
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
    permutation = rng.permutation(bundle["train_idx"])
    total_loss = 0.0
    total_examples = 0
    updates = 0
    labels = bundle["labels"]
    model.train()
    for start in range(0, len(permutation), args.batch_size):
        batch_cpu = torch.as_tensor(permutation[start : start + args.batch_size], dtype=torch.long)
        feats = batch_dense_channels(
            bundle["features"], model.feat_keys, architecture["feature_dims"], batch_cpu, device
        )
        label_feats = batch_dense_channels(
            bundle["label_features"], model.label_feat_keys, architecture["label_dims"], batch_cpu, device
        )
        targets = labels[batch_cpu].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_cpu.to(device), feats, label_feats, mask)
        loss = criterion(logits, targets)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        n = int(batch_cpu.numel())
        total_loss += float(loss.detach().cpu()) * n
        total_examples += n
        updates += 1
    return total_loss / max(total_examples, 1), updates


def evaluate_variant(
    model: nn.Module,
    bundle: Mapping[str, Any],
    split: str,
    architecture: Mapping[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    mask: torch.Tensor,
) -> Tuple[Dict[str, Any], torch.Tensor]:
    indices = bundle[f"{split}_idx"]
    logits = predict_indices(model, bundle, indices, architecture, args.eval_batch_size, device, mask)
    metrics = classification_metrics(logits, bundle["labels"][torch.as_tensor(indices)])
    return metrics, logits


def make_model(args: argparse.Namespace, architecture: Mapping[str, Any], device: torch.device) -> nn.Module:
    model = SeHGNN(
        dataset="IMDB",
        nfeat=args.embed_size,
        hidden=args.hidden,
        nclass=int(architecture["num_classes"]),
        feat_keys=architecture["feature_keys"],
        label_feat_keys=architecture["label_feature_keys"],
        tgt_type=TGT_TYPE,
        dropout=args.dropout,
        input_drop=args.input_drop,
        att_drop=args.att_drop,
        n_fp_layers=args.n_fp_layers,
        n_task_layers=args.n_task_layers,
        act=args.act,
        residual=args.residual,
        data_size=architecture["feature_dims"],
    )
    return model.to(device)


def save_variant_outputs(
    seed_dir: Path,
    variant: str,
    bundle: Mapping[str, Any],
    split_logits: Mapping[str, torch.Tensor],
    split_metrics: Mapping[str, Mapping[str, Any]],
    seed: int,
) -> Dict[str, str]:
    arrays: Dict[str, np.ndarray] = {"training_seed": np.asarray(seed, dtype=np.int64)}
    for split in ("train", "val", "test"):
        idx = bundle[f"{split}_idx"]
        logits = split_logits[split].numpy().astype(np.float32, copy=False)
        labels = bundle["labels"][torch.as_tensor(idx)].numpy().astype(np.int64, copy=False)
        arrays[f"{split}_idx"] = idx
        arrays[f"{split}_labels"] = labels
        arrays[f"{split}_logits"] = logits
    logits_path = seed_dir / f"{variant}_logits.npz"
    np.savez_compressed(logits_path, **arrays)

    test_logits = arrays["test_logits"]
    probabilities = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
    frame = pd.DataFrame(
        {
            "movie_id": arrays["test_idx"],
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
    model = make_model(args, architecture, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta)
    rng = np.random.RandomState(seed)
    masks = {name: channel_mask(model, bundle, device) for name, bundle in bundles.items()}

    parameter_signature = state_shape_signature(model)
    parameter_stats = model_memory_stats(model)
    seed_dir = output_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = seed_dir / "shared_checkpoint.pt"
    latest_path = seed_dir / "latest_training_state.pt"

    run_config = {
        "dataset": "IMDB NC",
        "variants": list(bundles),
        "seed": seed,
        "architecture": architecture,
        "model": {
            "embed_size": args.embed_size,
            "hidden": args.hidden,
            "n_fp_layers": args.n_fp_layers,
            "n_task_layers": args.n_task_layers,
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
        "data_dirs": {name: bundle["base_dir"] for name, bundle in bundles.items()},
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

    reset_cuda_peak(device)
    segment_start = time.perf_counter()
    variant_names = list(bundles)
    batches_per_variant = {
        name: int(math.ceil(len(bundle["train_idx"]) / args.batch_size))
        for name, bundle in bundles.items()
    }
    updates_per_super_epoch = sum(batches_per_variant.values())

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
                model, bundles[name], "val", architecture, args, device, masks[name]
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
                        "dataset": "IMDB NC",
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
                f"seed={seed} super_epoch={super_epochs_ran:03d} "
                f"steps={optimizer_steps} mean_val_loss={mean_val_loss:.6f} "
                f"mean_val_macro_f1={mean_val_macro_f1:.4f} order={order}",
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
                model, bundle, split, architecture, args, device, masks[name]
            )
            split_metrics[split] = metrics
            split_logits[split] = logits
        split_metrics["graph"] = {
            "edge_count": bundle["edge_count"],
            "active_feature_channels": len(bundle["features"]),
            "active_label_channels": len(bundle["label_features"]),
            "active_total_channels": int(masks[name].sum().item()),
            "global_total_channels": int(model.num_channels),
        }
        per_variant_split_metrics[name] = split_metrics
        per_variant_artifacts[name] = save_variant_outputs(
            seed_dir, name, bundle, split_logits, split_metrics, seed
        )
        test_outputs[name] = {
            "item_ids": bundle["test_idx"],
            "logits": split_logits["test"].numpy(),
        }

    invariance = pairwise_invariance(test_outputs)
    pd.DataFrame(invariance).to_csv(seed_dir / "pairwise_invariance.csv", index=False)
    test_metric_rows = [
        {"variant": name, **mean_scalar_dict([metrics["test"]])}
        for name, metrics in per_variant_split_metrics.items()
    ]
    pd.DataFrame(test_metric_rows).to_csv(seed_dir / "test_metrics_by_variant.csv", index=False)
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
    summary = {
        "format_version": 1,
        "dataset": "IMDB NC",
        "model": "SeHGNN shared graph-variant augmentation",
        "seed": seed,
        "variants": variant_names,
        "data_dirs": {name: bundle["base_dir"] for name, bundle in bundles.items()},
        "parameter_compatibility": {
            "strategy": architecture["strategy"],
            "parameter_shape_signature": parameter_signature,
            "same_checkpoint_for_all_variants": True,
            "global_total_channels": architecture["num_total_channels"],
            "per_variant_active_channels": {
                name: int(masks[name].sum().item()) for name in variant_names
            },
        },
        "architecture": architecture,
        "hyperparameters": {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (str, int, float, bool)) and key not in {"resume"}
        },
        "selection_metric": "mean_validation_cross_entropy",
        "best_mean_val_loss": stopper.best,
        "best_super_epoch": stopper.best_super_epoch,
        "epoch_accounting": {
            "definition": "one super-epoch visits every variant once; every training minibatch performs one optimizer update",
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
            "labels_sha256": array_sha256(next(iter(bundles.values()))["labels"].numpy()),
            **{
                f"{split}_idx_sha256": array_sha256(next(iter(bundles.values()))[f"{split}_idx"])
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
    parser = argparse.ArgumentParser(description="Shared-checkpoint SeHGNN augmentation for IMDB NC")
    parser.add_argument("--variants", default="1,2,3,4", help="Original IMDB graph variants only")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/sehgnn_augmentation/IMDB_NC"))
    parser.add_argument("--seeds", default="1566911444,20241017,20251017")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--super-epochs", type=int, default=200)
    parser.add_argument("--embed-size", type=int, default=512)
    parser.add_argument("--num-hops", type=int, default=4)
    parser.add_argument("--label-num-hops", type=int, default=4)
    parser.add_argument("--n-fp-layers", type=int, default=2)
    parser.add_argument("--n-task-layers", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=512)
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
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = parse_csv_ints(args.variants)
    unknown = set(variants) - set(ORIGINAL_VARIANTS)
    if unknown:
        raise ValueError(
            f"This augmentation runner intentionally supports original variants 1-4 only; got {sorted(unknown)}"
        )
    seeds = parse_csv_ints(args.seeds)
    args.data_root = args.data_root.resolve() if args.data_root is not None else None
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing IMDB variants {variants} and canonical semantic channels...", flush=True)
    bundles, architecture = prepare_bundles(args, variants)
    atomic_write_json(args.output_dir / "architecture_manifest.json", architecture)
    for name, bundle in bundles.items():
        print(
            f"  {name}: edges={bundle['edge_count']:,}, feature_channels={len(bundle['features'])}, "
            f"label_channels={len(bundle['label_features'])}",
            flush=True,
        )
    print(f"  shared total channels={architecture['num_total_channels']}", flush=True)

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
            "dataset": "IMDB NC",
            "model": "SeHGNN shared graph-variant augmentation",
            "variants": list(bundles),
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
    print(f"[OK] IMDB augmentation results: {args.output_dir}")


if __name__ == "__main__":
    main()
