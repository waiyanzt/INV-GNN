#!/usr/bin/env python3
"""Train SeHGNN on preprocessed Freebase MAGNN-aligned semantic channels.

Place this file in ``src/baselines/SeHGNN`` (or invoke it with that directory on
``PYTHONPATH``). It reuses the repository's unmodified ``model.SeHGNN`` class.
The preprocessor gives every channel a unique key ending in its source node-type
id, which is exactly what SeHGNN's sparse Freebase branch uses to choose the
trainable source-type embedding.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

try:
    from torch_sparse import SparseTensor
except ImportError as exc:  # pragma: no cover - environment-specific dependency
    raise ImportError(
        "run_freebase_magnn_channels.py requires torch_sparse, matching the existing "
        "SeHGNN Freebase implementation. Install the build compatible with your PyTorch/CUDA."
    ) from exc


def import_sehgnn():
    """Import the repository model without duplicating or silently changing it."""
    try:
        from model import SeHGNN  # type: ignore
        return SeHGNN
    except Exception as first_error:
        script_dir = Path(__file__).resolve().parent
        candidates = [
            script_dir,
            script_dir.parent / "SeHGNN",
            Path.cwd(),
        ]
        for candidate in candidates:
            if (candidate / "model").is_dir() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
                try:
                    from model import SeHGNN  # type: ignore
                    return SeHGNN
                except Exception:
                    pass
        raise ImportError(
            "Could not import model.SeHGNN. Copy this script into src/baselines/SeHGNN "
            "or add that directory to PYTHONPATH."
        ) from first_error


SeHGNN = import_sehgnn()


def set_random_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def scipy_to_sparse_tensor(matrix: sp.csr_matrix) -> SparseTensor:
    coo = matrix.tocoo()
    row = torch.from_numpy(coo.row.astype(np.int64, copy=False))
    col = torch.from_numpy(coo.col.astype(np.int64, copy=False))
    value = torch.from_numpy(coo.data.astype(np.float32, copy=False))
    return SparseTensor(
        row=row,
        col=col,
        value=value,
        sparse_sizes=(int(matrix.shape[0]), int(matrix.shape[1])),
        is_sorted=False,
    ).coalesce()


def load_preprocessed(data_dir: Path):
    manifest_path = data_dir / "channels_manifest.json"
    dataset_path = data_dir / "dataset.npz"
    if not manifest_path.exists() or not dataset_path.exists():
        raise FileNotFoundError(
            f"Expected channels_manifest.json and dataset.npz under {data_dir}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_npz = np.load(dataset_path)
    dataset = {key: dataset_npz[key] for key in dataset_npz.files}

    features: Dict[str, SparseTensor] = {}
    source_type_by_key: Dict[str, int] = {}
    for channel in manifest["channels"]:
        key = str(channel["model_key"])
        source_type = int(channel["source_type"])
        if key != str(manifest["target_type"]) and not key.endswith(str(source_type)):
            raise ValueError(
                f"Channel key {key!r} must end in source type {source_type} for repository SeHGNN"
            )
        matrix = sp.load_npz(data_dir / channel["matrix_file"]).tocsr().astype(np.float32)
        expected_shape = tuple(int(x) for x in channel["shape"])
        if matrix.shape != expected_shape:
            raise ValueError(f"Shape mismatch for channel {key}: {matrix.shape} vs {expected_shape}")
        features[key] = scipy_to_sparse_tensor(matrix)
        source_type_by_key[key] = source_type

    target_key = str(manifest["target_type"])
    if target_key not in features:
        raise KeyError(f"Identity target channel {target_key!r} is missing")
    return manifest, dataset, features, source_type_by_key


def batch_feature_dict(
    features: Mapping[str, SparseTensor],
    batch_cpu: torch.Tensor,
    device: torch.device,
) -> Dict[str, SparseTensor]:
    return {key: matrix[batch_cpu].to(device) for key, matrix in features.items()}


def parameter_estimate(
    num_channels: int,
    embed_size: int,
    hidden: int,
    n_fp_layers: int,
    n_task_layers: int,
    num_classes: int,
) -> int:
    """Conservative estimate for the repository SeHGNN parameter count."""
    embeddings_and_projection = num_channels * embed_size * hidden
    if n_fp_layers > 1:
        embeddings_and_projection += (n_fp_layers - 1) * num_channels * hidden * hidden
    concat = num_channels * hidden * hidden
    transformer = 2 * hidden * (hidden // 4) + hidden * hidden
    task = max(1, n_task_layers) * hidden * hidden + hidden * num_classes
    return int(embeddings_and_projection + concat + transformer + task)


def bytes_to_mib(num_bytes: int | float) -> float:
    return float(num_bytes) / (1024.0 ** 2)


def model_memory_stats(model: nn.Module) -> Dict[str, float | int]:
    """Return exact parameter/buffer counts and storage for the instantiated model."""
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    buffer_count = sum(buffer.numel() for buffer in model.buffers())
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    return {
        "parameter_count": int(parameter_count),
        "trainable_parameter_count": int(trainable_parameter_count),
        "buffer_count": int(buffer_count),
        "parameter_mib": bytes_to_mib(parameter_bytes),
        "buffer_mib": bytes_to_mib(buffer_bytes),
        "static_model_mib": bytes_to_mib(parameter_bytes + buffer_bytes),
    }


def model_parameter_mib(model: nn.Module) -> float:
    """Backward-compatible helper used by older backfill scripts."""
    return float(model_memory_stats(model)["parameter_mib"])


def array_sha256(array: np.ndarray) -> str:
    """Stable fingerprint used to verify split/logit alignment across variants."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def safe_checkpoint_token(value: Any) -> str:
    text = str(value if value not in (None, "") else "none")
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def build_sehgnn_model(
    config: argparse.Namespace | Mapping[str, Any],
    manifest: Mapping[str, Any],
    feature_keys: Iterable[str],
) -> nn.Module:
    """Construct exactly the SeHGNN architecture used for training/backfill."""
    def get(name: str, default: Any) -> Any:
        if isinstance(config, Mapping):
            return config.get(name, default)
        return getattr(config, name, default)

    num_classes = int(manifest["num_classes"])
    target_type = int(manifest["target_type"])
    node_counts = {int(k): int(v) for k, v in manifest["node_counts"].items()}
    return SeHGNN(
        dataset="Freebase",
        nfeat=int(get("embed_size", 512)),
        hidden=int(get("hidden", 512)),
        nclass=num_classes,
        feat_keys=feature_keys,
        label_feat_keys=[],
        tgt_type=str(target_type),
        dropout=float(get("dropout", 0.5)),
        input_drop=float(get("input_drop", 0.0)),
        att_drop=float(get("att_drop", 0.0)),
        n_fp_layers=int(get("n_fp_layers", 2)),
        n_task_layers=int(get("n_task_layers", 4)),
        act=str(get("act", "none")),
        residual=bool(get("residual", False)),
        data_size=node_counts,
        num_heads=int(get("num_heads", 1)),
    )


def extract_model_state(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict", "model", "net", "network"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise KeyError("Could not locate a model state_dict in checkpoint")


def topk_mrr(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float, float]:
    order = torch.argsort(logits, dim=1, descending=True)
    labels_col = labels.view(-1, 1)
    matches = order.eq(labels_col)
    ranks = matches.float().argmax(dim=1) + 1
    hit1 = (ranks <= 1).float().mean().item()
    hit3 = (ranks <= min(3, logits.shape[1])).float().mean().item()
    mrr = (1.0 / ranks.float()).mean().item()
    return hit1, hit3, mrr


def metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
    """Compute the complete multiclass NC metric set used by the aggregator."""
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError(f"Bad logits/labels shapes: {tuple(logits.shape)} and {tuple(labels.shape)}")
    logits_cpu = logits.detach().cpu()
    labels_cpu = labels.detach().cpu()
    preds = logits_cpu.argmax(dim=-1).numpy()
    truth = labels_cpu.numpy()
    num_classes = int(logits_cpu.shape[1])
    class_ids = np.arange(num_classes)
    hit1, hit3, mrr = topk_mrr(logits_cpu, labels_cpu)
    per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
        truth, preds, labels=class_ids, average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(truth, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, preds)),
        "macro_precision": float(
            precision_score(truth, preds, labels=class_ids, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(truth, preds, labels=class_ids, average="macro", zero_division=0)
        ),
        "micro_precision": float(
            precision_score(truth, preds, labels=class_ids, average="micro", zero_division=0)
        ),
        "micro_recall": float(
            recall_score(truth, preds, labels=class_ids, average="micro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(truth, preds, labels=class_ids, average="micro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(truth, preds, labels=class_ids, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(truth, preds, labels=class_ids, average="weighted", zero_division=0)
        ),
        "hit_at_1": float(hit1),
        "hit_at_3": float(hit3),
        "mrr": float(mrr),
        "per_class_precision": per_p.astype(float).tolist(),
        "per_class_recall": per_r.astype(float).tolist(),
        "per_class_f1": per_f1.astype(float).tolist(),
        "per_class_support": per_support.astype(int).tolist(),
        "confusion_matrix": confusion_matrix(
            truth, preds, labels=class_ids
        ).astype(int).tolist(),
    }


def predict_indices(
    model: nn.Module,
    features: Mapping[str, SparseTensor],
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    outputs: List[torch.Tensor] = []
    loader = DataLoader(
        torch.as_tensor(indices, dtype=torch.long),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    with torch.no_grad():
        for batch_cpu in loader:
            batch_feats = batch_feature_dict(features, batch_cpu, device)
            logits = model(batch_cpu.to(device), batch_feats, {}, None)
            outputs.append(logits.cpu())
    if not outputs:
        return torch.empty((0, 0))
    return torch.cat(outputs, dim=0)


def train_one_seed(
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    dataset: Mapping[str, np.ndarray],
    features: Mapping[str, SparseTensor],
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    set_random_seed(seed)
    labels = torch.from_numpy(dataset["labels"].astype(np.int64, copy=False))
    train_idx = dataset["train_idx"].astype(np.int64, copy=False)
    val_idx = dataset["val_idx"].astype(np.int64, copy=False)
    test_idx = dataset["test_idx"].astype(np.int64, copy=False)
    num_classes = int(manifest["num_classes"])
    model = build_sehgnn_model(args, manifest, features.keys()).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        torch.as_tensor(train_idx, dtype=torch.long),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / (
        f"sehgnn_freebase_{safe_checkpoint_token(manifest.get('pipeline'))}_"
        f"{safe_checkpoint_token(manifest.get('flavor'))}_"
        f"{safe_checkpoint_token(manifest.get('variant', 'universal'))}_"
        f"upto_{safe_checkpoint_token(manifest.get('up_to_variant'))}_"
        f"k{manifest['k']}_{safe_checkpoint_token(manifest.get('channel_identity'))}_"
        f"seed{seed}.pt"
    )

    best_val_loss = math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    train_start = time.time()
    epochs_run = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        seen = 0
        for batch_cpu in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batch_feats = batch_feature_dict(features, batch_cpu, device)
            logits = model(batch_cpu.to(device), batch_feats, {}, None)
            target = labels[batch_cpu].to(device)
            loss = criterion(logits, target)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            batch_n = len(batch_cpu)
            epoch_loss += float(loss.item()) * batch_n
            seen += batch_n

        epochs_run = epoch + 1
        val_logits = predict_indices(
            model, features, val_idx, args.eval_batch_size, device
        )
        val_loss = float(criterion(val_logits, labels[val_idx]).item())
        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        if epoch == 0 or (epoch + 1) % args.log_every == 0:
            val_metrics = metrics_from_logits(val_logits, labels[val_idx])
            print(
                f"seed={seed} epoch={epoch + 1:03d} "
                f"train_loss={epoch_loss / max(seen, 1):.6f} "
                f"val_loss={val_loss:.6f} val_macro_f1={val_metrics['macro_f1']:.4f}"
            )
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}; best epoch={best_epoch + 1}")
            break

    # Synchronize before stopping the timer and capture the historical training peak
    # before checkpoint loading can allocate a second copy of the state_dict.
    peak_training_gpu_mib = None
    peak_training_reserved_gpu_mib = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_training_gpu_mib = bytes_to_mib(torch.cuda.max_memory_allocated(device))
        peak_training_reserved_gpu_mib = bytes_to_mib(torch.cuda.max_memory_reserved(device))
    train_time = time.time() - train_start

    if not checkpoint_path.exists():
        raise RuntimeError("No checkpoint was saved")
    try:
        checkpoint_object = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint_object = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(extract_model_state(checkpoint_object), strict=True)
    del checkpoint_object
    model.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    inference_start = time.time()
    split_results: Dict[str, Any] = {}
    split_logits: Dict[str, np.ndarray] = {}
    for split_name, indices in (
        ("train", train_idx),
        ("val", val_idx),
        ("test", test_idx),
    ):
        logits = predict_indices(model, features, indices, args.eval_batch_size, device)
        metrics = metrics_from_logits(logits, labels[indices])
        metrics["loss"] = float(criterion(logits, labels[indices]).item())
        split_results[split_name] = metrics
        split_logits[split_name] = logits.numpy().astype(np.float32, copy=False)

    peak_inference_gpu_mib = None
    peak_inference_reserved_gpu_mib = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_inference_gpu_mib = bytes_to_mib(torch.cuda.max_memory_allocated(device))
        peak_inference_reserved_gpu_mib = bytes_to_mib(torch.cuda.max_memory_reserved(device))
    inference_time = time.time() - inference_start

    logits_path = checkpoint_path.with_name(checkpoint_path.stem + "_logits.npz")
    logits_payload: Dict[str, np.ndarray] = {
        "train_logits": split_logits["train"],
        "val_logits": split_logits["val"],
        "test_logits": split_logits["test"],
        "train_labels": labels[train_idx].numpy().astype(np.int64, copy=False),
        "val_labels": labels[val_idx].numpy().astype(np.int64, copy=False),
        "test_labels": labels[test_idx].numpy().astype(np.int64, copy=False),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "training_seed": np.asarray(seed, dtype=np.int64),
        "split_seed": np.asarray(int(manifest.get("split_seed", -1)), dtype=np.int64),
    }
    # Preserve global target ids when the preprocessor provides them. This makes
    # cross-variant alignment independent of local target-node ordering.
    for global_key in ("target_global_ids", "primary_target_nodes", "target_idx"):
        if global_key in dataset:
            global_ids = np.asarray(dataset[global_key], dtype=np.int64)
            logits_payload["target_global_ids"] = global_ids
            if global_ids.ndim == 1 and len(global_ids) > int(max(test_idx.max(), val_idx.max(), train_idx.max())):
                logits_payload["train_global_ids"] = global_ids[train_idx]
                logits_payload["val_global_ids"] = global_ids[val_idx]
                logits_payload["test_global_ids"] = global_ids[test_idx]
            break
    np.savez_compressed(logits_path, **logits_payload)

    memory_stats = model_memory_stats(model)
    result = {
        "seed": seed,
        "best_epoch": best_epoch + 1,
        "epochs_run": epochs_run,
        "best_val_loss": best_val_loss,
        "train_time_sec": train_time,
        "inference_time_sec": inference_time,
        "checkpoint": str(checkpoint_path),
        "logits_file": str(logits_path),
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "logits_file": str(logits_path),
        },
        "split_fingerprints": {
            "train_idx_sha256": array_sha256(train_idx),
            "val_idx_sha256": array_sha256(val_idx),
            "test_idx_sha256": array_sha256(test_idx),
            "train_labels_sha256": array_sha256(logits_payload["train_labels"]),
            "val_labels_sha256": array_sha256(logits_payload["val_labels"]),
            "test_labels_sha256": array_sha256(logits_payload["test_labels"]),
        },
        "memory": {
            "checkpoint_bytes": int(checkpoint_path.stat().st_size),
            "checkpoint_mib": bytes_to_mib(checkpoint_path.stat().st_size),
            **memory_stats,
            "peak_training_gpu_mib": peak_training_gpu_mib,
            "peak_training_reserved_gpu_mib": peak_training_reserved_gpu_mib,
            "peak_training_is_historical": True,
            "peak_inference_gpu_mib": peak_inference_gpu_mib,
            "peak_inference_reserved_gpu_mib": peak_inference_reserved_gpu_mib,
        },
        "splits": split_results,
    }
    return result


def _summary_stats(values: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": np.nanmean(values, axis=0).tolist() if values.ndim > 1 else float(np.nanmean(values)),
        "std": (
            np.nanstd(values, axis=0, ddof=1).tolist()
            if values.ndim > 1 and values.shape[0] > 1
            else (float(np.nanstd(values, ddof=1)) if values.ndim == 1 and values.size > 1 else (np.zeros(values.shape[1:]).tolist() if values.ndim > 1 else 0.0))
        ),
        "values": values.tolist(),
        "ddof": 1,
    }


def summarize(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize all metrics/resources while preserving the raw per-run values."""
    summary: Dict[str, Any] = {"std_ddof": 1}
    for split in ("train", "val", "test"):
        split_summary: Dict[str, Any] = {}
        keys = results[0]["splits"][split].keys()
        for key in keys:
            values = np.asarray([r["splits"][split][key] for r in results], dtype=float)
            split_summary[key] = _summary_stats(values)
        summary[split] = split_summary
    for key in ("train_time_sec", "inference_time_sec", "epochs_run", "best_epoch", "best_val_loss"):
        values = np.asarray([r[key] for r in results], dtype=float)
        summary[key] = _summary_stats(values)
    memory_keys = results[0]["memory"].keys()
    summary["memory"] = {}
    for key in memory_keys:
        raw = [r["memory"].get(key) for r in results]
        if all(value is None for value in raw):
            summary["memory"][key] = {"mean": None, "std": None, "values": raw, "ddof": 1}
        elif all(isinstance(value, (bool, np.bool_)) for value in raw):
            summary["memory"][key] = {"values": [bool(value) for value in raw]}
        elif any(value is None for value in raw):
            summary["memory"][key] = {"mean": None, "std": None, "values": raw, "ddof": 1}
        else:
            summary["memory"][key] = _summary_stats(np.asarray(raw, dtype=float))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SeHGNN Freebase NC on MAGNN-aligned channels")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoint"))
    parser.add_argument("--seeds", default="1566911444,20241017,20251017,20261017")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
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
    parser.add_argument(
        "--max-estimated-parameters",
        type=int,
        default=500_000_000,
        help="Refuse obviously infeasible full-K configurations before allocating the model",
    )
    parser.add_argument("--allow-large-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not args.seeds:
        raise ValueError("At least one seed is required")
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    manifest, dataset, features, _ = load_preprocessed(args.data_dir)
    estimate = parameter_estimate(
        int(manifest["num_channels"]),
        args.embed_size,
        args.hidden,
        args.n_fp_layers,
        args.n_task_layers,
        int(manifest["num_classes"]),
    )
    print(
        f"Loaded {manifest['num_channels']} channels from {args.data_dir}; "
        f"estimated SeHGNN parameters ~= {estimate:,}; device={device}"
    )
    if estimate > args.max_estimated_parameters and not args.allow_large_model:
        raise RuntimeError(
            f"Estimated parameters {estimate:,} exceed --max-estimated-parameters="
            f"{args.max_estimated_parameters:,}. Lower K/hidden size, use restricted_k, or pass "
            "--allow-large-model after checking memory."
        )

    results = [
        train_one_seed(args, manifest, dataset, features, seed, device)
        for seed in args.seeds
    ]
    manifest_keys = (
        "pipeline",
        "variant",
        "variants",
        "up_to_variant",
        "flavor",
        "k",
        "channel_identity",
        "num_channels",
        "split_seed",
        "target_type",
        "num_classes",
    )
    payload = {
        "format_version": 2,
        "dataset": "Freebase NC",
        "model": "SeHGNN",
        "data_dir": str(args.data_dir.resolve()),
        "device": str(device),
        "environment": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "manifest": {key: manifest.get(key) for key in manifest_keys},
        "hyperparameters": {
            key: value
            for key, value in vars(args).items()
            if key not in {"seeds", "data_dir", "output_json", "checkpoint_dir"}
            and isinstance(value, (str, int, float, bool))
        },
        "input_configuration": {
            "label_propagation_enabled": False,
            "label_num_hops": 0,
            "feature_channel_source": manifest.get("flavor"),
        },
        "metric_definitions": {
            "precision_and_recall_in_latex": "macro averaged over all Freebase classes",
            "classification": "accuracy, balanced accuracy, macro/micro precision and recall, micro/macro/weighted F1, Hit@1, Hit@3, MRR, loss, per-class metrics, and confusion matrix",
            "logits": "raw best-checkpoint pre-softmax logits",
        },
        "memory_definitions": {
            "checkpoint_mib": "checkpoint file bytes / 2^20",
            "parameter_mib": "parameter tensor storage only",
            "static_model_mib": "parameter plus registered-buffer tensor storage",
            "peak_training_gpu_mib": "historical max_memory_allocated during the actual training loop",
            "peak_inference_gpu_mib": "max_memory_allocated after loading the best checkpoint and running train/val/test inference",
        },
        "seeds": args.seeds,
        "num_runs": len(args.seeds),
        "estimated_parameters": estimate,
        "runs": results,
        "summary": summarize(results),
    }
    output_json = args.output_json or (args.data_dir / "sehgnn_results.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved results: {output_json}")
    test = payload["summary"]["test"]
    print(
        "Test: "
        f"accuracy={test['accuracy']['mean']:.4f} +/- {test['accuracy']['std']:.4f}, "
        f"micro-F1={test['micro_f1']['mean']:.4f} +/- {test['micro_f1']['std']:.4f}, "
        f"macro-F1={test['macro_f1']['mean']:.4f} +/- {test['macro_f1']['std']:.4f}"
    )


if __name__ == "__main__":
    main()