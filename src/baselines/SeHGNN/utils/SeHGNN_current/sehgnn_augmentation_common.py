#!/usr/bin/env python3
"""Shared utilities for SeHGNN graph-variant data-augmentation experiments.

The augmentation runners keep one model/optimizer/checkpoint across graph
variants.  A super-epoch is one shuffled, balanced visit to every variant,
followed by validation of the same checkpoint on every variant.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import resource
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import kendalltau
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


def set_determinism(seed: int) -> None:
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


def resolve_device(cpu: bool, gpu: int) -> torch.device:
    if cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{gpu}")


@dataclass
class EarlyStopper:
    """Minimize a monitored validation quantity at super-epoch boundaries."""

    patience: int
    min_delta: float = 0.0
    best: float = math.inf
    bad_super_epochs: int = 0
    best_super_epoch: int = 0
    should_stop: bool = False

    def update(self, value: float, super_epoch: int) -> bool:
        improved = float(value) < float(self.best) - float(self.min_delta)
        if improved:
            self.best = float(value)
            self.bad_super_epochs = 0
            self.best_super_epoch = int(super_epoch)
        else:
            self.bad_super_epochs += 1
            if self.bad_super_epochs >= self.patience:
                self.should_stop = True
        return improved

    def state_dict(self) -> Dict[str, Any]:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best": self.best,
            "bad_super_epochs": self.bad_super_epochs,
            "best_super_epoch": self.best_super_epoch,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["patience"]) != self.patience:
            raise ValueError("Early-stopper patience differs from the saved run")
        if float(state["min_delta"]) != float(self.min_delta):
            raise ValueError("Early-stopper min_delta differs from the saved run")
        self.best = float(state["best"])
        self.bad_super_epochs = int(state["bad_super_epochs"])
        self.best_super_epoch = int(state["best_super_epoch"])
        self.should_stop = bool(state["should_stop"])


def bytes_to_mib(num_bytes: int | float) -> float:
    return float(num_bytes) / (1024.0 ** 2)


def model_memory_stats(model: nn.Module) -> Dict[str, Any]:
    parameter_count = sum(p.numel() for p in model.parameters())
    trainable_parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_count = sum(b.numel() for b in model.buffers())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return {
        "parameter_count": int(parameter_count),
        "trainable_parameter_count": int(trainable_parameter_count),
        "buffer_count": int(buffer_count),
        "parameter_bytes": int(parameter_bytes),
        "buffer_bytes": int(buffer_bytes),
        "parameter_mib": bytes_to_mib(parameter_bytes),
        "buffer_mib": bytes_to_mib(buffer_bytes),
        "static_model_mib": bytes_to_mib(parameter_bytes + buffer_bytes),
    }


def process_peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB, macOS bytes. The target HPC environment is Linux.
    return value * 1024 if os.name == "posix" and value < 10**12 else value


def reset_cuda_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def cuda_memory_stats(device: torch.device) -> Dict[str, Any]:
    if device.type != "cuda":
        return {
            "allocated_mib": None,
            "reserved_mib": None,
            "peak_allocated_mib": None,
            "peak_reserved_mib": None,
        }
    torch.cuda.synchronize(device)
    return {
        "allocated_mib": bytes_to_mib(torch.cuda.memory_allocated(device)),
        "reserved_mib": bytes_to_mib(torch.cuda.memory_reserved(device)),
        "peak_allocated_mib": bytes_to_mib(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_mib": bytes_to_mib(torch.cuda.max_memory_reserved(device)),
    }


def merge_peak_memory(old: Mapping[str, Any], new: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in set(old) | set(new):
        values = [x for x in (old.get(key), new.get(key)) if x is not None]
        out[key] = max(values) if values else None
    return out


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        torch.save(dict(payload), temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows))
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        frame.to_csv(temp_path, index=False)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def torch_load(path: Path, map_location: Any = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def state_shape_signature(model: nn.Module) -> str:
    payload = [
        (name, tuple(tensor.shape), str(tensor.dtype))
        for name, tensor in model.state_dict().items()
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def capture_rng_state(local_rng: np.random.RandomState) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "numpy_local": local_rng.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any], local_rng: np.random.RandomState) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    local_rng.set_state(state["numpy_local"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def canonical_config(config: Mapping[str, Any]) -> str:
    return json.dumps(json_ready(config), sort_keys=True, separators=(",", ":"))


def save_latest_training_state(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    early_stopper: EarlyStopper,
    local_rng: np.random.RandomState,
    run_config: Mapping[str, Any],
    completed_super_epoch: int,
    counters: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    training_seconds_elapsed: float,
    peak_rss_bytes: int,
    training_gpu: Mapping[str, Any],
) -> None:
    atomic_torch_save(
        {
            "format_version": 1,
            "run_config": dict(run_config),
            "run_config_canonical": canonical_config(run_config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "early_stopper": early_stopper.state_dict(),
            "rng": capture_rng_state(local_rng),
            "completed_super_epoch": int(completed_super_epoch),
            "counters": dict(counters),
            "history": list(history),
            "training_seconds_elapsed": float(training_seconds_elapsed),
            "peak_rss_bytes": int(peak_rss_bytes),
            "training_gpu": dict(training_gpu),
        },
        path,
    )


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_latest_training_state(
    path: Path,
    *,
    resume: bool,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    early_stopper: EarlyStopper,
    local_rng: np.random.RandomState,
    run_config: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, Any] | None:
    if not resume:
        if path.exists():
            raise FileExistsError(
                f"{path} already exists. Use --resume or select a new output directory."
            )
        return None
    if not path.exists():
        return None
    state = torch_load(path, map_location="cpu")
    saved = state.get("run_config_canonical", canonical_config(state["run_config"]))
    current = canonical_config(run_config)
    if saved != current:
        raise ValueError(
            "Saved training configuration differs from the requested run. Only the maximum "
            "number of super-epochs and device may change when resuming."
        )
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    optimizer_to_device(optimizer, device)
    early_stopper.load_state_dict(state["early_stopper"])
    restore_rng_state(state["rng"], local_rng)
    return state


def _topk_mrr_numpy(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    order = np.argsort(-logits, axis=1, kind="stable")
    ranks = np.argmax(order == labels[:, None], axis=1) + 1
    return (
        float(np.mean(ranks <= 1)),
        float(np.mean(ranks <= min(3, logits.shape[1]))),
        float(np.mean(1.0 / ranks)),
    )


def classification_metrics(logits: torch.Tensor | np.ndarray, labels: torch.Tensor | np.ndarray) -> Dict[str, Any]:
    logits_np = logits.detach().cpu().numpy() if torch.is_tensor(logits) else np.asarray(logits)
    labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
    logits_np = np.asarray(logits_np, dtype=np.float64)
    labels_np = np.asarray(labels_np, dtype=np.int64)
    if logits_np.ndim != 2 or labels_np.ndim != 1 or logits_np.shape[0] != labels_np.shape[0]:
        raise ValueError(f"Bad logits/labels shapes: {logits_np.shape} and {labels_np.shape}")
    num_classes = int(logits_np.shape[1])
    class_ids = np.arange(num_classes)
    pred = np.argmax(logits_np, axis=1)
    hit1, hit3, mrr = _topk_mrr_numpy(logits_np, labels_np)
    per_p, per_r, per_f1, support = precision_recall_fscore_support(
        labels_np, pred, labels=class_ids, average=None, zero_division=0
    )
    shifted = logits_np - np.max(logits_np, axis=1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=1)) + np.max(logits_np, axis=1)
    loss = float(np.mean(logsumexp - logits_np[np.arange(len(labels_np)), labels_np]))
    return {
        "accuracy": float(accuracy_score(labels_np, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_np, pred)),
        "macro_precision": float(precision_score(labels_np, pred, labels=class_ids, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels_np, pred, labels=class_ids, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(labels_np, pred, labels=class_ids, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(labels_np, pred, labels=class_ids, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(labels_np, pred, labels=class_ids, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(labels_np, pred, labels=class_ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels_np, pred, labels=class_ids, average="weighted", zero_division=0)),
        "hit_at_1": hit1,
        "hit_at_3": hit3,
        "mrr": mrr,
        "loss": loss,
        "per_class_precision": per_p.astype(float).tolist(),
        "per_class_recall": per_r.astype(float).tolist(),
        "per_class_f1": per_f1.astype(float).tolist(),
        "per_class_support": support.astype(int).tolist(),
        "confusion_matrix": confusion_matrix(labels_np, pred, labels=class_ids).astype(int).tolist(),
    }


def stable_tau(a: np.ndarray, b: np.ndarray) -> float:
    result = kendalltau(a, b, variant="b", nan_policy="omit").correlation
    if result is None or not np.isfinite(result):
        return 1.0 if np.allclose(a, b, equal_nan=True) else 0.0
    return float(result)


def invariance_metrics(logits_a: np.ndarray, logits_b: np.ndarray, top_k: int = 3) -> Dict[str, float]:
    a = np.asarray(logits_a, dtype=np.float64)
    b = np.asarray(logits_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"Expected aligned 2-D logits, got {a.shape} and {b.shape}")
    all_tau: List[float] = []
    top_tau: List[float] = []
    top1: List[float] = []
    actual_k = min(int(top_k), a.shape[1])
    for row_a, row_b in zip(a, b):
        all_tau.append(stable_tau(row_a, row_b))
        top1.append(float(np.argmax(row_a) == np.argmax(row_b)))
        union = np.union1d(
            np.argsort(-row_a, kind="stable")[:actual_k],
            np.argsort(-row_b, kind="stable")[:actual_k],
        )
        top_tau.append(1.0 if union.size <= 1 else stable_tau(row_a[union], row_b[union]))
    a_shift = a - np.max(a, axis=1, keepdims=True)
    b_shift = b - np.max(b, axis=1, keepdims=True)
    pa = np.exp(a_shift); pa /= pa.sum(axis=1, keepdims=True)
    pb = np.exp(b_shift); pb /= pb.sum(axis=1, keepdims=True)
    diff = a - b
    pdiff = pa - pb
    return {
        # Native SeHGNN invariance metrics.
        "kendall_tau": float(np.mean(all_tau)),
        "kendall_tau_at_1": float(np.mean(top1)),
        "kendall_tau_at_3": float(np.mean(top_tau)),
        # RGCN augmentation compatibility metrics.
        "kendall_tau_flat_logits": stable_tau(a.ravel(), b.ravel()),
        "kendall_tau_confidence": stable_tau(pa.max(axis=1), pb.max(axis=1)),
        "prediction_agreement": float(np.mean(np.argmax(a, axis=1) == np.argmax(b, axis=1))),
        "mean_abs_logit_diff": float(np.mean(np.abs(diff))),
        "max_abs_logit_diff": float(np.max(np.abs(diff))),
        "mean_l2_logit_diff": float(np.mean(np.linalg.norm(diff, axis=1))),
        "mean_abs_probability_diff": float(np.mean(np.abs(pdiff))),
        "max_abs_probability_diff": float(np.max(np.abs(pdiff))),
    }


def pairwise_invariance(outputs: Mapping[str, Mapping[str, np.ndarray]]) -> List[Dict[str, Any]]:
    variants = list(outputs)
    rows: List[Dict[str, Any]] = []
    for index, left in enumerate(variants):
        for right in variants[index + 1 :]:
            a = outputs[left]
            b = outputs[right]
            if not np.array_equal(a["item_ids"], b["item_ids"]):
                raise ValueError(f"Evaluation ids differ between {left} and {right}")
            rows.append({"variant_a": left, "variant_b": right, **invariance_metrics(a["logits"], b["logits"])})
    return rows


def scalar_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))
    }


def mean_scalar_dict(metric_dicts: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    if not metric_dicts:
        return {}
    common = set.intersection(*(set(scalar_metrics(item)) for item in metric_dicts))
    return {
        key: float(np.mean([float(item[key]) for item in metric_dicts]))
        for key in sorted(common)
    }


def mean_std(values: Sequence[float | int | None]) -> Dict[str, Any]:
    raw = list(values)
    if not raw or any(value is None for value in raw):
        return {"mean": None, "std": None, "values": raw, "ddof": 1}
    array = np.asarray(raw, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "values": array.tolist(),
        "ddof": 1,
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value
