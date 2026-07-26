#!/usr/bin/env python3
"""Shared utilities for RGCN graph-variant data augmentation experiments.

The key design choice is a *global native relation vocabulary*.  Each raw graph
variant keeps its own edges, but a relation name always maps to the same RGCN
relation ID.  This prevents local DGL relation numbering from silently changing
what a learned relation weight means between variants.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import resource
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    import dgl
    from dgl.nn import RelGraphConv
except ImportError:
    dgl = None
    RelGraphConv = None

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DBLP_RELATIONS: Tuple[str, ...] = (
    "author-paper",
    "paper-author",
    "paper-conference",
    "conference-paper",
    "paper-term",
    "term-paper",
    "paper-area",
    "area-paper",
    "conference-area",
    "area-conference",
    "author-area",
    "area-author",
)

IMDB_RELATIONS: Tuple[str, ...] = (
    "actor-movie",
    "movie-actor",
    "movie-link",
    "link-movie",
    "movie-director",
    "director-movie",
    "actor-link",
    "link-actor",
    "link-director",
    "director-link",
)


# Global relation vocabulary for the IMDb link-prediction graphs.  It is the
# union of every native relation that the upstream runner may construct for
# md, mg, and ml.  A shared checkpoint requires one stable semantic ID per
# relation even when individual variants contain only a subset.
IMDB_LP_RELATIONS: Tuple[str, ...] = (
    "movie-actor",
    "actor-movie",
    "movie-director",
    "director-movie",
    "movie-link",
    "link-movie",
    "movie-genre",
    "genre-movie",
    "link-director",
    "director-link",
    "link-actor",
    "actor-link",
)


def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def relation_to_id(relations: Sequence[str]) -> Dict[str, int]:
    if len(set(relations)) != len(relations):
        raise ValueError("Global relation vocabulary contains duplicates.")
    return {name: idx for idx, name in enumerate(relations)}


def to_homogeneous_with_global_relations(
    graph: dgl.DGLHeteroGraph,
    global_relation_to_id: Mapping[str, int],
):
    """Convert a heterograph while remapping local DGL relation IDs globally."""
    if dgl is None:
        raise ImportError("DGL is required for IMDb/DBLP graph conversion")
    homogeneous = dgl.to_homogeneous(graph)
    local_etypes = (
        homogeneous.edata[dgl.ETYPE]
        if dgl.ETYPE in homogeneous.edata
        else homogeneous.edata["_TYPE"]
    ).long()
    ntype = (
        homogeneous.ndata[dgl.NTYPE]
        if dgl.NTYPE in homogeneous.ndata
        else homogeneous.ndata["_TYPE"]
    ).long()
    nid = (
        homogeneous.ndata[dgl.NID]
        if dgl.NID in homogeneous.ndata
        else homogeneous.ndata["_ID"]
    ).long()

    local_to_global = []
    for canonical_etype in graph.canonical_etypes:
        relation_name = canonical_etype[1]
        if relation_name not in global_relation_to_id:
            raise KeyError(
                f"Relation {relation_name!r} is absent from the global vocabulary. "
                f"Known relations: {sorted(global_relation_to_id)}"
            )
        local_to_global.append(global_relation_to_id[relation_name])
    local_to_global_tensor = torch.tensor(local_to_global, dtype=torch.long)
    global_etypes = local_to_global_tensor[local_etypes.cpu()]

    indexers: Dict[str, torch.Tensor] = {}
    for name in graph.ntypes:
        type_id = graph.get_ntype_id(name)
        mask = ntype == type_id
        homogeneous_indices = mask.nonzero(as_tuple=False).squeeze(1)
        local_ids = nid[mask]
        order = torch.argsort(local_ids)
        indexers[name] = homogeneous_indices[order]

    return homogeneous, global_etypes, indexers


def assert_same_tensor(name: str, values: Mapping[str, torch.Tensor]) -> None:
    variants = list(values)
    reference_name = variants[0]
    reference = values[reference_name].cpu()
    for variant in variants[1:]:
        current = values[variant].cpu()
        if reference.shape != current.shape or not torch.equal(reference, current):
            raise ValueError(
                f"{name} differs between {reference_name} and {variant}: "
                f"{tuple(reference.shape)} vs {tuple(current.shape)}"
            )


def assert_same_numpy(name: str, values: Mapping[str, np.ndarray]) -> None:
    variants = list(values)
    reference_name = variants[0]
    reference = np.asarray(values[reference_name])
    for variant in variants[1:]:
        current = np.asarray(values[variant])
        if reference.shape != current.shape or not np.array_equal(reference, current):
            raise ValueError(
                f"{name} differs between {reference_name} and {variant}: "
                f"{reference.shape} vs {current.shape}"
            )


def assert_same_indexers(indexers_by_variant: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
    variants = list(indexers_by_variant)
    reference_variant = variants[0]
    reference = indexers_by_variant[reference_variant]
    for variant in variants[1:]:
        current = indexers_by_variant[variant]
        if set(reference) != set(current):
            raise ValueError(
                f"Node-type sets differ: {reference_variant}={sorted(reference)}, "
                f"{variant}={sorted(current)}"
            )
        for node_type in reference:
            if not torch.equal(reference[node_type].cpu(), current[node_type].cpu()):
                raise ValueError(
                    f"Homogeneous node indexing for type {node_type!r} differs "
                    f"between {reference_variant} and {variant}. A single shared "
                    "embedding table would not be semantically aligned."
                )


class RGCNEncoder(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_rels: int,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        num_layers: int,
        num_bases: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if RelGraphConv is None:
            raise ImportError("DGL is required to construct RGCNEncoder")
        self.emb = nn.Embedding(num_nodes, in_dim)
        nn.init.xavier_uniform_(self.emb.weight)

        def make_layer(din: int, dout: int, activation, layer_dropout: float):
            kwargs = dict(
                regularizer="basis",
                num_bases=min(num_bases, num_rels),
                self_loop=True,
                dropout=layer_dropout,
                activation=activation,
            )
            try:
                return RelGraphConv(din, dout, num_rels, low_mem=True, **kwargs)
            except TypeError:
                return RelGraphConv(din, dout, num_rels, **kwargs)

        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(make_layer(in_dim, out_dim, None, dropout))
        else:
            self.layers.append(make_layer(in_dim, hid_dim, F.relu, dropout))
            for _ in range(num_layers - 2):
                self.layers.append(make_layer(hid_dim, hid_dim, F.relu, dropout))
            self.layers.append(make_layer(hid_dim, out_dim, None, dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_types: torch.Tensor,
        edge_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden = self.emb.weight
        for layer in self.layers:
            if edge_norm is None:
                hidden = layer(graph, hidden, edge_types)
            else:
                # DGL RelGraphConv accepts an optional per-edge normalization
                # tensor with shape (num_edges, 1).
                hidden = layer(graph, hidden, edge_types, edge_norm)
        return self.dropout(hidden)


class NodeClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs)


class EarlyStopper:
    def __init__(self, mode: str, patience: int, min_delta: float = 1e-6) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf if mode == "min" else -math.inf
        self.bad_cycles = 0

    def update(self, value: float) -> bool:
        improved = (
            value < self.best - self.min_delta
            if self.mode == "min"
            else value > self.best + self.min_delta
        )
        if improved:
            self.best = value
            self.bad_cycles = 0
        else:
            self.bad_cycles += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.bad_cycles >= self.patience


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor) -> Dict[str, float]:
    selected_logits = logits[indices]
    y_true = labels[indices].detach().cpu().numpy()
    y_pred = selected_logits.argmax(dim=1).detach().cpu().numpy()
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "Micro_F1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "Macro_F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _safe_tau(a: np.ndarray, b: np.ndarray) -> float:
    tau = kendalltau(np.asarray(a), np.asarray(b), nan_policy="omit").statistic
    return float(tau) if tau is not None and np.isfinite(tau) else float("nan")


def classification_invariance_rows(outputs: Mapping[str, Dict[str, np.ndarray]]) -> List[Dict[str, Any]]:
    variants = list(outputs)
    rows: List[Dict[str, Any]] = []
    for i, va in enumerate(variants):
        for vb in variants[i + 1 :]:
            a = outputs[va]
            b = outputs[vb]
            if not np.array_equal(a["item_id"], b["item_id"]):
                raise ValueError(f"Test item IDs differ between {va} and {vb}.")
            if a["logits"].shape != b["logits"].shape:
                raise ValueError(f"Logit shapes differ between {va} and {vb}.")
            difference = a["logits"] - b["logits"]
            rows.append(
                {
                    "variant_a": va,
                    "variant_b": vb,
                    "kendall_tau_flat_logits": _safe_tau(a["logits"].ravel(), b["logits"].ravel()),
                    "kendall_tau_confidence": _safe_tau(a["confidence"], b["confidence"]),
                    "prediction_agreement": float(np.mean(a["prediction"] == b["prediction"])),
                    "max_abs_logit_diff": float(np.max(np.abs(difference))),
                    "mean_abs_logit_diff": float(np.mean(np.abs(difference))),
                    "mean_l2_logit_diff": float(np.linalg.norm(difference, axis=1).mean()),
                }
            )
    return rows


def pairwise_loss(pos_logit: torch.Tensor, neg_logit: torch.Tensor) -> torch.Tensor:
    return -(F.logsigmoid(pos_logit).mean() + F.logsigmoid(-neg_logit).mean())


def link_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(np.int64)
    return {
        "AUC": float(roc_auc_score(y_true, y_score)),
        "AP": float(average_precision_score(y_true, y_score)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
    }


def link_ranking_metrics(
    positive_edges: np.ndarray,
    negative_edges: np.ndarray,
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
) -> Dict[str, float]:
    candidates: MutableMapping[int, List[Tuple[float, int]]] = defaultdict(list)
    for edge, score in zip(positive_edges, positive_scores):
        candidates[int(edge[0])].append((float(score), 1))
    for edge, score in zip(negative_edges, negative_scores):
        candidates[int(edge[0])].append((float(score), 0))

    hits_1 = hits_3 = hits_5 = 0
    reciprocal_rank = 0.0
    query_count = 0
    for items in candidates.values():
        items.sort(key=lambda item: item[0], reverse=True)
        positive_ranks = [rank + 1 for rank, (_, label) in enumerate(items) if label == 1]
        if not positive_ranks:
            continue
        rank = min(positive_ranks)
        query_count += 1
        reciprocal_rank += 1.0 / rank
        hits_1 += int(rank <= 1)
        hits_3 += int(rank <= 3)
        hits_5 += int(rank <= 5)

    denominator = max(1, query_count)
    return {
        "Hits@1": hits_1 / denominator,
        "Hits@3": hits_3 / denominator,
        "Hits@5": hits_5 / denominator,
        "MRR": reciprocal_rank / denominator,
        "ranking_queries": float(query_count),
    }


def link_invariance_rows(outputs: Mapping[str, pd.DataFrame], threshold: float) -> List[Dict[str, Any]]:
    variants = list(outputs)
    rows: List[Dict[str, Any]] = []
    keys = ["paper_id", "conf_id", "label"]
    for i, va in enumerate(variants):
        for vb in variants[i + 1 :]:
            merged = outputs[va].merge(outputs[vb], on=keys, suffixes=("_a", "_b"), how="inner")
            expected = len(outputs[va])
            if len(merged) != expected or len(outputs[vb]) != expected:
                raise ValueError(
                    f"Candidate sets do not align for {va} and {vb}: "
                    f"{len(outputs[va])}, {len(outputs[vb])}, merged={len(merged)}"
                )
            diff = merged["score_a"].to_numpy() - merged["score_b"].to_numpy()
            pred_a = merged["score_a"].to_numpy() >= threshold
            pred_b = merged["score_b"].to_numpy() >= threshold
            rows.append(
                {
                    "variant_a": va,
                    "variant_b": vb,
                    "kendall_tau_scores": _safe_tau(merged["score_a"], merged["score_b"]),
                    "prediction_agreement": float(np.mean(pred_a == pred_b)),
                    "max_abs_score_diff": float(np.max(np.abs(diff))),
                    "mean_abs_score_diff": float(np.mean(np.abs(diff))),
                }
            )
    return rows


def subsample_negatives_per_query(neg_edges: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0:
        return np.asarray(neg_edges, dtype=np.int64)
    rng = np.random.RandomState(seed)
    by_query: MutableMapping[int, List[Tuple[int, int]]] = defaultdict(list)
    for query, candidate in np.asarray(neg_edges):
        by_query[int(query)].append((int(query), int(candidate)))
    selected: List[Tuple[int, int]] = []
    for query in sorted(by_query):
        items = by_query[query]
        if len(items) <= maximum:
            selected.extend(items)
        else:
            indices = rng.choice(len(items), size=maximum, replace=False)
            selected.extend(items[int(index)] for index in indices)
    return np.asarray(selected, dtype=np.int64)


def group_negatives_by_query(neg_edges: np.ndarray) -> Dict[int, np.ndarray]:
    groups: MutableMapping[int, List[Tuple[int, int]]] = defaultdict(list)
    for query, candidate in np.asarray(neg_edges):
        groups[int(query)].append((int(query), int(candidate)))
    return {query: np.asarray(items, dtype=np.int64) for query, items in groups.items()}



def inverse_destination_degree_norm(graph: dgl.DGLGraph) -> torch.Tensor:
    """Return c_i-style 1/in_degree(dst) normalization for every edge."""
    _src, dst = graph.edges(order="eid")
    degrees = graph.in_degrees().clamp(min=1).to(torch.float32)
    return (1.0 / degrees[dst]).unsqueeze(1)


def triple_link_invariance_rows(
    outputs: Mapping[str, pd.DataFrame],
    threshold: float,
) -> List[Dict[str, Any]]:
    """Pairwise invariance metrics on a shared triple candidate table."""
    variants = list(outputs)
    rows: List[Dict[str, Any]] = []
    keys = ["query_id", "head", "relation", "tail", "label"]
    for i, va in enumerate(variants):
        for vb in variants[i + 1 :]:
            merged = outputs[va].merge(
                outputs[vb], on=keys, suffixes=("_a", "_b"), how="inner"
            )
            expected = len(outputs[va])
            if len(merged) != expected or len(outputs[vb]) != expected:
                raise ValueError(
                    f"Triple candidate sets do not align for {va} and {vb}: "
                    f"{len(outputs[va])}, {len(outputs[vb])}, merged={len(merged)}"
                )
            score_a = merged["score_a"].to_numpy()
            score_b = merged["score_b"].to_numpy()
            diff = score_a - score_b
            rows.append(
                {
                    "variant_a": va,
                    "variant_b": vb,
                    "kendall_tau_scores": _safe_tau(score_a, score_b),
                    "prediction_agreement": float(
                        np.mean((score_a >= threshold) == (score_b >= threshold))
                    ),
                    "max_abs_score_diff": float(np.max(np.abs(diff))),
                    "mean_abs_score_diff": float(np.mean(np.abs(diff))),
                }
            )
    return rows

def model_memory_bytes(model: nn.Module) -> Dict[str, int]:
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    return {
        "parameter_bytes": int(parameter_bytes),
        "buffer_bytes": int(buffer_bytes),
        "static_model_bytes": int(parameter_bytes + buffer_bytes),
    }


def process_peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    return int(value if sys.platform == "darwin" else value * 1024)


def reset_cuda_peak(device: torch.device) -> None:
    if device.type != "cuda":
        return
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)


def cuda_memory_stats(device: torch.device) -> Dict[str, int]:
    if device.type != "cuda":
        return {
            "gpu_allocated_bytes": 0,
            "gpu_reserved_bytes": 0,
            "gpu_peak_allocated_bytes": 0,
            "gpu_peak_reserved_bytes": 0,
        }
    torch.cuda.synchronize(device)
    return {
        "gpu_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "gpu_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }



def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    """Atomically replace a PyTorch checkpoint.

    The previous valid checkpoint remains in place if the process is interrupted
    while writing the temporary file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def torch_load_full(path: Path, map_location: Any = "cpu") -> Any:
    """Load a complete training-state checkpoint across PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move restored optimizer tensors, including Adam moments, to device."""
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _capture_local_numpy_rng(rng) -> Dict[str, Any]:
    if isinstance(rng, np.random.Generator):
        return {"kind": "generator", "state": rng.bit_generator.state}
    if isinstance(rng, np.random.RandomState):
        return {"kind": "random_state", "state": rng.get_state()}
    raise TypeError(f"Unsupported NumPy RNG type: {type(rng)!r}")


def _restore_local_numpy_rng(saved, rng) -> None:
    # Backward compatibility with resume states produced before Generator
    # support, where numpy_local was the raw RandomState tuple.
    if not isinstance(saved, Mapping):
        if isinstance(rng, np.random.RandomState):
            rng.set_state(saved)
            return
        raise ValueError(
            "Legacy RandomState resume data cannot restore a NumPy Generator"
        )
    kind = saved.get("kind")
    if kind == "generator" and isinstance(rng, np.random.Generator):
        rng.bit_generator.state = saved["state"]
        return
    if kind == "random_state" and isinstance(rng, np.random.RandomState):
        rng.set_state(saved["state"])
        return
    raise ValueError(
        f"Saved NumPy RNG kind {kind!r} is incompatible with {type(rng)!r}"
    )


def capture_rng_state(rng) -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "numpy_local": _capture_local_numpy_rng(rng),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any], rng) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    _restore_local_numpy_rng(state["numpy_local"], rng)
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_state = state.get("torch_cuda")
    if torch.cuda.is_available() and cuda_state is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_state])


def early_stopper_state(stopper: "EarlyStopper") -> Dict[str, Any]:
    return {
        "mode": stopper.mode,
        "patience": stopper.patience,
        "min_delta": stopper.min_delta,
        "best": stopper.best,
        "bad_cycles": stopper.bad_cycles,
    }


def restore_early_stopper(stopper: "EarlyStopper", state: Mapping[str, Any]) -> None:
    if stopper.mode != state["mode"]:
        raise ValueError(f"Early-stopper mode mismatch: {stopper.mode} vs {state['mode']}")
    if stopper.patience != int(state["patience"]):
        raise ValueError(
            f"Early-stopper patience mismatch: {stopper.patience} vs {state['patience']}"
        )
    if abs(stopper.min_delta - float(state["min_delta"])) > 1e-15:
        raise ValueError("Early-stopper min_delta differs from the saved run")
    stopper.best = float(state["best"])
    stopper.bad_cycles = int(state["bad_cycles"])


def _canonical_config(config: Mapping[str, Any]) -> str:
    return json.dumps(json_ready(dict(config)), sort_keys=True, separators=(",", ":"))


def load_latest_training_state(
    path: Path,
    *,
    resume: bool,
    run_config: Mapping[str, Any],
    modules: Mapping[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    early_stopper: "EarlyStopper",
    rng,
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    """Restore an exact state saved at a completed super-epoch boundary."""
    path = Path(path)
    if not path.exists():
        existing_outputs = [
            candidate
            for candidate in (path.parent / "shared_checkpoint.pt", path.parent / "training_history.csv")
            if candidate.exists()
        ]
        if existing_outputs:
            names = ", ".join(str(candidate) for candidate in existing_outputs)
            raise RuntimeError(
                "This seed directory contains outputs from a non-resumable or incomplete "
                f"run ({names}), but no {path.name}. Exact resume is impossible. "
                "Choose a new --output-dir or move the old seed directory before restarting."
            )
        if resume:
            print(f"[resume] No latest state at {path}; starting a new run.", flush=True)
        return None
    if not resume:
        raise RuntimeError(
            f"Existing resumable state found at {path}. Use --resume to continue it, "
            "or choose a new --output-dir for a fresh run."
        )

    state = torch_load_full(path, map_location=device)
    if int(state.get("state_version", 0)) != 1:
        raise ValueError(f"Unsupported resume-state version in {path}")
    if _canonical_config(state["run_config"]) != _canonical_config(run_config):
        raise ValueError(
            "Resume configuration differs from the saved run. Architecture, optimizer, "
            "batching, variants, seed, patience, and evaluation settings must match. "
            "You may only increase --super-epochs or change the device."
        )
    saved_modules = state["modules"]
    if set(saved_modules) != set(modules):
        raise ValueError(
            f"Saved module keys {sorted(saved_modules)} do not match {sorted(modules)}"
        )
    for name, module in modules.items():
        module.load_state_dict(saved_modules[name])
    optimizer.load_state_dict(state["optimizer"])
    optimizer_to_device(optimizer, device)
    restore_early_stopper(early_stopper, state["early_stopper"])
    restore_rng_state(state["rng_state"], rng)
    print(
        f"[resume] Restored {path} after completed super-epoch "
        f"{state['completed_super_epoch']}.",
        flush=True,
    )
    return state


def save_latest_training_state(
    path: Path,
    *,
    dataset: str,
    run_config: Mapping[str, Any],
    modules: Mapping[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    early_stopper: "EarlyStopper",
    rng,
    completed_super_epoch: int,
    counters: Mapping[str, int],
    history: Sequence[Mapping[str, Any]],
    training_seconds_elapsed: float,
    process_peak_rss_bytes_value: int,
    training_gpu: Mapping[str, int],
) -> None:
    """Save the latest exact restart point after a complete super-epoch."""
    atomic_torch_save(
        {
            "state_version": 1,
            "dataset": dataset,
            "run_config": dict(run_config),
            "modules": {name: module.state_dict() for name, module in modules.items()},
            "optimizer": optimizer.state_dict(),
            "early_stopper": early_stopper_state(early_stopper),
            "rng_state": capture_rng_state(rng),
            "completed_super_epoch": int(completed_super_epoch),
            "counters": {str(key): int(value) for key, value in counters.items()},
            "history": [dict(row) for row in history],
            "training_seconds_elapsed": float(training_seconds_elapsed),
            "process_peak_rss_bytes": int(process_peak_rss_bytes_value),
            "training_gpu": {str(key): int(value) for key, value in training_gpu.items()},
        },
        Path(path),
    )


def merge_cuda_memory_stats(
    previous: Optional[Mapping[str, int]], current: Mapping[str, int]
) -> Dict[str, int]:
    previous = previous or {}
    return {
        "gpu_allocated_bytes": int(current.get("gpu_allocated_bytes", 0)),
        "gpu_reserved_bytes": int(current.get("gpu_reserved_bytes", 0)),
        "gpu_peak_allocated_bytes": max(
            int(previous.get("gpu_peak_allocated_bytes", 0)),
            int(current.get("gpu_peak_allocated_bytes", 0)),
        ),
        "gpu_peak_reserved_bytes": max(
            int(previous.get("gpu_peak_reserved_bytes", 0)),
            int(current.get("gpu_peak_reserved_bytes", 0)),
        ),
    }

def checkpoint_size_bytes(path: Path) -> int:
    return int(path.stat().st_size) if path.exists() else 0


def sha256_tensor(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(dict(payload)), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def mean_dict(metric_dicts: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not metric_dicts:
        return {}
    keys = metric_dicts[0].keys()
    return {key: float(np.mean([metrics[key] for metrics in metric_dicts])) for key in keys}
