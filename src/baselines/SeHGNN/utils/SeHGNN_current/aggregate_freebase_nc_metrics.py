#!/usr/bin/env python3
"""Aggregate four-run SeHGNN Freebase NC experiments and print LaTeX rows.

Groups
------
``k``
    One independently trained SeHGNN per transformed graph variant.  The script
    additionally creates ``Output Fusion`` per aligned run by averaging the raw
    best-checkpoint logits across variants before argmax/softmax.

``full_k`` / ``restricted_k``
    One universal-graph model per run.  Its metrics are duplicated for every
    considered variant and Output Fusion, as requested.  Pairwise invariance is
    exactly one because each comparison references the same prediction tensor.

Kendall definitions
-------------------
* Kendall tau: node-macro Kendall tau-b over all class logits.
* Kendall tau at 1: top-1 class agreement rate (tau is undefined for a one-item list).
* Kendall tau at 3: node-macro Kendall tau-b over the union of each model's top-3 classes.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
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

GROUP_ALIASES = {
    "k": "k",
    "full_k": "full_k",
    "fullk": "full_k",
    "restricted_k": "restricted_k",
    "restrictedk": "restricted_k",
}
MODEL_LABELS = {
    "k": "SeHGNN",
    "full_k": "SeHGNN* + fullK",
    "restricted_k": "MAGNN* + restrictedK",
}
TABLE_METRICS = ("accuracy", "macro_precision", "macro_recall", "micro_f1", "macro_f1")
RESOURCE_KEYS = (
    "train_time_sec",
    "epochs_run",
    "checkpoint_mib",
    "parameter_mib",
    "peak_training_gpu_mib",
    "peak_inference_gpu_mib",
)


def display_name(name: str) -> str:
    if name == "Output Fusion":
        return name
    return name.replace("_", " ").strip().title()


def mean_std(values: Sequence[float | int | None]) -> Dict[str, Any]:
    if not values or all(v is None for v in values):
        return {"mean": None, "std": None, "values": list(values), "ddof": 1}
    if any(v is None for v in values):
        return {"mean": None, "std": None, "values": list(values), "ddof": 1}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0,
        "values": arr.tolist(),
        "ddof": 1,
    }


def classification_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """Recompute the runner's complete test metric set from raw logits."""
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError(f"Bad logits/labels shapes: {logits.shape} and {labels.shape}")
    num_classes = int(logits.shape[1])
    pred = np.argmax(logits, axis=1)
    order = np.argsort(-logits, axis=1, kind="stable")
    rank = np.argmax(order == labels[:, None], axis=1) + 1
    per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
        labels,
        pred,
        labels=np.arange(num_classes),
        average=None,
        zero_division=0,
    )
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=1)) + np.max(logits, axis=1)
    cross_entropy = float(np.mean(logsumexp - logits[np.arange(len(labels)), labels]))
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "macro_precision": float(precision_score(labels, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(labels, pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(labels, pred, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(labels, pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, pred, average="weighted", zero_division=0)),
        "hit_at_1": float(np.mean(rank <= 1)),
        "hit_at_3": float(np.mean(rank <= min(3, num_classes))),
        "mrr": float(np.mean(1.0 / rank)),
        "loss": cross_entropy,
        "per_class_precision": per_p.astype(float).tolist(),
        "per_class_recall": per_r.astype(float).tolist(),
        "per_class_f1": per_f1.astype(float).tolist(),
        "per_class_support": per_support.astype(int).tolist(),
        "confusion_matrix": confusion_matrix(
            labels, pred, labels=np.arange(num_classes)
        ).astype(int).tolist(),
    }


def stable_tau(a: np.ndarray, b: np.ndarray) -> float:
    value = kendalltau(a, b, variant="b", nan_policy="omit").correlation
    if value is None or not np.isfinite(value):
        return 1.0 if np.allclose(a, b, equal_nan=True) else 0.0
    return float(value)


def kendall_metrics(logits_a: np.ndarray, logits_b: np.ndarray, top_k: int = 3) -> Dict[str, float]:
    if logits_a.shape != logits_b.shape:
        raise ValueError(f"Logit shape mismatch: {logits_a.shape} vs {logits_b.shape}")
    if logits_a.ndim != 2:
        raise ValueError("Expected [num_nodes, num_classes] logits")
    all_tau: List[float] = []
    top3_tau: List[float] = []
    top1_equal: List[float] = []
    actual_k = min(top_k, logits_a.shape[1])
    for row_a, row_b in zip(logits_a, logits_b):
        all_tau.append(stable_tau(row_a, row_b))
        top1_equal.append(float(int(np.argmax(row_a)) == int(np.argmax(row_b))))
        a_top = np.argsort(-row_a, kind="stable")[:actual_k]
        b_top = np.argsort(-row_b, kind="stable")[:actual_k]
        union = np.union1d(a_top, b_top)
        top3_tau.append(1.0 if union.size <= 1 else stable_tau(row_a[union], row_b[union]))
    return {
        "kendall_tau": float(np.mean(all_tau)),
        "kendall_tau_at_1": float(np.mean(top1_equal)),
        "kendall_tau_at_3": float(np.mean(top3_tau)),
    }


def parse_result_specs(items: Sequence[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--result must be NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty result name in {item!r}")
        if name in out:
            raise ValueError(f"Duplicate result name {name!r}")
        out[name] = Path(path).expanduser().resolve()
    return out


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_run_file(raw_path: str, result_json: Path) -> Path:
    path = Path(raw_path).expanduser()
    candidates = [path, result_json.parent / path, result_json.parent / path.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not locate run artifact {raw_path!r}; checked {[str(x) for x in candidates]}"
    )


def _candidate_logits_paths(run: Mapping[str, Any], result_json: Path) -> List[Path]:
    """Return explicit and legacy-inferred logits artifact candidates."""
    candidates: List[Path] = []
    explicit_values = [
        run.get("logits_file"),
        run.get("logits_path"),
        run.get("artifacts", {}).get("logits_file")
        if isinstance(run.get("artifacts"), Mapping)
        else None,
    ]
    for raw in explicit_values:
        if raw not in (None, ""):
            path = Path(str(raw)).expanduser()
            candidates.extend([path, result_json.parent / path, result_json.parent / path.name])

    checkpoint_raw = run.get("checkpoint") or run.get("checkpoint_path")
    if checkpoint_raw not in (None, ""):
        checkpoint = Path(str(checkpoint_raw)).expanduser()
        inferred_names = [
            checkpoint.with_name(checkpoint.stem + "_logits.npz"),
            checkpoint.with_suffix(checkpoint.suffix + ".logits.npz"),
            result_json.parent / "logits" / (checkpoint.stem + "_logits.npz"),
            result_json.parent / (checkpoint.stem + "_logits.npz"),
        ]
        candidates.extend(inferred_names)

    out: List[Path] = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            out.append(resolved)
    return out


def load_run_logits(run: Mapping[str, Any], result_json: Path) -> Dict[str, np.ndarray]:
    candidates = _candidate_logits_paths(run, result_json)
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        seed = run.get("seed", "unknown")
        raise ValueError(
            f"Run seed={seed} has no usable logits artifact. Older versions of "
            "run_freebase_magnn_channels.py saved checkpoints and metrics but did not "
            "save raw logits, which are required for Output Fusion and Kendall tau. "
            "Backfill them without retraining using:\n\n"
            f"  python backfill_freebase_nc_logits.py --result-json {result_json}\n\n"
            f"Checked candidates: {[str(x) for x in candidates]}"
        )
    with np.load(path) as data:
        required = ("test_logits", "test_labels", "test_idx")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"Missing {missing} in {path}")
        return {key: data[key].copy() for key in data.files}


def runs_by_seed(payload: Mapping[str, Any], expected_runs: int) -> Dict[int, Mapping[str, Any]]:
    runs = payload.get("runs", [])
    if len(runs) != expected_runs:
        raise ValueError(f"Expected {expected_runs} runs, got {len(runs)}")
    out = {int(run["seed"]): run for run in runs}
    if len(out) != len(runs):
        raise ValueError("Duplicate seeds in result JSON")
    return out


def _first_numeric(*values: Any) -> float | None:
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def run_resources(
    run: Mapping[str, Any], result_json: Path | None = None
) -> Dict[str, float | None]:
    """Read current and legacy memory keys with an exact checkpoint-size fallback."""
    memory = run.get("memory", {})
    if not isinstance(memory, Mapping):
        memory = {}
    memory_stats = run.get("memory_stats", {})
    if not isinstance(memory_stats, Mapping):
        memory_stats = {}

    checkpoint_mib = _first_numeric(
        memory.get("checkpoint_mib"),
        memory.get("checkpoint_size_mib"),
        memory_stats.get("checkpoint_mib"),
        memory_stats.get("checkpoint_size_mib"),
        run.get("checkpoint_mib"),
        run.get("checkpoint_size_mib"),
    )
    if checkpoint_mib is None and result_json is not None:
        checkpoint_raw = run.get("checkpoint") or run.get("checkpoint_path")
        if checkpoint_raw not in (None, ""):
            try:
                checkpoint_path = resolve_run_file(str(checkpoint_raw), result_json)
                checkpoint_mib = float(checkpoint_path.stat().st_size) / (1024.0 ** 2)
            except FileNotFoundError:
                pass

    return {
        "train_time_sec": _first_numeric(
            run.get("train_time_sec"), run.get("training_time_sec"), run.get("train_time")
        ),
        "epochs_run": _first_numeric(
            run.get("epochs_run"), run.get("epochs"), run.get("num_epochs")
        ),
        "checkpoint_mib": checkpoint_mib,
        "parameter_mib": _first_numeric(
            memory.get("parameter_mib"),
            memory.get("model_parameter_mib"),
            memory.get("model_parameter_memory_mib"),
            memory_stats.get("parameter_mib"),
            memory_stats.get("model_parameter_mib"),
            run.get("parameter_mib"),
            run.get("model_parameter_memory_mib"),
        ),
        "peak_training_gpu_mib": _first_numeric(
            memory.get("peak_training_gpu_mib"),
            memory.get("peak_training_allocated_mib"),
            memory.get("peak_train_gpu_mib"),
            memory_stats.get("peak_training_gpu_mib"),
            memory_stats.get("peak_training_allocated_mib"),
            run.get("peak_training_gpu_mib"),
            run.get("peak_training_allocated_mib"),
        ),
        "peak_inference_gpu_mib": _first_numeric(
            memory.get("peak_inference_gpu_mib"),
            memory.get("peak_inference_allocated_mib"),
            memory.get("peak_eval_gpu_mib"),
            memory_stats.get("peak_inference_gpu_mib"),
            memory_stats.get("peak_inference_allocated_mib"),
            run.get("peak_inference_gpu_mib"),
            run.get("peak_inference_allocated_mib"),
        ),
    }


def aggregate_run_metrics(run_metrics: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate every scalar metric; detailed arrays are handled separately."""
    keys = sorted(set.intersection(*(set(x) for x in run_metrics)))
    scalar_keys = [
        key
        for key in keys
        if all(isinstance(x[key], (int, float, np.integer, np.floating)) for x in run_metrics)
    ]
    return {key: mean_std([float(x[key]) for x in run_metrics]) for key in scalar_keys}


def aggregate_detailed_metrics(run_metrics: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-class arrays and confusion matrices over aligned runs."""
    out: Dict[str, Any] = {}
    for key in ("per_class_precision", "per_class_recall", "per_class_f1", "per_class_support"):
        arrays = np.asarray([x[key] for x in run_metrics], dtype=np.float64)
        out[key] = [
            mean_std(arrays[:, class_index].tolist())
            for class_index in range(arrays.shape[1])
        ]
    matrices = np.asarray([x["confusion_matrix"] for x in run_metrics], dtype=np.float64)
    out["confusion_matrix"] = {
        "mean": np.mean(matrices, axis=0).tolist(),
        "std": np.std(matrices, axis=0, ddof=1).tolist() if len(matrices) > 1 else np.zeros_like(matrices[0]).tolist(),
        "sum": np.sum(matrices, axis=0).astype(int).tolist(),
        "ddof": 1,
    }
    return out


def aggregate_resources(run_resources_list: Sequence[Mapping[str, float | None]]) -> Dict[str, Any]:
    return {
        key: mean_std([row.get(key) for row in run_resources_list])
        for key in RESOURCE_KEYS
    }


def combine_optional(values: Sequence[float | None], operation: str) -> float | None:
    if any(value is None for value in values):
        return None
    numeric = [float(value) for value in values if value is not None]
    if operation == "sum":
        return float(sum(numeric))
    if operation == "max":
        return float(max(numeric))
    raise ValueError(operation)


def fusion_resources(rows: Sequence[Mapping[str, float | None]]) -> Dict[str, float | None]:
    """Resource accounting for the explicit all-checkpoints-loaded fusion design.

    Training is independent, so elapsed time/epochs and stored model size are
    additive.  Peak training is the maximum observed separate training peak.
    Output-fusion inference follows the user's pseudocode that loads all models,
    so the table uses the conservative sum of per-model inference peaks.
    """
    return {
        "train_time_sec": combine_optional([r["train_time_sec"] for r in rows], "sum"),
        "epochs_run": combine_optional([r["epochs_run"] for r in rows], "sum"),
        "checkpoint_mib": combine_optional([r["checkpoint_mib"] for r in rows], "sum"),
        "parameter_mib": combine_optional([r["parameter_mib"] for r in rows], "sum"),
        "peak_training_gpu_mib": combine_optional([r["peak_training_gpu_mib"] for r in rows], "max"),
        "peak_inference_gpu_mib": combine_optional([r["peak_inference_gpu_mib"] for r in rows], "sum"),
    }


def format_pm(stat: Mapping[str, Any], decimals: int) -> str:
    mean, std = stat.get("mean"), stat.get("std")
    if mean is None or std is None:
        return r"$\text{N/A}$"
    return f"${float(mean):.{decimals}f} \\pm {float(std):.{decimals}f}$"


def node_row(name: str, stats: Mapping[str, Mapping[str, Any]]) -> str:
    values = " & ".join(format_pm(stats[key], 4) for key in TABLE_METRICS)
    return f"& {display_name(name)} & {values} \\\\"


def invariance_row(comparison: str, stats: Mapping[str, Mapping[str, Any]]) -> str:
    values = " & ".join(
        format_pm(stats[key], 4)
        for key in ("kendall_tau", "kendall_tau_at_1", "kendall_tau_at_3")
    )
    return f"& {comparison} & {values} \\\\"


def resource_row(name: str, stats: Mapping[str, Mapping[str, Any]]) -> str:
    values = " & ".join(format_pm(stats[key], 2) for key in RESOURCE_KEYS)
    return f"& {display_name(name)} & {values} \\\\"


def _resolve_k_variant(
    payload: Mapping[str, Any],
    expected_variant: str | None,
) -> tuple[str | None, str]:
    """Resolve the per-graph K variant, including older result JSON files.

    Older versions of ``run_freebase_magnn_channels.py`` accidentally omitted
    ``manifest.variant`` when copying the preprocessing manifest into
    ``sehgnn_results.json``.  The preprocessing output itself still contains the
    correct value.  This resolver therefore checks, in order:

    1. ``sehgnn_results.json`` -> ``manifest.variant``;
    2. ``data_dir/channels_manifest.json`` -> ``variant``;
    3. the final directory component of ``data_dir``;
    4. a singleton ``manifest.variants`` list.

    The returned source string is included in validation errors so a malformed
    or mismatched result file remains easy to diagnose.
    """
    manifest = payload.get("manifest", {})

    explicit = manifest.get("variant")
    if explicit not in (None, "", "None"):
        return str(explicit), "result manifest"

    data_dir_value = payload.get("data_dir")
    if data_dir_value not in (None, ""):
        data_dir = Path(str(data_dir_value))
        channel_manifest_path = data_dir / "channels_manifest.json"
        if channel_manifest_path.is_file():
            try:
                channel_manifest = load_json(channel_manifest_path)
                channel_variant = channel_manifest.get("variant")
                if channel_variant not in (None, "", "None"):
                    return str(channel_variant), str(channel_manifest_path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                # Continue to path-based resolution.  The main aggregation data
                # are already loaded, so a stale or unavailable preprocessing
                # directory should not prevent aggregation of old completed runs.
                pass

        if expected_variant is not None and data_dir.name == str(expected_variant):
            return str(expected_variant), "data_dir basename"

    listed_variants = manifest.get("variants")
    if isinstance(listed_variants, list) and len(listed_variants) == 1:
        return str(listed_variants[0]), "singleton manifest.variants"

    return None, "unresolved"


def validate_payload(
    payload: Mapping[str, Any],
    expected_group: str,
    expected_variant: str | None,
) -> None:
    manifest = payload.get("manifest", {})
    flavor = str(manifest.get("flavor"))
    if flavor != expected_group:
        raise ValueError(f"Expected flavor {expected_group!r}, got {flavor!r}")

    if expected_group == "k":
        actual_variant, source = _resolve_k_variant(payload, expected_variant)
        if actual_variant is None:
            raise ValueError(
                f"Expected k variant {expected_variant!r}, but the result JSON does not "
                "contain manifest.variant and it could not be recovered from "
                "data_dir/channels_manifest.json or the data_dir basename. "
                "This normally means the --result mapping points to the wrong file."
            )
        if str(actual_variant) != str(expected_variant):
            raise ValueError(
                f"Expected k variant {expected_variant!r}, got {actual_variant!r} "
                f"(resolved from {source})"
            )


def aggregate_k(
    variants: Sequence[str],
    result_paths: Mapping[str, Path],
    expected_runs: int,
) -> Tuple[Dict[str, Any], List[int]]:
    payloads: Dict[str, Dict[str, Any]] = {}
    seed_maps: Dict[str, Dict[int, Mapping[str, Any]]] = {}
    for variant in variants:
        if variant not in result_paths:
            raise KeyError(f"Missing --result {variant}=PATH")
        payload = load_json(result_paths[variant])
        validate_payload(payload, "k", variant)
        payloads[variant] = payload
        seed_maps[variant] = runs_by_seed(payload, expected_runs)

    seeds = sorted(seed_maps[variants[0]])
    for variant in variants[1:]:
        if sorted(seed_maps[variant]) != seeds:
            raise ValueError(f"Aligned seeds differ for {variant}: {sorted(seed_maps[variant])} vs {seeds}")

    run_logits: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {v: {} for v in variants}
    node_run_metrics: Dict[str, List[Dict[str, float]]] = {v: [] for v in variants}
    resource_runs: Dict[str, List[Dict[str, float | None]]] = {v: [] for v in variants}
    consistency_audit: List[Dict[str, Any]] = []

    for variant in variants:
        for seed in seeds:
            run = seed_maps[variant][seed]
            arrays = load_run_logits(run, result_paths[variant])
            run_logits[variant][seed] = arrays
            recomputed = classification_metrics(arrays["test_logits"], arrays["test_labels"])
            node_run_metrics[variant].append(recomputed)
            resource_runs[variant].append(run_resources(run, result_paths[variant]))
            stored = run.get("splits", {}).get("test", {})
            max_abs = max(
                abs(float(stored.get(key, recomputed[key])) - recomputed[key])
                for key in TABLE_METRICS
            )
            consistency_audit.append({"variant": variant, "seed": seed, "max_abs_metric_diff": max_abs})

    # Validate split alignment once per seed before fusion and invariance.
    for seed in seeds:
        anchor = run_logits[variants[0]][seed]
        for variant in variants[1:]:
            other = run_logits[variant][seed]
            for key in ("test_idx", "test_labels"):
                if not np.array_equal(anchor[key], other[key]):
                    raise RuntimeError(f"{key} differs for seed={seed}: {variants[0]} vs {variant}")

    fusion_metric_runs: List[Dict[str, float]] = []
    fusion_resource_runs: List[Dict[str, float | None]] = []
    fusion_logits_by_seed: Dict[int, np.ndarray] = {}
    for seed_index, seed in enumerate(seeds):
        logits = np.mean(
            np.stack([run_logits[variant][seed]["test_logits"] for variant in variants], axis=0),
            axis=0,
        )
        fusion_logits_by_seed[seed] = logits
        labels = run_logits[variants[0]][seed]["test_labels"]
        fusion_metric_runs.append(classification_metrics(logits, labels))
        fusion_resource_runs.append(
            fusion_resources([resource_runs[variant][seed_index] for variant in variants])
        )

    # Compare Output Fusion directly against every independently trained graph
    # variant, in addition to the original variant-vs-variant comparisons.

    invariance_logits: Dict[str, Dict[int, np.ndarray]] = {
        variant: {
            seed: run_logits[variant][seed]["test_logits"]
            for seed in seeds
        }
        for variant in variants
    }
    invariance_logits["Output Fusion"] = fusion_logits_by_seed

    invariance_runs: Dict[str, List[Dict[str, float]]] = {}
    comparison_names = list(variants) + ["Output Fusion"]
    for left, right in itertools.combinations(comparison_names, 2):
        comparison = f"{display_name(left)} vs {display_name(right)}"
        invariance_runs[comparison] = [
            kendall_metrics(
                invariance_logits[left][seed],
                invariance_logits[right][seed],
            )
            for seed in seeds
        ]

    node_summary = {variant: aggregate_run_metrics(node_run_metrics[variant]) for variant in variants}
    node_summary["Output Fusion"] = aggregate_run_metrics(fusion_metric_runs)
    node_detailed = {
        variant: aggregate_detailed_metrics(node_run_metrics[variant]) for variant in variants
    }
    node_detailed["Output Fusion"] = aggregate_detailed_metrics(fusion_metric_runs)
    resource_summary = {variant: aggregate_resources(resource_runs[variant]) for variant in variants}
    resource_summary["Output Fusion"] = aggregate_resources(fusion_resource_runs)
    invariance_summary = {
        comparison: aggregate_run_metrics(values)
        for comparison, values in invariance_runs.items()
    }
    return {
        "node_prediction": node_summary,
        "node_prediction_detailed": node_detailed,
        "invariance": invariance_summary,
        "training_and_memory": resource_summary,
        "per_run": {
            "node_prediction": {**node_run_metrics, "Output Fusion": fusion_metric_runs},
            "invariance": invariance_runs,
            "training_and_memory": {**resource_runs, "Output Fusion": fusion_resource_runs},
        },
        "consistency_audit": consistency_audit,
        "resource_accounting": {
            "output_fusion_train_time_epochs_checkpoint_parameter": "sum across variant models per aligned run",
            "output_fusion_peak_training_gpu": "maximum of separately trained variant models per run",
            "output_fusion_peak_inference_gpu": "sum of per-model peaks, matching concurrent loading in the requested pseudocode",
        },
    }, seeds


def aggregate_universal(
    group: str,
    variants: Sequence[str],
    result_path: Path,
    expected_runs: int,
) -> Tuple[Dict[str, Any], List[int]]:
    payload = load_json(result_path)
    validate_payload(payload, group, None)
    seed_map = runs_by_seed(payload, expected_runs)
    seeds = sorted(seed_map)
    metric_runs: List[Dict[str, float]] = []
    resource_runs: List[Dict[str, float | None]] = []
    for seed in seeds:
        run = seed_map[seed]
        arrays = load_run_logits(run, result_path)
        metric_runs.append(classification_metrics(arrays["test_logits"], arrays["test_labels"]))
        resource_runs.append(run_resources(run, result_path))
    metric_summary = aggregate_run_metrics(metric_runs)
    metric_detailed = aggregate_detailed_metrics(metric_runs)
    resource_summary_one = aggregate_resources(resource_runs)
    names = list(variants) + ["Output Fusion"]
    invariance_one = {
        key: mean_std([1.0] * expected_runs)
        for key in ("kendall_tau", "kendall_tau_at_1", "kendall_tau_at_3")
    }
    comparison_names = list(variants) + ["Output Fusion"]
    invariance = {
        f"{display_name(left)} vs {display_name(right)}": invariance_one
        for left, right in itertools.combinations(comparison_names, 2)
    }
    return {
        "node_prediction": {name: metric_summary for name in names},
        "node_prediction_detailed": {name: metric_detailed for name in names},
        "invariance": invariance,
        "training_and_memory": {name: resource_summary_one for name in names},
        "per_run": {
            "node_prediction": {name: metric_runs for name in names},
            "invariance": {
                comparison: [
                    {"kendall_tau": 1.0, "kendall_tau_at_1": 1.0, "kendall_tau_at_3": 1.0}
                    for _ in seeds
                ]
                for comparison in invariance
            },
            "training_and_memory": {name: resource_runs for name in names},
        },
        "resource_accounting": {
            "duplicated_rows": "all variants and Output Fusion reference the same universal-graph model in each run"
        },
    }, seeds



def missing_resource_fields(aggregated: Mapping[str, Any]) -> Dict[str, List[str]]:
    missing: Dict[str, List[str]] = {}
    for name, stats in aggregated.get("training_and_memory", {}).items():
        absent = [
            key
            for key in RESOURCE_KEYS
            if stats.get(key, {}).get("mean") is None
        ]
        if absent:
            missing[name] = absent
    return missing

def render_report(group: str, variants: Sequence[str], aggregated: Mapping[str, Any]) -> str:
    lines = ["Freebase NC", MODEL_LABELS[group], "", "node prediction results"]
    for name in list(variants) + ["Output Fusion"]:
        lines.append(node_row(name, aggregated["node_prediction"][name]))
    lines.extend(["", "invariance results"])
    for comparison, stats in aggregated["invariance"].items():
        lines.append(invariance_row(comparison, stats))
    lines.extend(["", "training and memory"])
    for name in list(variants) + ["Output Fusion"]:
        lines.append(resource_row(name, aggregated["training_and_memory"][name]))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate four-run Freebase NC SeHGNN metrics")
    parser.add_argument("--group", required=True, help="k, fullK/full_k, or restrictedK/restricted_k")
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="NAME=sehgnn_results.json. K needs one per variant; universal groups use universal=PATH.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalized = args.group.strip().lower().replace("-", "_")
    if normalized not in GROUP_ALIASES:
        raise ValueError(f"Unknown group {args.group!r}; choose k, fullK, or restrictedK")
    group = GROUP_ALIASES[normalized]
    result_paths = parse_result_specs(args.result)
    variants = list(dict.fromkeys(args.variants))
    if len(variants) < 1:
        raise ValueError("At least one variant is required")

    if group == "k":
        aggregated, seeds = aggregate_k(variants, result_paths, args.expected_runs)
    else:
        if "universal" in result_paths:
            result_path = result_paths["universal"]
        elif len(result_paths) == 1:
            result_path = next(iter(result_paths.values()))
        else:
            raise ValueError("Universal groups need --result universal=PATH")
        aggregated, seeds = aggregate_universal(
            group, variants, result_path, args.expected_runs
        )

    payload = {
        "format_version": 1,
        "dataset": "Freebase NC",
        "model_label": MODEL_LABELS[group],
        "group": group,
        "variants": variants,
        "seeds": seeds,
        "num_runs": len(seeds),
        "std_ddof": 1,
        "metric_definitions": {
            "precision": "macro-averaged multiclass precision in the LaTeX row; macro and micro versions are both stored",
            "recall": "macro-averaged multiclass recall in the LaTeX row; macro and micro versions are both stored",
            "classification_metrics": "accuracy, balanced accuracy, macro/micro precision and recall, micro/macro/weighted F1, Hit@1, Hit@3, MRR, loss, per-class metrics, and confusion matrix",
            "kendall_tau": "test-node macro average of Kendall tau-b over all eight class logits",
            "kendall_tau_at_1": "test-node top-1 class agreement rate",
            "kendall_tau_at_3": "test-node macro Kendall tau-b over the union of each model's top-3 classes",
            "output_fusion": "arithmetic mean of raw best-checkpoint logits before softmax/argmax",
            "output_fusion_invariance": "Kendall metrics are reported for every variant pair and for each variant against Output Fusion; universal groups are exactly 1 because all rows reference the same logits",
        },
        **aggregated,
    }
    report = render_report(group, variants, aggregated)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "aggregated_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (args.output_dir / "latex_rows.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    missing = missing_resource_fields(aggregated)
    if missing:
        print("\nWARNING: some resource fields are missing from one or more runs:")
        for name, fields in missing.items():
            print(f"  {name}: {', '.join(fields)}")
        print(
            "Run backfill_freebase_nc_logits.py on the result JSON files with a CUDA GPU "
            "to recover checkpoint/parameter memory and replay peak training/inference memory."
        )
    print(f"\nSaved: {args.output_dir / 'aggregated_metrics.json'}")
    print(f"Saved: {args.output_dir / 'latex_rows.txt'}")


if __name__ == "__main__":
    main()