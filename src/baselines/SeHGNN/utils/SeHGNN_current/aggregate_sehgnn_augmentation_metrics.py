#!/usr/bin/env python3
"""Aggregate shared-checkpoint SeHGNN graph-augmentation experiments.

The runner writes one summary per seed.  This script aggregates complete node
classification metrics, pairwise invariance, epoch accounting, and resource
usage without duplicating parameter/checkpoint memory across graph variants:
there is one shared model and one shared checkpoint in each seed.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from sehgnn_augmentation_common import atomic_write_csv, atomic_write_json, mean_std, scalar_metrics

TABLE_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "micro_f1",
    "macro_f1",
    "weighted_f1",
    "hit_at_1",
    "hit_at_3",
    "mrr",
    "loss",
)
DETAIL_METRICS = (
    "per_class_precision",
    "per_class_recall",
    "per_class_f1",
    "per_class_support",
    "confusion_matrix",
)
INVARIANCE_METRICS = (
    "kendall_tau",
    "kendall_tau_at_1",
    "kendall_tau_at_3",
    "kendall_tau_flat_logits",
    "kendall_tau_confidence",
    "prediction_agreement",
    "mean_abs_logit_diff",
    "max_abs_logit_diff",
    "mean_l2_logit_diff",
    "mean_abs_probability_diff",
    "max_abs_probability_diff",
)


def parse_csv(spec: str | None) -> List[str] | None:
    if spec is None:
        return None
    values = [value.strip() for value in spec.split(",") if value.strip()]
    return list(dict.fromkeys(values))


def load_runs(input_dir: Path) -> List[Dict[str, Any]]:
    combined = input_dir / "all_seed_summaries.json"
    if combined.is_file():
        payload = json.loads(combined.read_text(encoding="utf-8"))
        runs = payload.get("runs", [])
        if not isinstance(runs, list) or not runs:
            raise ValueError(f"No runs found in {combined}")
        return runs
    paths = sorted(input_dir.glob("seed_*/summary.json"))
    if not paths:
        raise FileNotFoundError(
            f"No all_seed_summaries.json or seed_*/summary.json files under {input_dir}"
        )
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def validate_and_filter(
    runs: Sequence[Mapping[str, Any]],
    requested_seeds: Sequence[str] | None,
    requested_variants: Sequence[str] | None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    selected: List[Dict[str, Any]] = []
    seed_set = set(requested_seeds or [])
    for raw in runs:
        run = dict(raw)
        if requested_seeds is not None and str(run["seed"]) not in seed_set:
            continue
        selected.append(run)
    if not selected:
        raise ValueError("No runs remain after seed filtering")

    dataset = selected[0]["dataset"]
    model = selected[0]["model"]
    signatures = set()
    architecture_json = set()
    for run in selected:
        if run["dataset"] != dataset or run["model"] != model:
            raise ValueError("Mixed datasets/models cannot be aggregated together")
        variants = list(run["variants"])
        if requested_variants is not None:
            missing = set(requested_variants) - set(variants)
            if missing:
                raise ValueError(f"Run seed={run['seed']} is missing variants {sorted(missing)}")
        compatibility = run.get("parameter_compatibility", {})
        if compatibility.get("same_checkpoint_for_all_variants") is not True:
            raise ValueError(f"Run seed={run['seed']} is not marked as shared-checkpoint training")
        signatures.add(str(compatibility.get("parameter_shape_signature")))
        architecture_json.add(json.dumps(run.get("architecture", {}), sort_keys=True))
    if len(signatures) != 1:
        raise ValueError("Parameter shape signatures differ across seeds")
    if len(architecture_json) != 1:
        raise ValueError("Canonical architectures differ across seeds")

    variants = list(requested_variants) if requested_variants is not None else list(selected[0]["variants"])
    for run in selected:
        if requested_variants is None and list(run["variants"]) != variants:
            raise ValueError("Variant order/set differs across runs; pass --variants explicitly if intentional")
    seeds = [int(run["seed"]) for run in selected]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate seed summaries found: {seeds}")
    selected.sort(key=lambda run: seeds.index(int(run["seed"])))
    return selected, variants


def aggregate_arrays(values: Sequence[Any]) -> Dict[str, Any]:
    arrays = [np.asarray(value, dtype=np.float64) for value in values]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"Detailed metric shapes differ: {sorted(shapes)}")
    stacked = np.stack(arrays, axis=0)
    return {
        "mean": np.mean(stacked, axis=0).tolist(),
        "std": (np.std(stacked, axis=0, ddof=1) if len(arrays) > 1 else np.zeros_like(stacked[0])).tolist(),
        "values": [array.tolist() for array in arrays],
        "ddof": 1,
    }


def nested_get(mapping: Mapping[str, Any], path: str) -> Any:
    current: Any = mapping
    for key in path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def fmt_pm(stat: Mapping[str, Any], digits: int = 4) -> str:
    mean = stat.get("mean")
    std = stat.get("std")
    if mean is None or std is None:
        return "--"
    return f"{float(mean):.{digits}f} $\\pm$ {float(std):.{digits}f}"


def display_name(name: str) -> str:
    if name == "Mean over variants":
        return name
    if name.startswith("v") and name[1:].isdigit():
        return f"Variant {name[1:]}"
    return name.replace("_", " ").title()


def aggregate(runs: Sequence[Mapping[str, Any]], variants: Sequence[str]) -> Dict[str, Any]:
    split_names = ("train", "val", "test")
    node_prediction: Dict[str, Dict[str, Dict[str, Any]]] = {}
    node_detailed: Dict[str, Dict[str, Dict[str, Any]]] = {}
    node_rows: List[Dict[str, Any]] = []

    for variant in variants:
        node_prediction[variant] = {}
        node_detailed[variant] = {}
        for split in split_names:
            metric_runs = [run["per_variant_split_metrics"][variant][split] for run in runs]
            scalar_names = sorted(set.intersection(*(set(scalar_metrics(item)) for item in metric_runs)))
            node_prediction[variant][split] = {
                metric: mean_std([item[metric] for item in metric_runs]) for metric in scalar_names
            }
            node_detailed[variant][split] = {
                metric: aggregate_arrays([item[metric] for item in metric_runs])
                for metric in DETAIL_METRICS
                if all(metric in item for item in metric_runs)
            }
            for metric in scalar_names:
                stat = node_prediction[variant][split][metric]
                node_rows.append(
                    {
                        "dataset": runs[0]["dataset"],
                        "variant": variant,
                        "split": split,
                        "metric": metric,
                        "mean": stat["mean"],
                        "std": stat["std"],
                        "ddof": stat["ddof"],
                        "values": json.dumps(stat["values"]),
                    }
                )

    mean_test_runs: List[Dict[str, float]] = []
    for run in runs:
        chosen = [run["per_variant_split_metrics"][variant]["test"] for variant in variants]
        common = set.intersection(*(set(scalar_metrics(item)) for item in chosen))
        mean_test_runs.append(
            {metric: float(np.mean([float(item[metric]) for item in chosen])) for metric in common}
        )
    mean_over_variants = {
        metric: mean_std([item[metric] for item in mean_test_runs])
        for metric in sorted(set.intersection(*(set(item) for item in mean_test_runs)))
    }

    pair_names = [
        (str(row["variant_a"]), str(row["variant_b"]))
        for row in runs[0].get("pairwise_invariance", [])
        if row["variant_a"] in variants and row["variant_b"] in variants
    ]
    invariance: Dict[str, Dict[str, Any]] = {}
    invariance_rows: List[Dict[str, Any]] = []
    for left, right in pair_names:
        records = []
        for run in runs:
            matches = [
                row for row in run.get("pairwise_invariance", [])
                if str(row["variant_a"]) == left and str(row["variant_b"]) == right
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one invariance row for {left} vs {right}, seed={run['seed']}")
            records.append(matches[0])
        pair = f"{left} vs {right}"
        invariance[pair] = {}
        for metric in INVARIANCE_METRICS:
            if all(metric in record for record in records):
                stat = mean_std([record[metric] for record in records])
                invariance[pair][metric] = stat
                invariance_rows.append(
                    {
                        "dataset": runs[0]["dataset"],
                        "variant_a": left,
                        "variant_b": right,
                        "metric": metric,
                        "mean": stat["mean"],
                        "std": stat["std"],
                        "ddof": stat["ddof"],
                        "values": json.dumps(stat["values"]),
                    }
                )

    resource_paths = (
        "training_seconds",
        "inference_seconds",
        "best_super_epoch",
        "best_mean_val_loss",
        "epoch_accounting.super_epochs_ran",
        "epoch_accounting.variant_epochs_ran",
        "epoch_accounting.optimizer_steps",
        "epoch_accounting.updates_per_super_epoch",
        "memory.parameter_count",
        "memory.trainable_parameter_count",
        "memory.parameter_mib",
        "memory.buffer_mib",
        "memory.static_model_mib",
        "memory.checkpoint_mib",
        "memory.process_peak_rss_mib",
        "memory.training_gpu.peak_allocated_mib",
        "memory.training_gpu.peak_reserved_mib",
        "memory.inference_gpu.peak_allocated_mib",
        "memory.inference_gpu.peak_reserved_mib",
    )
    resources: Dict[str, Any] = {}
    resource_rows: List[Dict[str, Any]] = []
    for path in resource_paths:
        values = [nested_get(run, path) for run in runs]
        stat = mean_std(values)
        resources[path] = stat
        resource_rows.append(
            {
                "dataset": runs[0]["dataset"],
                "scope": "one shared model/checkpoint per seed",
                "metric": path,
                "mean": stat["mean"],
                "std": stat["std"],
                "ddof": stat["ddof"],
                "values": json.dumps(stat["values"]),
            }
        )

    active_channels: Dict[str, Any] = {}
    graph_stats: Dict[str, Any] = {}
    for variant in variants:
        active_values = []
        graph_records = []
        for run in runs:
            compatibility = run["parameter_compatibility"]
            per_variant = compatibility.get("per_variant_active_channels", {})
            active_values.append(per_variant.get(variant))
            graph_records.append(run["per_variant_split_metrics"][variant].get("graph", {}))
        active_channels[variant] = mean_std(active_values)
        scalar_graph_keys = sorted(set.intersection(*(set(scalar_metrics(item)) for item in graph_records)))
        graph_stats[variant] = {
            key: mean_std([record[key] for record in graph_records]) for key in scalar_graph_keys
        }

    return {
        "node_prediction": node_prediction,
        "node_prediction_detailed": node_detailed,
        "mean_over_variants_test": mean_over_variants,
        "invariance": invariance,
        "training_and_memory": resources,
        "active_channels": active_channels,
        "graph_statistics": graph_stats,
        "csv_rows": {
            "node": node_rows,
            "invariance": invariance_rows,
            "resources": resource_rows,
        },
    }


def render_report(dataset: str, variants: Sequence[str], aggregated: Mapping[str, Any]) -> str:
    lines = [
        dataset,
        "SeHGNN shared-checkpoint graph-variant data augmentation",
        "",
        "Test node prediction (mean $\\pm$ sample std across seeds)",
        "Variant & Accuracy & Balanced Acc. & Macro Precision & Macro Recall & Micro F1 & Macro F1 & Weighted F1 & Hit@1 & Hit@3 & MRR & Loss \\\\",
    ]
    ordered_metrics = TABLE_METRICS
    for variant in variants:
        stats = aggregated["node_prediction"][variant]["test"]
        values = " & ".join(fmt_pm(stats[metric]) for metric in ordered_metrics)
        lines.append(f"{display_name(variant)} & {values} \\\\")
    mean_stats = aggregated["mean_over_variants_test"]
    values = " & ".join(fmt_pm(mean_stats[metric]) for metric in ordered_metrics)
    lines.append(f"Mean over variants & {values} \\\\")

    lines.extend([
        "",
        "Pairwise invariance",
        'Comparison & Kendall $\\tau$ & $\\tau$@1 & $\\tau$@3 & Flat-logit $\\tau$ & Confidence $\\tau$ & Prediction agreement & Mean $|\\Delta$ logit$|$ & Max $|\\Delta$ logit$|$ & Mean L2 logit & Mean $|\\Delta p|$ & Max $|\\Delta p|$ \\\\',
    ])
    for pair, stats in aggregated["invariance"].items():
        values = " & ".join(fmt_pm(stats[metric]) for metric in INVARIANCE_METRICS)
        lines.append(f"{pair.replace('_', ' ')} & {values} \\\\")

    resources = aggregated["training_and_memory"]
    lines.extend([
        "",
        "Shared training and memory (reported once per seed, not once per variant)",
        "Train sec. & Super-epochs & Variant-epochs & Optimizer steps & Checkpoint MiB & Parameter MiB & Peak train GPU MiB & Peak inference GPU MiB \\\\",
    ])
    resource_order = (
        "training_seconds",
        "epoch_accounting.super_epochs_ran",
        "epoch_accounting.variant_epochs_ran",
        "epoch_accounting.optimizer_steps",
        "memory.checkpoint_mib",
        "memory.parameter_mib",
        "memory.training_gpu.peak_allocated_mib",
        "memory.inference_gpu.peak_allocated_mib",
    )
    lines.append(" & ".join(fmt_pm(resources[key], digits=2) for key in resource_order) + " \\\\")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate SeHGNN shared-checkpoint augmentation metrics")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seeds", default=None, help="Optional comma-separated seed filter")
    parser.add_argument("--variants", default=None, help="Optional comma-separated ordered variant subset")
    parser.add_argument("--expected-runs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or (input_dir / "aggregate")).resolve()
    runs = load_runs(input_dir)
    runs, variants = validate_and_filter(runs, parse_csv(args.seeds), parse_csv(args.variants))
    if args.expected_runs is not None and len(runs) != args.expected_runs:
        raise ValueError(f"Expected {args.expected_runs} runs, found {len(runs)}")

    aggregated = aggregate(runs, variants)
    payload = {
        "format_version": 1,
        "dataset": runs[0]["dataset"],
        "model": runs[0]["model"],
        "flavor": runs[0].get("flavor", "original"),
        "variants": list(variants),
        "seeds": [int(run["seed"]) for run in runs],
        "num_runs": len(runs),
        "std_ddof": 1,
        "selection_metric": runs[0]["selection_metric"],
        "parameter_compatibility": {
            "strategy": runs[0]["parameter_compatibility"]["strategy"],
            "parameter_shape_signature": runs[0]["parameter_compatibility"]["parameter_shape_signature"],
            "same_checkpoint_for_all_variants": True,
            "resource_accounting": "parameter size, checkpoint size, and shared training memory are counted once per seed",
        },
        "metric_definitions": {
            "classification": "accuracy, balanced accuracy, macro/micro precision and recall, micro/macro/weighted F1, Hit@1, Hit@3, MRR, cross-entropy, per-class metrics, and confusion matrix",
            "selection": "minimum mean validation cross-entropy across all variants, evaluated after each complete shuffled super-epoch",
            "kendall_tau": "test-node macro Kendall tau-b over all class logits",
            "kendall_tau_at_1": "test-node top-1 agreement",
            "kendall_tau_at_3": "test-node macro Kendall tau-b over the union of top-3 classes",
            "kendall_tau_flat_logits": "RGCN-compatible Kendall tau-b over flattened aligned test logits",
            "kendall_tau_confidence": "RGCN-compatible Kendall tau-b over maximum class probabilities",
        },
        **{key: value for key, value in aggregated.items() if key != "csv_rows"},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "aggregated_metrics.json", payload)
    atomic_write_csv(output_dir / "node_metrics.csv", aggregated["csv_rows"]["node"])
    atomic_write_csv(output_dir / "invariance_metrics.csv", aggregated["csv_rows"]["invariance"])
    atomic_write_csv(output_dir / "training_memory_metrics.csv", aggregated["csv_rows"]["resources"])
    report = render_report(runs[0]["dataset"], variants, aggregated)
    (output_dir / "latex_rows.txt").write_text(report + "\n", encoding="utf-8")

    print(report)
    print(f"\n[OK] Aggregated {len(runs)} runs: {output_dir}")


if __name__ == "__main__":
    main()
