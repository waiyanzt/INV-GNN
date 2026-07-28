#!/usr/bin/env python3
"""Aggregate SlotGAT WordNet LP runs, invariance, and output fusion.

Expected inputs are the self-describing checkpoints and per-run JSON files
written by ``run_wordnet_lp.py``.  For every selected seed, this script:

1. loads each graph-variant checkpoint;
2. recomputes validation/test metrics as a compatibility check;
3. computes macro per-query Kendall tau on shared candidate rankings;
4. computes Kendall tau on per-query Hit@1 and Hit@3 indicators;
5. compares ``Output Fusion`` with every component variant using the same
   aligned candidate logits and Hit@1/Hit@3 indicators;
6. evaluates ``Output Fusion`` by averaging raw DistMult logits from all
   selected variants before thresholding or any probability conversion; and
7. aggregates every numeric metric over seeds as mean +/- population std.

It prints publication-ready LaTeX rows in the format requested for:

* link prediction results;
* invariance results; and
* training and memory.

Output-fusion resource accounting
---------------------------------
Training time, epochs, checkpoint size, and model-parameter memory are summed
across component variants.  Because the component models are trained and
encoded sequentially, peak training memory is the maximum component peak.
Peak inference memory is the maximum of (a) the component inference peaks and
(b) the measured fused-score evaluation peak.  Both summed component peaks and
the primary sequential-policy values are retained in the output JSON.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import kendalltau

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_wordnet_lp as runner  # noqa: E402

AGGREGATE_FORMAT = "slotgat_wordnet_lp_aggregate_v5_checked"
OUTPUT_FUSION = "output_fusion"

DISPLAY_NAMES = {
    "no_changes": "Unchanged",
    "all_inverse_edges": "All Inverse Edges",
    "transitive_edges": "Transitive Edges",
    "universal_edges": "Universal Edges",
    OUTPUT_FUSION: "Output Fusion",
}

LINK_TABLE_METRICS = (
    "precision",
    "recall",
    "f1",
    "Hits@1",
    "Hits@3",
    "filtered_MRR",
)

RESOURCE_TABLE_METRICS = (
    "training_time_sec",
    "epochs_trained",
    "checkpoint_mib",
    "parameter_mib",
    "peak_training_gpu_allocated_mib",
    "peak_inference_gpu_allocated_mib",
)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8"
    )


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Run JSON not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return value


def checkpoint_path(checkpoint_dir: Path, variant: str, seed: int) -> Path:
    return checkpoint_dir / f"slotgat_wordnet_lp_{variant}_seed{seed}.pt"


def result_path(results_dir: Path, variant: str, seed: int) -> Path:
    return results_dir / variant / f"seed_{seed}.json"


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.number)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def summarize_numeric_dicts(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate every top-level numeric field over records."""
    keys: set[str] = set()
    for record in records:
        keys.update(str(key) for key in record)

    summary: dict[str, Any] = {}
    for key in sorted(keys):
        values = [finite_float(record.get(key)) for record in records]
        finite = np.asarray([value for value in values if value is not None], dtype=float)
        if finite.size == 0:
            continue
        summary[key] = {
            "mean": float(finite.mean()),
            "std": float(finite.std(ddof=0)),
            "n": int(finite.size),
            "values": finite.tolist(),
        }
    return summary


def stat_value(summary: Mapping[str, Any], key: str) -> tuple[float, float] | None:
    item = summary.get(key)
    if not isinstance(item, Mapping):
        return None
    mean = finite_float(item.get("mean"))
    std = finite_float(item.get("std"))
    if mean is None or std is None:
        return None
    return mean, std


def latex_stat(summary: Mapping[str, Any], key: str, decimals: int = 4) -> str:
    pair = stat_value(summary, key)
    if pair is None:
        return "--"
    mean, std = pair
    return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"


def display_name(name: str) -> str:
    if name in DISPLAY_NAMES:
        return DISPLAY_NAMES[name]
    return name.replace("_", " ").title()


def move_representation(
    representation: runner.EncodedRepresentation,
    device: torch.device,
) -> runner.EncodedRepresentation:
    return runner.EncodedRepresentation(
        entity_embeddings=representation.entity_embeddings.to(device),
        relation_embeddings=representation.relation_embeddings.to(device),
    )


def cpu_copy_representation(
    representation: runner.EncodedRepresentation,
) -> runner.EncodedRepresentation:
    return runner.EncodedRepresentation(
        entity_embeddings=representation.entity_embeddings.detach().cpu().clone(),
        relation_embeddings=representation.relation_embeddings.detach().cpu().clone(),
    )


def metric_max_abs_difference(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> float:
    differences: list[float] = []
    for key in set(first) & set(second):
        left = finite_float(first[key])
        right = finite_float(second[key])
        if left is not None and right is not None:
            differences.append(abs(left - right))
    return max(differences, default=0.0)


def metric_abs_differences(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, float]:
    output: dict[str, float] = {}
    for key in sorted(set(first) & set(second)):
        left = finite_float(first[key])
        right = finite_float(second[key])
        if left is not None and right is not None:
            output[str(key)] = abs(left - right)
    return output


def largest_metric_differences(
    differences: Mapping[str, float], limit: int = 5
) -> list[tuple[str, float]]:
    return sorted(differences.items(), key=lambda item: item[1], reverse=True)[:limit]


def validate_score_payloads(
    first: Mapping[str, np.ndarray], second: Mapping[str, np.ndarray]
) -> None:
    for key in ("query_ids", "candidate_ids", "labels"):
        if not np.array_equal(np.asarray(first[key]), np.asarray(second[key])):
            raise AssertionError(
                f"Kendall candidate alignment failed: payload field {key!r} differs. "
                "All variants must use the same test queries and candidate seed."
            )


def robust_kendall(first: np.ndarray, second: np.ndarray) -> float:
    """Kendall tau-b for two aligned one-dimensional arrays."""
    a = np.asarray(first).reshape(-1)
    b = np.asarray(second).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"Kendall arrays differ in shape: {a.shape} vs {b.shape}")
    if a.size < 2:
        return float("nan")
    # scipy returns NaN for two identical constant vectors, although identical
    # predictions represent perfect agreement.
    if np.array_equal(a, b):
        return 1.0
    tau = kendalltau(a, b, nan_policy="omit").statistic
    return float(tau) if tau is not None and np.isfinite(tau) else float("nan")


def macro_query_kendall(
    first_scores: np.ndarray, second_scores: np.ndarray
) -> tuple[float, int, int]:
    """Average Kendall tau-b across test queries."""
    first = np.asarray(first_scores)
    second = np.asarray(second_scores)
    if first.shape != second.shape:
        raise ValueError(
            f"Kendall score matrices differ in shape: {first.shape} vs {second.shape}"
        )
    if first.ndim != 2:
        raise ValueError(
            f"Kendall score payload must be 2D (queries, candidates); found {first.shape}"
        )

    values: list[float] = []
    for row_first, row_second in zip(first, second):
        value = robust_kendall(row_first, row_second)
        if math.isfinite(value):
            values.append(value)

    valid = len(values)
    total = int(first.shape[0])
    return (
        float(np.mean(values)) if values else float("nan"),
        valid,
        total,
    )


def hit_indicators(payload: Mapping[str, np.ndarray], k: int) -> np.ndarray:
    scores = np.asarray(payload["scores"])
    labels = np.asarray(payload["labels"])
    if scores.shape != labels.shape:
        raise ValueError("Score and label matrices must have equal shape")
    true_columns = labels.argmax(axis=1)
    true_scores = scores[np.arange(len(scores)), true_columns]
    # Match the ranking convention in run_wordnet_lp.py: ties count against the
    # true item because rank is the number of scores >= the true score.
    ranks = (scores >= true_scores[:, None]).sum(axis=1)
    return (ranks <= k).astype(np.int64)


def invariance_metrics(
    first: Mapping[str, np.ndarray], second: Mapping[str, np.ndarray]
) -> dict[str, float]:
    validate_score_payloads(first, second)
    macro_tau, valid_queries, total_queries = macro_query_kendall(
        first["scores"], second["scores"]
    )
    return {
        # Primary ranking-agreement metric: per-query tau-b, macro averaged.
        "kendall_tau": macro_tau,
        "kendall_tau_valid_queries": float(valid_queries),
        "kendall_tau_total_queries": float(total_queries),
        # Diagnostic only: this also reflects cross-query score calibration.
        "kendall_tau_global_flattened": robust_kendall(
            first["scores"], second["scores"]
        ),
        "kendall_tau_at_1": robust_kendall(
            hit_indicators(first, 1), hit_indicators(second, 1)
        ),
        "kendall_tau_at_3": robust_kendall(
            hit_indicators(first, 3), hit_indicators(second, 3)
        ),
    }


def numeric_resource_fields(resources: Mapping[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in resources.items():
        number = finite_float(value)
        if number is not None:
            output[str(key)] = number
    return output


def sum_resource(
    resources: Sequence[Mapping[str, Any]], key: str
) -> float | None:
    values = [finite_float(resource.get(key)) for resource in resources]
    finite = [value for value in values if value is not None]
    return float(sum(finite)) if len(finite) == len(resources) else None


def max_resource(
    resources: Sequence[Mapping[str, Any]], key: str
) -> float | None:
    values = [finite_float(resource.get(key)) for resource in resources]
    finite = [value for value in values if value is not None]
    return float(max(finite)) if finite else None


def build_fusion_resources(
    component_resources: Sequence[Mapping[str, Any]],
    measured_fusion_inference_allocated_mib: float | None,
    measured_fusion_inference_reserved_mib: float | None,
) -> dict[str, Any]:
    component_peak_train_allocated = max_resource(
        component_resources, "peak_training_gpu_allocated_mib"
    )
    component_peak_train_reserved = max_resource(
        component_resources, "peak_training_gpu_reserved_mib"
    )
    component_peak_infer_allocated = max_resource(
        component_resources, "peak_inference_gpu_allocated_mib"
    )
    component_peak_infer_reserved = max_resource(
        component_resources, "peak_inference_gpu_reserved_mib"
    )

    inference_allocated_values = [
        value
        for value in (
            component_peak_infer_allocated,
            measured_fusion_inference_allocated_mib,
        )
        if value is not None
    ]
    inference_reserved_values = [
        value
        for value in (
            component_peak_infer_reserved,
            measured_fusion_inference_reserved_mib,
        )
        if value is not None
    ]

    return {
        "training_time_sec": sum_resource(component_resources, "training_time_sec"),
        "epochs_trained": sum_resource(component_resources, "epochs_trained"),
        "checkpoint_mib": sum_resource(component_resources, "checkpoint_mib"),
        "parameter_mib": sum_resource(component_resources, "parameter_mib"),
        "buffer_mib": sum_resource(component_resources, "buffer_mib"),
        "static_model_mib": sum_resource(component_resources, "static_model_mib"),
        "peak_training_gpu_allocated_mib": component_peak_train_allocated,
        "peak_training_gpu_reserved_mib": component_peak_train_reserved,
        "peak_inference_gpu_allocated_mib": (
            max(inference_allocated_values) if inference_allocated_values else None
        ),
        "peak_inference_gpu_reserved_mib": (
            max(inference_reserved_values) if inference_reserved_values else None
        ),
        "measured_fused_scoring_peak_inference_allocated_mib": (
            measured_fusion_inference_allocated_mib
        ),
        "measured_fused_scoring_peak_inference_reserved_mib": (
            measured_fusion_inference_reserved_mib
        ),
        "sum_component_peak_training_gpu_allocated_mib": sum_resource(
            component_resources, "peak_training_gpu_allocated_mib"
        ),
        "sum_component_peak_inference_gpu_allocated_mib": sum_resource(
            component_resources, "peak_inference_gpu_allocated_mib"
        ),
        "resource_policy": (
            "sum time/epochs/checkpoint/parameter memory; max peak memory under "
            "sequential checkpoint execution"
        ),
    }


def print_latex(
    variants: Sequence[str],
    aggregate: Mapping[str, Any],
    invariance: Mapping[str, Any],
) -> str:
    lines: list[str] = ["WordNet LP and SlotGAT", "link prediction results"]
    for variant in [*variants, OUTPUT_FUSION]:
        test_summary = aggregate[variant]["test"]
        values = " & ".join(latex_stat(test_summary, key) for key in LINK_TABLE_METRICS)
        lines.append(f"& {display_name(variant)} & {values} \\\\")

    lines.extend(["", "invariance results"])
    for first, second in itertools.combinations(variants, 2):
        comparison = f"{first}__vs__{second}"
        summary = invariance[comparison]
        values = " & ".join(
            latex_stat(summary, key)
            for key in ("kendall_tau", "kendall_tau_at_1", "kendall_tau_at_3")
        )
        lines.append(
            f"& {display_name(first)} vs {display_name(second)} & {values} \\\\"
        )

    lines.extend(["", "output fusion invariance results"])
    for variant in variants:
        comparison = f"{OUTPUT_FUSION}__vs__{variant}"
        summary = invariance[comparison]
        values = " & ".join(
            latex_stat(summary, key)
            for key in ("kendall_tau", "kendall_tau_at_1", "kendall_tau_at_3")
        )
        lines.append(
            f"& {display_name(OUTPUT_FUSION)} vs {display_name(variant)} "
            f"& {values} \\\\"
        )

    lines.extend(["", "training and memory"])
    for variant in [*variants, OUTPUT_FUSION]:
        resource_summary = aggregate[variant]["resources"]
        values = " & ".join(
            latex_stat(resource_summary, key) for key in RESOURCE_TABLE_METRICS
        )
        lines.append(f"& {display_name(variant)} & {values} \\\\")

    text = "\n".join(lines)
    print(text)
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate four-run WordNet SlotGAT metrics and output fusion."
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(runner.CANONICAL_VARIANTS),
        help="Variants to aggregate/fuse (aliases accepted).",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(runner.DEFAULT_SEEDS)
    )
    parser.add_argument("--data-root", type=Path, default=SCRIPT_DIR / "data")
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=SCRIPT_DIR / "checkpoint"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=SCRIPT_DIR / "results" / "wordnet_lp"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=SCRIPT_DIR / "results" / "wordnet_lp" / "aggregate_metrics.json",
    )
    parser.add_argument(
        "--latex-output",
        type=Path,
        default=SCRIPT_DIR / "results" / "wordnet_lp" / "latex_rows.txt",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--backend-override", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--binary-negatives", type=int, default=50)
    parser.add_argument("--binary-candidate-seed", type=int, default=42)
    parser.add_argument(
        "--metric-tolerance",
        type=float,
        default=1e-5,
        help="Tolerance for checkpoint-vs-JSON metric reproduction.",
    )
    parser.add_argument(
        "--strict-metric-check",
        action="store_true",
        help="Fail instead of warning when recomputed metrics exceed the tolerance.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = runner.canonicalize_variants(args.variants)
    if len(variants) < 2:
        raise ValueError("Output fusion and invariance require at least two variants")
    if len(args.seeds) != 4:
        print(
            f"Warning: requested protocol uses four runs, but {len(args.seeds)} seeds "
            f"were supplied: {args.seeds}"
        )

    data_root = args.data_root.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    latex_output = args.latex_output.expanduser().resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    datasets = {
        variant: runner.load_variant_data(data_root, variant) for variant in variants
    }
    runner.validate_shared_eval_splits(list(datasets.values()))
    reference = datasets[variants[0]]
    # Base-relation train triples are shared.  Derived relation IDs do not alter
    # filters for the held-out base-relation queries.
    tail_filters, head_filters = runner.build_filter_dicts(
        reference.train_pos, reference.val_pos, reference.test_pos
    )

    raw_by_variant: dict[str, list[dict[str, Any]]] = {
        variant: [] for variant in [*variants, OUTPUT_FUSION]
    }
    invariance_by_comparison: defaultdict[str, list[dict[str, float]]] = defaultdict(list)
    compatibility_checks: list[dict[str, Any]] = []

    print("WordNet LP and SlotGAT")
    print(f"Aggregator: {Path(__file__).resolve()}")
    print(f"Aggregate format: {AGGREGATE_FORMAT}")
    print(f"Runner module: {Path(runner.__file__).resolve()}")
    print(f"Variants: {variants}")
    print(f"Seeds: {args.seeds}")
    print(f"Device: {device}")

    for seed in args.seeds:
        print(f"\n=== Aggregating seed {seed} ===")
        cpu_representations: dict[str, runner.EncodedRepresentation] = {}
        score_payloads: dict[str, dict[str, np.ndarray]] = {}
        component_resources: list[Mapping[str, Any]] = []

        for variant in variants:
            run_json_path = result_path(results_dir, variant, seed)
            checkpoint = checkpoint_path(checkpoint_dir, variant, seed)
            run_result = load_json(run_json_path)
            training_config = run_result.get("training_config", {})
            if isinstance(training_config, Mapping):
                stored_binary_negatives = training_config.get("binary_negatives")
                stored_candidate_seed = training_config.get("binary_candidate_seed")
                if (
                    stored_binary_negatives is not None
                    and int(stored_binary_negatives) != args.binary_negatives
                ):
                    raise ValueError(
                        f"{run_json_path} used binary_negatives={stored_binary_negatives}; "
                        f"rerun aggregation with --binary-negatives {stored_binary_negatives}."
                    )
                if (
                    stored_candidate_seed is not None
                    and int(stored_candidate_seed) != args.binary_candidate_seed
                ):
                    raise ValueError(
                        f"{run_json_path} used binary_candidate_seed={stored_candidate_seed}; "
                        f"rerun aggregation with --binary-candidate-seed {stored_candidate_seed}."
                    )
            if run_result.get("variant") != variant or int(run_result.get("seed")) != seed:
                raise ValueError(
                    f"Run JSON identity mismatch in {run_json_path}: "
                    f"variant={run_result.get('variant')} seed={run_result.get('seed')}"
                )

            dataset = datasets[variant]
            encoder, relation_embedding, graph_bundle, checkpoint_meta = (
                runner.build_model_from_checkpoint(
                    checkpoint,
                    dataset=dataset,
                    device=device,
                    backend_override=args.backend_override,
                )
            )
            encoder.eval()
            relation_embedding.eval()
            with torch.no_grad():
                live_representation = runner.representation_from_live_model(
                    encoder, relation_embedding, graph_bundle
                )
                recomputed_val, _ = runner.evaluate_representations(
                    [live_representation],
                    dataset.val_pos,
                    device=device,
                    tail_filters=tail_filters,
                    head_filters=head_filters,
                    num_entities=dataset.num_entities,
                    eval_batch_size=args.eval_batch_size,
                    binary_negatives=args.binary_negatives,
                    binary_candidate_seed=args.binary_candidate_seed + 1,
                )
                recomputed_test, payload = runner.evaluate_representations(
                    [live_representation],
                    dataset.test_pos,
                    device=device,
                    tail_filters=tail_filters,
                    head_filters=head_filters,
                    num_entities=dataset.num_entities,
                    eval_batch_size=args.eval_batch_size,
                    binary_negatives=args.binary_negatives,
                    binary_candidate_seed=args.binary_candidate_seed,
                    return_candidate_scores=True,
                )
            assert payload is not None
            cpu_representations[variant] = cpu_copy_representation(live_representation)
            score_payloads[variant] = payload

            stored_validation = dict(run_result.get("validation", {}))
            stored_test = dict(run_result.get("test", {}))
            validation_differences = metric_abs_differences(
                stored_validation, recomputed_val
            )
            test_differences = metric_abs_differences(stored_test, recomputed_test)
            val_difference = max(validation_differences.values(), default=0.0)
            test_difference = max(test_differences.values(), default=0.0)
            compatibility_checks.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "validation_max_abs_difference": val_difference,
                    "test_max_abs_difference": test_difference,
                    "largest_validation_differences": largest_metric_differences(
                        validation_differences
                    ),
                    "largest_test_differences": largest_metric_differences(
                        test_differences
                    ),
                    "validation_differences": validation_differences,
                    "test_differences": test_differences,
                    "checkpoint_format": checkpoint_meta.get("format_version"),
                }
            )
            if max(val_difference, test_difference) > args.metric_tolerance:
                message = (
                    f"Checkpoint metrics do not reproduce {run_json_path} within "
                    f"tolerance {args.metric_tolerance:g}: val diff={val_difference:.3g}, "
                    f"test diff={test_difference:.3g}."
                )
                if args.strict_metric_check:
                    raise AssertionError(message)
                print(f"  Warning: {message}")
                print(
                    "    largest val differences: "
                    f"{largest_metric_differences(validation_differences)}"
                )
                print(
                    "    largest test differences: "
                    f"{largest_metric_differences(test_differences)}"
                )

            resources = run_result.get("resources", {})
            if not isinstance(resources, Mapping):
                raise TypeError(f"resources must be an object in {run_json_path}")
            component_resources.append(resources)
            raw_by_variant[variant].append(
                {
                    "seed": seed,
                    # Recomputed best-checkpoint metrics are the source of truth.
                    "validation": dict(recomputed_val),
                    "test": dict(recomputed_test),
                    # Stored JSON values remain available for audit.
                    "stored_validation": stored_validation,
                    "stored_test": stored_test,
                    "resources": dict(resources),
                    "checkpoint": str(checkpoint),
                    "run_json": str(run_json_path),
                }
            )
            print(
                f"  {variant:22s} reproduced test MRR="
                f"{recomputed_test['filtered_MRR']:.6f}"
            )

            del live_representation, encoder, relation_embedding, graph_bundle
            runner.clear_memory(device)

        for first, second in itertools.combinations(variants, 2):
            comparison_key = f"{first}__vs__{second}"
            pair_metrics = invariance_metrics(
                score_payloads[first], score_payloads[second]
            )
            invariance_by_comparison[comparison_key].append(
                {"seed": float(seed), **pair_metrics}
            )

        # Output fusion is exactly the arithmetic mean of raw relation-score
        # logits.  No sigmoid/softmax is applied before averaging.
        runner.clear_memory(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        device_representations = [
            move_representation(cpu_representations[variant], device)
            for variant in variants
        ]
        with torch.no_grad():
            fusion_val, _ = runner.evaluate_representations(
                device_representations,
                reference.val_pos,
                device=device,
                tail_filters=tail_filters,
                head_filters=head_filters,
                num_entities=reference.num_entities,
                eval_batch_size=args.eval_batch_size,
                binary_negatives=args.binary_negatives,
                binary_candidate_seed=args.binary_candidate_seed + 1,
            )
            fusion_test, fusion_payload = runner.evaluate_representations(
                device_representations,
                reference.test_pos,
                device=device,
                tail_filters=tail_filters,
                head_filters=head_filters,
                num_entities=reference.num_entities,
                eval_batch_size=args.eval_batch_size,
                binary_negatives=args.binary_negatives,
                binary_candidate_seed=args.binary_candidate_seed,
                return_candidate_scores=True,
            )
        assert fusion_payload is not None

        # Compare output fusion with every component variant using exactly the
        # same test queries, candidate IDs, and deterministic candidate seed.
        for variant in variants:
            comparison_key = f"{OUTPUT_FUSION}__vs__{variant}"
            pair_metrics = invariance_metrics(
                fusion_payload, score_payloads[variant]
            )
            invariance_by_comparison[comparison_key].append(
                {"seed": float(seed), **pair_metrics}
            )
        if device.type == "cuda":
            runner.synchronize(device)
            fused_peak_allocated = runner.bytes_to_mib(
                torch.cuda.max_memory_allocated(device)
            )
            fused_peak_reserved = runner.bytes_to_mib(
                torch.cuda.max_memory_reserved(device)
            )
        else:
            fused_peak_allocated = None
            fused_peak_reserved = None

        fusion_resources = build_fusion_resources(
            component_resources,
            fused_peak_allocated,
            fused_peak_reserved,
        )
        raw_by_variant[OUTPUT_FUSION].append(
            {
                "seed": seed,
                "validation": fusion_val,
                "test": fusion_test,
                "resources": fusion_resources,
                "component_variants": list(variants),
            }
        )
        print(
            f"  {'output_fusion':22s} test MRR={fusion_test['filtered_MRR']:.6f}"
        )

        del device_representations, cpu_representations, score_payloads, fusion_payload
        runner.clear_memory(device)
        gc.collect()

    aggregate: dict[str, Any] = {}
    for variant, records in raw_by_variant.items():
        aggregate[variant] = {
            "validation": summarize_numeric_dicts(
                [record["validation"] for record in records]
            ),
            "test": summarize_numeric_dicts([record["test"] for record in records]),
            "resources": summarize_numeric_dicts(
                [numeric_resource_fields(record["resources"]) for record in records]
            ),
        }

    invariance_aggregate = {
        comparison: summarize_numeric_dicts(records)
        for comparison, records in invariance_by_comparison.items()
    }

    latex_text = print_latex(variants, aggregate, invariance_aggregate)
    latex_output.parent.mkdir(parents=True, exist_ok=True)
    latex_output.write_text(latex_text + "\n", encoding="utf-8")

    payload = {
        "format_version": AGGREGATE_FORMAT,
        "dataset": "WordNet LP",
        "model": "SlotGAT",
        "variants": variants,
        "seeds": list(args.seeds),
        "num_runs_per_variant": len(args.seeds),
        "output_fusion": {
            "definition": (
                "arithmetic mean of raw DistMult logits across selected variant "
                "checkpoints before thresholding or probability conversion"
            ),
            "component_variants": variants,
            "kendall_comparisons": [
                f"{OUTPUT_FUSION}__vs__{variant}" for variant in variants
            ],
            "kendall_definition": (
                "macro mean of per-query Kendall tau-b on aligned deterministic "
                "test-candidate logits; tau@1 and tau@3 are Kendall tau-b on "
                "per-query Hit@1 and Hit@3 indicators using the same candidate sets"
            ),
        },
        "metric_source_policy": (
            "All aggregate validation/test rows are recomputed from the selected "
            "best checkpoints. Stored run JSON metrics are retained only for audit."
        ),
        "aggregate": aggregate,
        "invariance": invariance_aggregate,
        "raw_runs": raw_by_variant,
        "raw_invariance": dict(invariance_by_comparison),
        "compatibility_checks": compatibility_checks,
        "latex_output": str(latex_output),
    }
    write_json(output_json, payload)
    print(f"\nAggregate JSON: {output_json}")
    print(f"LaTeX rows:     {latex_output}")


if __name__ == "__main__":
    main()