#!/usr/bin/env python3
"""Recompute Kendall rank invariance from saved RGCN augmentation scores.

The augmentation runners train one shared model on several graph variants and
save one test-score CSV per input variant.  There is no separate "combined
graph" prediction.  Consequently, a result labelled with training regime
``IMDb1--4`` compares the rankings produced by that shared model when it is
evaluated on IMDb1, IMDb2, IMDb3, and IMDb4.

This postprocessor follows the metric definition in
``src/baselines/RGCN/A8. new Experiment Details.tex``:

* Kendall tau is computed independently for every test instance/query.
* Link-prediction tau@K is computed on the union of the two top-K candidate
  sets, with K in {1, 3}.
* Query values are averaged within a seed.
* Reported uncertainty is the sample standard deviation across seeds.

Only the Python standard library is required.

Examples
--------
Run every available benchmark using the repository's standard results root:

    python rgcn_data_augmentation/calculate_kendall_tau.py

Run only IMDb node classification and IMDb link prediction:

    python rgcn_data_augmentation/calculate_kendall_tau.py \
        --datasets imdb_nc imdb_lp_ml imdb_lp_md

The script writes ``kendall_tau_per_seed.csv`` and
``kendall_tau_summary.csv`` in each selected benchmark results directory.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ScoreMap = Dict[Tuple[str, ...], Tuple[str, List[float]]]
QueryScoreMap = Dict[str, Dict[Tuple[str, ...], float]]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    relative_dir: str
    task: str
    file_glob: str
    variant_regex: str
    variant_order: Tuple[str, ...]
    variant_labels: Mapping[str, str]
    training_label: str
    item_column: Optional[str] = None
    score_prefixes: Tuple[str, ...] = ()
    query_column: Optional[str] = None
    identity_columns: Tuple[str, ...] = ()
    score_column: Optional[str] = None


SPECS: Dict[str, DatasetSpec] = {
    "imdb_nc": DatasetSpec(
        key="imdb_nc",
        relative_dir="IMDB",
        task="classification",
        file_glob="test_scores_*.csv",
        variant_regex=r"test_scores_(v[1-4])\.csv",
        variant_order=("v1", "v2", "v3", "v4"),
        variant_labels={"v1": "IMDb1", "v2": "IMDb2", "v3": "IMDb3", "v4": "IMDb4"},
        training_label="IMDb1--4",
        item_column="movie_id",
        score_prefixes=("logit_class_",),
    ),
    "imdb_lp_ml": DatasetSpec(
        key="imdb_lp_ml",
        relative_dir="IMDB_LP/ml",
        task="link_prediction",
        file_glob="test_scores_*.csv",
        variant_regex=r"test_scores_(v[1-4])\.csv",
        variant_order=("v1", "v2", "v3", "v4"),
        variant_labels={"v1": "IMDb1", "v2": "IMDb2", "v3": "IMDb3", "v4": "IMDb4"},
        training_label="IMDb1--4",
        query_column="query_row",
        identity_columns=("candidate_id", "movie_local", "link_local", "label"),
        score_column="score",
    ),
    "imdb_lp_md": DatasetSpec(
        key="imdb_lp_md",
        relative_dir="IMDB_LP/md",
        task="link_prediction",
        file_glob="test_scores_*.csv",
        variant_regex=r"test_scores_(v[13])\.csv",
        variant_order=("v1", "v3"),
        variant_labels={"v1": "IMDb1", "v3": "IMDb3"},
        training_label="IMDb1--3",
        query_column="query_row",
        identity_columns=("candidate_id", "movie_local", "director_local", "label"),
        score_column="score",
    ),
    "dblp_lp": DatasetSpec(
        key="dblp_lp",
        relative_dir="DBLP",
        task="link_prediction",
        file_glob="test_scores_*.csv",
        variant_regex=r"test_scores_(v[1-3])\.csv",
        variant_order=("v1", "v2", "v3"),
        variant_labels={"v1": "DBLP1", "v2": "DBLP2", "v3": "DBLP3"},
        training_label="DBLP1--3",
        query_column="paper_id",
        identity_columns=("conf_id", "label"),
        score_column="score",
    ),
    "wordnet_lp": DatasetSpec(
        key="wordnet_lp",
        relative_dir="WORDNET_3VAR_UPDATED_UNIVERSAL_NEG1",
        task="link_prediction",
        file_glob="shared_candidate_test_scores_*.csv",
        variant_regex=r"shared_candidate_test_scores_(.+)\.csv",
        variant_order=("no_changes", "all_inverse_edges", "universal_edges"),
        variant_labels={
            "no_changes": "No Changes",
            "all_inverse_edges": "All Inverse Edges",
            "universal_edges": "Universal",
        },
        training_label="No Changes + All Inverse + Universal",
        query_column="query_id",
        identity_columns=("head", "relation", "tail", "label"),
        score_column="score",
    ),
    "freebase_nc": DatasetSpec(
        key="freebase_nc",
        relative_dir="FREEBASE_chunked_recompute",
        task="classification",
        file_glob="test_scores_*.csv",
        variant_regex=r"test_scores_(unchanged|exact_2)\.csv",
        variant_order=("unchanged", "exact_2"),
        variant_labels={"unchanged": "Unchanged", "exact_2": "Exact 2"},
        training_label="Unchanged + Exact 2",
        item_column="node_id",
        score_prefixes=("log_probability_class_", "logit_class_"),
    ),
}


def sign(value: float) -> int:
    return (value > 0.0) - (value < 0.0)


def kendall_tau_b(scores_a: Sequence[float], scores_b: Sequence[float]) -> Optional[float]:
    """Return Kendall tau-b, or None when one ranking has no ordering."""
    if len(scores_a) != len(scores_b):
        raise ValueError("Kendall inputs have different lengths.")
    concordant = discordant = ties_a = ties_b = 0
    for left, right in itertools.combinations(range(len(scores_a)), 2):
        order_a = sign(scores_a[left] - scores_a[right])
        order_b = sign(scores_b[left] - scores_b[right])
        if order_a == 0 and order_b == 0:
            continue
        if order_a == 0:
            ties_a += 1
        elif order_b == 0:
            ties_b += 1
        elif order_a == order_b:
            concordant += 1
        else:
            discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_a)
        * (concordant + discordant + ties_b)
    )
    if denominator == 0.0:
        return None
    return (concordant - discordant) / denominator


def average_defined(values: Iterable[Optional[float]]) -> Tuple[float, int]:
    defined = [value for value in values if value is not None and math.isfinite(value)]
    if not defined:
        return float("nan"), 0
    return statistics.fmean(defined), len(defined)


def top_k_union_tau(
    candidates: Sequence[Tuple[str, ...]],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    k: int,
) -> Optional[float]:
    """Compute tau-b on the union of two top-k candidate sets.

    For K=1, an identical singleton has no candidate pair.  We assign 1.0 to
    that exact agreement; different top-1 candidates form a two-item union and
    necessarily receive -1.0.
    """
    order_a = sorted(range(len(candidates)), key=lambda i: (-scores_a[i], candidates[i]))
    order_b = sorted(range(len(candidates)), key=lambda i: (-scores_b[i], candidates[i]))
    # A top-k set is not uniquely defined when a score tie crosses its cutoff.
    # Omit that query instead of silently resolving the tie by candidate ID.
    if k < len(candidates):
        if scores_a[order_a[k - 1]] == scores_a[order_a[k]]:
            return None
        if scores_b[order_b[k - 1]] == scores_b[order_b[k]]:
            return None
    selected = set(order_a[: min(k, len(order_a))])
    selected.update(order_b[: min(k, len(order_b))])
    indices = sorted(selected, key=lambda i: candidates[i])
    if len(indices) < 2:
        return 1.0
    tau = kendall_tau_b(
        [scores_a[index] for index in indices],
        [scores_b[index] for index in indices],
    )
    # Two entirely tied restrictions contain no ordering information.  They
    # are exact agreements only when the tied score patterns match.
    if tau is None:
        pattern_a = [
            sign(scores_a[left] - scores_a[right])
            for left, right in itertools.combinations(indices, 2)
        ]
        pattern_b = [
            sign(scores_b[left] - scores_b[right])
            for left, right in itertools.combinations(indices, 2)
        ]
        return 1.0 if pattern_a == pattern_b else 0.0
    return tau


def variant_from_path(path: Path, spec: DatasetSpec) -> Optional[str]:
    match = re.fullmatch(spec.variant_regex, path.name)
    return match.group(1) if match else None


def score_columns(fieldnames: Sequence[str], prefixes: Sequence[str]) -> List[str]:
    for prefix in prefixes:
        columns = [name for name in fieldnames if name.startswith(prefix)]
        if columns:
            return sorted(columns, key=lambda name: int(name.rsplit("_", 1)[1]))
    raise ValueError(
        f"None of the score prefixes {prefixes!r} occur in CSV columns {fieldnames!r}."
    )


def read_classification(path: Path, spec: DatasetSpec) -> ScoreMap:
    if spec.item_column is None:
        raise ValueError(f"{spec.key} has no item column.")
    output: ScoreMap = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header.")
        columns = score_columns(reader.fieldnames, spec.score_prefixes)
        for row in reader:
            item = (row[spec.item_column],)
            if item in output:
                raise ValueError(f"Duplicate item {item} in {path}.")
            output[item] = (row.get("label", ""), [float(row[column]) for column in columns])
    return output


def read_link_prediction(path: Path, spec: DatasetSpec) -> QueryScoreMap:
    if spec.query_column is None or spec.score_column is None:
        raise ValueError(f"{spec.key} lacks link-prediction columns.")
    output: QueryScoreMap = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header.")
        required = {
            spec.query_column,
            spec.score_column,
            *spec.identity_columns,
        }
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}.")
        for row in reader:
            query = row[spec.query_column]
            candidate = tuple(row[column] for column in spec.identity_columns)
            if candidate in output[query]:
                raise ValueError(f"Duplicate candidate {candidate} for query {query} in {path}.")
            output[query][candidate] = float(row[spec.score_column])
    return dict(output)


def validate_same_keys(
    keys_a: Iterable[Tuple[str, ...] | str],
    keys_b: Iterable[Tuple[str, ...] | str],
    context: str,
) -> None:
    set_a, set_b = set(keys_a), set(keys_b)
    if set_a != set_b:
        only_a = list(sorted(set_a - set_b))[:3]
        only_b = list(sorted(set_b - set_a))[:3]
        raise ValueError(
            f"Candidate alignment failed for {context}: "
            f"only_a={only_a}, only_b={only_b}."
        )


def classification_pair(
    output_a: ScoreMap,
    output_b: ScoreMap,
) -> Dict[str, float | int]:
    validate_same_keys(output_a, output_b, "classification items")
    taus: List[Optional[float]] = []
    for item in sorted(output_a):
        label_a, scores_a = output_a[item]
        label_b, scores_b = output_b[item]
        if label_a != label_b:
            raise ValueError(f"Labels differ for classification item {item}.")
        taus.append(kendall_tau_b(scores_a, scores_b))
    mean_tau, valid = average_defined(taus)
    return {
        "kendall_tau": mean_tau,
        "kendall_tau_at_1": float("nan"),
        "kendall_tau_at_3": float("nan"),
        "instance_count": len(taus),
        "valid_tau_count": valid,
        "valid_tau_at_1_count": 0,
        "valid_tau_at_3_count": 0,
    }


def link_pair(
    output_a: QueryScoreMap,
    output_b: QueryScoreMap,
) -> Dict[str, float | int]:
    validate_same_keys(output_a, output_b, "link-prediction queries")
    full_taus: List[Optional[float]] = []
    top1_taus: List[Optional[float]] = []
    top3_taus: List[Optional[float]] = []
    for query in sorted(output_a):
        scores_by_candidate_a = output_a[query]
        scores_by_candidate_b = output_b[query]
        validate_same_keys(
            scores_by_candidate_a,
            scores_by_candidate_b,
            f"link-prediction query {query}",
        )
        candidates = sorted(scores_by_candidate_a)
        scores_a = [scores_by_candidate_a[candidate] for candidate in candidates]
        scores_b = [scores_by_candidate_b[candidate] for candidate in candidates]
        full_taus.append(kendall_tau_b(scores_a, scores_b))
        top1_taus.append(top_k_union_tau(candidates, scores_a, scores_b, 1))
        top3_taus.append(top_k_union_tau(candidates, scores_a, scores_b, 3))
    mean_tau, valid = average_defined(full_taus)
    mean_tau1, valid_tau1 = average_defined(top1_taus)
    mean_tau3, valid_tau3 = average_defined(top3_taus)
    return {
        "kendall_tau": mean_tau,
        "kendall_tau_at_1": mean_tau1,
        "kendall_tau_at_3": mean_tau3,
        "instance_count": len(full_taus),
        "valid_tau_count": valid,
        "valid_tau_at_1_count": valid_tau1,
        "valid_tau_at_3_count": valid_tau3,
    }


def discover_seed_outputs(
    result_dir: Path,
    spec: DatasetSpec,
) -> Dict[int, Dict[str, Path]]:
    seeds: Dict[int, Dict[str, Path]] = {}
    for seed_dir in sorted(result_dir.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        try:
            seed = int(seed_dir.name.removeprefix("seed_"))
        except ValueError:
            continue
        variants: Dict[str, Path] = {}
        for path in seed_dir.glob(spec.file_glob):
            variant = variant_from_path(path, spec)
            if variant in spec.variant_order:
                variants[variant] = path
        missing = [variant for variant in spec.variant_order if variant not in variants]
        if missing:
            raise FileNotFoundError(
                f"{seed_dir} is missing score CSVs for variants {missing}."
            )
        seeds[seed] = variants
    if not seeds:
        raise FileNotFoundError(f"No seed directories found in {result_dir}.")
    return seeds


def sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def finite_values(rows: Sequence[Mapping[str, object]], field: str) -> List[float]:
    values = [float(row[field]) for row in rows]
    return [value for value in values if math.isfinite(value)]


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def calculate_dataset(
    results_root: Path,
    spec: DatasetSpec,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    result_dir = results_root / spec.relative_dir
    seed_paths = discover_seed_outputs(result_dir, spec)
    per_seed: List[Dict[str, object]] = []
    for seed, variant_paths in sorted(seed_paths.items()):
        if spec.task == "classification":
            outputs = {
                variant: read_classification(variant_paths[variant], spec)
                for variant in spec.variant_order
            }
        else:
            outputs = {
                variant: read_link_prediction(variant_paths[variant], spec)
                for variant in spec.variant_order
            }
        for variant_a, variant_b in itertools.combinations(spec.variant_order, 2):
            if spec.task == "classification":
                metrics = classification_pair(outputs[variant_a], outputs[variant_b])
            else:
                metrics = link_pair(outputs[variant_a], outputs[variant_b])
            per_seed.append(
                {
                    "dataset": spec.key,
                    "training_regime": spec.training_label,
                    "seed": seed,
                    "variant_a": variant_a,
                    "variant_b": variant_b,
                    "comparison": (
                        f"{spec.variant_labels[variant_a]} vs "
                        f"{spec.variant_labels[variant_b]}"
                    ),
                    **metrics,
                }
            )

    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in per_seed:
        grouped[(str(row["variant_a"]), str(row["variant_b"]))].append(row)

    summary: List[Dict[str, object]] = []
    for variant_a, variant_b in itertools.combinations(spec.variant_order, 2):
        rows = grouped[(variant_a, variant_b)]
        tau = finite_values(rows, "kendall_tau")
        tau1 = finite_values(rows, "kendall_tau_at_1")
        tau3 = finite_values(rows, "kendall_tau_at_3")
        complete_tau = len(tau) == len(rows)
        complete_tau1 = len(tau1) == len(rows)
        complete_tau3 = len(tau3) == len(rows)
        summary.append(
            {
                "dataset": spec.key,
                "training_regime": spec.training_label,
                "variant_a": variant_a,
                "variant_b": variant_b,
                "comparison": (
                    f"{spec.variant_labels[variant_a]} vs "
                    f"{spec.variant_labels[variant_b]}"
                ),
                "seed_count": len(rows),
                "valid_seed_count": len(tau),
                "kendall_tau_mean": (
                    statistics.fmean(tau) if complete_tau else float("nan")
                ),
                "kendall_tau_std": (
                    sample_std(tau) if complete_tau else float("nan")
                ),
                "kendall_tau_at_1_mean": (
                    statistics.fmean(tau1) if complete_tau1 else float("nan")
                ),
                "kendall_tau_at_1_std": (
                    sample_std(tau1) if complete_tau1 else float("nan")
                ),
                "kendall_tau_at_3_mean": (
                    statistics.fmean(tau3) if complete_tau3 else float("nan")
                ),
                "kendall_tau_at_3_std": (
                    sample_std(tau3) if complete_tau3 else float("nan")
                ),
                "instance_count": rows[0]["instance_count"],
                "min_valid_tau_count": min(
                    int(row["valid_tau_count"]) for row in rows
                ),
                "min_valid_tau_at_1_count": min(
                    int(row["valid_tau_at_1_count"]) for row in rows
                ),
                "min_valid_tau_at_3_count": min(
                    int(row["valid_tau_at_3_count"]) for row in rows
                ),
            }
        )

    per_seed_fields = (
        "dataset",
        "training_regime",
        "seed",
        "variant_a",
        "variant_b",
        "comparison",
        "kendall_tau",
        "kendall_tau_at_1",
        "kendall_tau_at_3",
        "instance_count",
        "valid_tau_count",
        "valid_tau_at_1_count",
        "valid_tau_at_3_count",
    )
    summary_fields = (
        "dataset",
        "training_regime",
        "variant_a",
        "variant_b",
        "comparison",
        "seed_count",
        "valid_seed_count",
        "kendall_tau_mean",
        "kendall_tau_std",
        "kendall_tau_at_1_mean",
        "kendall_tau_at_1_std",
        "kendall_tau_at_3_mean",
        "kendall_tau_at_3_std",
        "instance_count",
        "min_valid_tau_count",
        "min_valid_tau_at_1_count",
        "min_valid_tau_at_3_count",
    )
    write_csv(result_dir / "kendall_tau_per_seed.csv", per_seed, per_seed_fields)
    write_csv(result_dir / "kendall_tau_summary.csv", summary, summary_fields)
    return per_seed, summary


def format_metric(row: Mapping[str, object], prefix: str) -> str:
    mean = float(row[f"{prefix}_mean"])
    std = float(row[f"{prefix}_std"])
    return "--" if not math.isfinite(mean) else f"{mean:.4f} +/- {std:.4f}"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_root = script_dir.parent / "results" / "rgcn_augmentation"
    parser = argparse.ArgumentParser(
        description="Recompute per-query Kendall tau for RGCN augmentation runs."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=default_root,
        help=f"RGCN augmentation results root (default: {default_root})",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(SPECS),
        default=tuple(SPECS),
        help="Benchmarks to process (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        spec = SPECS[dataset]
        _, summary = calculate_dataset(args.results_root, spec)
        print(f"\n{dataset} | shared training: {spec.training_label}")
        for row in summary:
            line = (
                f"  {row['comparison']}: "
                f"tau={format_metric(row, 'kendall_tau')}"
            )
            if spec.task == "link_prediction":
                line += (
                    f", tau@1={format_metric(row, 'kendall_tau_at_1')}, "
                    f"tau@3={format_metric(row, 'kendall_tau_at_3')}"
                )
            print(line)


if __name__ == "__main__":
    main()
