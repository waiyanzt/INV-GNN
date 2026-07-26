#!/usr/bin/env python3
"""Audit existing RGCN variants for joint augmentation training.

The repository's IMDb node-classification, IMDb link-prediction, and DBLP
models learn a homogeneous node embedding table.  This script verifies the ID,
split, and graph-schema assumptions required to share that table and one model
checkpoint across graph variants.  It does not regenerate the repository's raw
preprocessing outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

from rgcn_aug_common import (
    DBLP_RELATIONS,
    IMDB_LP_RELATIONS,
    IMDB_RELATIONS,
    json_ready,
    sha256_tensor,
    write_json,
)


DEFAULT_BASES = {
    "DBLP": {
        "v1": "data/preprocessed/DBLP_rgcn_v1",
        "v2": "data/preprocessed/DBLP_rgcn_v2",
        "v3": "data/preprocessed/DBLP_rgcn_v3",
    },
    "IMDB": {
        "v1": "data/preprocessed/IMDB_rgcn_v1",
        "v2": "data/preprocessed/IMDB_rgcn_v2",
        "v3": "data/preprocessed/IMDB_rgcn_v3",
        "v4": "data/preprocessed/IMDB_rgcn_v4",
    },
}

IMDB_LP_VARIANTS = {
    "md": ("v1", "v3"),
    "mg": ("v1", "v2", "v3", "v4"),
    "ml": ("v1", "v2", "v3", "v4"),
}

IMDB_LP_KNOWN_GRAPH_KEYS = {
    "movie-actor",
    "movie-director",
    "movie-link",
    "movie-genre",
    "link-director",
    "link-actor",
}

EXPECTED_GRAPH_KEYS = {
    "DBLP": {
        "v1": {"author-paper", "paper-conference", "paper-term", "paper-area"},
        "v2": {"author-paper", "paper-conference", "paper-term", "conference-area"},
        "v3": {"author-paper", "paper-conference", "paper-term", "author-area"},
    },
    "IMDB": {
        "v1": {"actor-movie", "movie-link", "movie-director"},
        "v2": {"actor-link", "link-movie", "link-director"},
        "v3": {"actor-link", "link-movie", "movie-director"},
        "v4": {"actor-movie", "movie-link", "link-director"},
    },
}


def imdb_lp_default_bases(task: str) -> Dict[str, str]:
    return {
        variant: f"data/preprocessed/IMDB_rgcn_lp_{task}_{variant}"
        for variant in IMDB_LP_VARIANTS[task]
    }


def load_variant(path: Path):
    graph_path = path / "graph_data.pt"
    meta_path = path / "meta.pt"
    if not graph_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Expected graph_data.pt and meta.pt under {path}")
    graph_data = torch.load(graph_path, map_location="cpu")
    meta = torch.load(meta_path, map_location="cpu")
    return graph_data, meta


def normalize_num_nodes(num_nodes: Mapping[str, Any]) -> Dict[str, int]:
    return {str(key): int(value) for key, value in num_nodes.items()}


def edge_count(edge_pair) -> int:
    source, destination = edge_pair
    if len(source) != len(destination):
        raise ValueError("Source/destination edge arrays have different lengths.")
    return int(len(source))


def assert_equal_tensors(name: str, by_variant: Mapping[str, torch.Tensor]) -> None:
    variants = list(by_variant)
    reference_variant = variants[0]
    reference = by_variant[reference_variant].cpu()
    for variant in variants[1:]:
        current = by_variant[variant].cpu()
        if reference.shape != current.shape or not torch.equal(reference, current):
            raise ValueError(
                f"{name} differs between {reference_variant} and {variant}; "
                "joint training requires semantically aligned IDs/splits."
            )


def audit_standard_dataset(dataset: str, bases: Mapping[str, str]) -> Dict[str, Any]:
    loaded = {
        variant: load_variant(Path(directory)) for variant, directory in bases.items()
    }

    num_nodes_by_variant = {
        variant: normalize_num_nodes(meta["num_nodes"])
        for variant, (_, meta) in loaded.items()
    }
    first_variant = next(iter(num_nodes_by_variant))
    reference_num_nodes = num_nodes_by_variant[first_variant]
    for variant, counts in num_nodes_by_variant.items():
        if counts != reference_num_nodes:
            raise ValueError(
                f"num_nodes differs between {first_variant} and {variant}: "
                f"{reference_num_nodes} vs {counts}"
            )

    for variant, (graph_data, _) in loaded.items():
        missing = EXPECTED_GRAPH_KEYS[dataset][variant] - set(graph_data)
        if missing:
            raise ValueError(
                f"{dataset} {variant} is missing graph keys: {sorted(missing)}"
            )

    manifest: Dict[str, Any] = {
        "dataset": dataset,
        "feature_policy": (
            "learned homogeneous node embedding; no external feature "
            "preprocessing required"
        ),
        "num_nodes": reference_num_nodes,
        "total_homogeneous_nodes": int(sum(reference_num_nodes.values())),
        "variant_paths": dict(bases),
        "variants": {},
        "global_native_relation_vocabulary": list(
            DBLP_RELATIONS if dataset == "DBLP" else IMDB_RELATIONS
        ),
    }

    for variant, (graph_data, _) in loaded.items():
        manifest["variants"][variant] = {
            "graph_keys": sorted(graph_data),
            "edge_counts": {
                key: edge_count(value) for key, value in graph_data.items()
            },
        }

    if dataset == "IMDB":
        required = ["labels", "train_idx", "val_idx", "test_idx"]
        for key in required:
            assert_equal_tensors(
                key, {variant: meta[key] for variant, (_, meta) in loaded.items()}
            )
        reference_meta = loaded[first_variant][1]
        manifest["shared_supervision"] = {
            key: {
                "shape": list(reference_meta[key].shape),
                "sha256": sha256_tensor(reference_meta[key]),
            }
            for key in required
        }
    else:
        split_names = [
            "train_pos",
            "train_neg",
            "val_pos",
            "val_neg",
            "test_pos",
            "test_neg",
        ]
        for key in split_names:
            assert_equal_tensors(
                f"splits.{key}",
                {
                    variant: meta["splits"][key]
                    for variant, (_, meta) in loaded.items()
                },
            )
        reference_splits = loaded[first_variant][1]["splits"]
        manifest["shared_supervision"] = {
            key: {
                "shape": list(reference_splits[key].shape),
                "sha256": sha256_tensor(reference_splits[key]),
            }
            for key in split_names
        }

    return manifest


def audit_imdb_lp(task: str, bases: Mapping[str, str]) -> Dict[str, Any]:
    loaded = {
        variant: load_variant(Path(directory)) for variant, directory in bases.items()
    }
    first_variant = next(iter(loaded))
    reference_num_nodes = normalize_num_nodes(loaded[first_variant][1]["num_nodes"])

    required_tail = {"md": "director", "mg": "genre", "ml": "link"}[task]
    split_names = (
        "train_pos",
        "train_neg",
        "val_pos",
        "val_neg",
        "test_pos",
        "test_neg",
    )

    manifest: Dict[str, Any] = {
        "dataset": "IMDB_LP",
        "task": task,
        "feature_policy": (
            "learned homogeneous node embedding; no external feature "
            "preprocessing required"
        ),
        "num_nodes": reference_num_nodes,
        "total_homogeneous_nodes": int(sum(reference_num_nodes.values())),
        "variant_paths": dict(bases),
        "variants": {},
        "global_native_relation_vocabulary": list(IMDB_LP_RELATIONS),
    }

    if "movie" not in reference_num_nodes or required_tail not in reference_num_nodes:
        raise ValueError(
            f"IMDb-LP task={task} requires movie and {required_tail} node types"
        )

    for variant, (graph_data, meta) in loaded.items():
        counts = normalize_num_nodes(meta["num_nodes"])
        if counts != reference_num_nodes:
            raise ValueError(
                f"num_nodes differs between {first_variant} and {variant}: "
                f"{reference_num_nodes} vs {counts}"
            )
        unknown_keys = set(graph_data) - IMDB_LP_KNOWN_GRAPH_KEYS
        if unknown_keys:
            raise ValueError(
                f"IMDB_LP {task}/{variant} has unknown graph keys: "
                f"{sorted(unknown_keys)}"
            )
        if not graph_data:
            raise ValueError(f"IMDB_LP {task}/{variant} has no graph edges")
        if "splits" not in meta:
            raise ValueError(f"IMDB_LP {task}/{variant} meta.pt lacks splits")
        missing = set(split_names) - set(meta["splits"])
        if missing:
            raise ValueError(
                f"IMDB_LP {task}/{variant} is missing splits: {sorted(missing)}"
            )
        for prefix in ("train", "val", "test"):
            positive = meta["splits"][f"{prefix}_pos"]
            negative = meta["splits"][f"{prefix}_neg"]
            if positive.ndim != 2 or positive.shape[1] != 2:
                raise ValueError(
                    f"{task}/{variant} {prefix}_pos must have shape (N, 2)"
                )
            if negative.ndim != 2 or positive.shape[0] != negative.shape[0]:
                raise ValueError(
                    f"{task}/{variant} {prefix}_neg must have shape (N, K) "
                    "aligned with positives"
                )
        manifest["variants"][variant] = {
            "graph_keys": sorted(graph_data),
            "edge_counts": {
                key: edge_count(value) for key, value in graph_data.items()
            },
        }

    for key in split_names:
        assert_equal_tensors(
            f"splits.{key}",
            {
                variant: meta["splits"][key]
                for variant, (_, meta) in loaded.items()
            },
        )

    reference_splits = loaded[first_variant][1]["splits"]
    manifest["shared_supervision"] = {
        key: {
            "shape": list(reference_splits[key].shape),
            "sha256": sha256_tensor(reference_splits[key]),
        }
        for key in split_names
    }
    manifest["negative_candidates_per_positive"] = {
        prefix: int(reference_splits[f"{prefix}_neg"].shape[1])
        for prefix in ("train", "val", "test")
    }
    return manifest


def parse_path_overrides(values: Sequence[str]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(
                f"Invalid --path {value!r}; expected variant=/path/to/directory"
            )
        variant, path = value.split("=", 1)
        overrides[variant.strip().lower()] = path.strip()
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["DBLP", "IMDB", "IMDB_LP"], required=True
    )
    parser.add_argument(
        "--task",
        choices=["md", "mg", "ml"],
        default="",
        help="Required when --dataset IMDB_LP",
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help=(
            "Override a variant directory, e.g. "
            "--path v1=data/preprocessed/DBLP_rgcn_v1"
        ),
    )
    parser.add_argument("--output", default="", help="Manifest JSON path")
    args = parser.parse_args()

    if args.dataset == "IMDB_LP":
        if not args.task:
            raise SystemExit("--task is required with --dataset IMDB_LP")
        bases = imdb_lp_default_bases(args.task)
        bases.update(parse_path_overrides(args.path))
        expected_variants = set(IMDB_LP_VARIANTS[args.task])
        if set(bases) != expected_variants:
            raise SystemExit(
                f"Expected variants {sorted(expected_variants)}, got {sorted(bases)}"
            )
        manifest = audit_imdb_lp(args.task, bases)
        default_output = f"rgcn_aug_imdb_lp_{args.task}_manifest.json"
    else:
        if args.task:
            raise SystemExit("--task is only valid with --dataset IMDB_LP")
        bases = dict(DEFAULT_BASES[args.dataset])
        bases.update(parse_path_overrides(args.path))
        expected_variants = set(DEFAULT_BASES[args.dataset])
        if set(bases) != expected_variants:
            raise SystemExit(
                f"Expected variants {sorted(expected_variants)}, got {sorted(bases)}"
            )
        manifest = audit_standard_dataset(args.dataset, bases)
        default_output = f"rgcn_aug_{args.dataset.lower()}_manifest.json"

    output = Path(args.output or default_output)
    write_json(output, manifest)
    print(json.dumps(json_ready(manifest), indent=2, sort_keys=True))
    print(f"\n[OK] Wrote {output}")


if __name__ == "__main__":
    main()
