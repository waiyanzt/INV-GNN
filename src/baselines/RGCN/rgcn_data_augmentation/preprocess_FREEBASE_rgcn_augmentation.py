#!/usr/bin/env python3
"""Joint preprocessing for Freebase RGCN graph-variant augmentation.

This replaces the legacy per-variant reverse-relation offset with one global
relation vocabulary shared by every selected variant.  It also creates the
BOOK classification split once and reuses it for every variant.

Expected raw layout (HGB format):

    data/raw/dataset_variant_3hops_filter/<variant>/node.dat
    data/raw/dataset_variant_3hops_filter/<variant>/link.dat
    data/raw/dataset_variant_3hops_filter/<variant>/label.dat

Output layout:

    data/rgcn_augmentation/freebase/<variant>/rgcn_data.pt
    data/rgcn_augmentation/freebase/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split

BOOK_TYPE = 0
NUM_CLASSES = 8
DEFAULT_SEED = 1566911444


def read_nodes(path: Path) -> np.ndarray:
    rows: List[Tuple[int, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 3 or not parts[0]:
                continue
            rows.append((int(parts[0]), int(parts[2])))
    if not rows:
        raise ValueError(f"No nodes found in {path}")
    num_nodes = max(node_id for node_id, _ in rows) + 1
    node_type = np.full(num_nodes, -1, dtype=np.int64)
    for node_id, type_id in rows:
        node_type[node_id] = type_id
    if np.any(node_type < 0):
        raise ValueError(f"{path} contains non-contiguous node IDs")
    return node_type


def read_links(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: List[Tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 3 or not parts[0]:
                continue
            rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
    if not rows:
        raise ValueError(f"No edges found in {path}")
    array = np.asarray(rows, dtype=np.int64)
    return array[:, 0], array[:, 1], array[:, 2]


def read_book_labels(path: Path, node_type: np.ndarray) -> np.ndarray:
    labels = np.full(len(node_type), -1, dtype=np.int64)
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 4 or not parts[0]:
                continue
            node_id, type_id, class_id = int(parts[0]), int(parts[2]), int(parts[3])
            if type_id != BOOK_TYPE or node_type[node_id] != BOOK_TYPE:
                continue
            if 0 <= class_id < NUM_CLASSES:
                labels[node_id] = class_id
    if not np.any(labels >= 0):
        raise ValueError(f"No valid BOOK labels found in {path}")
    return labels


def stratified_split(labels: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labeled = np.flatnonzero(labels >= 0)
    y = labels[labeled]
    train_idx, rest_idx, _, rest_y = train_test_split(
        labeled,
        y,
        test_size=0.4,
        stratify=y,
        random_state=seed,
    )
    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=0.5,
        stratify=rest_y,
        random_state=seed,
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def mask_from_indices(num_nodes: int, indices: np.ndarray) -> np.ndarray:
    mask = np.zeros(num_nodes, dtype=bool)
    mask[indices] = True
    return mask


def relation_type_signatures(
    src: np.ndarray,
    dst: np.ndarray,
    rel: np.ndarray,
    node_type: np.ndarray,
) -> Dict[int, List[Tuple[int, int]]]:
    out: Dict[int, set] = {}
    for source, target, relation in zip(src.tolist(), dst.tolist(), rel.tolist()):
        out.setdefault(int(relation), set()).add(
            (int(node_type[source]), int(node_type[target]))
        )
    return {key: sorted(value) for key, value in out.items()}


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Freebase joint RGCN augmentation")
    parser.add_argument("--variants", nargs="+", default=["unchanged", "exact_2"])
    parser.add_argument(
        "--data-root",
        default="data/raw/dataset_variant_3hops_filter",
        help="Directory containing one HGB directory per graph variant",
    )
    parser.add_argument(
        "--output-root",
        default="data/rgcn_augmentation/freebase",
    )
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = list(dict.fromkeys(args.variants))
    if len(variants) != len(args.variants):
        raise SystemExit("--variants contains duplicates")

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    raw: Dict[str, Dict[str, np.ndarray]] = {}
    max_relation_id = -1
    for variant in variants:
        directory = data_root / variant
        node_type = read_nodes(directory / "node.dat")
        src, dst, rel = read_links(directory / "link.dat")
        labels = read_book_labels(directory / "label.dat", node_type)
        if int(src.max(initial=0)) >= len(node_type) or int(dst.max(initial=0)) >= len(node_type):
            raise ValueError(f"Edge endpoint outside node range for {variant}")
        max_relation_id = max(max_relation_id, int(rel.max()))
        raw[variant] = {
            "node_type": node_type,
            "src": src,
            "dst": dst,
            "rel": rel,
            "labels": labels,
        }

    reference = variants[0]
    for variant in variants[1:]:
        for key in ("node_type", "labels"):
            if not np.array_equal(raw[reference][key], raw[variant][key]):
                raise ValueError(f"{key} is not aligned between {reference} and {variant}")

    # Numeric relation IDs are assumed to retain their HGB meaning.  Check their
    # source/target node-type signatures wherever the same relation occurs.
    signatures = {
        variant: relation_type_signatures(
            raw[variant]["src"], raw[variant]["dst"], raw[variant]["rel"], raw[variant]["node_type"]
        )
        for variant in variants
    }
    for relation_id in range(max_relation_id + 1):
        observed = {
            variant: signatures[variant][relation_id]
            for variant in variants
            if relation_id in signatures[variant]
        }
        if len({tuple(value) for value in observed.values()}) > 1:
            raise ValueError(
                f"Relation ID {relation_id} has incompatible node-type signatures: {observed}"
            )

    num_forward_relations = max_relation_id + 1
    num_relations = 2 * num_forward_relations
    node_type = raw[reference]["node_type"]
    labels = raw[reference]["labels"]
    train_idx, val_idx, test_idx = stratified_split(labels, args.split_seed)
    train_mask = mask_from_indices(len(node_type), train_idx)
    val_mask = mask_from_indices(len(node_type), val_idx)
    test_mask = mask_from_indices(len(node_type), test_idx)

    manifest = {
        "dataset": "FREEBASE",
        "variants": variants,
        "num_nodes": int(len(node_type)),
        "num_forward_relations": num_forward_relations,
        "num_relations_with_reverse": num_relations,
        "reverse_relation_rule": "reverse_id = forward_id + global_num_forward_relations",
        "split_seed": args.split_seed,
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "labels_sha256": sha256_array(labels),
        "node_type_sha256": sha256_array(node_type),
        "variants_detail": {},
    }

    for variant in variants:
        src = raw[variant]["src"]
        dst = raw[variant]["dst"]
        rel = raw[variant]["rel"]
        edge_index = np.stack(
            [np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0
        )
        edge_type = np.concatenate([rel, rel + num_forward_relations])
        payload = {
            "edge_index": torch.from_numpy(edge_index).long(),
            "edge_type": torch.from_numpy(edge_type).long(),
            "y": torch.from_numpy(labels).long(),
            "train_mask": torch.from_numpy(train_mask),
            "val_mask": torch.from_numpy(val_mask),
            "test_mask": torch.from_numpy(test_mask),
            "node_type": torch.from_numpy(node_type).long(),
            "num_nodes": int(len(node_type)),
            "num_relations": int(num_relations),
            "num_forward_relations": int(num_forward_relations),
            "num_classes": NUM_CLASSES,
            "variant": variant,
            "split_seed": args.split_seed,
        }
        variant_dir = output_root / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        torch.save(payload, variant_dir / "rgcn_data.pt")
        manifest["variants_detail"][variant] = {
            "raw_forward_edges": int(len(src)),
            "directed_edges_with_reverse": int(edge_index.shape[1]),
            "forward_relation_ids_present": sorted(np.unique(rel).astype(int).tolist()),
            "relation_type_signatures": {
                str(key): value for key, value in signatures[variant].items()
            },
        }
        print(
            f"[{variant}] nodes={len(node_type)} raw_edges={len(src)} "
            f"directed_edges={edge_index.shape[1]}"
        )

    with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[OK] Wrote Freebase augmentation data under {output_root}")


if __name__ == "__main__":
    main()
