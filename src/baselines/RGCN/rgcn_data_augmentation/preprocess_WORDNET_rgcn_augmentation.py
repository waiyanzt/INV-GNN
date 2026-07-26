#!/usr/bin/env python3
"""Create a leakage-free NPZ for the four generated WordNet variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

VARIANTS = [
    "no_changes",
    "all_inverse_edges",
    "transitive_edges",
    "universal_edges",
]
TRANSITIVE_PREFIX = "__transitive__"
FORMAT_VERSION = "wordnet_lp_four_variants_v1"


def load_dict(path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                raise ValueError(f"Bad dictionary line {line_idx} in {path}: {line!r}")
            raw_id, name = parts
            mapping[name] = int(raw_id)
    return mapping


def validate_dense_ids(name: str, mapping: dict[str, int]) -> None:
    """Require IDs to be unique and contiguous so array indexing is safe."""
    ids = sorted(mapping.values())
    expected = list(range(len(mapping)))
    if ids != expected:
        raise ValueError(
            f"{name} IDs must be contiguous 0..{len(mapping) - 1}; "
            f"found first IDs {ids[:10]}"
        )


def load_triples(
    path: Path,
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
) -> np.ndarray:
    rows: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"Bad triple line {line_idx} in {path}: {line!r}")
            h, r, t = parts
            try:
                rows.append((entity_to_id[h], relation_to_id[r], entity_to_id[t]))
            except KeyError as exc:
                raise KeyError(
                    f"Unknown entity/relation {exc.args[0]!r} at line {line_idx} in {path}"
                ) from exc
    if not rows:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(rows, dtype=np.int32)


def triples_to_set(array: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(map(int, row)) for row in array}


def ensure_unique(name: str, array: np.ndarray):
    unique = len(triples_to_set(array))
    if unique != len(array):
        raise AssertionError(f"{name} contains {len(array) - unique} duplicate triples")


def validate(
    train_arrays: dict[str, np.ndarray],
    official_train: np.ndarray,
    val_pos: np.ndarray,
    test_pos: np.ndarray,
    relation_vocab: list[str],
    base_relation_ids: set[int],
):
    arrays = {
        "official_train": official_train,
        "validation": val_pos,
        "test": test_pos,
        **{f"train_{name}": array for name, array in train_arrays.items()},
    }
    for name, array in arrays.items():
        ensure_unique(name, array)

    official_train_set = triples_to_set(official_train)
    val_set = triples_to_set(val_pos)
    test_set = triples_to_set(test_pos)
    heldout_set = val_set | test_set

    if val_set & test_set:
        raise AssertionError("Validation and test splits overlap")
    if triples_to_set(train_arrays["no_changes"]) != official_train_set:
        raise AssertionError("no_changes/data.txt must exactly equal official train.txt")

    for variant, array in train_arrays.items():
        variant_set = triples_to_set(array)
        missing = official_train_set - variant_set
        if missing:
            raise AssertionError(
                f"{variant} is missing {len(missing)} official training triples"
            )
        direct_overlap = variant_set & heldout_set
        if direct_overlap:
            raise AssertionError(
                f"{variant} contains {len(direct_overlap)} held-out original triples"
            )

    for split_name, split in (("validation", val_pos), ("test", test_pos)):
        bad = {int(r) for _, r, _ in split if int(r) not in base_relation_ids}
        if bad:
            raise AssertionError(
                f"{split_name} contains derived relation IDs: {sorted(bad)}"
            )

    relation_to_id = {name: idx for idx, name in enumerate(relation_vocab)}
    inverse_id_by_base_id = {}
    for base_id in base_relation_ids:
        base_name = relation_vocab[base_id]
        inverse_name = f"{base_name}__inv"
        if inverse_name in relation_to_id:
            inverse_id_by_base_id[base_id] = relation_to_id[inverse_name]
    heldout_inverse = {
        (int(t), inverse_id_by_base_id[int(r)], int(h))
        for h, r, t in np.concatenate([val_pos, test_pos], axis=0)
        if int(r) in inverse_id_by_base_id
    }
    for variant in ("all_inverse_edges", "universal_edges"):
        overlap = triples_to_set(train_arrays[variant]) & heldout_inverse
        if overlap:
            raise AssertionError(
                f"{variant} contains {len(overlap)} inverses of held-out triples"
            )

    inverse_ids = {
        idx for idx, name in enumerate(relation_vocab) if name.endswith("__inv")
    }
    shortcut_ids = {
        idx for idx, name in enumerate(relation_vocab) if name.startswith(TRANSITIVE_PREFIX)
    }
    variant_relation_ids = {
        variant: {int(r) for _, r, _ in array}
        for variant, array in train_arrays.items()
    }
    if variant_relation_ids["no_changes"] - base_relation_ids:
        raise AssertionError("no_changes contains derived relation types")
    if variant_relation_ids["all_inverse_edges"] - (base_relation_ids | inverse_ids):
        raise AssertionError("all_inverse_edges contains non-base/non-inverse relations")
    if variant_relation_ids["transitive_edges"] & inverse_ids:
        raise AssertionError("transitive_edges must not contain inverse edges")
    if variant_relation_ids["transitive_edges"] - (base_relation_ids | shortcut_ids):
        raise AssertionError("transitive_edges contains unexpected relation types")
    if variant_relation_ids["universal_edges"] - (
        base_relation_ids | inverse_ids | shortcut_ids
    ):
        raise AssertionError("universal_edges contains unexpected relation types")


def count_relation_categories(array: np.ndarray, relation_vocab: list[str]):
    counts = {"base": 0, "inverse": 0, "shortcut": 0}
    for _, relation_id, _ in array:
        relation_name = relation_vocab[int(relation_id)]
        if relation_name.startswith(TRANSITIVE_PREFIX):
            counts["shortcut"] += 1
        elif relation_name.endswith("__inv"):
            counts["inverse"] += 1
        else:
            counts["base"] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Create WordNet link-prediction NPZ for four leakage-free variants."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-splits-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    source_dir = (
        args.source_splits_dir.resolve()
        if args.source_splits_dir is not None
        else data_dir / "original_splits"
    )
    output = (
        args.output.resolve()
        if args.output is not None
        else data_dir / "wordnet_splits.npz"
    )

    entity_to_id = load_dict(source_dir / "entities.dict")
    base_relation_to_local_id = load_dict(source_dir / "relations.dict")
    shared_relation_to_id = load_dict(data_dir / "shared_relations.dict")

    validate_dense_ids("entity", entity_to_id)
    validate_dense_ids("base relation", base_relation_to_local_id)
    validate_dense_ids("shared relation", shared_relation_to_id)

    num_entities = len(entity_to_id)
    num_relations = len(shared_relation_to_id)
    entity_vocab = [""] * num_entities
    for name, idx in entity_to_id.items():
        entity_vocab[idx] = name
    relation_vocab = [""] * num_relations
    for name, idx in shared_relation_to_id.items():
        relation_vocab[idx] = name

    base_relation_names = [
        name
        for name, _ in sorted(
            base_relation_to_local_id.items(), key=lambda item: item[1]
        )
    ]
    base_relation_ids = {shared_relation_to_id[name] for name in base_relation_names}

    official_train = load_triples(
        source_dir / "train.txt", entity_to_id, shared_relation_to_id
    )
    val_pos = load_triples(
        source_dir / "valid.txt", entity_to_id, shared_relation_to_id
    )
    test_pos = load_triples(
        source_dir / "test.txt", entity_to_id, shared_relation_to_id
    )
    train_arrays = {
        variant: load_triples(
            data_dir / variant / "data.txt", entity_to_id, shared_relation_to_id
        )
        for variant in VARIANTS
    }

    validate(
        train_arrays,
        official_train=official_train,
        val_pos=val_pos,
        test_pos=test_pos,
        relation_vocab=relation_vocab,
        base_relation_ids=base_relation_ids,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        train_pos_no_changes=train_arrays["no_changes"],
        train_pos_all_inverse_edges=train_arrays["all_inverse_edges"],
        train_pos_transitive_edges=train_arrays["transitive_edges"],
        train_pos_universal_edges=train_arrays["universal_edges"],
        val_pos=val_pos,
        test_pos=test_pos,
        entity_vocab=np.asarray(entity_vocab),
        relation_vocab=np.asarray(relation_vocab),
        num_entities=np.asarray(num_entities),
        num_relations=np.asarray(num_relations),
        num_base_relations=np.asarray(len(base_relation_ids)),
        base_relation_ids=np.asarray(sorted(base_relation_ids), dtype=np.int32),
        variant_names=np.asarray(VARIANTS),
        format_version=np.asarray(FORMAT_VERSION),
    )

    stats = {
        "official_split_counts": {
            "train": int(len(official_train)),
            "validation": int(len(val_pos)),
            "test": int(len(test_pos)),
        },
        "num_entities": num_entities,
        "num_base_relations": len(base_relation_ids),
        "num_shared_relations": num_relations,
        "variant_training_edges": {
            variant: int(len(array)) for variant, array in train_arrays.items()
        },
        "variant_edge_categories": {
            variant: count_relation_categories(array, relation_vocab)
            for variant, array in train_arrays.items()
        },
        "validation_and_test_shared_across_variants": True,
        "evaluation_relations_are_base_only": True,
        "leakage_checks_passed": True,
        "format_version": FORMAT_VERSION,
        "npz_variant_names": VARIANTS,
    }
    with (data_dir / "wordnet_split_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")

    print(f"Saved {output}")
    print(f"Saved {data_dir / 'wordnet_split_stats.json'}")
    print("Leakage and variant-definition checks passed.")


if __name__ == "__main__":
    main()
