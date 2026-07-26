"""Dataset loader for the four WordNet link-prediction graph variants.

The loader consumes ``wordnet_splits.npz`` produced by ``preprocess_wordnet_lp.py``
and exposes PyTorch/PyG-compatible training graph tensors.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch import Tensor

CANONICAL_VARIANTS: Final[tuple[str, ...]] = (
    "no_changes",
    "all_inverse_edges",
    "transitive_edges",
    "universal_edges",
)

VARIANT_ALIASES: Final[dict[str, str]] = {
    "no_changes": "no_changes",
    "unchanged": "no_changes",
    "all_inverse_edges": "all_inverse_edges",
    "inverse": "all_inverse_edges",
    "transitive_edges": "transitive_edges",
    "transitive": "transitive_edges",
    "universal_edges": "universal_edges",
    "universal": "universal_edges",
}

TRAIN_KEY_BY_VARIANT: Final[dict[str, str]] = {
    variant: f"train_pos_{variant}" for variant in CANONICAL_VARIANTS
}

REQUIRED_COMMON_KEYS: Final[tuple[str, ...]] = (
    "val_pos",
    "test_pos",
    "entity_vocab",
    "relation_vocab",
    "num_entities",
    "num_relations",
)


def canonicalize_variant(variant: str) -> str:
    """Return the canonical directory/NPZ name for a paper or code alias."""
    try:
        return VARIANT_ALIASES[variant]
    except KeyError as exc:
        allowed = ", ".join(sorted(VARIANT_ALIASES))
        raise ValueError(f"Unknown WordNet variant {variant!r}. Allowed: {allowed}") from exc


def _validate_triples(
    name: str,
    array: np.ndarray,
    *,
    num_entities: int,
    num_relations: int,
) -> np.ndarray:
    """Validate and normalize one ``(head, relation, tail)`` array."""
    array = np.asarray(array)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3); found {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integer IDs; found dtype {array.dtype}")

    # Keep the compact int32 representation written by the preprocessor.
    array = np.ascontiguousarray(array, dtype=np.int32)
    if array.size:
        heads = array[:, 0]
        relations = array[:, 1]
        tails = array[:, 2]
        if heads.min() < 0 or tails.min() < 0 or relations.min() < 0:
            raise ValueError(f"{name} contains negative IDs")
        if heads.max() >= num_entities or tails.max() >= num_entities:
            raise ValueError(
                f"{name} contains entity ID outside [0, {num_entities - 1}]"
            )
        if relations.max() >= num_relations:
            raise ValueError(
                f"{name} contains relation ID outside [0, {num_relations - 1}]"
            )
    return array


class WordNetLPDataset:
    """Load one graph variant while sharing the same validation/test queries."""

    def __init__(self, variant: str, splits_path: str | os.PathLike[str] | None = None):
        """
        Args:
            variant: Canonical name or paper alias. Supported aliases are
                ``unchanged``, ``inverse``, ``transitive``, and ``universal``.
            splits_path: Path to ``wordnet_splits.npz``. When omitted, use the
                repository-relative default expected by ``run_wordnet_lp.py``.
        """
        self.requested_variant = variant
        self.variant = canonicalize_variant(variant)

        if splits_path is None:
            splits_path = Path(__file__).resolve().parent / "data" / (
                "wordnet_3hops_augmented_full"
            ) / "wordnet_splits.npz"
        self.splits_path = Path(splits_path).expanduser().resolve()
        if not self.splits_path.is_file():
            raise FileNotFoundError(f"WordNet split file not found: {self.splits_path}")

        with np.load(self.splits_path, allow_pickle=False) as data:
            train_key = TRAIN_KEY_BY_VARIANT[self.variant]
            required = set(REQUIRED_COMMON_KEYS) | {train_key}
            missing = sorted(required - set(data.files))
            if missing:
                raise KeyError(
                    f"{self.splits_path} is missing required NPZ keys for "
                    f"{self.variant}: {missing}. Available keys: {sorted(data.files)}"
                )

            self.num_entities = int(np.asarray(data["num_entities"]).item())
            self.num_relations = int(np.asarray(data["num_relations"]).item())
            if self.num_entities <= 0 or self.num_relations <= 0:
                raise ValueError(
                    f"Invalid vocabulary sizes: entities={self.num_entities}, "
                    f"relations={self.num_relations}"
                )

            self.entity_vocab = np.asarray(data["entity_vocab"]).astype(str)
            self.relation_vocab = np.asarray(data["relation_vocab"]).astype(str)
            if len(self.entity_vocab) != self.num_entities:
                raise ValueError(
                    "entity_vocab length does not match num_entities: "
                    f"{len(self.entity_vocab)} != {self.num_entities}"
                )
            if len(self.relation_vocab) != self.num_relations:
                raise ValueError(
                    "relation_vocab length does not match num_relations: "
                    f"{len(self.relation_vocab)} != {self.num_relations}"
                )

            self._train_pos = _validate_triples(
                train_key,
                data[train_key],
                num_entities=self.num_entities,
                num_relations=self.num_relations,
            )
            self.val_pos = _validate_triples(
                "val_pos",
                data["val_pos"],
                num_entities=self.num_entities,
                num_relations=self.num_relations,
            )
            self.test_pos = _validate_triples(
                "test_pos",
                data["test_pos"],
                num_entities=self.num_entities,
                num_relations=self.num_relations,
            )

            self.num_base_relations = (
                int(np.asarray(data["num_base_relations"]).item())
                if "num_base_relations" in data.files
                else None
            )
            self.base_relation_ids = (
                np.asarray(data["base_relation_ids"], dtype=np.int32)
                if "base_relation_ids" in data.files
                else None
            )
            self.format_version = (
                str(np.asarray(data["format_version"]).item())
                if "format_version" in data.files
                else "legacy"
            )

    def get_train_graph(self, device=None) -> tuple[Tensor, Tensor]:
        """Return ``edge_index`` and ``edge_type`` tensors for RGCN propagation."""
        train = torch.from_numpy(self._train_pos).long()
        edge_index = torch.stack((train[:, 0], train[:, 2]), dim=0)
        edge_type = train[:, 1]
        if device is not None:
            edge_index = edge_index.to(device)
            edge_type = edge_type.to(device)
        return edge_index, edge_type

    @property
    def train_pos(self) -> np.ndarray:
        return self._train_pos

    def __repr__(self) -> str:
        return (
            f"WordNetLPDataset(variant={self.variant}, "
            f"num_entities={self.num_entities}, "
            f"num_relations={self.num_relations}, "
            f"train={len(self._train_pos)}, "
            f"val={len(self.val_pos)}, "
            f"test={len(self.test_pos)}, "
            f"format={self.format_version})"
        )
