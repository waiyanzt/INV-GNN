#!/usr/bin/env python3
"""SlotGAT entry point for joint WordNet graph-variant augmentation.

The model-independent training/evaluation protocol is shared with the RGCN
runner so splits, negative sampling, checkpoint selection, exact resume, and
reported invariance metrics cannot drift between the two baselines.
"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SLOTGAT_ROOT = SCRIPT_DIR.parent
REPOSITORY_ROOT = SLOTGAT_ROOT.parents[2]
RGCN_AUGMENTATION_DIR = (
    SLOTGAT_ROOT.parent / "RGCN" / "rgcn_data_augmentation"
)


def has_option(*names: str) -> bool:
    return any(
        argument == name or argument.startswith(name + "=")
        for argument in sys.argv[1:]
        for name in names
    )


def add_default(option: str, value: str, *aliases: str) -> None:
    if not has_option(option, *aliases):
        sys.argv.extend((option, value))


def main() -> None:
    sys.path.insert(0, str(RGCN_AUGMENTATION_DIR))

    default_data_root = REPOSITORY_ROOT / "data" / "wordnet_3hops_augmented_full"
    raw_data_root = REPOSITORY_ROOT / "data" / "raw" / (
        "wordnet_3hops_augmented_full"
    )
    if not default_data_root.exists() and raw_data_root.exists():
        default_data_root = raw_data_root

    add_default("--encoder", "slotgat")
    add_default(
        "--variants",
        "no_changes,universal_edges,all_inverse_edges",
    )
    add_default("--candidate-known-scope", "selected")
    add_default("--data-root", str(default_data_root))
    add_default(
        "--output-dir",
        str(SLOTGAT_ROOT / "results" / "slotgat_augmentation" / "WORDNET"),
    )

    # Architecture/optimizer defaults are inherited from the supplied SlotGAT
    # Freebase runner. WordNet-specific evaluation and training scheduling
    # defaults remain owned by the shared protocol.
    add_default("--hidden-dim", "64", "--hidden_dim")
    add_default("--lr", "0.005")
    add_default("--weight-decay", "0.001", "--weight_decay")

    from run_WORDNET_rgcn_augmentation import main as shared_main

    shared_main()


if __name__ == "__main__":
    main()
