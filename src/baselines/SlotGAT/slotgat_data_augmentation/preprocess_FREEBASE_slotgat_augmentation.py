#!/usr/bin/env python3
"""Prepare globally aligned Freebase variants for joint SlotGAT training."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SLOTGAT_ROOT = SCRIPT_DIR.parent
REPOSITORY_ROOT = SLOTGAT_ROOT.parents[2]
RGCN_AUGMENTATION_DIR = (
    SLOTGAT_ROOT.parent / "RGCN" / "rgcn_data_augmentation"
)


def has_option(name: str) -> bool:
    return any(
        argument == name or argument.startswith(name + "=")
        for argument in sys.argv[1:]
    )


def add_default(option: str, *values: str) -> None:
    if not has_option(option):
        sys.argv.extend((option, *values))


def main() -> None:
    sys.path.insert(0, str(RGCN_AUGMENTATION_DIR))
    add_default(
        "--data-root",
        str(REPOSITORY_ROOT / "data" / "raw" / "dataset_variant_3hops_filter"),
    )
    add_default(
        "--output-root",
        str(REPOSITORY_ROOT / "data" / "slotgat_augmentation" / "freebase"),
    )
    add_default("--variants", "unchanged", "exact_2")

    from preprocess_FREEBASE_rgcn_augmentation import main as shared_main

    shared_main()


if __name__ == "__main__":
    main()
