#!/usr/bin/env python3
"""Create the shared leakage-free WordNet augmentation NPZ for SlotGAT."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SLOTGAT_ROOT = SCRIPT_DIR.parent
REPOSITORY_ROOT = SLOTGAT_ROOT.parents[2]
RGCN_AUGMENTATION_DIR = (
    SLOTGAT_ROOT.parent / "RGCN" / "rgcn_data_augmentation"
)


def option_present(name: str) -> bool:
    return any(
        argument == name or argument.startswith(name + "=")
        for argument in sys.argv[1:]
    )


def main() -> None:
    sys.path.insert(0, str(RGCN_AUGMENTATION_DIR))
    data_dir = REPOSITORY_ROOT / "data" / "wordnet_3hops_augmented_full"
    raw_data_dir = REPOSITORY_ROOT / "data" / "raw" / (
        "wordnet_3hops_augmented_full"
    )
    if not data_dir.exists() and raw_data_dir.exists():
        data_dir = raw_data_dir

    if not option_present("--data-dir"):
        sys.argv.extend(("--data-dir", str(data_dir)))

    from preprocess_WORDNET_rgcn_augmentation import main as shared_main

    shared_main()


if __name__ == "__main__":
    main()
