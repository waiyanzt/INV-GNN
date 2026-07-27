#!/usr/bin/env python3
"""SlotGAT entry point for joint Freebase graph-variant augmentation."""

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
    add_default("--encoder", "slotgat")
    add_default("--variants", "unchanged,exact_2")
    add_default(
        "--data-root",
        str(REPOSITORY_ROOT / "data" / "slotgat_augmentation" / "freebase"),
    )
    add_default(
        "--output-dir",
        str(SLOTGAT_ROOT / "results" / "slotgat_augmentation" / "FREEBASE"),
    )
    add_default("--hidden-dim", "64")
    add_default("--lr", "0.005")
    add_default("--weight-decay", "0.001")

    from run_FREEBASE_rgcn_augmentation import main as shared_main

    shared_main()


if __name__ == "__main__":
    main()
