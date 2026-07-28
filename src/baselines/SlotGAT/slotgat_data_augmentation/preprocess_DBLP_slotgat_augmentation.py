#!/usr/bin/env python3
"""Prepare shared DBLP1--3 graph artifacts for SlotGAT augmentation."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SLOTGAT_ROOT = SCRIPT_DIR.parent
REPOSITORY_ROOT = SLOTGAT_ROOT.parents[2]
RGCN_ROOT = SLOTGAT_ROOT.parent / "RGCN"
MAGNN_ROOT = SLOTGAT_ROOT.parent / "MAGNN"


def option_present(name: str) -> bool:
    return any(
        argument == name or argument.startswith(name + "=")
        for argument in sys.argv[1:]
    )


def add_default(option: str, value: str) -> None:
    if not option_present(option):
        sys.argv.extend((option, value))


def main() -> None:
    rgcn_splits = (
        RGCN_ROOT
        / "data"
        / "preprocessed"
        / "DBLP_shared_splits"
        / "DBLP_pc_shared_splits.npz"
    )
    magnn_splits = (
        MAGNN_ROOT
        / "data"
        / "preprocessed"
        / "DBLP_shared_splits"
        / "DBLP_pc_shared_splits.npz"
    )
    shared_splits = (
        rgcn_splits if rgcn_splits.is_file() else magnn_splits
    )

    add_default("--variant", "v1,v2,v3")
    add_default(
        "--raw-dir",
        str(REPOSITORY_ROOT / "data" / "raw" / "DBLP"),
    )
    add_default("--shared-npz", str(shared_splits))
    add_default("--min-conf", "0")
    add_default("--out-dir", str(RGCN_ROOT / "data" / "preprocessed"))

    sys.path.insert(0, str(RGCN_ROOT))
    from preprocess_DBLP_rgcn import main as shared_main

    shared_main()


if __name__ == "__main__":
    main()
