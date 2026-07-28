#!/usr/bin/env python3
"""Prepare CMPNN-aligned IMDb md/ml graphs shared by SlotGAT and RGCN."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


SCRIPT_DIR = Path(__file__).resolve().parent
SLOTGAT_ROOT = SCRIPT_DIR.parent
REPOSITORY_ROOT = SLOTGAT_ROOT.parents[2]
RGCN_ROOT = SLOTGAT_ROOT.parent / "RGCN"
CMPNN_ROOT = SLOTGAT_ROOT.parent / "CMPNN"


def option_present(name: str) -> bool:
    return any(
        argument == name or argument.startswith(name + "=")
        for argument in sys.argv[1:]
    )


def option_value(name: str) -> Optional[str]:
    for index, argument in enumerate(sys.argv[1:], start=1):
        if argument.startswith(name + "="):
            return argument.split("=", 1)[1]
        if argument == name and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
    return None


def add_default(option: str, value: str) -> None:
    if not option_present(option):
        sys.argv.extend((option, value))


def main() -> None:
    task = option_value("--task")
    if task is not None:
        task = task.strip().lower()
        if task not in {"md", "ml"}:
            raise SystemExit("SlotGAT IMDb-LP currently supports task md or ml")
        add_default(
            "--variant",
            "v1,v3" if task == "md" else "v1,v2,v3,v4",
        )
        add_default(
            "--shared-npz",
            str(CMPNN_ROOT / f"IMDB_{task}_shared_splits.npz"),
        )

    add_default(
        "--csv",
        str(
            REPOSITORY_ROOT
            / "data"
            / "raw"
            / "IMDB"
            / "movie_metadata.csv"
        ),
    )
    add_default("--out-dir", str(RGCN_ROOT / "data" / "preprocessed"))

    sys.path.insert(0, str(RGCN_ROOT))
    from preprocess_IMDB_rgcn_lp import main as shared_main

    shared_main()


if __name__ == "__main__":
    main()
