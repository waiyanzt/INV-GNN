#!/usr/bin/env python3
"""SlotGAT entry point for joint DBLP paper-conference prediction."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SLOTGAT_ROOT = SCRIPT_DIR.parent
RGCN_ROOT = SLOTGAT_ROOT.parent / "RGCN"
RGCN_AUGMENTATION_DIR = RGCN_ROOT / "rgcn_data_augmentation"


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
    add_default("--variants", "v1,v2,v3")
    add_default("--data-root", str(RGCN_ROOT / "data" / "preprocessed"))
    add_default(
        "--output-dir",
        str(
            SLOTGAT_ROOT
            / "results"
            / "slotgat_augmentation"
            / "DBLP"
        ),
    )
    add_default("--super-epochs", "300")
    add_default("--patience", "40")
    add_default("--batch-size", "0")
    add_default("--neg-per-paper", "3")
    add_default("--hidden-dim", "64")
    add_default("--num-layers", "2")
    add_default("--num-heads", "8")
    add_default("--edge-feats", "64")
    add_default("--dropout-feat", "0.5")
    add_default("--dropout-attn", "0.2")
    add_default("--slope", "0.05")
    add_default("--alpha", "0.05")
    add_default("--aggregator", "SA")
    add_default("--sa-att-dim", "3")
    add_default("--slotgat-edge-chunk-size", "0")
    add_default("--slotgat-decomposed-layers", "1")
    add_default("--lr", "0.005")
    add_default("--weight-decay", "0.001")
    add_default("--grad-clip", "0")

    from run_DBLP_rgcn_augmentation import main as shared_main

    shared_main()


if __name__ == "__main__":
    main()
