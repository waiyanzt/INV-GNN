#!/usr/bin/env python3
"""
IMDB CMPNN link prediction — **skip** for **md** and **ml** only.

  python preprocess_IMDB_cmpnn_lp_skip.py --task md --variant v1,v3 \\
      --from-rgcn-root ../MAGNN/data/preprocessed
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

# RGCN LP skip (single source of truth for skip topology + neg sampling)
_MAGNN = Path(__file__).resolve().parent.parent / "MAGNN"
if str(_MAGNN) not in sys.path:
    sys.path.insert(0, str(_MAGNN))

import preprocess_IMDB_rgcn_lp as rgcn_lp  # noqa: E402
from preprocess_IMDB_rgcn_lp_skip import build_universal_lp_graph  # noqa: E402

SEED = 1566911444


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _parse_task(s: str) -> str:
    t = str(s).strip().lower()
    if t not in {"md", "ml"}:
        raise SystemExit("task must be md or ml")
    return t


def _parse_variants(s: str, task: str) -> list[str]:
    vals = [x.strip().lower() for x in str(s).split(",") if x.strip()]
    good = {"v1", "v3"} if task == "md" else {"v1", "v2", "v3", "v4"}
    bad = [v for v in vals if v not in good]
    if bad:
        raise SystemExit(f"bad variant(s) {bad}; expected subset of {sorted(good)}")
    return vals


def rgcn_skip_graph_to_cmpnn_triplets(
    graph_data: dict, task: str, num_nodes: dict
) -> np.ndarray:
    """
    Map RGCN ``graph_data`` (local type ids per edge kind) to CMPNN directed triplets.

    Global indexing (same as legacy CMPNN skip): movie 0..M-1, director off_d..,
    actor off_a.., link off_l..off_l+M-1.
    """
    M = int(num_nodes["movie"])
    Dn = int(num_nodes["director"])
    An = int(num_nodes["actor"])
    off_d = M
    off_a = M + Dn
    off_l = M + Dn + An

    if task == "md":
        REL_MA, REL_ML, REL_LD, REL_LA, REL_MD = 0, 1, 2, 3, 4
    else:
        REL_MD, REL_MA, REL_ML, REL_LD, REL_LA = 0, 1, 2, 3, 4

    rows: list[tuple[int, int, int]] = []

    if "movie-actor" in graph_data:
        ua, va = graph_data["movie-actor"]
        for i in range(len(ua)):
            rows.append((int(ua[i]), off_a + int(va[i]), REL_MA))

    if "movie-link" in graph_data:
        um, vm = graph_data["movie-link"]
        for i in range(len(um)):
            rows.append((int(um[i]), off_l + int(vm[i]), REL_ML))

    if "link-director" in graph_data:
        ul, vd = graph_data["link-director"]
        for i in range(len(ul)):
            rows.append((off_l + int(ul[i]), off_d + int(vd[i]), REL_LD))

    if "link-actor" in graph_data:
        ul, va = graph_data["link-actor"]
        for i in range(len(ul)):
            rows.append((off_l + int(ul[i]), off_a + int(va[i]), REL_LA))

    if "movie-director" in graph_data:
        um, vd = graph_data["movie-director"]
        for i in range(len(um)):
            rows.append((int(um[i]), off_d + int(vd[i]), REL_MD))

    if not rows:
        return np.zeros((0, 3), dtype=np.int64)
    arr = np.asarray(rows, dtype=np.int64)
    return np.unique(arr, axis=0)


def cmpnn_meta_from_rgcn_skip(
    task: str, meta_rgcn: dict, num_triplets: int, num_nodes: dict
) -> dict:
    M = int(num_nodes["movie"])
    Dn = int(num_nodes["director"])
    An = int(num_nodes["actor"])
    off_d = M
    off_a = M + Dn
    off_l = M + Dn + An
    N = off_l + M

    base = {
        "neg_k": int(meta_rgcn["neg_k"]),
        "skip_lp": True,
        "kendall_keys": list(meta_rgcn["kendall_keys"]),
        "graph_source": "preprocess_IMDB_rgcn_lp_skip.build_universal_lp_graph",
        "num_nodes_rgcn": dict(num_nodes),
    }

    if task == "md":
        return {
            **base,
            "task": "md",
            "num_entity": int(N),
            "num_relation": 5,
            "rel_md": 4,
            "off_d": int(off_d),
            "sizes": {"M": M, "Dn": Dn, "An": An},
            "counts": {"triplets": int(num_triplets)},
        }

    return {
        **base,
        "task": "ml",
        "num_entity": int(N),
        "num_relation": 5,
        "rel_ml": 2,
        "off_l": int(off_l),
        "counts": {"triplets": int(num_triplets)},
    }


def save_copies(edge_list: torch.Tensor, splits: dict, meta_base: dict, task: str, variants: list[str], out_root: Path):
    meta_tensors = {k: torch.tensor(v, dtype=torch.long) for k, v in splits.items()}
    for v in variants:
        out_dir = out_root / f"IMDB_cmpnn_lp_skip_{task}_{v}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = dict(meta_base)
        meta["variant"] = v
        meta["splits"] = meta_tensors
        torch.save(edge_list.clone(), out_dir / "edge_list.pt")
        torch.save(meta, out_dir / "meta.pt")
        print(f"Wrote {out_dir}", flush=True)


def _torch_load_compat(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_rgcn_skip_preprocessed(rgcn_root: Path, task: str, variant: str) -> tuple[dict, dict[str, np.ndarray], dict]:
    """
    Load ``graph_data.pt`` + ``meta.pt`` from a directory written by
    ``preprocess_IMDB_rgcn_lp_skip.save_variant_copies`` (same layout as
    ``run_IMDB_rgcn_lp_skip.load_preprocessed``).
    """
    d = rgcn_root / f"IMDB_rgcn_lp_skip_{task}_{variant}"
    if not d.is_dir():
        raise SystemExit(f"Missing RGCN skip directory: {d}")
    gd_path = d / "graph_data.pt"
    meta_path = d / "meta.pt"
    if not gd_path.is_file() or not meta_path.is_file():
        raise SystemExit(f"Need graph_data.pt and meta.pt under {d}")
    graph_data = _torch_load_compat(gd_path)
    meta_full = _torch_load_compat(meta_path)
    if meta_full.get("task") != task:
        raise SystemExit(f"{meta_path}: meta.task={meta_full.get('task')!r} != {task!r}")
    splits_raw = meta_full["splits"]
    splits = {
        k: (v.cpu().numpy() if torch.is_tensor(v) else np.asarray(v, dtype=np.int64))
        for k, v in splits_raw.items()
    }
    meta_rgcn = {k: v for k, v in meta_full.items() if k != "splits"}
    return graph_data, splits, meta_rgcn


def main():
    ap = argparse.ArgumentParser(description="IMDB CMPNN universal skip LP preprocess (md, ml), RGCN-aligned graph.")
    ap.add_argument("--task", type=_parse_task, required=True)
    ap.add_argument("--variant", default="", help="Comma list; default md=v1,v3 ml=v1,v2,v3,v4")
    ap.add_argument("--csv", default="../MAGNN/data/raw/IMDB/movie_metadata.csv")
    ap.add_argument("--shared-npz", default="", help="IMDB_md_shared_splits.npz or IMDB_ml_shared_splits.npz")
    ap.add_argument(
        "--from-rgcn-root",
        default="",
        help="Directory containing IMDB_rgcn_lp_skip_{task}_{variant}/ (e.g. ../MAGNN/data/preprocessed). "
        "If set, reads graph_data.pt + meta.pt from RGCN skip preprocess; CSV/npz not used.",
    )
    ap.add_argument("--out-dir", default="data/preprocessed")
    ap.add_argument("--neg-k", type=int, default=19)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    task = args.task
    default_var = "v1,v3" if task == "md" else "v1,v2,v3,v4"
    variants = _parse_variants(args.variant.strip() if args.variant.strip() else default_var, task)
    out_root = Path(args.out_dir)

    if args.from_rgcn_root.strip():
        rgcn_root = Path(args.from_rgcn_root)
        graph_data, splits, meta_rgcn = load_rgcn_skip_preprocessed(rgcn_root, task, variants[0])
        num_nodes = meta_rgcn["num_nodes"]
        for v in variants[1:]:
            _, splits_v, _ = load_rgcn_skip_preprocessed(rgcn_root, task, v)
            for key in splits:
                if splits[key].shape != splits_v[key].shape or not np.array_equal(splits[key], splits_v[key]):
                    raise SystemExit(
                        f"Splits differ between {variants[0]!r} and {v!r} on {key!r}; "
                        "RGCN skip should copy identical splits to each variant folder."
                    )
        arr = rgcn_skip_graph_to_cmpnn_triplets(graph_data, task, num_nodes)
        edge_list = torch.tensor(arr, dtype=torch.long)
        meta = cmpnn_meta_from_rgcn_skip(task, meta_rgcn, len(arr), num_nodes)
        save_copies(edge_list, splits, meta, task, variants, out_root)
        print(
            "Done. CMPNN edge_list converted from RGCN graph_data.pt (same as run_IMDB_rgcn_lp_skip).",
            flush=True,
        )
        return

    defaults = {"md": "IMDB_md_shared_splits.npz", "ml": "IMDB_ml_shared_splits.npz"}
    sp = Path(args.shared_npz.strip() if args.shared_npz.strip() else defaults[task])
    if not sp.is_file():
        raise SystemExit(f"Missing shared npz: {sp}")

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"Missing csv: {csv_path}")

    set_seed(args.seed)
    z = np.load(sp)
    train_pos = z["train_pos"].astype(np.int64)
    val_pos = z["val_pos"].astype(np.int64)
    test_pos = z["test_pos"].astype(np.int64)

    if task == "md":
        movies = rgcn_lp.read_imdb_frame_md_mg(str(csv_path))
    else:
        movies = rgcn_lp.read_imdb_frame_ml(str(csv_path))

    graph_data, splits, meta_rgcn = build_universal_lp_graph(
        movies, task, train_pos, val_pos, test_pos, args.neg_k, args.seed
    )
    num_nodes = meta_rgcn["num_nodes"]
    arr = rgcn_skip_graph_to_cmpnn_triplets(graph_data, task, num_nodes)
    edge_list = torch.tensor(arr, dtype=torch.long)
    meta = cmpnn_meta_from_rgcn_skip(task, meta_rgcn, len(arr), num_nodes)

    save_copies(edge_list, splits, meta, task, variants, out_root)
    print("Done. Rebuilt via build_universal_lp_graph → same topology as IMDB_rgcn_lp_skip.", flush=True)


if __name__ == "__main__":
    main()
