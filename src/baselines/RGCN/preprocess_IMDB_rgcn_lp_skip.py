#!/usr/bin/env python3
"""
  python preprocess_IMDB_rgcn_lp_skip.py --task md --variant v1,v3 \\
      --shared-npz ../CMPNN/IMDB_md_shared_splits.npz
  python preprocess_IMDB_rgcn_lp_skip.py --task ml --variant v1,v2,v3,v4 \\
      --shared-npz ../CMPNN/IMDB_ml_shared_splits.npz
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import preprocess_IMDB_rgcn_lp as rgcn_lp

SEED = 1566911444


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _parse_task(s: str) -> str:
    t = str(s).strip().lower()
    if t not in {"md", "mg", "ml"}:
        raise SystemExit("task must be one of: md, mg, ml")
    return t


def _parse_variants(s: str, task: str) -> list[str]:
    """Output folder names only; graph is shared (see module docstring)."""
    vals = [x.strip().lower() for x in str(s).split(",") if x.strip()]
    if task == "md":
        good = {"v1", "v3"}
    else:
        good = {"v1", "v2", "v3", "v4"}
    bad = [v for v in vals if v not in good]
    if bad:
        raise SystemExit(f"Unknown variant(s) for task={task}: {bad}; expected subset of {sorted(good)}")
    return vals


def _copy_graph_data(gd: dict) -> dict:
    return {k: (np.asarray(u, dtype=np.int64).copy(), np.asarray(v, dtype=np.int64).copy()) for k, (u, v) in gd.items()}


def build_universal_lp_graph(
    movies: pd.DataFrame,
    task: str,
    train_pos: np.ndarray,
    val_pos: np.ndarray,
    test_pos: np.ndarray,
    neg_k: int,
    seed: int,
) -> tuple[dict, dict[str, np.ndarray], dict]:
    """
    One heterograph per task: movie/actor/director/link (+ genre for mg) with universal
    skip (link hub) edges; LP supervision edges as above.
    """
    raw: dict[str, list] = {}

    if task == "md":
        directors = sorted(set(movies["director_name"].dropna()))
        actors = sorted(
            set(
                movies["actor_1_name"].dropna().tolist()
                + movies["actor_2_name"].dropna().tolist()
                + movies["actor_3_name"].dropna().tolist()
            )
        )
        M = len(movies)
        Dn = len(directors)
        An = len(actors)
        dmap = {n: i for i, n in enumerate(directors)}
        amap = {n: i for i, n in enumerate(actors)}
        train_md = {(int(r[0]), int(r[1])) for r in train_pos}

        for mi, row in movies.iterrows():
            mi = int(mi)
            li = mi
            for acol in ("actor_1_name", "actor_2_name", "actor_3_name"):
                if pd.notna(row[acol]) and row[acol] in amap:
                    rgcn_lp._add_pair(raw, "movie-actor", np.array([mi]), np.array([amap[row[acol]]]))
            rgcn_lp._add_pair(raw, "movie-link", np.array([mi]), np.array([li]))
            d_name = row["director_name"]
            if pd.notna(d_name) and d_name in dmap:
                # Director ↔ movie only via train ``movie-director`` below; link hub gives L–D.
                rgcn_lp._add_pair(raw, "link-director", np.array([li]), np.array([dmap[d_name]]))
            for acol in ("actor_1_name", "actor_2_name", "actor_3_name"):
                if pd.notna(row[acol]) and row[acol] in amap:
                    rgcn_lp._add_pair(raw, "link-actor", np.array([li]), np.array([amap[row[acol]]]))

        for m, d in train_md:
            rgcn_lp._add_pair(raw, "movie-director", np.array([m]), np.array([d]))

        num_nodes = {"movie": M, "actor": An, "director": Dn, "link": M}
        rng = np.random.RandomState(seed)
        rng2 = np.random.RandomState(seed + 7)
        splits: dict[str, np.ndarray] = {
            "train_pos": train_pos.astype(np.int64),
            "val_pos": val_pos.astype(np.int64),
            "test_pos": test_pos.astype(np.int64),
            "train_neg": rgcn_lp.sample_negs_md(train_pos, Dn, neg_k, rng),
            "val_neg": rgcn_lp.sample_negs_md(val_pos, Dn, neg_k, rng2),
            "test_neg": rgcn_lp.sample_negs_md(test_pos, Dn, neg_k, rng2),
        }
        kendall_keys = ["movie_local", "director_local"]

    elif task == "mg":
        directors, actors = rgcn_lp._directors_actors_mg(movies)
        M = len(movies)
        Dn = len(directors)
        An = len(actors)
        Gn = rgcn_lp.NUM_GENRES
        dmap = {n: i for i, n in enumerate(directors)}
        amap = {n: i for i, n in enumerate(actors)}
        train_mg = {(int(r[0]), int(r[1])) for r in train_pos}

        for mi, row in movies.iterrows():
            mi = int(mi)
            li = mi
            rgcn_lp._add_pair(raw, "movie-director", np.array([mi]), np.array([dmap[row["director_name"]]]))
            for acol in ("actor_1_name", "actor_2_name", "actor_3_name"):
                if pd.notna(row[acol]) and row[acol] in amap:
                    rgcn_lp._add_pair(raw, "movie-actor", np.array([mi]), np.array([amap[row[acol]]]))
            rgcn_lp._add_pair(raw, "movie-link", np.array([mi]), np.array([li]))
            d_name = row["director_name"]
            if pd.notna(d_name) and d_name in dmap:
                rgcn_lp._add_pair(raw, "link-director", np.array([li]), np.array([dmap[d_name]]))
            for acol in ("actor_1_name", "actor_2_name", "actor_3_name"):
                if pd.notna(row[acol]) and row[acol] in amap:
                    rgcn_lp._add_pair(raw, "link-actor", np.array([li]), np.array([amap[row[acol]]]))

        for m, g in train_mg:
            rgcn_lp._add_pair(raw, "movie-genre", np.array([m]), np.array([g]))

        num_nodes = {"movie": M, "actor": An, "director": Dn, "link": M, "genre": Gn}
        rng = np.random.RandomState(seed)
        rng2 = np.random.RandomState(seed + 7)
        splits = {
            "train_pos": train_pos.astype(np.int64),
            "val_pos": val_pos.astype(np.int64),
            "test_pos": test_pos.astype(np.int64),
            "train_neg": rgcn_lp.sample_negs_mg(train_pos, neg_k, rng),
            "val_neg": rgcn_lp.sample_negs_mg(val_pos, neg_k, rng2),
            "test_neg": rgcn_lp.sample_negs_mg(test_pos, neg_k, rng2),
        }
        kendall_keys = ["movie_local", "genre_id"]

    else:  # ml
        directors, actors = rgcn_lp._directors_actors_ml(movies)
        M = len(movies)
        Dn = len(directors)
        An = len(actors)
        dmap = {n: i for i, n in enumerate(directors)}
        amap = {n: i for i, n in enumerate(actors)}
        train_ml = {(int(r[0]), int(r[1])) for r in train_pos}
        train_movies = {int(r[0]) for r in train_pos}

        for i, row in movies.iterrows():
            i = int(i)
            if i not in train_movies:
                continue
            li = i
            rgcn_lp._add_pair(raw, "movie-director", np.array([i]), np.array([dmap[row["director_name"]]]))
            rgcn_lp._add_pair(raw, "movie-actor", np.array([i]), np.array([amap[row["actor_1_name"]]]))
            d_name = row["director_name"]
            if pd.notna(d_name) and d_name in dmap:
                rgcn_lp._add_pair(raw, "link-director", np.array([li]), np.array([dmap[d_name]]))
            a1 = row["actor_1_name"]
            if pd.notna(a1) and a1 in amap:
                rgcn_lp._add_pair(raw, "link-actor", np.array([li]), np.array([amap[a1]]))

        for m, l in train_ml:
            rgcn_lp._add_pair(raw, "movie-link", np.array([m]), np.array([l]))

        num_nodes = {"movie": M, "actor": An, "director": Dn, "link": M}
        all_true = (
            set(map(tuple, train_pos.tolist()))
            | set(map(tuple, val_pos.tolist()))
            | set(map(tuple, test_pos.tolist()))
        )
        k = min(int(neg_k), max(1, len(movies) - 1))
        rng = np.random.RandomState(seed)
        rng2 = np.random.RandomState(seed + 7)
        splits = {
            "train_pos": train_pos.astype(np.int64),
            "val_pos": val_pos.astype(np.int64),
            "test_pos": test_pos.astype(np.int64),
            "train_neg": rgcn_lp.sample_negs_ml(train_pos, len(movies), all_true, k, rng),
            "val_neg": rgcn_lp.sample_negs_ml(val_pos, len(movies), all_true, k, rng2),
            "test_neg": rgcn_lp.sample_negs_ml(test_pos, len(movies), all_true, k, rng2),
        }
        kendall_keys = ["movie_local", "link_local"]

    graph_data = rgcn_lp._finalize_pairs(raw)

    meta = {
        "task": task,
        "variant": "universal",
        "num_nodes": num_nodes,
        "kendall_keys": kendall_keys,
        "neg_k": int(neg_k) if task != "ml" else int(k),
        "skip_lp": True,
        "skip_lp_graph": "universal_link_hub",
    }
    return graph_data, splits, meta


def save_variant_copies(
    graph_data: dict,
    splits: dict[str, np.ndarray],
    meta_base: dict,
    task: str,
    variants: list[str],
    out_root: Path,
) -> None:
    """Write identical graph + splits under each ``IMDB_rgcn_lp_skip_{task}_{v}/``."""
    for v in variants:
        out_dir = out_root / f"IMDB_rgcn_lp_skip_{task}_{v}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        meta = {**meta_base, "variant": v}
        meta["splits"] = {k: torch.tensor(splits[k], dtype=torch.long) for k in splits}

        torch.save(_copy_graph_data(graph_data), out_dir / "graph_data.pt")
        torch.save(meta, out_dir / "meta.pt")
        print(f"Saved {out_dir} (universal graph copy for variant={v})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="IMDB RGCN LP — one universal graph per task, copied to each variant dir.")
    ap.add_argument("--task", type=_parse_task, required=True)
    ap.add_argument("--variant", default="v1", help="md: v1,v3; mg/ml: v1–v4 (folder names only).")
    ap.add_argument("--csv", default="data/raw/IMDB/movie_metadata.csv")
    ap.add_argument(
        "--shared-npz",
        default="",
        help="Path to shared splits npz (same as non-skip LP).",
    )
    ap.add_argument("--out-dir", default="data/preprocessed")
    ap.add_argument("--neg-k", type=int, default=19, help="Negatives per positive (MD/ML); MG typically 2.")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    task = args.task
    variants = _parse_variants(args.variant, task)
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"Missing csv: {csv_path}")

    defaults = {
        "md": "../CMPNN/IMDB_md_shared_splits.npz",
        "mg": "../CMPNN/IMDB_mg_shared_splits.npz",
        "ml": "../CMPNN/IMDB_ml_shared_splits.npz",
    }
    shared = Path(args.shared_npz) if args.shared_npz.strip() else Path(defaults[task])
    if not shared.is_file():
        raise SystemExit(
            f"Missing shared npz: {shared}. Pass --shared-npz or place file next to CMPNN build outputs."
        )

    if task == "mg" and (args.neg_k < 1 or args.neg_k > 2):
        print(f"[warn] mg usually uses neg-k in [1,2] (three genres); got {args.neg_k}", flush=True)

    set_seed(args.seed)
    z = np.load(shared)
    train_pos = z["train_pos"].astype(np.int64)
    val_pos = z["val_pos"].astype(np.int64)
    test_pos = z["test_pos"].astype(np.int64)

    if task in ("md", "mg"):
        movies = rgcn_lp.read_imdb_frame_md_mg(str(csv_path))
    else:
        movies = rgcn_lp.read_imdb_frame_ml(str(csv_path))

    print(f"=== IMDB RGCN LP skip (universal) | task={task} variants={variants} ===", flush=True)
    graph_data, splits, meta_base = build_universal_lp_graph(
        movies, task, train_pos, val_pos, test_pos, args.neg_k, args.seed
    )
    save_variant_copies(graph_data, splits, meta_base, task, variants, Path(args.out_dir))


if __name__ == "__main__":
    main()
