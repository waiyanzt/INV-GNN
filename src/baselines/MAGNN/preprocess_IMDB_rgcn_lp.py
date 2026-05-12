#!/usr/bin/env python3
"""
  python preprocess_IMDB_rgcn_lp.py --task md --variant v1 \\
    --shared-npz ../CMPNN/IMDB_md_shared_splits.npz
  python preprocess_IMDB_rgcn_lp.py --task ml --variant v1,v2,v3,v4 \\
    --shared-npz ../CMPNN/IMDB_ml_shared_splits.npz
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SEED = 1566911444
NUM_GENRES = 3


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
    vals = [x.strip().lower() for x in str(s).split(",") if x.strip()]
    if task == "md":
        good = {"v1", "v3"}
    else:
        good = {"v1", "v2", "v3", "v4"}
    bad = [v for v in vals if v not in good]
    if bad:
        raise SystemExit(f"Unknown variant(s) for task={task}: {bad}; expected subset of {sorted(good)}")
    return vals


# ----- Movie tables (match CMPNN) -----

def read_imdb_frame_md_mg(csv_path: str) -> pd.DataFrame:
    movies = (
        pd.read_csv(csv_path, encoding="utf-8")
        .drop_duplicates(subset=["movie_imdb_link"])
        .dropna(axis=0, subset=["actor_1_name", "director_name"])
        .reset_index(drop=True)
    )
    labels = np.full(len(movies), -1, dtype=int)
    for idx, genres in movies["genres"].items():
        for g in str(genres).split("|"):
            if g == "Action":
                labels[idx] = 0
                break
            elif g == "Comedy":
                labels[idx] = 1
                break
            elif g == "Drama":
                labels[idx] = 2
                break
    keep = labels != -1
    movies = movies[keep].reset_index(drop=True)
    return movies


def read_imdb_frame_ml(csv_path: str) -> pd.DataFrame:
    movies = pd.read_csv(csv_path, encoding="utf-8")
    movies = movies.drop_duplicates(subset="movie_imdb_link").dropna(
        subset=["movie_imdb_link", "actor_1_name", "director_name", "genres"]
    ).reset_index(drop=True)
    labels = np.full(len(movies), -1, dtype=np.int64)
    for i, genres in movies["genres"].astype(str).items():
        for g in genres.split("|"):
            g = g.strip()
            if g == "Action":
                labels[i] = 0
                break
            elif g == "Comedy":
                labels[i] = 1
                break
            elif g == "Drama":
                labels[i] = 2
                break
    keep = np.where(labels >= 0)[0]
    return movies.iloc[keep].reset_index(drop=True)


def _directors_actors_mg(movies: pd.DataFrame) -> tuple[list, list]:
    directors = sorted(set(movies["director_name"].dropna()))
    actors = sorted(
        set(
            movies["actor_1_name"].dropna().tolist()
            + movies["actor_2_name"].dropna().tolist()
            + movies["actor_3_name"].dropna().tolist()
        )
    )
    return directors, actors


def _directors_actors_ml(movies: pd.DataFrame) -> tuple[list, list]:
    """CMPNN ML: director + actor_1 only (same as ``run_CMPNN_IMDB_ml``)."""
    directors = sorted(set(movies["director_name"].dropna().tolist()))
    actors = sorted(set(movies["actor_1_name"].dropna().tolist()))
    return directors, actors


# ----- Negative sampling: (N, K) like CMPNN ML -----

def sample_negs_md(pos: np.ndarray, n_directors: int, k: int, rng: np.random.RandomState) -> np.ndarray:
    N = len(pos)
    neg = np.zeros((N, k), dtype=np.int64)
    for i, (_m, d_true) in enumerate(pos):
        cand = [d for d in range(n_directors) if d != int(d_true)]
        if not cand:
            cand = list(range(n_directors))
        neg[i] = rng.choice(cand, size=k, replace=(k > len(cand)))
    return neg


def sample_negs_mg(pos: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
    all_g = np.arange(NUM_GENRES, dtype=np.int64)
    N = len(pos)
    neg = np.zeros((N, k), dtype=np.int64)
    for i, (_m, g_true) in enumerate(pos):
        gt = int(g_true)
        cand = all_g[all_g != gt]
        neg[i] = rng.choice(cand, size=k, replace=(k > len(cand)))
    return neg


def sample_negs_ml(
    pos: np.ndarray, n_movies: int, true_set: set[tuple[int, int]], k: int, rng: np.random.RandomState
) -> np.ndarray:
    L_all = np.arange(n_movies, dtype=np.int64)
    N = len(pos)
    neg = np.zeros((N, k), dtype=np.int64)
    for i, (mL, l_true) in enumerate(pos):
        mL, l_true = int(mL), int(l_true)
        cand = [x for x in L_all if (mL, int(x)) not in true_set]
        if not cand:
            cand = list(L_all)
        neg[i] = rng.choice(cand, size=k, replace=(k > len(cand)))
    return neg


# ----- graph_data builders (local ids per ntype) -----

def _add_pair(d: dict, key: str, u: np.ndarray, v: np.ndarray) -> None:
    if key not in d:
        d[key] = ([], [])
    d[key][0].extend(u.tolist())
    d[key][1].extend(v.tolist())


def _finalize_pairs(d: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for k, (a, b) in d.items():
        if not a:
            continue
        out[k] = (np.array(a, dtype=np.int64), np.array(b, dtype=np.int64))
    return out


def build_md(
    movies: pd.DataFrame, variant: str, train_pos: np.ndarray, neg_k: int, seed: int
) -> tuple[dict, dict, dict[str, np.ndarray]]:
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

    raw: dict[str, list] = {}
    train_md = {(int(r[0]), int(r[1])) for r in train_pos}

    if variant == "v1":
        for mi, row in movies.iterrows():
            mi = int(mi)
            for acol in ("actor_1_name", "actor_2_name", "actor_3_name"):
                if pd.notna(row[acol]) and row[acol] in amap:
                    _add_pair(raw, "movie-actor", np.array([mi]), np.array([amap[row[acol]]]))
            _add_pair(raw, "movie-link", np.array([mi]), np.array([mi]))
        for m, d in train_md:
            _add_pair(raw, "movie-director", np.array([m]), np.array([d]))
        num_nodes = {"movie": M, "actor": An, "director": Dn, "link": M}
    elif variant == "v3":
        for mi, row in movies.iterrows():
            mi = int(mi)
            _add_pair(raw, "movie-link", np.array([mi]), np.array([mi]))
            for acol in ("actor_1_name", "actor_2_name", "actor_3_name"):
                if pd.notna(row[acol]) and row[acol] in amap:
                    _add_pair(raw, "link-actor", np.array([mi]), np.array([amap[row[acol]]]))
        for m, d in train_md:
            _add_pair(raw, "movie-director", np.array([m]), np.array([d]))
        num_nodes = {"movie": M, "actor": An, "director": Dn, "link": M}
    else:
        raise ValueError(variant)

    graph_data = _finalize_pairs(raw)
    rng = np.random.RandomState(seed)
    splits = {
        "train_pos": train_pos.astype(np.int64),
        "val_pos": None,  # filled below
        "test_pos": None,
        "train_neg": sample_negs_md(train_pos, Dn, neg_k, rng),
        "val_neg": None,
        "test_neg": None,
    }
    meta = {
        "task": "md",
        "variant": variant,
        "num_nodes": num_nodes,
        "kendall_keys": ["movie_local", "director_local"],
        "neg_k": int(neg_k),
    }
    return graph_data, meta, splits


def build_mg(
    movies: pd.DataFrame, variant: str, train_pos: np.ndarray, neg_k: int, seed: int
) -> tuple[dict, dict, dict[str, np.ndarray]]:
    directors, actors = _directors_actors_mg(movies)
    M = len(movies)
    Dn = len(directors)
    An = len(actors)
    Gn = NUM_GENRES
    dmap = {n: i for i, n in enumerate(directors)}
    amap = {n: i for i, n in enumerate(actors)}

    raw: dict[str, list] = {}
    train_mg = {(int(r[0]), int(r[1])) for r in train_pos}

    if variant == "v1":
        for mi, row in movies.iterrows():
            mi = int(mi)
            _add_pair(raw, "movie-director", np.array([mi]), np.array([dmap[row["director_name"]]]))
            for acol in ("actor_1_name", "actor_2_name", "actor_3_name"):
                if pd.notna(row[acol]) and row[acol] in amap:
                    _add_pair(raw, "movie-actor", np.array([mi]), np.array([amap[row[acol]]]))
            _add_pair(raw, "movie-link", np.array([mi]), np.array([mi]))
        for m, g in train_mg:
            _add_pair(raw, "movie-genre", np.array([m]), np.array([g]))
    elif variant == "v2":
        for mi, row in movies.iterrows():
            mi = int(mi)
            li = mi
            _add_pair(raw, "link-director", np.array([li]), np.array([dmap[row["director_name"]]]))
            for acol in ("actor_1_name", "actor_2_name", "actor_3_name"):
                if pd.notna(row[acol]) and row[acol] in amap:
                    _add_pair(raw, "link-actor", np.array([li]), np.array([amap[row[acol]]]))
            _add_pair(raw, "movie-link", np.array([mi]), np.array([li]))
        for m, g in train_mg:
            _add_pair(raw, "movie-genre", np.array([m]), np.array([g]))
    elif variant == "v3":
        for mi, row in movies.iterrows():
            mi = int(mi)
            li = mi
            _add_pair(raw, "movie-link", np.array([mi]), np.array([li]))
            for acol in ("actor_1_name", "actor_2_name", "actor_3_name"):
                if pd.notna(row[acol]) and row[acol] in amap:
                    _add_pair(raw, "link-actor", np.array([li]), np.array([amap[row[acol]]]))
            _add_pair(raw, "movie-director", np.array([mi]), np.array([dmap[row["director_name"]]]))
        for m, g in train_mg:
            _add_pair(raw, "movie-genre", np.array([m]), np.array([g]))
    elif variant == "v4":
        for mi, row in movies.iterrows():
            mi = int(mi)
            li = mi
            for acol in ("actor_1_name", "actor_2_name", "actor_3_name"):
                if pd.notna(row[acol]) and row[acol] in amap:
                    _add_pair(raw, "movie-actor", np.array([mi]), np.array([amap[row[acol]]]))
            _add_pair(raw, "movie-link", np.array([mi]), np.array([li]))
            _add_pair(raw, "link-director", np.array([li]), np.array([dmap[row["director_name"]]]))
        for m, g in train_mg:
            _add_pair(raw, "movie-genre", np.array([m]), np.array([g]))
    else:
        raise ValueError(variant)

    graph_data = _finalize_pairs(raw)
    num_nodes = {"movie": M, "actor": An, "director": Dn, "link": M, "genre": Gn}
    rng = np.random.RandomState(seed)
    splits = {
        "train_pos": train_pos.astype(np.int64),
        "val_pos": None,
        "test_pos": None,
        "train_neg": sample_negs_mg(train_pos, neg_k, rng),
        "val_neg": None,
        "test_neg": None,
    }
    meta = {
        "task": "mg",
        "variant": variant,
        "num_nodes": num_nodes,
        "kendall_keys": ["movie_local", "genre_id"],
        "neg_k": int(neg_k),
    }
    return graph_data, meta, splits


def build_ml(
    movies: pd.DataFrame, variant: str, train_pos: np.ndarray, neg_k: int, seed: int
) -> tuple[dict, dict, dict[str, np.ndarray]]:
    directors, actors = _directors_actors_ml(movies)
    M = len(movies)
    Dn = len(directors)
    An = len(actors)
    dmap = {n: i for i, n in enumerate(directors)}
    amap = {n: i for i, n in enumerate(actors)}

    raw: dict[str, list] = {}
    train_ml = {(int(r[0]), int(r[1])) for r in train_pos}

    if variant == "v1":
        for i, row in movies.iterrows():
            i = int(i)
            _add_pair(raw, "movie-director", np.array([i]), np.array([dmap[row["director_name"]]]))
            _add_pair(raw, "movie-actor", np.array([i]), np.array([amap[row["actor_1_name"]]]))
            _add_pair(raw, "movie-link", np.array([i]), np.array([i]))
        for m, l in train_ml:
            _add_pair(raw, "movie-link", np.array([m]), np.array([l]))
    elif variant == "v2":
        for i, row in movies.iterrows():
            i = int(i)
            li = i
            _add_pair(raw, "link-director", np.array([li]), np.array([dmap[row["director_name"]]]))
            _add_pair(raw, "link-actor", np.array([li]), np.array([amap[row["actor_1_name"]]]))
            _add_pair(raw, "movie-link", np.array([i]), np.array([li]))
        for m, l in train_ml:
            _add_pair(raw, "movie-link", np.array([m]), np.array([l]))
    elif variant == "v3":
        for i, row in movies.iterrows():
            i = int(i)
            li = i
            _add_pair(raw, "movie-link", np.array([i]), np.array([li]))
            _add_pair(raw, "link-actor", np.array([li]), np.array([amap[row["actor_1_name"]]]))
            _add_pair(raw, "movie-director", np.array([i]), np.array([dmap[row["director_name"]]]))
        for m, l in train_ml:
            _add_pair(raw, "movie-link", np.array([m]), np.array([l]))
    elif variant == "v4":
        for i, row in movies.iterrows():
            i = int(i)
            li = i
            _add_pair(raw, "movie-actor", np.array([i]), np.array([amap[row["actor_1_name"]]]))
            _add_pair(raw, "movie-link", np.array([i]), np.array([li]))
            _add_pair(raw, "link-director", np.array([li]), np.array([dmap[row["director_name"]]]))
        for m, l in train_ml:
            _add_pair(raw, "movie-link", np.array([m]), np.array([l]))
    else:
        raise ValueError(variant)

    graph_data = _finalize_pairs(raw)
    num_nodes = {"movie": M, "actor": An, "director": Dn, "link": M}
    rng = np.random.RandomState(seed)
    splits = {
        "train_pos": train_pos.astype(np.int64),
        "val_pos": None,
        "test_pos": None,
        "train_neg": None,
        "val_neg": None,
        "test_neg": None,
    }
    meta = {
        "task": "ml",
        "variant": variant,
        "num_nodes": num_nodes,
        "kendall_keys": ["movie_local", "link_local"],
        "neg_k": int(neg_k),
    }
    return graph_data, meta, splits


def preprocess_one(
    csv_path: Path,
    shared_npz: Path,
    task: str,
    variant: str,
    out_root: Path,
    neg_k: int,
    seed: int,
) -> None:
    out_dir = out_root / f"IMDB_rgcn_lp_{task}_{variant}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== IMDB RGCN LP preprocess | task={task} variant={variant} ===", flush=True)
    z = np.load(shared_npz)
    train_pos = z["train_pos"].astype(np.int64)
    val_pos = z["val_pos"].astype(np.int64)
    test_pos = z["test_pos"].astype(np.int64)

    if task in ("md", "mg"):
        movies = read_imdb_frame_md_mg(str(csv_path))
    else:
        movies = read_imdb_frame_ml(str(csv_path))

    if task == "md":
        graph_data, meta, splits = build_md(movies, variant, train_pos, neg_k, seed)
    elif task == "mg":
        graph_data, meta, splits = build_mg(movies, variant, train_pos, neg_k, seed)
    else:
        graph_data, meta, splits = build_ml(movies, variant, train_pos, neg_k, seed)
        all_true = (
            set(map(tuple, train_pos.tolist()))
            | set(map(tuple, val_pos.tolist()))
            | set(map(tuple, test_pos.tolist()))
        )
        rng = np.random.RandomState(seed)
        k = min(int(neg_k), max(1, len(movies) - 1))
        splits["train_neg"] = sample_negs_ml(train_pos, len(movies), all_true, k, rng)
        splits["val_neg"] = sample_negs_ml(val_pos, len(movies), all_true, k, rng)
        splits["test_neg"] = sample_negs_ml(test_pos, len(movies), all_true, k, rng)
        meta["neg_k"] = k

    splits["val_pos"] = val_pos
    splits["test_pos"] = test_pos
    rng = np.random.RandomState(seed + 7)
    if task == "md":
        Dn = meta["num_nodes"]["director"]
        splits["val_neg"] = sample_negs_md(val_pos, Dn, neg_k, rng)
        splits["test_neg"] = sample_negs_md(test_pos, Dn, neg_k, rng)
    elif task == "mg":
        splits["val_neg"] = sample_negs_mg(val_pos, neg_k, rng)
        splits["test_neg"] = sample_negs_mg(test_pos, neg_k, rng)

    meta["splits"] = {k: torch.tensor(v, dtype=torch.long) for k, v in splits.items() if v is not None}

    torch.save(graph_data, out_dir / "graph_data.pt")
    torch.save(meta, out_dir / "meta.pt")
    print(f"Saved {out_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="IMDB RGCN link prediction preprocessing (CMPNN-aligned).")
    ap.add_argument("--task", type=_parse_task, required=True)
    ap.add_argument("--variant", default="v1", help="Comma-separated; md allows v1,v3; mg/ml allow v1-v4.")
    ap.add_argument("--csv", default="data/raw/IMDB/movie_metadata.csv")
    ap.add_argument(
        "--shared-npz",
        default="",
        help="Path to shared splits npz (e.g. ../CMPNN/IMDB_md_shared_splits.npz).",
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
    out_root = Path(args.out_dir)
    for v in variants:
        preprocess_one(csv_path, shared, task, v, out_root, args.neg_k, args.seed)


if __name__ == "__main__":
    main()
