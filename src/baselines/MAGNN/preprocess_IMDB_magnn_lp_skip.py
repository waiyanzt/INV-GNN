#!/usr/bin/env python3
from __future__ import annotations

"""
IMDB MAGNN link prediction preprocessing (SKIP-NODE / universal graph).
"""

import argparse
import os
import pickle
import random
import shutil
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

SEED = 1566911444


# ---------------------------------------------------------------------
# Skip-node metapaths (FULL traversal then COMPRESS)
# ---------------------------------------------------------------------
SKIP_METAPATHS = {
    "v1": [
        [
            [0, 1, 0],
            [0, 3, [0], 1, [0], 3, 0],
            [0, 2, 0],
            [0, 3, [0], 2, [0], 3, 0],
            [0, 3, 0],
        ],
        [
            [1, 0, 1],
            [1, [0], 3, 0, 3, [0], 1],
            [1, 0, 2, 0, 1],
            [1, [0], 3, [0], 2, [0], 3, [0], 1],
            [1, [0], 3, 0, 2, 0, 3, [0], 1],
            [1, 0, 3, [0], 2, [0], 3, 0, 1],
            [1, 0, 3, 0, 1],
            [1, [0], 3, [0], 1],
        ],
        [
            [2, 0, 2],
            [2, [0], 3, 0, 3, [0], 2],
            [2, 0, 1, 0, 2],
            [2, 0, 3, [0], 1, [0], 3, 0, 2],
            [2, [0], 3, 0, 1, 0, 3, [0], 2],
            [2, [0], 3, [0], 1, [0], 3, [0], 2],
            [2, 0, 3, 0, 2],
            [2, [0], 3, [0], 2],
        ],
        [
            [3, 0, 3],
            [3, 0, 1, 0, 3],
            [3, [0], 1, [0], 3],
            [3, 0, 2, 0, 3],
            [3, [0], 2, [0], 3],
        ],
    ],
    "v2": [
        [
            [0, [3], 1, [3], 0],
            [0, 3, 1, 3, 0],
            [0, [3], 2, [3], 0],
            [0, 3, 2, 3, 0],
            [0, 3, 0],
        ],
        [
            [1, [3], 0, [3], 1],
            [1, 3, 0, 3, 1],
            [1, [3], 0, [3], 2, [3], 0, [3], 1],
            [1, 3, 2, 3, 1],
            [1, 3, 0, [3], 2, [3], 0, 3, 1],
            [1, [3], 0, 3, 2, 3, 0, [3], 1],
            [1, [3], 0, 3, 0, [3], 1],
            [1, 3, 1],
        ],
        [
            [2, [3], 0, [3], 2],
            [2, 3, 0, 3, 2],
            [2, [3], 0, [3], 1, [3], 0, [3], 2],
            [2, [3], 0, 3, 1, 3, 0, [3], 2],
            [2, 3, 0, [3], 1, [3], 0, 3, 2],
            [2, 3, 1, 3, 2],
            [2, [3], 0, 3, 0, [3], 2],
            [2, 3, 2],
        ],
        [
            [3, 0, 3],
            [3, 0, [3], 1, [3], 0, 3],
            [3, 1, 3],
            [3, 0, [3], 2, [3], 0, 3],
            [3, 2, 3],
        ],
    ],
    "v3": [
        [
            [0, 1, 0],
            [0, 3, [0], 1, [0], 3, 0],
            [0, [3], 2, [3], 0],
            [0, 3, 2, 3, 0],
            [0, 3, 0],
        ],
        [
            [1, 0, 1],
            [1, [0], 3, 0, 3, [0], 1],
            [1, 0, [3], 2, [3], 0, 1],
            [1, [0], 3, 2, 3, [0], 1],
            [1, [0], 3, 0, [3], 2, [3], 0, 3, [0], 1],
            [1, 0, 3, 2, 3, 0, 1],
            [1, 0, 3, 0, 1],
            [1, [0], 3, [0], 1],
        ],
        [
            [2, [3], 0, [3], 2],
            [2, 3, 0, 3, 2],
            [2, [3], 0, 1, 0, [3], 2],
            [2, [3], 0, 3, [0], 1, [0], 3, 0, [3], 2],
            [2, 3, 0, 1, 0, 3, 2],
            [2, 3, [0], 1, [0], 3, 2],
            [2, [3], 0, 3, 0, [3], 2],
            [2, 3, 2],
        ],
        [
            [3, 0, 3],
            [3, 0, 1, 0, 3],
            [3, [0], 1, [0], 3],
            [3, 0, [3], 2, [3], 0, 3],
            [3, 2, 3],
        ],
    ],
    "v4": [
        [
            [0, [3], 1, [3], 0],
            [0, 3, 1, 3, 0],
            [0, 2, 0],
            [0, 3, [0], 2, [0], 3, 0],
            [0, 3, 0],
        ],
        [
            [1, [3], 0, [3], 1],
            [1, 3, 0, 3, 1],
            [1, [3], 0, 2, 0, [3], 1],
            [1, 3, [0], 2, [0], 3, 1],
            [1, 3, 0, 2, 0, 3, 1],
            [1, [3], 0, 3, [0], 2, [0], 3, 0, [3], 1],
            [1, [3], 0, 3, 0, [3], 1],
            [1, 3, 1],
        ],
        [
            [2, 0, 2],
            [2, [0], 3, 0, 3, [0], 2],
            [2, 0, [3], 1, [3], 0, 2],
            [2, 0, 3, 1, 3, 0, 2],
            [2, [0], 3, 0, [3], 1, [3], 0, 3, [0], 2],
            [2, [0], 3, 1, 3, [0], 2],
            [2, 0, 3, 0, 2],
            [2, [0], 3, [0], 2],
        ],
        [
            [3, 0, 3],
            [3, 0, [3], 1, [3], 0, 3],
            [3, 1, 3],
            [3, 0, 2, 0, 3],
            [3, [0], 2, [0], 3],
        ],
    ],
}


# ---------------------------------------------------------------------
# Helpers (skip metapath parsing + FULL traversal exact typed walks)
# ---------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def parse_task(s: str) -> str:
    t = s.strip().lower()
    if t not in ("md", "ml"):
        raise SystemExit("task must be md or ml")
    return t


def parse_variants(s: str, task: str):
    vs = [x.strip().lower() for x in s.split(",") if x.strip()]
    good = {"v1", "v3"} if task == "md" else {"v1", "v2", "v3", "v4"}
    bad = [v for v in vs if v not in good]
    if bad:
        raise SystemExit(f"invalid variants for task={task}: {bad}; expected subset of {sorted(good)}")
    return vs


def parse_skip_metapath(spec):
    full = []
    skip_pos = set()
    for i, x in enumerate(spec):
        if isinstance(x, list):
            if len(x) != 1:
                raise ValueError(f"Skip notation must be single-item list, got {x}")
            full.append(x[0])
            skip_pos.add(i)
        else:
            full.append(x)
    full = tuple(int(x) for x in full)
    semantic = tuple(full[i] for i in range(len(full)) if i not in skip_pos)
    return full, skip_pos, semantic


def compress_full_path(path, skip_pos):
    return tuple(path[i] for i in range(len(path)) if i not in skip_pos)


def validate_symmetric_metapath(full):
    if tuple(full) != tuple(full[::-1]):
        raise ValueError(f"Metapath must be symmetric: {full}")


def get_metapath_neighbor_pairs_full(adjM, type_mask, metapath_specs: list[list[int | list[int]]]):
    """
    Return per-spec dicts containing:
      full, skip_pos, semantic, neighbor_pairs (endpoints -> list(full_paths))
    """
    outs = []
    typed_nbr_cache: dict[tuple[int, int], np.ndarray] = {}

    def neighbors_of_type(node_id: int, ntype: int) -> np.ndarray:
        key = (int(node_id), int(ntype))
        if key in typed_nbr_cache:
            return typed_nbr_cache[key]
        nbrs = np.where((adjM[int(node_id)] > 0) & (type_mask == int(ntype)))[0].astype(np.int32)
        typed_nbr_cache[key] = nbrs
        return nbrs

    def enumerate_half_paths_exact(full: tuple[int, ...]) -> dict[int, list[tuple[int, ...]]]:
        half_len = (len(full) + 1) // 2
        source_type = int(full[0])
        source_nodes = np.where(type_mask == source_type)[0]
        by_target: dict[int, list[tuple[int, ...]]] = defaultdict(list)
        for source in source_nodes:
            paths = [(int(source),)]
            for depth in range(1, half_len):
                req_t = int(full[depth])
                new_paths = []
                for p in paths:
                    last = p[-1]
                    nbrs = neighbors_of_type(last, req_t)
                    if len(nbrs) == 0:
                        continue
                    for nb in nbrs:
                        new_paths.append(p + (int(nb),))
                paths = new_paths
                if not paths:
                    break
            for p in paths:
                by_target[int(p[-1])].append(p)
        return by_target

    for spec in metapath_specs:
        full, skip_pos, semantic = parse_skip_metapath(spec)
        validate_symmetric_metapath(full)
        print(f"    Full metapath: {full}", flush=True)
        print(f"    Skip positions: {sorted(skip_pos) if skip_pos else 'none'}", flush=True)
        print(f"    Semantic metapath: {semantic}", flush=True)
        metapath_to_target = enumerate_half_paths_exact(full)
        metapath_neighbor_pairs: dict[tuple[int, int], list[tuple[int, ...]]] = {}
        for _, half_paths in metapath_to_target.items():
            for p1 in half_paths:
                for p2 in half_paths:
                    full_path = tuple(p1 + p2[-2::-1])
                    pair = (p1[0], p2[0])
                    metapath_neighbor_pairs[pair] = metapath_neighbor_pairs.get(pair, []) + [full_path]
        print(f"      -> {len(metapath_neighbor_pairs)} neighbor pairs found", flush=True)
        outs.append(
            {
                "spec": spec,
                "full": full,
                "skip_pos": skip_pos,
                "semantic": semantic,
                "neighbor_pairs": metapath_neighbor_pairs,
            }
        )
    return outs


def save_metapath_mode(
    out_dir: str,
    mp_infos: list[dict],
    type_mask: np.ndarray,
    ctr_ntype: int,
):
    """
    Save `.adjlist` and `_idx.pickle` for each semantic metapath.
    idx pickle maps local-center-node-id -> np.ndarray(paths, L_semantic) of COMPRESSED paths.
    """
    ctr_nodes = np.where(type_mask == int(ctr_ntype))[0]
    off = int(ctr_nodes.min())
    n = len(ctr_nodes)
    os.makedirs(out_dir, exist_ok=True)

    for info in mp_infos:
        semantic = tuple(int(x) for x in info["semantic"])
        skip_pos = set(int(x) for x in info["skip_pos"])
        mp_pairs = info["neighbor_pairs"]

        name = "-".join(map(str, semantic))
        per_src_paths: dict[int, list[np.ndarray]] = defaultdict(list)
        per_src_nbrs: dict[int, set[int]] = defaultdict(set)

        for (src_g, dst_g), full_paths in mp_pairs.items():
            if src_g < off or src_g >= off + n:
                continue
            s = int(src_g - off)
            d = int(dst_g - off)
            per_src_nbrs[s].add(d)
            for p in full_paths:
                comp = compress_full_path(p, skip_pos)
                per_src_paths[s].append(np.array(comp[::-1], dtype=np.int32))

        # adjlist: neighbors per source
        with open(os.path.join(out_dir, f"{name}.adjlist"), "w") as f:
            for s in range(n):
                nbrs = sorted(per_src_nbrs.get(s, []))
                if nbrs:
                    f.write(f"{s} " + " ".join(map(str, nbrs)) + "\n")
                else:
                    f.write(f"{s}\n")

        # idx pickle: stack compressed paths
        node2paths = {}
        L = len(semantic)
        for s in range(n):
            arr = per_src_paths.get(s, [])
            node2paths[s] = np.stack(arr, axis=0) if arr else np.empty((0, L), dtype=np.int32)
        with open(os.path.join(out_dir, f"{name}_idx.pickle"), "wb") as pf:
            pickle.dump(node2paths, pf)


# ---------------------------------------------------------------------
# IMDB-specific graph + features + splits (copied from non-skip LP preprocessor)
# ---------------------------------------------------------------------
def read_movies(task: str, csv_path: str) -> pd.DataFrame:
    if task == "ml":
        movies = (
            pd.read_csv(csv_path, encoding="utf-8")
            .drop_duplicates(subset=["movie_imdb_link"])
            .dropna(subset=["movie_imdb_link", "actor_1_name", "director_name", "genres"])
            .reset_index(drop=True)
        )
    else:
        movies = (
            pd.read_csv(csv_path, encoding="utf-8")
            .drop_duplicates(subset=["movie_imdb_link"])
            .dropna(subset=["actor_1_name", "director_name"])
            .reset_index(drop=True)
        )

    labels = np.full(len(movies), -1, dtype=np.int64)
    for i, genres in movies["genres"].astype(str).items():
        for g in genres.split("|"):
            g = g.strip()
            if g == "Action":
                labels[i] = 0
                break
            if g == "Comedy":
                labels[i] = 1
                break
            if g == "Drama":
                labels[i] = 2
                break
    keep = np.where(labels >= 0)[0]
    return movies.iloc[keep].reset_index(drop=True)


def primary_genre_id(genres_val) -> int:
    for g in str(genres_val).split("|"):
        g = g.strip()
        if g == "Action":
            return 0
        if g == "Comedy":
            return 1
        if g == "Drama":
            return 2
    return -1


def sample_neg_md(pos, n_tail, k, rng):
    out = np.zeros((len(pos), k), dtype=np.int64)
    for i, (_, t) in enumerate(pos):
        cand = [x for x in range(n_tail) if x != int(t)]
        out[i] = rng.choice(cand, size=k, replace=(k > len(cand)))
    return out


def sample_neg_ml(pos, n_links, all_true, k, rng):
    out = np.zeros((len(pos), k), dtype=np.int64)
    links = np.arange(n_links, dtype=np.int64)
    for i, (m, _) in enumerate(pos):
        cand = [x for x in links if (int(m), int(x)) not in all_true]
        if not cand:
            cand = links.tolist()
        out[i] = rng.choice(cand, size=k, replace=(k > len(cand)))
    return out


def build_star_aligned_bow_features(
    movies: pd.DataFrame, adjM: np.ndarray, M: int, Dn: int, An: int, task: str
) -> list[sp.csr_matrix]:
    kw = movies["plot_keywords"].fillna("").apply(lambda x: str(x).replace("|", " ")).values
    try:
        vectorizer = CountVectorizer(min_df=2)
        movie_X = vectorizer.fit_transform(kw)
    except ValueError:
        movie_X = None
    if movie_X is None or movie_X.shape[1] == 0:
        try:
            vectorizer = CountVectorizer(min_df=1)
            movie_X = vectorizer.fit_transform(kw)
        except ValueError:
            movie_X = None
    if movie_X is None or movie_X.shape[1] == 0:
        movie_X = sp.csr_matrix(np.ones((len(movies), 1), dtype=np.float32))

    adj_da_m = sp.csr_matrix(adjM[M : M + Dn + An, :M], dtype=np.float64)
    rs = np.asarray(adj_da_m.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    adj_da_m_norm = sp.diags(1.0 / rs).dot(adj_da_m)
    director_actor_X = adj_da_m_norm.dot(movie_X)

    adj_m_da = sp.csr_matrix(adjM[:M, M : M + Dn + An], dtype=np.float64)
    cs = np.asarray(adj_m_da.sum(axis=1)).ravel()
    cs[cs == 0] = 1.0
    adj_m_da_norm = sp.diags(1.0 / cs).dot(adj_m_da)
    link_X = adj_m_da_norm.dot(director_actor_X)

    movie_X = movie_X.astype(np.float32)
    director_actor_X = director_actor_X.astype(np.float32)
    link_X = link_X.astype(np.float32)

    out: list[sp.csr_matrix] = [
        movie_X,
        director_actor_X[:Dn],
        director_actor_X[Dn:],
        link_X,
    ]
    if task == "md":
        out.append(sp.eye(3, dtype=np.float32, format="csr"))
    return out


def edge_type_id(u: int, v: int) -> int:
    """
    Universal semantic edge types (same as skip node classification):
      0:M→D  1:D→M  2:M→A  3:A→M  4:M→L  5:L→M  6:D→L  7:L→D  8:A→L  9:L→A
    """
    m = {
        (0, 1): 0,
        (1, 0): 1,
        (0, 2): 2,
        (2, 0): 3,
        (0, 3): 4,
        (3, 0): 5,
        (1, 3): 6,
        (3, 1): 7,
        (2, 3): 8,
        (3, 2): 9,
    }
    if (u, v) not in m:
        raise ValueError(f"Unknown edge type for ({u}->{v})")
    return m[(u, v)]


def semantic_etypes_for_metapath(semantic: Iterable[int]) -> list[int]:
    s = list(map(int, semantic))
    return [edge_type_id(s[i], s[i + 1]) for i in range(len(s) - 1)]


def canonical_semantic_infos(
    variant: str,
    metapath_specs: list[list[int | list[int]]],
    canonical_by_sem: dict[tuple[int, ...], list[int | list[int]]],
) -> list[list[int | list[int]]]:
    """
    For universal outputs, we ignore variant-specific skip specs and instead
    use the canonical (v1) spec for each semantic metapath.
    """
    sems = []
    for spec in metapath_specs:
        _, __, sem = parse_skip_metapath(spec)
        sems.append(tuple(int(x) for x in sem))
    uniq = list(dict.fromkeys(sems))
    missing = [s for s in uniq if s not in canonical_by_sem]
    if missing:
        raise SystemExit(f"variant={variant} has semantic metapaths not in canonical(v1): {missing}")
    return [canonical_by_sem[s] for s in uniq]


def build_graph_and_splits(task: str, movies: pd.DataFrame, shared_npz: str, neg_k: int, seed: int):
    """
    UNIVERSAL graph construction (no variant-specific structural differences):
    - movie–actor edges (all)
    - movie–link identity + link-hub edges are TRAIN-ONLY for ml (leakage control)
    - movie–director edges:
        - md: train-only supervision edges
        - ml: CSV structural context edges (all)
    """
    z = np.load(shared_npz)
    train_pos = z["train_pos"].astype(np.int64)
    val_pos = z["val_pos"].astype(np.int64)
    test_pos = z["test_pos"].astype(np.int64)
    rng = np.random.RandomState(seed)

    directors = sorted(set(movies["director_name"].dropna().tolist()))
    actors_all = sorted(
        set(
            movies["actor_1_name"].dropna().tolist()
            + movies.get("actor_2_name", pd.Series(dtype=str)).dropna().tolist()
            + movies.get("actor_3_name", pd.Series(dtype=str)).dropna().tolist()
        )
    )
    actors_ml = sorted(set(movies["actor_1_name"].dropna().tolist()))
    M = len(movies)
    Dn = len(directors)
    An = len(actors_ml if task == "ml" else actors_all)
    Ln = M
    Gn = 3

    dmap = {x: i for i, x in enumerate(directors)}
    amap = {x: i for i, x in enumerate(actors_ml if task == "ml" else actors_all)}

    off_m = 0
    off_d = M
    off_a = M + Dn
    off_l = M + Dn + An
    off_g = M + Dn + An + Ln

    use_genre = task == "md"
    N = M + Dn + An + Ln + (Gn if use_genre else 0)
    type_mask = np.zeros(N, dtype=np.int32)
    type_mask[off_d : off_d + Dn] = 1
    type_mask[off_a : off_a + An] = 2
    type_mask[off_l : off_l + Ln] = 3
    if use_genre:
        type_mask[off_g : off_g + Gn] = 4

    adjM = np.zeros((N, N), dtype=np.int32)

    def add(u, v):
        adjM[u, v] = 1
        adjM[v, u] = 1

    train_movies = {int(r[0]) for r in train_pos.tolist()} if task == "ml" else None

    for mi, row in movies.iterrows():
        mi = int(mi)
        # universal movie–actor
        acols = ("actor_1_name",) if task == "ml" else ("actor_1_name", "actor_2_name", "actor_3_name")
        for c in acols:
            if pd.notna(row.get(c)) and row[c] in amap:
                add(off_m + mi, off_a + amap[row[c]])

        # ml-only: movie–director context edges from CSV
        if task == "ml":
            di = dmap.get(row["director_name"])
            if di is not None:
                add(off_m + mi, off_d + di)

        # link hub edges (train-only for ml leakage control; always on for md as it's not the target)
        allow_link = train_movies is None or mi in train_movies
        if allow_link:
            add(off_m + mi, off_l + mi)  # movie-link identity
            # link-director + link-actor (universal union)
            di = dmap.get(row["director_name"])
            if di is not None:
                add(off_l + mi, off_d + di)
            for c in acols:
                if pd.notna(row.get(c)) and row[c] in amap:
                    add(off_l + mi, off_a + amap[row[c]])

        if task == "md":
            gid = primary_genre_id(row.get("genres"))
            if gid >= 0:
                add(off_m + mi, off_g + gid)

    # task supervision edges and negatives
    if task == "md":
        for m, d in train_pos:
            add(off_m + int(m), off_d + int(d))
        train_neg = sample_neg_md(train_pos, Dn, neg_k, rng)
        val_neg = sample_neg_md(val_pos, Dn, neg_k, rng)
        test_neg = sample_neg_md(test_pos, Dn, neg_k, rng)
    elif task == "ml":
        for m, l in train_pos:
            add(off_m + int(m), off_l + int(l))
        all_true = set(map(tuple, train_pos.tolist())) | set(map(tuple, val_pos.tolist())) | set(map(tuple, test_pos.tolist()))
        k = min(int(neg_k), max(1, Ln - 1))
        train_neg = sample_neg_ml(train_pos, Ln, all_true, k, rng)
        val_neg = sample_neg_ml(val_pos, Ln, all_true, k, rng)
        test_neg = sample_neg_ml(test_pos, Ln, all_true, k, rng)
    else:
        raise SystemExit(f"unknown task {task}")

    return (
        adjM,
        type_mask,
        {"M": M, "D": Dn, "A": An, "L": Ln, "G": Gn if use_genre else 0},
        train_pos,
        val_pos,
        test_pos,
        train_neg,
        val_neg,
        test_neg,
        {"off_m": off_m, "off_d": off_d, "off_a": off_a, "off_l": off_l, "off_g": off_g},
    )


def build_config(task: str, variant: str):
    """
    Build semantic metapaths for mode0/mode1 using the provided SKIP_METAPATHS,
    but canonicalize specs to v1 per semantic metapath so outputs are universal.
    """
    # canonical per semantic: from v1
    canon = {}
    for group in SKIP_METAPATHS["v1"]:
        for spec in group:
            _, __, sem = parse_skip_metapath(spec)
            sem = tuple(int(x) for x in sem)
            canon[sem] = spec

    groups = SKIP_METAPATHS[variant]

    if task == "md":
        mode0_specs = canonical_semantic_infos(variant, groups[0], canon)
        mode1_specs = canonical_semantic_infos(variant, groups[1], canon)
        tail_type = 1
        num_ntypes = 5
        num_etypes = 10
    else:
        mode0_specs = canonical_semantic_infos(variant, groups[0], canon)
        mode1_specs = canonical_semantic_infos(variant, groups[3], canon)
        tail_type = 3
        num_ntypes = 4
        num_etypes = 10

    # semantic metapaths (tuples of node types) and their etypes
    mode0_sem = [parse_skip_metapath(s)[2] for s in mode0_specs]
    mode1_sem = [parse_skip_metapath(s)[2] for s in mode1_specs]
    mode0_ets = [semantic_etypes_for_metapath(mp) for mp in mode0_sem]
    mode1_ets = [semantic_etypes_for_metapath(mp) for mp in mode1_sem]

    return {
        "num_ntypes": int(num_ntypes),
        "tail_type": int(tail_type),
        "metapaths": [mode0_sem, mode1_sem],
        "etypes": [mode0_ets, mode1_ets],
        "num_etypes": int(num_etypes),
        "skip_specs": {"mode0": mode0_specs, "mode1": mode1_specs},
    }


def main():
    ap = argparse.ArgumentParser(description="Preprocess IMDB MAGNN LP (skip-node, universal).")
    ap.add_argument("--task", required=True)
    ap.add_argument("--variants", default="v1")
    ap.add_argument("--csv", default="data/raw/IMDB/movie_metadata.csv")
    ap.add_argument("--shared-npz", default="")
    ap.add_argument("--out-root", default="data/preprocessed")
    ap.add_argument("--neg-k", type=int, default=19)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--feats", choices=("bow", "identity"), default="bow")
    args = ap.parse_args()

    task = parse_task(args.task)
    variants = parse_variants(args.variants, task)
    defaults = {
        "md": "../CMPNN/IMDB_md_shared_splits.npz",
        "ml": "../CMPNN/IMDB_ml_shared_splits.npz",
    }
    shared = args.shared_npz.strip() or defaults[task]
    if not os.path.isfile(shared):
        raise SystemExit(f"missing shared npz: {shared}")
    if not os.path.isfile(args.csv):
        raise SystemExit(f"missing csv: {args.csv}")

    set_seed(args.seed)
    print("== IMDB MAGNN LP skip preprocess ==", flush=True)
    print(f"task={task} variants={variants} seed={args.seed} neg_k={args.neg_k}", flush=True)
    print(f"csv={args.csv}", flush=True)
    print(f"shared_npz={shared}", flush=True)
    movies = read_movies(task, args.csv)
    print(f"loaded movies after genre filter: {len(movies)}", flush=True)

    for v in variants:
        print("\n" + "=" * 80, flush=True)
        print(f"variant={v} | task={task}", flush=True)
        print("=" * 80, flush=True)
        cfg = build_config(task, v)
        out_dir = os.path.join(args.out_root, f"IMDB_magnn_lp_skip_{task}_{v}")
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(os.path.join(out_dir, "0"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "1"), exist_ok=True)
        print(f"output dir: {out_dir}", flush=True)
        print("building universal graph + train/val/test splits ...", flush=True)

        (
            adjM,
            type_mask,
            counts,
            train_pos,
            val_pos,
            test_pos,
            train_neg,
            val_neg,
            test_neg,
            offsets,
        ) = build_graph_and_splits(task, movies, shared, args.neg_k, args.seed)
        print(
            f"counts: M={counts['M']} D={counts['D']} A={counts['A']} L={counts['L']} G={counts['G']}",
            flush=True,
        )
        print(
            f"splits: train_pos={len(train_pos)} val_pos={len(val_pos)} test_pos={len(test_pos)} "
            f"| neg_shape train={train_neg.shape} val={val_neg.shape} test={test_neg.shape}",
            flush=True,
        )

        np.save(os.path.join(out_dir, "node_types.npy"), type_mask)
        sp.save_npz(os.path.join(out_dir, "adjM.npz"), sp.csr_matrix(adjM))
        np.savez(os.path.join(out_dir, "train_val_test_pos.npz"), train_pos=train_pos, val_pos=val_pos, test_pos=test_pos)
        np.savez(
            os.path.join(out_dir, "train_val_test_neg.npz"),
            train_neg=train_neg,
            val_neg=val_neg,
            test_neg=test_neg,
        )

        # features
        if args.feats == "bow":
            feats = build_star_aligned_bow_features(movies, adjM, counts["M"], counts["D"], counts["A"], task)
            for t, feat in enumerate(feats):
                sp.save_npz(os.path.join(out_dir, f"features_{t}.npz"), feat)
        else:
            for t in range(cfg["num_ntypes"]):
                n = int((type_mask == t).sum())
                feat = sp.eye(n, n, dtype=np.float32, format="csr")
                sp.save_npz(os.path.join(out_dir, f"features_{t}.npz"), feat)

        # metapaths: FULL traversal using canonical specs; save COMPRESSED semantic paths
        print("building metapaths for mode 0 (query side, ctr=0) ...", flush=True)
        infos0 = get_metapath_neighbor_pairs_full(adjM, type_mask, cfg["skip_specs"]["mode0"])
        print("building metapaths for mode 1 (tail side) ...", flush=True)
        infos1 = get_metapath_neighbor_pairs_full(adjM, type_mask, cfg["skip_specs"]["mode1"])
        print("saving metapath adjlists / idx pickles ...", flush=True)
        save_metapath_mode(os.path.join(out_dir, "0"), infos0, type_mask, ctr_ntype=0)
        save_metapath_mode(os.path.join(out_dir, "1"), infos1, type_mask, ctr_ntype=int(cfg["tail_type"]))

        with open(os.path.join(out_dir, "config.pkl"), "wb") as f:
            pickle.dump(cfg, f)
        with open(os.path.join(out_dir, "offsets.pkl"), "wb") as f:
            pickle.dump(offsets, f)
        print(f"done: saved {out_dir}", flush=True)

    print("\nall requested variants finished.", flush=True)


if __name__ == "__main__":
    main()

