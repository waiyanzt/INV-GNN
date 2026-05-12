#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
import random
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

SEED = 1566911444


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
    """Action/Comedy/Drama label id (same rule as read_movies); return -1 if none."""
    for g in str(genres_val).split("|"):
        g = g.strip()
        if g == "Action":
            return 0
        if g == "Comedy":
            return 1
        if g == "Drama":
            return 2
    return -1


def get_metapath_neighbor_pairs(M, type_mask, expected_metapaths):
    """
    Enumerate *exact-length* typed walks for symmetric metapaths.

    This is intentionally NOT shortest-path based: MAGNN metapath instances are defined
    by following the exact type sequence, not by graph distance.
    """
    outs = []
    typed_nbr_cache: dict[tuple[int, int], np.ndarray] = {}

    def neighbors_of_type(node_id: int, ntype: int) -> np.ndarray:
        key = (int(node_id), int(ntype))
        if key in typed_nbr_cache:
            return typed_nbr_cache[key]
        nbrs = np.where((M[int(node_id)] > 0) & (type_mask == int(ntype)))[0].astype(np.int32)
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

    for mi, metapath in enumerate(expected_metapaths, start=1):
        full = tuple(int(x) for x in metapath)
        if full != full[::-1]:
            raise ValueError(f"Metapath must be symmetric: {full}")
        print(f"    [metapath {mi}/{len(expected_metapaths)}] building paths for {full}", flush=True)
        metapath_to_target = enumerate_half_paths_exact(full)

        metapath_neighbor_pairs: dict[tuple[int, int], list[tuple[int, ...]]] = {}
        for _, half_paths in metapath_to_target.items():
            for p1 in half_paths:
                for p2 in half_paths:
                    full_path = tuple(p1 + p2[-2::-1])
                    pair = (p1[0], p2[0])
                    metapath_neighbor_pairs[pair] = metapath_neighbor_pairs.get(pair, []) + [full_path]

        print(f"      done {full}: {len(metapath_neighbor_pairs)} neighbor-pairs", flush=True)
        outs.append(metapath_neighbor_pairs)
    return outs


def save_metapath_mode(out_dir, neighbor_pairs, metapaths, type_mask, ctr_ntype):
    ctr_nodes = np.where(type_mask == ctr_ntype)[0]
    off = int(ctr_nodes.min())
    n = len(ctr_nodes)
    os.makedirs(out_dir, exist_ok=True)

    for mp, mp_pairs in zip(metapaths, neighbor_pairs):
        name = "-".join(map(str, mp))
        per_src_paths = defaultdict(list)
        per_src_nbrs = defaultdict(set)

        for (src_g, dst_g), paths in mp_pairs.items():
            if src_g < off or src_g >= off + n:
                continue
            s = int(src_g - off)
            d = int(dst_g - off)
            per_src_nbrs[s].add(d)
            for p in paths:
                per_src_paths[s].append(np.array(p[::-1], dtype=np.int32))

        with open(os.path.join(out_dir, f"{name}.adjlist"), "w") as f:
            for s in range(n):
                nbrs = sorted(per_src_nbrs.get(s, []))
                if nbrs:
                    f.write(f"{s} " + " ".join(map(str, nbrs)) + "\n")
                else:
                    f.write(f"{s}\n")

        node2paths = {}
        L = len(mp)
        for s in range(n):
            arr = per_src_paths.get(s, [])
            node2paths[s] = np.stack(arr, axis=0) if arr else np.empty((0, L), dtype=np.int32)
        with open(os.path.join(out_dir, f"{name}_idx.pickle"), "wb") as pf:
            pickle.dump(node2paths, pf)


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
    """
    Same bag-of-words + neighbor smoothing as ``preprocess_IMDB_star.py``:
    movies get plot_keywords BoW; director/actor rows are normalized DA→movie aggregates;
    link rows are normalized movie→DA @ director_actor_X.
    """
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


def build_config(task, variant):
    # Node type ids:
    # md: 0 movie, 1 director, 2 actor, 3 link, 4 genre (CSV auxiliary)
    # ml: 0 movie, 1 director, 2 actor, 3 link
    if task == "md":
        # Movie-side metapaths use genre type 4 (CSV labels). Tail-side: director (pairs are movie, director).
        by_v = {
            "v1": (
                [[(0, 1, 0), (0, 2, 0), (0, 3, 0), (0, 4, 0)]],
                [[[0, 1], [2, 3], [4, 5], [10, 11]]],
            ),
            "v3": (
                [[(0, 1, 0), (0, 3, 0), (0, 3, 2, 3, 0), (0, 4, 0)]],
                [[[0, 1], [4, 5], [4, 8, 9, 5], [10, 11]]],
            ),
        }
        # Tail-side metapaths must match *available* structure per variant.
        # v1 has movie–actor edges, so director→movie→actor→movie→director is valid.
        # v3 does NOT have movie–actor edges; it reaches actors through link-hub:
        # director→movie→link→actor→link→movie→director.
        if variant == "v1":
            tail_mps = [(1, 0, 1), (1, 0, 2, 0, 1), (1, 0, 3, 0, 1)]
            tail_ets = [[1, 0], [1, 2, 3, 0], [1, 4, 5, 0]]
        elif variant == "v3":
            tail_mps = [(1, 0, 1), (1, 0, 3, 2, 3, 0, 1), (1, 0, 3, 0, 1)]
            tail_ets = [[1, 0], [1, 4, 8, 9, 5, 0], [1, 4, 5, 0]]
        else:
            raise SystemExit(f"invalid variant for md: {variant} (expected v1 or v3)")
        mode0_mps, mode0_ets = by_v[variant]
        return dict(
            num_ntypes=5,
            tail_type=1,
            metapaths=[mode0_mps[0], tail_mps],
            etypes=[mode0_ets[0], tail_ets],
            num_etypes=12,
        )

    if task == "ml":
        by_v = {
            "v1": (
                [[(0, 1, 0), (0, 2, 0), (0, 3, 0)], [(3, 0, 3), (3, 0, 1, 0, 3), (3, 0, 2, 0, 3)]],
                [[[0, 1], [2, 3], [4, 5]], [[5, 4], [5, 0, 1, 4], [5, 2, 3, 4]]],
            ),
            "v2": (
                # MAGNN star-aligned v2: actor/director are attached to the link hub.
                [[(0, 3, 0), (0, 3, 1, 3, 0), (0, 3, 2, 3, 0)], [(3, 0, 3), (3, 1, 3), (3, 2, 3)]],
                [[[4, 5], [4, 6, 7, 5], [4, 8, 9, 5]], [[5, 4], [6, 7], [8, 9]]],
            ),
            "v3": (
                [[(0, 1, 0), (0, 3, 0), (0, 3, 2, 3, 0)], [(3, 0, 3), (3, 0, 1, 0, 3), (3, 2, 3)]],
                [[[0, 1], [4, 5], [4, 8, 9, 5]], [[5, 4], [5, 0, 1, 4], [8, 9]]],
            ),
            "v4": (
                [[(0, 2, 0), (0, 3, 0), (0, 3, 1, 3, 0)], [(3, 0, 3), (3, 1, 3), (3, 0, 2, 0, 3)]],
                [[[2, 3], [4, 5], [4, 6, 7, 5]], [[5, 4], [6, 7], [5, 2, 3, 4]]],
            ),
        }
        mps, ets = by_v[variant]
        return dict(num_ntypes=4, tail_type=3, metapaths=mps, etypes=ets, num_etypes=10)

    raise SystemExit(f"unknown task {task}")


def build_graph_and_splits(task, variant, movies, shared_npz, neg_k, seed):
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
    type_mask[off_d:off_d + Dn] = 1
    type_mask[off_a:off_a + An] = 2
    type_mask[off_l:off_l + Ln] = 3
    if use_genre:
        type_mask[off_g:off_g + Gn] = 4

    adjM = np.zeros((N, N), dtype=np.int32)

    def add(u, v):
        adjM[u, v] = 1
        adjM[v, u] = 1

    # For MAGNN LP we keep the full structural graph (including movie–link),
    # matching the original MAGNN star preprocessors. Supervision still uses shared splits.
    train_movies = None

    for mi, row in movies.iterrows():
        mi = int(mi)
        # ML: structural movie–director edges from CSV (context; not the LP target).
        # MD: train-only supervision edges are added below.
        if task == "ml":
            di = dmap.get(row["director_name"])
            if di is not None:
                add(off_m + mi, off_d + di)
        if variant in ("v1", "v4"):
            acols = ("actor_1_name",) if task == "ml" else ("actor_1_name", "actor_2_name", "actor_3_name")
            for c in acols:
                if pd.notna(row.get(c)) and row[c] in amap:
                    add(off_m + mi, off_a + amap[row[c]])

        # movie↔link exists for all movies (row-aligned link id = movie index)
        add(off_m + mi, off_l + mi)

        # Link hub context edges (variant-specific).
        if variant in ("v2", "v4"):
            di = dmap.get(row["director_name"])
            if di is not None:
                add(off_l + mi, off_d + di)
        if variant in ("v2", "v3"):
            acols = ("actor_1_name",) if task == "ml" else ("actor_1_name", "actor_2_name", "actor_3_name")
            for c in acols:
                if pd.notna(row.get(c)) and row[c] in amap:
                    add(off_l + mi, off_a + amap[row[c]])
        if task == "md":
            gid = primary_genre_id(row.get("genres"))
            if gid >= 0:
                add(off_m + mi, off_g + gid)

    # task LP train edges only
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
    )


def main():
    ap = argparse.ArgumentParser(description="Preprocess IMDB MAGNN LP (md/ml).")
    ap.add_argument("--task", required=True)
    ap.add_argument("--variants", default="v1")
    ap.add_argument("--csv", default="data/raw/IMDB/movie_metadata.csv")
    ap.add_argument("--shared-npz", default="")
    ap.add_argument("--out-root", default="data/preprocessed")
    ap.add_argument("--neg-k", type=int, default=19)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--feats",
        choices=("bow", "identity"),
        default="bow",
        help="bow: plot_keywords BoW + star-style propagation (default); identity: one-hot per node type",
    )
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
    print(f"== IMDB MAGNN LP preprocess ==", flush=True)
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
        out_dir = os.path.join(args.out_root, f"IMDB_magnn_lp_{task}_{v}")
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "0"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "1"), exist_ok=True)
        print(f"output dir: {out_dir}", flush=True)
        print("building graph + train/val/test splits ...", flush=True)

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
        ) = build_graph_and_splits(task, v, movies, shared, args.neg_k, args.seed)
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
        np.savez(
            os.path.join(out_dir, "train_val_test_pos.npz"),
            train_pos=train_pos,
            val_pos=val_pos,
            test_pos=test_pos,
        )
        np.savez(
            os.path.join(out_dir, "train_val_test_neg.npz"),
            train_neg=train_neg,
            val_neg=val_neg,
            test_neg=test_neg,
        )

        if args.feats == "bow":
            print("saving star-aligned BoW features per node type ...", flush=True)
            feats = build_star_aligned_bow_features(
                movies, adjM, counts["M"], counts["D"], counts["A"], task
            )
            for t, feat in enumerate(feats):
                sp.save_npz(os.path.join(out_dir, f"features_{t}.npz"), feat)
        else:
            print("saving identity features per node type ...", flush=True)
            for t in range(cfg["num_ntypes"]):
                n = int((type_mask == t).sum())
                feat = sp.eye(n, n, dtype=np.float32, format="csr")
                sp.save_npz(os.path.join(out_dir, f"features_{t}.npz"), feat)

        # metapaths for 2 modes
        print("building metapaths for mode 0 (query side) ...", flush=True)
        neigh_mode0 = get_metapath_neighbor_pairs(adjM, type_mask, cfg["metapaths"][0])
        print("building metapaths for mode 1 (target side) ...", flush=True)
        neigh_mode1 = get_metapath_neighbor_pairs(adjM, type_mask, cfg["metapaths"][1])
        print("saving metapath adjlists / idx pickles ...", flush=True)
        save_metapath_mode(os.path.join(out_dir, "0"), neigh_mode0, cfg["metapaths"][0], type_mask, ctr_ntype=0)
        save_metapath_mode(
            os.path.join(out_dir, "1"), neigh_mode1, cfg["metapaths"][1], type_mask, ctr_ntype=cfg["tail_type"]
        )

        with open(os.path.join(out_dir, "config.pkl"), "wb") as f:
            pickle.dump(cfg, f)
        print(f"done: saved {out_dir}", flush=True)

    print("\nall requested variants finished.", flush=True)


if __name__ == "__main__":
    main()
