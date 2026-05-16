"""IMDB movie->director LP for CMPNN (v1 and v3 only; no genre nodes)."""

import numpy as np
import pandas as pd


def read_imdb_frame(csv_path):
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
    labels = labels[keep]
    movies["label"] = labels
    return movies


def _offsets(movies):
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
    off_d = M
    off_a = M + Dn
    off_l = M + Dn + An
    N = off_l + M
    return M, Dn, An, off_d, off_a, off_l, N, directors, actors


def build_triplets_md(movies, variant, train_pos):
    M, Dn, An, off_d, off_a, off_l, N, directors, actors = _offsets(movies)

    def d_idx(name):
        return off_d + directors.index(name)

    def a_idx(name):
        return off_a + actors.index(name)

    triplets = []

    if variant == "v1":
        rel_md = 2
        for mi, row in movies.iterrows():
            for acol in ("actor_1_name", "actor_2_name", "actor_3_name"):
                if pd.notna(row[acol]):
                    triplets.append((mi, a_idx(row[acol]), 0))
            triplets.append((mi, off_l + mi, 1))
        for h, t in train_pos:
            triplets.append((h, d_idx(directors[t]), rel_md))
        num_relation = 3

    elif variant == "v3":
        rel_md = 2
        for mi, row in movies.iterrows():
            li = off_l + mi
            triplets.append((mi, li, 0))
            for acol in ("actor_1_name", "actor_2_name", "actor_3_name"):
                if pd.notna(row[acol]):
                    triplets.append((li, a_idx(row[acol]), 1))
        for h, t in train_pos:
            triplets.append((h, d_idx(directors[t]), rel_md))
        num_relation = 3
    else:
        raise ValueError(f"variant must be v1 or v3, got {variant}")

    triplets = np.array(triplets, dtype=int)
    meta = dict(
        num_entity=N,
        num_relation=num_relation,
        rel_md=rel_md,
        off_d=off_d,
        sizes=dict(M=M, Dn=Dn, An=An),
        counts=dict(triplets=len(triplets)),
    )
    return triplets, meta


def _sample_k_negs(pos_pairs, D_all, true_set, k, rng):
    neg_pairs = []
    D_arr = np.array(D_all)
    for h, t in pos_pairs:
        cand = D_arr[D_arr != t]
        if len(cand) == 0:
            cand = D_arr
        if k <= len(cand):
            chosen = rng.choice(cand, size=k, replace=False)
        else:
            chosen = rng.choice(cand, size=k, replace=True)
        for d in chosen:
            neg_pairs.append((h, d))
    return np.array(neg_pairs, dtype=int)


def preprocess_imdb_md(csv_path, shared_npz, variant, neg_k, seed):
    movies = read_imdb_frame(csv_path)
    data = np.load(shared_npz)
    train_pos = data["train_pos"]
    val_pos = data["val_pos"]
    test_pos = data["test_pos"]

    directors = sorted(set(movies["director_name"].dropna()))
    D_all = list(range(len(directors)))
    true_set = {(h, t) for h, t in train_pos}

    triplets, meta = build_triplets_md(movies, variant, train_pos)

    rng = np.random.RandomState(seed)
    train_neg = _sample_k_negs(train_pos, D_all, true_set, neg_k, rng)
    val_neg = _sample_k_negs(val_pos, D_all, true_set, neg_k, rng)
    test_neg = _sample_k_negs(test_pos, D_all, true_set, neg_k, rng)

    splits = dict(
        train_pos=train_pos,
        train_neg=train_neg,
        val_pos=val_pos,
        val_neg=val_neg,
        test_pos=test_pos,
        test_neg=test_neg,
    )
    return triplets, meta, splits
