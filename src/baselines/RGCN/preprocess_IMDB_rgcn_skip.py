#!/usr/bin/env python3
"""
IMDb RGCN preprocessing with skip-node canonical collapse for 4 variations.

to run the preprocessing:
python preprocess_IMDB_rgcn_skip.py --variants v1,v2,v3,v4

"""

import os
import shutil
import argparse
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


SEED = 1566911444
RAW = "data/raw/IMDB/movie_metadata.csv"
OUT_BASE = "data/preprocessed"

VARIANT_OUT = {
    "v1": os.path.join(OUT_BASE, "IMDB_rgcn_skip_v1"),
    "v2": os.path.join(OUT_BASE, "IMDB_rgcn_skip_v2"),
    "v3": os.path.join(OUT_BASE, "IMDB_rgcn_skip_v3"),
    "v4": os.path.join(OUT_BASE, "IMDB_rgcn_skip_v4"),
}

# relation IDs for canonical semantic graph
REL = {
    "movie_to_director": 0,
    "director_to_movie": 1,
    "movie_to_actor": 2,
    "actor_to_movie": 3,
    "movie_to_link": 4,
    "link_to_movie": 5,
    "link_to_director": 6,
    "director_to_link": 7,
    "link_to_actor": 8,
    "actor_to_link": 9,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_and_filter_movies():
    movies = (
        pd.read_csv(RAW, encoding="utf-8")
        .drop_duplicates(subset=["movie_imdb_link"])
        .dropna(axis=0, subset=["director_name", "actor_1_name", "movie_imdb_link"])
        .reset_index(drop=True)
    )

    # labels: 0=Action, 1=Comedy, 2=Drama; keep only these
    labels = np.full(len(movies), -1, dtype=np.int64)
    for i, genres in enumerate(movies["genres"].fillna("")):
        for g in str(genres).split("|"):
            if g == "Action":
                labels[i] = 0
                break
            elif g == "Comedy":
                labels[i] = 1
                break
            elif g == "Drama":
                labels[i] = 2
                break

    keep = labels != -1
    movies = movies.loc[keep].reset_index(drop=True)
    labels = labels[keep]
    return movies, labels


def build_vocabularies(movies: pd.DataFrame):
    directors = sorted(set(movies["director_name"].dropna()))
    actors = sorted(set(
        movies["actor_1_name"].dropna().tolist() +
        movies["actor_2_name"].dropna().tolist() +
        movies["actor_3_name"].dropna().tolist()
    ))
    links = sorted(set(movies["movie_imdb_link"].dropna()))

    director_to_idx = {x: i for i, x in enumerate(directors)}
    actor_to_idx = {x: i for i, x in enumerate(actors)}
    link_to_idx = {x: i for i, x in enumerate(links)}

    return directors, actors, links, director_to_idx, actor_to_idx, link_to_idx


def build_global_node_index(movies, directors, actors, links):
    num_movies = len(movies)
    num_directors = len(directors)
    num_actors = len(actors)
    num_links = len(links)

    off_movie = 0
    off_director = num_movies
    off_actor = num_movies + num_directors
    off_link = num_movies + num_directors + num_actors

    n_total = num_movies + num_directors + num_actors + num_links

    type_mask = np.zeros(n_total, dtype=np.int64)
    type_mask[off_director:off_actor] = 1
    type_mask[off_actor:off_link] = 2
    type_mask[off_link:] = 3

    return {
        "num_movies": num_movies,
        "num_directors": num_directors,
        "num_actors": num_actors,
        "num_links": num_links,
        "n_total": n_total,
        "off_movie": off_movie,
        "off_director": off_director,
        "off_actor": off_actor,
        "off_link": off_link,
        "type_mask": type_mask,
    }


def build_movie_features(movies: pd.DataFrame):
    vectorizer = CountVectorizer(min_df=2)
    movie_X = vectorizer.fit_transform(
        movies["plot_keywords"].fillna("").apply(lambda x: str(x).replace("|", " ")).values
    ).astype(np.float32)
    return movie_X


def build_all_node_features(movie_X, movies, director_to_idx, actor_to_idx, link_to_idx, info):
    num_movies = info["num_movies"]
    num_directors = info["num_directors"]
    num_actors = info["num_actors"]
    num_links = info["num_links"]
    n_total = info["n_total"]
    off_director = info["off_director"]
    off_actor = info["off_actor"]
    off_link = info["off_link"]

    dim = movie_X.shape[1]
    X = np.zeros((n_total, dim), dtype=np.float32)
    X[:num_movies] = movie_X.toarray()

    # aggregate movie features into director / actor / link nodes
    dir_movies = defaultdict(list)
    actor_movies = defaultdict(list)
    link_movies = defaultdict(list)

    for m_idx, row in movies.iterrows():
        d = row["director_name"]
        l = row["movie_imdb_link"]
        if pd.notna(d) and d in director_to_idx:
            dir_movies[director_to_idx[d]].append(m_idx)
        if pd.notna(l) and l in link_to_idx:
            link_movies[link_to_idx[l]].append(m_idx)

        for acol in ("actor_1_name", "actor_2_name", "actor_3_name"):
            a = row.get(acol)
            if pd.notna(a) and a in actor_to_idx:
                actor_movies[actor_to_idx[a]].append(m_idx)

    for d_local, m_list in dir_movies.items():
        X[off_director + d_local] = X[np.array(m_list)].mean(axis=0)
    for a_local, m_list in actor_movies.items():
        X[off_actor + a_local] = X[np.array(m_list)].mean(axis=0)
    for l_local, m_list in link_movies.items():
        X[off_link + l_local] = X[np.array(m_list)].mean(axis=0)

    return X


def build_raw_edges_for_variant(movies, director_to_idx, actor_to_idx, link_to_idx, info, variant):
    """
    Returns raw adjacency dictionaries by typed endpoints.

    Raw variants:
      v1: M-D, M-A, M-L
      v2: M-L, L-D, L-A
      v3: M-D, M-L, L-A
      v4: M-A, M-L, L-D
    """
    off_movie = info["off_movie"]
    off_director = info["off_director"]
    off_actor = info["off_actor"]
    off_link = info["off_link"]

    raw = defaultdict(set)

    for m_idx, row in movies.iterrows():
        m = off_movie + m_idx

        d_name = row["director_name"]
        l_name = row["movie_imdb_link"]
        director_node = None
        link_node = None

        if pd.notna(d_name) and d_name in director_to_idx:
            director_node = off_director + director_to_idx[d_name]
        if pd.notna(l_name) and l_name in link_to_idx:
            link_node = off_link + link_to_idx[l_name]

        actor_nodes = []
        for acol in ("actor_1_name", "actor_2_name", "actor_3_name"):
            a = row.get(acol)
            if pd.notna(a) and a in actor_to_idx:
                actor_nodes.append(off_actor + actor_to_idx[a])

        if variant == "v1":
            if director_node is not None:
                raw[("movie", "director")].add((m, director_node))
                raw[("director", "movie")].add((director_node, m))
            for a in actor_nodes:
                raw[("movie", "actor")].add((m, a))
                raw[("actor", "movie")].add((a, m))
            if link_node is not None:
                raw[("movie", "link")].add((m, link_node))
                raw[("link", "movie")].add((link_node, m))

        elif variant == "v2":
            if link_node is not None:
                raw[("movie", "link")].add((m, link_node))
                raw[("link", "movie")].add((link_node, m))
            if director_node is not None and link_node is not None:
                raw[("link", "director")].add((link_node, director_node))
                raw[("director", "link")].add((director_node, link_node))
            for a in actor_nodes:
                if link_node is not None:
                    raw[("link", "actor")].add((link_node, a))
                    raw[("actor", "link")].add((a, link_node))

        elif variant == "v3":
            if director_node is not None:
                raw[("movie", "director")].add((m, director_node))
                raw[("director", "movie")].add((director_node, m))
            if link_node is not None:
                raw[("movie", "link")].add((m, link_node))
                raw[("link", "movie")].add((link_node, m))
            for a in actor_nodes:
                if link_node is not None:
                    raw[("link", "actor")].add((link_node, a))
                    raw[("actor", "link")].add((a, link_node))

        elif variant == "v4":
            for a in actor_nodes:
                raw[("movie", "actor")].add((m, a))
                raw[("actor", "movie")].add((a, m))
            if link_node is not None:
                raw[("movie", "link")].add((m, link_node))
                raw[("link", "movie")].add((link_node, m))
            if director_node is not None and link_node is not None:
                raw[("link", "director")].add((link_node, director_node))
                raw[("director", "link")].add((director_node, link_node))

        else:
            raise ValueError(f"Unknown variant {variant}")

    return raw


def collapse_to_canonical_semantic_edges(raw, variant):
    """
    Full traversal + skip-node collapse into canonical semantic edges.
    """
    semantic = defaultdict(set)

    # movie-link is present directly in all four variants
    for e in raw.get(("movie", "link"), set()):
        semantic["movie_to_link"].add(e)
    for e in raw.get(("link", "movie"), set()):
        semantic["link_to_movie"].add(e)

    # movie-director
    if variant in ("v1", "v3"):
        for e in raw.get(("movie", "director"), set()):
            semantic["movie_to_director"].add(e)
        for e in raw.get(("director", "movie"), set()):
            semantic["director_to_movie"].add(e)
    else:
        # v2, v4: recover via movie-link-director
        ml = raw.get(("movie", "link"), set())
        ld = raw.get(("link", "director"), set())

        by_link_to_director = defaultdict(set)
        for l, d in ld:
            by_link_to_director[l].add(d)

        for m, l in ml:
            for d in by_link_to_director.get(l, []):
                semantic["movie_to_director"].add((m, d))
                semantic["director_to_movie"].add((d, m))

    # movie-actor
    if variant in ("v1", "v4"):
        for e in raw.get(("movie", "actor"), set()):
            semantic["movie_to_actor"].add(e)
        for e in raw.get(("actor", "movie"), set()):
            semantic["actor_to_movie"].add(e)
    else:
        # v2, v3: recover via movie-link-actor
        ml = raw.get(("movie", "link"), set())
        la = raw.get(("link", "actor"), set())

        by_link_to_actor = defaultdict(set)
        for l, a in la:
            by_link_to_actor[l].add(a)

        for m, l in ml:
            for a in by_link_to_actor.get(l, []):
                semantic["movie_to_actor"].add((m, a))
                semantic["actor_to_movie"].add((a, m))

    return semantic


def add_universal_link_entity_edges(semantic, movies, director_to_idx, actor_to_idx, link_to_idx, info):
    """
    Add link <-> director and link <-> actor for each movie row (same for all variants).
    """
    off_director = info["off_director"]
    off_actor = info["off_actor"]
    off_link = info["off_link"]

    for m_idx, row in movies.iterrows():
        d_name = row["director_name"]
        l_name = row["movie_imdb_link"]
        if pd.notna(l_name) and l_name in link_to_idx:
            l_node = off_link + link_to_idx[l_name]
            if pd.notna(d_name) and d_name in director_to_idx:
                d_node = off_director + director_to_idx[d_name]
                semantic["link_to_director"].add((l_node, d_node))
                semantic["director_to_link"].add((d_node, l_node))
            for acol in ("actor_1_name", "actor_2_name", "actor_3_name"):
                a = row.get(acol)
                if pd.notna(a) and a in actor_to_idx:
                    a_node = off_actor + actor_to_idx[a]
                    semantic["link_to_actor"].add((l_node, a_node))
                    semantic["actor_to_link"].add((a_node, l_node))


def build_pyg_edge_tensors(semantic):
    edge_src = []
    edge_dst = []
    edge_type = []

    relation_order = [
        ("movie_to_director", REL["movie_to_director"]),
        ("director_to_movie", REL["director_to_movie"]),
        ("movie_to_actor", REL["movie_to_actor"]),
        ("actor_to_movie", REL["actor_to_movie"]),
        ("movie_to_link", REL["movie_to_link"]),
        ("link_to_movie", REL["link_to_movie"]),
        ("link_to_director", REL["link_to_director"]),
        ("director_to_link", REL["director_to_link"]),
        ("link_to_actor", REL["link_to_actor"]),
        ("actor_to_link", REL["actor_to_link"]),
    ]

    for rel_name, rel_id in relation_order:
        edges = sorted(semantic[rel_name])
        for u, v in edges:
            edge_src.append(u)
            edge_dst.append(v)
            edge_type.append(rel_id)

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    return edge_index, edge_type


def build_shared_splits(num_movies):
    idx = np.arange(num_movies)
    train_idx, val_idx = train_test_split(
        idx, test_size=int(0.1 * len(idx)), random_state=SEED
    )
    train_idx, test_idx = train_test_split(
        train_idx, test_size=int(0.2 * len(idx)), random_state=SEED
    )
    train_idx.sort()
    val_idx.sort()
    test_idx.sort()
    return train_idx, val_idx, test_idx


def _build_universal_semantic_edges(movies, director_to_idx, actor_to_idx, link_to_idx, info):
    """
    Build a single *universal* canonical semantic graph:
      movie <-> director, movie <-> actor, movie <-> link,
      link <-> director, link <-> actor

    This ignores v1-v4 raw layouts entirely so all variant output dirs can be byte-identical.
    """
    off_movie = info["off_movie"]
    off_director = info["off_director"]
    off_actor = info["off_actor"]
    off_link = info["off_link"]

    semantic = defaultdict(set)
    for m_idx, row in movies.iterrows():
        m = off_movie + int(m_idx)

        d_name = row["director_name"]
        l_name = row["movie_imdb_link"]
        director_node = None
        link_node = None

        if pd.notna(d_name) and d_name in director_to_idx:
            director_node = off_director + director_to_idx[d_name]
            semantic["movie_to_director"].add((m, director_node))
            semantic["director_to_movie"].add((director_node, m))

        if pd.notna(l_name) and l_name in link_to_idx:
            link_node = off_link + link_to_idx[l_name]
            semantic["movie_to_link"].add((m, link_node))
            semantic["link_to_movie"].add((link_node, m))

        actor_nodes = []
        for acol in ("actor_1_name", "actor_2_name", "actor_3_name"):
            a = row.get(acol)
            if pd.notna(a) and a in actor_to_idx:
                actor_nodes.append(off_actor + actor_to_idx[a])

        for a_node in actor_nodes:
            semantic["movie_to_actor"].add((m, a_node))
            semantic["actor_to_movie"].add((a_node, m))

        # Link entity connectivity (only if link exists for this movie)
        if link_node is not None:
            if director_node is not None:
                semantic["link_to_director"].add((link_node, director_node))
                semantic["director_to_link"].add((director_node, link_node))
            for a_node in actor_nodes:
                semantic["link_to_actor"].add((link_node, a_node))
                semantic["actor_to_link"].add((a_node, link_node))

    return semantic


def _write_variant_dir(out_dir, X, edge_index, edge_type, y, train_mask, val_mask, test_mask, type_mask, meta):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(torch.tensor(X, dtype=torch.float32), os.path.join(out_dir, "x.pt"))
    torch.save(edge_index, os.path.join(out_dir, "edge_index.pt"))
    torch.save(edge_type, os.path.join(out_dir, "edge_type.pt"))
    torch.save(torch.tensor(y, dtype=torch.long), os.path.join(out_dir, "y.pt"))
    torch.save(torch.tensor(train_mask), os.path.join(out_dir, "train_mask.pt"))
    torch.save(torch.tensor(val_mask), os.path.join(out_dir, "val_mask.pt"))
    torch.save(torch.tensor(test_mask), os.path.join(out_dir, "test_mask.pt"))
    np.save(os.path.join(out_dir, "type_mask.npy"), type_mask)
    torch.save(meta, os.path.join(out_dir, "meta.pt"))


def preprocess_universal(variants):
    movies, labels = load_and_filter_movies()
    directors, actors, links, director_to_idx, actor_to_idx, link_to_idx = build_vocabularies(movies)
    info = build_global_node_index(movies, directors, actors, links)

    movie_X = build_movie_features(movies)
    X = build_all_node_features(movie_X, movies, director_to_idx, actor_to_idx, link_to_idx, info)

    semantic = _build_universal_semantic_edges(movies, director_to_idx, actor_to_idx, link_to_idx, info)
    edge_index, edge_type = build_pyg_edge_tensors(semantic)

    train_idx, val_idx, test_idx = build_shared_splits(info["num_movies"])

    # Masks only on movie nodes
    train_mask = np.zeros(info["n_total"], dtype=bool)
    val_mask = np.zeros(info["n_total"], dtype=bool)
    test_mask = np.zeros(info["n_total"], dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    y = np.full(info["n_total"], -1, dtype=np.int64)
    y[:info["num_movies"]] = labels

    meta = {
        "num_nodes": info["n_total"],
        "num_movies": info["num_movies"],
        "num_relations": len(REL),
        "num_classes": 3,
        "variant": "universal",
    }

    for v in variants:
        out_dir = VARIANT_OUT[v]
        _write_variant_dir(
            out_dir,
            X,
            edge_index,
            edge_type,
            y,
            train_mask,
            val_mask,
            test_mask,
            info["type_mask"],
            meta,
        )
        print(f"[{v}] wrote UNIVERSAL graph to {out_dir}", flush=True)

    print("Universal semantic edge counts:", flush=True)
    for k in ["movie_to_director", "movie_to_actor", "movie_to_link", "link_to_director", "link_to_actor"]:
        print(f"  {k}: {len(semantic[k])}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMDb RGCN preprocessing with skip-node collapse")
    parser.add_argument(
        "--variants",
        default="v1,v2,v3,v4",
        help="Comma list of v1,v2,v3,v4 output dirs to write (all will be identical universal graph).",
    )
    args = parser.parse_args()

    set_seed(SEED)
    variants = [x.strip().lower() for x in str(args.variants).split(",") if x.strip()]
    good = {"v1", "v2", "v3", "v4"}
    bad = [v for v in variants if v not in good]
    if bad:
        raise SystemExit(f"Unknown variants: {bad}; expected subset of {sorted(good)}")
    preprocess_universal(variants)