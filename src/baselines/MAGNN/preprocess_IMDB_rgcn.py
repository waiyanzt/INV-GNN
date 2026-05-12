#!/usr/bin/env python3
"""
to run: python preprocess_IMDB_rgcn.py --variant v1,v2,v3,v4

"""
import os
import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

SEED = 1566911444


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _parse_variants(s: str):
    vals = [x.strip().lower() for x in str(s).split(",") if x.strip()]
    good = {"v1", "v2", "v3", "v4"}
    bad = [v for v in vals if v not in good]
    if bad:
        raise SystemExit(f"Unknown variant(s): {bad}; expected subset of {sorted(good)}")
    return vals


def maybe_make_splits(labels_np, seed: int, split_npz: str | None):
    idx = np.arange(len(labels_np), dtype=np.int64)

    if split_npz and os.path.exists(split_npz):
        z = np.load(split_npz)
        return (
            z["train_idx"].astype(np.int64),
            z["val_idx"].astype(np.int64),
            z["test_idx"].astype(np.int64),
        )

    train_idx, temp_idx, y_train, y_temp = train_test_split(
        idx, labels_np, test_size=0.4, stratify=labels_np, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=y_temp, random_state=seed
    )
    return (
        np.sort(train_idx.astype(np.int64)),
        np.sort(val_idx.astype(np.int64)),
        np.sort(test_idx.astype(np.int64)),
    )


def preprocess_one(args, variant: str):
    out_dir = Path(args.out_dir) / f"IMDB_rgcn_{variant}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== preprocess IMDB RGCN {variant} ===", flush=True)

    csv_path = Path(args.raw_dir) / args.movie_metadata_file
    movies = (
        pd.read_csv(csv_path, encoding="utf-8")
        .drop_duplicates(subset=["movie_imdb_link"])
        .dropna(axis=0, subset=["actor_1_name", "director_name", "movie_imdb_link"])
        .reset_index(drop=True)
    )

    # labels: Action=0, Comedy=1, Drama=2, others removed
    labels = np.full(len(movies), -1, dtype=int)
    for movie_idx, genres in movies["genres"].fillna("").items():
        for genre in str(genres).split("|"):
            if genre == "Action":
                labels[movie_idx] = 0
                break
            elif genre == "Comedy":
                labels[movie_idx] = 1
                break
            elif genre == "Drama":
                labels[movie_idx] = 2
                break

    keep_idx = np.where(labels != -1)[0]
    movies = movies.iloc[keep_idx].reset_index(drop=True)
    labels = labels[keep_idx]

    directors = sorted(set(movies["director_name"].dropna().tolist()))
    actors = sorted(set(
        movies["actor_1_name"].dropna().tolist()
        + movies["actor_2_name"].dropna().tolist()
        + movies["actor_3_name"].dropna().tolist()
    ))
    imdb_links = sorted(set(movies["movie_imdb_link"].dropna().tolist()))

    movie_map = {i: i for i in range(len(movies))}
    actor_map = {x: i for i, x in enumerate(actors)}
    director_map = {x: i for i, x in enumerate(directors)}
    link_map = {x: i for i, x in enumerate(imdb_links)}

    num_nodes = {
        "movie": len(movie_map),
        "actor": len(actor_map),
        "director": len(director_map),
        "link": len(link_map),
    }

    print(
        f"Movies={num_nodes['movie']} Actors={num_nodes['actor']} "
        f"Directors={num_nodes['director']} Links={num_nodes['link']}",
        flush=True,
    )

    train_idx, val_idx, test_idx = maybe_make_splits(labels, args.seed, args.split_npz)

    # canonical local edge tables from movie_metadata.csv
    actor_movie_pairs = set()
    movie_director_pairs = set()
    movie_link_pairs = set()

    for movie_idx, row in movies.iterrows():
        if row["director_name"] in director_map:
            movie_director_pairs.add((movie_idx, director_map[row["director_name"]]))

        if row["movie_imdb_link"] in link_map:
            movie_link_pairs.add((movie_idx, link_map[row["movie_imdb_link"]]))

        for acol in ("actor_1_name", "actor_2_name", "actor_3_name"):
            val = row.get(acol)
            if pd.notna(val) and val in actor_map:
                actor_movie_pairs.add((actor_map[val], movie_idx))

    graph_data = {}

    if variant == "v1":
        # actor-movie, movie-link, movie-director
        am = sorted(actor_movie_pairs)
        md = sorted(movie_director_pairs)
        ml = sorted(movie_link_pairs)

        graph_data["actor-movie"] = (
            np.array([a for a, m in am], dtype=np.int64),
            np.array([m for a, m in am], dtype=np.int64),
        )
        graph_data["movie-director"] = (
            np.array([m for m, d in md], dtype=np.int64),
            np.array([d for m, d in md], dtype=np.int64),
        )
        graph_data["movie-link"] = (
            np.array([m for m, l in ml], dtype=np.int64),
            np.array([l for m, l in ml], dtype=np.int64),
        )

    elif variant == "v2":
        # actor-link, link-movie, link-director
        movie_to_link = {m: l for m, l in movie_link_pairs}
        movie_to_director = {m: d for m, d in movie_director_pairs}

        actor_link_pairs = set()
        link_director_pairs = set()

        for a, m in actor_movie_pairs:
            if m in movie_to_link:
                actor_link_pairs.add((a, movie_to_link[m]))

        for m, l in movie_to_link.items():
            if m in movie_to_director:
                link_director_pairs.add((l, movie_to_director[m]))

        al = sorted(actor_link_pairs)
        lm = sorted((l, m) for m, l in movie_link_pairs)
        ld = sorted(link_director_pairs)

        graph_data["actor-link"] = (
            np.array([a for a, l in al], dtype=np.int64),
            np.array([l for a, l in al], dtype=np.int64),
        )
        graph_data["link-movie"] = (
            np.array([l for l, m in lm], dtype=np.int64),
            np.array([m for l, m in lm], dtype=np.int64),
        )
        graph_data["link-director"] = (
            np.array([l for l, d in ld], dtype=np.int64),
            np.array([d for l, d in ld], dtype=np.int64),
        )

    elif variant == "v3":
        # actor-link, link-movie, movie-director
        movie_to_link = {m: l for m, l in movie_link_pairs}

        actor_link_pairs = set()
        for a, m in actor_movie_pairs:
            if m in movie_to_link:
                actor_link_pairs.add((a, movie_to_link[m]))

        al = sorted(actor_link_pairs)
        lm = sorted((l, m) for m, l in movie_link_pairs)
        md = sorted(movie_director_pairs)

        graph_data["actor-link"] = (
            np.array([a for a, l in al], dtype=np.int64),
            np.array([l for a, l in al], dtype=np.int64),
        )
        graph_data["link-movie"] = (
            np.array([l for l, m in lm], dtype=np.int64),
            np.array([m for l, m in lm], dtype=np.int64),
        )
        graph_data["movie-director"] = (
            np.array([m for m, d in md], dtype=np.int64),
            np.array([d for m, d in md], dtype=np.int64),
        )

    elif variant == "v4":
        # actor-movie, movie-link, link-director
        movie_to_link = {m: l for m, l in movie_link_pairs}
        movie_to_director = {m: d for m, d in movie_director_pairs}

        link_director_pairs = set()
        for m, l in movie_to_link.items():
            if m in movie_to_director:
                link_director_pairs.add((l, movie_to_director[m]))

        am = sorted(actor_movie_pairs)
        ml = sorted(movie_link_pairs)
        ld = sorted(link_director_pairs)

        graph_data["actor-movie"] = (
            np.array([a for a, m in am], dtype=np.int64),
            np.array([m for a, m in am], dtype=np.int64),
        )
        graph_data["movie-link"] = (
            np.array([m for m, l in ml], dtype=np.int64),
            np.array([l for m, l in ml], dtype=np.int64),
        )
        graph_data["link-director"] = (
            np.array([l for l, d in ld], dtype=np.int64),
            np.array([d for l, d in ld], dtype=np.int64),
        )

    else:
        raise ValueError(variant)

    meta = {
        "num_nodes": num_nodes,
        "labels": torch.tensor(labels, dtype=torch.long),
        "train_idx": torch.tensor(train_idx, dtype=torch.long),
        "val_idx": torch.tensor(val_idx, dtype=torch.long),
        "test_idx": torch.tensor(test_idx, dtype=torch.long),
        "variant": variant,
    }

    torch.save(graph_data, out_dir / "graph_data.pt")
    torch.save(meta, out_dir / "meta.pt")
    print(f"Saved to {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Preprocess IMDB RGCN variants from movie_metadata.csv")
    ap.add_argument("--variant", default="v1,v2,v3,v4",
                    help="Comma-separated list like v1,v2,v3,v4")
    ap.add_argument("--raw-dir", default="data/raw/IMDB/")
    ap.add_argument("--movie-metadata-file", default="movie_metadata.csv")
    ap.add_argument("--split-npz", default="")
    ap.add_argument("--out-dir", default="data/preprocessed")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    set_seed(args.seed)
    for v in _parse_variants(args.variant):
        preprocess_one(args, v)


if __name__ == "__main__":
    main()