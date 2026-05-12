"""Build IMDB movie-link LP splits (shared across baselines)."""

import os
import argparse

import numpy as np
import pandas as pd


def read_filtered_movies(csv_path):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to movie_metadata.csv")
    parser.add_argument("--out", default="IMDB_ml_shared_splits.npz")
    parser.add_argument("--seed", type=int, default=1566911444)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.1)
    args = parser.parse_args()

    movies = read_filtered_movies(args.csv)
    n = len(movies)
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)

    n_train = max(1, int(round(args.train_frac * n)))
    n_val = max(1, int(round(args.val_frac * n)))
    n_test = n - n_train - n_val

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    def to_pairs(idxs):
        return np.stack([idxs, idxs], axis=1)

    train_pos = to_pairs(train_idx)
    val_pos = to_pairs(val_idx)
    test_pos = to_pairs(test_idx)

    np.savez(args.out, train_pos=train_pos, val_pos=val_pos, test_pos=test_pos)
    print(f"Saved {args.out}  train={len(train_pos)} val={len(val_pos)} test={len(test_pos)}")


if __name__ == "__main__":
    main()
