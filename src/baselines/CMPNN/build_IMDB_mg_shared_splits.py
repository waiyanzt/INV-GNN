"""Shared movie→genre LP splits for IMDB (CMPNN). Same 70/10/20 movie split."""
import os
import argparse
import numpy as np
import pandas as pd


def read_imdb_frame(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8')
    df = df.drop_duplicates(subset='movie_imdb_link').dropna(
        subset=['movie_imdb_link', 'actor_1_name', 'director_name', 'genres']
    ).reset_index(drop=True)

    labels = np.full(len(df), -1, dtype=np.int64)
    for i, genres in df['genres'].astype(str).items():
        for g in genres.split('|'):
            g = g.strip()
            if g == 'Action':
                labels[i] = 0
                break
            elif g == 'Comedy':
                labels[i] = 1
                break
            elif g == 'Drama':
                labels[i] = 2
                break

    keep = np.where(labels >= 0)[0]
    df = df.iloc[keep].reset_index(drop=True)
    df['label'] = labels[keep]
    return df


def main():
    ap = argparse.ArgumentParser(description='Build IMDB movie-genre shared splits')
    ap.add_argument('--csv', required=True, help='movie_metadata.csv (MAGNN IMDB raw)')
    ap.add_argument('--out', default='IMDB_mg_shared_splits.npz')
    ap.add_argument('--seed', type=int, default=1566911444)
    args = ap.parse_args()

    df = read_imdb_frame(args.csv)
    n = len(df)
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)

    n_train = max(1, int(round(0.7 * n)))
    n_val = max(1, min(int(round(0.1 * n)), n - n_train))

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    def pack(rows):
        pairs = np.stack([rows, df['label'].values[rows]], axis=1).astype(np.int64)
        return pairs

    train_pos = pack(train_idx)
    val_pos = pack(val_idx)
    test_pos = pack(test_idx)

    np.savez_compressed(
        args.out,
        train_pos=train_pos,
        val_pos=val_pos,
        test_pos=test_pos,
    )
    print(f"Saved {args.out}: train={len(train_pos)}, val={len(val_pos)}, test={len(test_pos)}")


if __name__ == '__main__':
    main()
