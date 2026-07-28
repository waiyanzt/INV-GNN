"""
Preprocess raw IMDB data into 6 graph variants for SeHGNN node classification.

Node types: 0=Movie, 1=Director, 2=Actor, 3=Link (IMDB link node)
Task:       Movie genre classification (Action/Comedy/Drama, 3 classes)

Variant connectivity (all share movie-link edges):
  1: actor-movie,  director-movie
  2: actor-link,   director-link
  3: actor-link,   director-movie
  4: actor-movie,  director-link
  5: actor-movie,  actor-link,  director-movie,  director-link  (universal)
  6: same as 5 — universal graph; no metapath filtering applied at runtime

Usage:
    python preprocess_IMDB.py
"""

import os
import pathlib

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

RAW_CSV = '../../../data/raw/IMDB/movie_metadata.csv'
SAVE_ROOT = 'data'
RAND_SEED = 1566911444

VARIANT_EDGES = {
    1: {'actor': ['movie'],         'director': ['movie']},
    2: {'actor': ['link'],          'director': ['link']},
    3: {'actor': ['link'],          'director': ['movie']},
    4: {'actor': ['movie'],         'director': ['link']},
    5: {'actor': ['movie', 'link'], 'director': ['movie', 'link']},  # universal (pruned at runtime)
    6: {'actor': ['movie', 'link'], 'director': ['movie', 'link']},  # universal (all 77 metapaths)
}


def load_and_clean():
    movies = (
        pd.read_csv(RAW_CSV, encoding='utf-8')
        .drop_duplicates(subset=['movie_imdb_link'])
        .dropna(axis=0, subset=['actor_1_name', 'director_name'])
        .reset_index(drop=True)
    )

    labels = np.zeros(len(movies), dtype=int)
    for idx, genres in movies['genres'].items():
        labels[idx] = -1
        for genre in genres.split('|'):
            if genre == 'Action':
                labels[idx] = 0
                break
            elif genre == 'Comedy':
                labels[idx] = 1
                break
            elif genre == 'Drama':
                labels[idx] = 2
                break

    unwanted = np.where(labels == -1)[0]
    movies = movies.drop(unwanted).reset_index(drop=True)
    labels = np.delete(labels, unwanted, 0)

    directors = sorted(set(movies['director_name'].dropna()))
    actors = sorted(set(
        movies['actor_1_name'].dropna().to_list() +
        movies['actor_2_name'].dropna().to_list() +
        movies['actor_3_name'].dropna().to_list()
    ))
    imdb_links = list(set(movies['movie_imdb_link'].dropna()))

    return movies, labels, directors, actors, imdb_links


def build_adjacency(movies, directors, actors, imdb_links, variant_cfg):
    n_m = len(movies)
    n_d = len(directors)
    n_a = len(actors)
    n_l = len(imdb_links)
    dim = n_m + n_d + n_a + n_l

    off_d = n_m
    off_a = n_m + n_d
    off_l = n_m + n_d + n_a

    type_mask = np.zeros(dim, dtype=int)
    type_mask[off_d:off_a] = 1
    type_mask[off_a:off_l] = 2
    type_mask[off_l:] = 3

    adjM = np.zeros((dim, dim), dtype=int)

    actor_targets = variant_cfg['actor']
    director_targets = variant_cfg['director']

    for movie_idx, row in movies.iterrows():
        link_idx = imdb_links.index(row['movie_imdb_link'])
        link_global = off_l + link_idx

        # movie-link (always present)
        adjM[movie_idx, link_global] = 1
        adjM[link_global, movie_idx] = 1

        # director edges
        if row['director_name'] in directors:
            d_idx = off_d + directors.index(row['director_name'])
            for tgt in director_targets:
                hub = movie_idx if tgt == 'movie' else link_global
                adjM[hub, d_idx] = 1
                adjM[d_idx, hub] = 1

        # actor edges (all 3 actors)
        for acol in ['actor_1_name', 'actor_2_name', 'actor_3_name']:
            aname = row.get(acol)
            if pd.notna(aname) and aname in actors:
                a_idx = off_a + actors.index(aname)
                for tgt in actor_targets:
                    hub = movie_idx if tgt == 'movie' else link_global
                    adjM[hub, a_idx] = 1
                    adjM[a_idx, hub] = 1

    return adjM, type_mask


def main():
    print('Loading raw IMDB data ...')
    movies, labels, directors, actors, imdb_links = load_and_clean()
    print(f'  Movies: {len(movies)}, Directors: {len(directors)}, '
          f'Actors: {len(actors)}, Links: {len(imdb_links)}')
    print(f'  Labels: {len(labels)}, classes: {np.unique(labels).tolist()}')

    # Movie BoW features (shared across all variants)
    vectorizer = CountVectorizer(min_df=2)
    movie_X = vectorizer.fit_transform(
        movies['plot_keywords'].fillna('').apply(
            lambda x: str(x).replace('|', ' ')
        ).values
    )
    print(f'  Movie BoW features: {movie_X.shape}')

    # Splits (shared across all variants)
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size=int(0.1 * len(labels)),
        random_state=RAND_SEED,
    )
    train_idx, test_idx = train_test_split(
        train_idx,
        test_size=int(0.2 * len(labels)),
        random_state=RAND_SEED,
    )
    train_idx.sort()
    val_idx.sort()
    test_idx.sort()
    print(f'  Splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}')

    for var_id, cfg in VARIANT_EDGES.items():
        a_tgts = '+'.join(cfg['actor'])
        d_tgts = '+'.join(cfg['director'])
        print(f'\n--- Variant {var_id}: actor->{a_tgts}, director->{d_tgts} ---')
        save_dir = os.path.join(SAVE_ROOT, f'IMDB_var{var_id}')
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

        adjM, type_mask = build_adjacency(movies, directors, actors, imdb_links, cfg)
        sparse_adj = scipy.sparse.csr_matrix(adjM)
        print(f'  adjM: {sparse_adj.shape}, nnz={sparse_adj.nnz}')

        scipy.sparse.save_npz(os.path.join(save_dir, 'adjM.npz'), sparse_adj)
        np.save(os.path.join(save_dir, 'node_types.npy'), type_mask)
        scipy.sparse.save_npz(os.path.join(save_dir, 'features_0.npz'), movie_X)
        np.save(os.path.join(save_dir, 'labels.npy'), labels)
        np.savez(os.path.join(save_dir, 'train_val_test_idx.npz'),
                 train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
        print(f'  Saved to {save_dir}/')

    print('\nDone.')


if __name__ == '__main__':
    main()
