"""
Preprocess raw IMDB data into 5 graph variants in HGB format for SlotGAT.

Node types: 0=Movie, 1=Director, 2=Actor, 3=Link (IMDB link node)
Task:       Movie genre classification (Action/Comedy/Drama, 3 classes)

Variant connectivity (all share movie-link edges):
  1: actor-movie,  director-movie
  2: actor-link,   director-link
  3: actor-link,   director-movie
  4: actor-movie,  director-link
  5: actor-movie,  actor-link, director-movie, director-link

Output per variant (HGB format):
  node.dat, link.dat, label.dat, label.dat.test, info.dat

Usage:
    python preprocess_IMDB.py
"""

import os
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_CSV = os.path.join(SCRIPT_DIR, '..', 'MAGNN', 'data', 'raw', 'IMDB', 'movie_metadata.csv')
SAVE_ROOT = os.path.join(SCRIPT_DIR, 'data')
RAND_SEED = 1566911444

VARIANT_EDGES = {
    1: {'actor': 'movie',  'director': 'movie'},
    2: {'actor': 'link',   'director': 'link'},
    3: {'actor': 'link',   'director': 'movie'},
    4: {'actor': 'movie',  'director': 'link'},
    5: {'actor': ('movie', 'link'), 'director': ('movie', 'link')},
}


def _normalize_targets(target_spec):
    if isinstance(target_spec, str):
        targets = [target_spec]
    else:
        targets = list(target_spec)

    seen = set()
    normalized = []
    for t in targets:
        if t not in {'movie', 'link'}:
            raise ValueError(f'Unsupported target type: {t}')
        if t not in seen:
            seen.add(t)
            normalized.append(t)
    return normalized


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


def build_edges(movies, directors, actors, imdb_links, variant_cfg):
    """Build list of (src_id, dst_id, relation_id) tuples for a variant."""
    n_m = len(movies)
    n_d = len(directors)
    n_a = len(actors)

    off_d = n_m
    off_a = n_m + n_d
    off_l = n_m + n_d + n_a

    actor_targets = _normalize_targets(variant_cfg['actor'])
    director_targets = _normalize_targets(variant_cfg['director'])

    edges = []
    rel_id = 0
    rel_meta = {}

    # movie-link edges (always present): rel 0 = movie->link, rel 1 = link->movie
    rel_meta[0] = (0, 3)
    rel_meta[1] = (3, 0)

    rel_lookup = {}
    next_rel = 2
    for role, targets in (('director', director_targets), ('actor', actor_targets)):
        for target in targets:
            if role == 'director':
                if target == 'movie':
                    rel_meta[next_rel] = (0, 1)      # movie->director
                    rel_meta[next_rel + 1] = (1, 0)  # director->movie
                else:
                    rel_meta[next_rel] = (3, 1)      # link->director
                    rel_meta[next_rel + 1] = (1, 3)  # director->link
            else:
                if target == 'movie':
                    rel_meta[next_rel] = (0, 2)      # movie->actor
                    rel_meta[next_rel + 1] = (2, 0)  # actor->movie
                else:
                    rel_meta[next_rel] = (3, 2)      # link->actor
                    rel_meta[next_rel + 1] = (2, 3)  # actor->link
            rel_lookup[(role, target, 'forward')] = next_rel
            rel_lookup[(role, target, 'backward')] = next_rel + 1
            next_rel += 2

    for movie_idx, row in movies.iterrows():
        link_idx = imdb_links.index(row['movie_imdb_link'])
        link_global = off_l + link_idx

        # movie-link
        edges.append((movie_idx, link_global, 0, 1.0))
        edges.append((link_global, movie_idx, 1, 1.0))

        # director edges
        if row['director_name'] in directors:
            d_global = off_d + directors.index(row['director_name'])
            for target in director_targets:
                if target == 'movie':
                    edges.append((movie_idx, d_global, rel_lookup[('director', 'movie', 'forward')], 1.0))
                    edges.append((d_global, movie_idx, rel_lookup[('director', 'movie', 'backward')], 1.0))
                else:
                    edges.append((link_global, d_global, rel_lookup[('director', 'link', 'forward')], 1.0))
                    edges.append((d_global, link_global, rel_lookup[('director', 'link', 'backward')], 1.0))

        # actor edges
        for acol in ['actor_1_name', 'actor_2_name', 'actor_3_name']:
            aname = row.get(acol)
            if pd.notna(aname) and aname in actors:
                a_global = off_a + actors.index(aname)
                for target in actor_targets:
                    if target == 'movie':
                        edges.append((movie_idx, a_global, rel_lookup[('actor', 'movie', 'forward')], 1.0))
                        edges.append((a_global, movie_idx, rel_lookup[('actor', 'movie', 'backward')], 1.0))
                    else:
                        edges.append((link_global, a_global, rel_lookup[('actor', 'link', 'forward')], 1.0))
                        edges.append((a_global, link_global, rel_lookup[('actor', 'link', 'backward')], 1.0))

    return edges, rel_meta


def write_node_dat(path, movies, directors, actors, imdb_links, movie_features):
    """Write node.dat in HGB format."""
    n_m = len(movies)
    n_d = len(directors)
    n_a = len(actors)
    off_d = n_m
    off_a = n_m + n_d
    off_l = n_m + n_d + n_a

    with open(path, 'w', encoding='utf-8') as f:
        for i in range(n_m):
            name = str(movies.iloc[i]['movie_title']).strip().replace('\t', '_').replace(' ', '_')
            feat_str = ','.join(f'{v:.6f}' for v in movie_features[i])
            f.write(f'{i}\t{name}\t0\t{feat_str}\n')
        for i, dname in enumerate(directors):
            nid = off_d + i
            name = dname.strip().replace('\t', '_').replace(' ', '_')
            f.write(f'{nid}\t{name}\t1\n')
        for i, aname in enumerate(actors):
            nid = off_a + i
            name = aname.strip().replace('\t', '_').replace(' ', '_')
            f.write(f'{nid}\t{name}\t2\n')
        for i, link in enumerate(imdb_links):
            nid = off_l + i
            name = link.strip().replace('\t', '_').replace(' ', '_')
            f.write(f'{nid}\t{name}\t3\n')


def write_link_dat(path, edges):
    """Write link.dat in HGB format."""
    with open(path, 'w', encoding='utf-8') as f:
        for (src, dst, rid, w) in edges:
            f.write(f'{src}\t{dst}\t{rid}\t{w}\n')


def write_label_dat(path, movie_indices, movies, labels):
    """Write label.dat or label.dat.test."""
    with open(path, 'w', encoding='utf-8') as f:
        for idx in movie_indices:
            name = str(movies.iloc[idx]['movie_title']).strip().replace('\t', '_').replace(' ', '_')
            f.write(f'{idx}\t{name}\t0\t{labels[idx]}\n')


def write_info_dat(path, movies, directors, actors, imdb_links, rel_meta, feat_dim):
    """Write info.dat with JSON metadata."""
    info = {
        "node.dat": {
            "node type": {
                "0": "movie",
                "1": "director",
                "2": "actor",
                "3": "link"
            },
            "num_nodes": {
                "0": len(movies),
                "1": len(directors),
                "2": len(actors),
                "3": len(imdb_links)
            },
            "feature_dim": feat_dim
        },
        "link.dat": {
            "link type": {}
        },
        "label.dat": {
            "num_classes": 3,
            "label type": {
                "0": "Action",
                "1": "Comedy",
                "2": "Drama"
            }
        }
    }
    type_names = {0: "movie", 1: "director", 2: "actor", 3: "link"}
    for rid, (h, t) in sorted(rel_meta.items()):
        info["link.dat"]["link type"][str(rid)] = f"{type_names[h]}-{type_names[t]}"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=4)


def main():
    print('Loading raw IMDB data ...')
    movies, labels, directors, actors, imdb_links = load_and_clean()
    print(f'  Movies: {len(movies)}, Directors: {len(directors)}, '
          f'Actors: {len(actors)}, Links: {len(imdb_links)}')
    print(f'  Labels: {len(labels)}, classes: {np.unique(labels).tolist()}')

    # Movie BoW features from plot_keywords (shared across variants)
    vectorizer = CountVectorizer(min_df=2)
    movie_X = vectorizer.fit_transform(
        movies['plot_keywords'].fillna('').apply(
            lambda x: str(x).replace('|', ' ')
        ).values
    )
    movie_features = movie_X.toarray().astype(float)
    feat_dim = movie_features.shape[1]
    print(f'  Movie BoW features: {movie_features.shape}')

    # Train/test split (matching SeHGNN preprocessing)
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
    # label.dat gets train+val, label.dat.test gets test
    train_val_idx = np.sort(np.concatenate([train_idx, val_idx]))
    print(f'  Splits: train+val={len(train_val_idx)}, test={len(test_idx)}')

    for var_id, cfg in VARIANT_EDGES.items():
        actor_targets = ','.join(_normalize_targets(cfg['actor']))
        director_targets = ','.join(_normalize_targets(cfg['director']))
        print(f'\n--- Variant {var_id}: actor->{actor_targets}, director->{director_targets} ---')
        save_dir = os.path.join(SAVE_ROOT, f'IMDB_var{var_id}')
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

        edges, rel_meta = build_edges(movies, directors, actors, imdb_links, cfg)
        print(f'  Edges: {len(edges)}, relation types: {len(rel_meta)}')

        write_node_dat(os.path.join(save_dir, 'node.dat'),
                       movies, directors, actors, imdb_links, movie_features)
        write_link_dat(os.path.join(save_dir, 'link.dat'), edges)
        write_label_dat(os.path.join(save_dir, 'label.dat'),
                        train_val_idx, movies, labels)
        write_label_dat(os.path.join(save_dir, 'label.dat.test'),
                        test_idx, movies, labels)
        write_info_dat(os.path.join(save_dir, 'info.dat'),
                       movies, directors, actors, imdb_links, rel_meta, feat_dim)

        np.save(os.path.join(save_dir, 'labels.npy'), labels)
        np.savez(os.path.join(save_dir, 'train_val_test_idx.npz'),
                 train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
        print(f'  Saved to {save_dir}/')

    print('\nDone.')


if __name__ == '__main__':
    main()
