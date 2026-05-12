#!/usr/bin/env python3
"""
DBLP IMDB skip-node preprocessing for all 4 graph variations
using FULL-TRAVERSAL-THEN-COMPRESS.

Meaning of skip node:
- graph traversal still uses the full real metapath
- saved metapath instance arrays remove skipped interior node positions

Node types:
  0 = Movie
  1 = Director
  2 = Actor
  3 = IMDBLink

Usage:
  python preprocess_IMDB_skip.py --variant v1
  python preprocess_IMDB_skip.py --variant v2
  python preprocess_IMDB_skip.py --variant v3
  python preprocess_IMDB_skip.py --variant v4
"""

import pathlib
import argparse
import numpy as np
import scipy.sparse
import pandas as pd
import networkx as nx
from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x


# ---------------------------------------------------------------------
# Skip-metapath helpers
# ---------------------------------------------------------------------
def parse_skip_metapath(spec):
    """
    Example:
      [1, [0], 3, [0], 1]
        full     = (1, 0, 3, 0, 1)
        skip_pos = {1, 3}
        semantic = (1, 3, 1)

    Traversal uses FULL.
    Saving uses SEMANTIC.
    """
    full = []
    skip_pos = set()

    for i, x in enumerate(spec):
        if isinstance(x, list):
            if len(x) != 1:
                raise ValueError(f"Skip notation must be a single-item list, got {x}")
            full.append(x[0])
            skip_pos.add(i)
        else:
            full.append(x)

    full = tuple(full)
    semantic = tuple(full[i] for i in range(len(full)) if i not in skip_pos)
    return full, skip_pos, semantic


def compress_full_path(path, skip_pos):
    """
    Remove skipped POSITIONS from a full metapath instance path.
    Example:
      full path: (dA, m1, l3, m2, dB)
      skip_pos = {1, 3}
      -> (dA, l3, dB)
    """
    return tuple(path[i] for i in range(len(path)) if i not in skip_pos)


def validate_symmetric_metapath(full):
    """
    This preprocessing assumes symmetric metapaths, e.g.:
      (0,1,0), (1,0,2,0,1), ...
    """
    if tuple(full) != tuple(full[::-1]):
        raise ValueError(f"Metapath must be symmetric: {full}")


# ---------------------------------------------------------------------
# Core path enumeration (FULL traversal)
# ---------------------------------------------------------------------
def build_masked_graph_for_full_metapath(adjM, type_mask, full_metapath):
    """
    Build a masked graph containing only edge types needed for the full metapath.
    """
    mask = np.zeros(adjM.shape, dtype=bool)

    for i in range((len(full_metapath) - 1) // 2):
        t1 = full_metapath[i]
        t2 = full_metapath[i + 1]

        temp = np.zeros(adjM.shape, dtype=bool)
        temp[np.ix_(type_mask == t1, type_mask == t2)] = True
        temp[np.ix_(type_mask == t2, type_mask == t1)] = True
        mask = np.logical_or(mask, temp)

    return nx.from_numpy_array((adjM * mask).astype(int))


def get_metapath_neighbor_pairs_full(adjM, type_mask, metapath_specs):
    """
    FULL traversal:
    - parse skip metapath spec into full path
    - enumerate valid full paths in the real graph
    - keep full paths in memory
    - compression is applied only when saving idx arrays
    """
    outs = []

    # Cache typed neighbors to avoid repeating expensive scans.
    typed_nbr_cache = {}

    def neighbors_of_type(node_id, ntype):
        key = (int(node_id), int(ntype))
        if key in typed_nbr_cache:
            return typed_nbr_cache[key]
        nbrs = np.where((adjM[int(node_id)] > 0) & (type_mask == ntype))[0].astype(np.int32)
        typed_nbr_cache[key] = nbrs
        return nbrs

    def enumerate_half_paths_exact(full):
        """
        Enumerate all paths following the exact half type sequence full[:half_len].
        This does NOT use shortest paths (important for skip-node equivalence).
        """
        half_len = (len(full) + 1) // 2
        source_type = int(full[0])
        source_nodes = np.where(type_mask == source_type)[0]
        by_target = defaultdict(list)

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
                by_target[p[-1]].append(p)
        return by_target

    for spec in metapath_specs:
        full, skip_pos, semantic = parse_skip_metapath(spec)
        validate_symmetric_metapath(full)

        print(f"  Full metapath: {full}")
        print(f"  Skip positions: {skip_pos if skip_pos else 'none'}")
        print(f"  Semantic metapath: {semantic}")

        metapath_to_target = enumerate_half_paths_exact(full)

        metapath_neighbor_pairs = {}
        for _, half_paths in metapath_to_target.items():
            for p1 in half_paths:
                for p2 in half_paths:
                    full_path = tuple(p1 + p2[-2::-1])
                    pair = (p1[0], p2[0])
                    metapath_neighbor_pairs[pair] = metapath_neighbor_pairs.get(pair, []) + [full_path]

        print(f"    -> {len(metapath_neighbor_pairs)} neighbor pairs found")

        outs.append({
            'spec': spec,
            'full': full,
            'skip_pos': skip_pos,
            'semantic': semantic,
            'neighbor_pairs': metapath_neighbor_pairs
        })

    return outs


def get_networkx_graph(neighbor_pairs_info, type_mask, ctr_ntype):
    """
    Build metapath-specific graph over center node type using FULL-path neighbor pairs.
    """
    indices = np.where(type_mask == ctr_ntype)[0]
    idx_mapping = {idx: i for i, idx in enumerate(indices)}

    G_list = []
    for mp_info in neighbor_pairs_info:
        metapaths = mp_info['neighbor_pairs']
        G = nx.MultiDiGraph()
        G.add_nodes_from(range(len(indices)))

        for (src, dst), paths in sorted(metapaths.items()):
            for _ in range(len(paths)):
                G.add_edge(idx_mapping[src], idx_mapping[dst])

        G_list.append(G)

    return G_list


def get_edge_metapath_idx_array(neighbor_pairs_info, deduplicate=False):
    """
    Save COMPRESSED semantic paths, but neighbor-pair construction uses FULL paths.
    """
    all_arrays = []

    for mp_info in neighbor_pairs_info:
        skip_pos = mp_info['skip_pos']
        mp_pairs = mp_info['neighbor_pairs']

        paths = []
        for _, p_list in sorted(mp_pairs.items()):
            for full_path in p_list:
                semantic_path = compress_full_path(full_path, skip_pos)
                paths.append(semantic_path)

        if deduplicate:
            paths = list(dict.fromkeys(paths))

        if not paths:
            arr = np.empty((0, 0), dtype=int)
        else:
            arr = np.array(paths, dtype=int)

        all_arrays.append(arr)
        if arr.size == 0 or arr.ndim < 2:
            print(f"    idx shape: {arr.shape} | unique endpoint pairs: 0")
        else:
            ep = np.unique(arr[:, [0, -1]], axis=0)
            print(f"    idx shape: {arr.shape} | unique endpoint pairs: {len(ep)}")

    return all_arrays


# ---------------------------------------------------------------------
# Variant-specific edge construction
# ---------------------------------------------------------------------
def build_edges_v1(adjM, movies, director_to_idx, actor_to_idx, link_to_idx,
                   n_m, n_d, n_a):
    """
    V1: M-D, M-A, M-L  (star on Movie)
    """
    d_off, a_off, l_off = n_m, n_m + n_d, n_m + n_d + n_a

    for movie_idx, row in movies.iterrows():
        director = row['director_name']
        if director in director_to_idx:
            di = d_off + director_to_idx[director]
            adjM[movie_idx, di] = 1
            adjM[di, movie_idx] = 1

        for acol in ('actor_1_name', 'actor_2_name', 'actor_3_name'):
            actor = row.get(acol)
            if pd.notna(actor) and actor in actor_to_idx:
                ai = a_off + actor_to_idx[actor]
                adjM[movie_idx, ai] = 1
                adjM[ai, movie_idx] = 1

        link = row['movie_imdb_link']
        if link in link_to_idx:
            li = l_off + link_to_idx[link]
            adjM[movie_idx, li] = 1
            adjM[li, movie_idx] = 1


def build_edges_v2(adjM, movies, director_to_idx, actor_to_idx, link_to_idx,
                   n_m, n_d, n_a):
    """
    V2: L-M, L-D, L-A  (star on Link)
    """
    d_off, a_off, l_off = n_m, n_m + n_d, n_m + n_d + n_a

    for movie_idx, row in movies.iterrows():
        link = row['movie_imdb_link']
        if link not in link_to_idx:
            continue

        li = l_off + link_to_idx[link]
        adjM[movie_idx, li] = 1
        adjM[li, movie_idx] = 1

        director = row['director_name']
        if director in director_to_idx:
            di = d_off + director_to_idx[director]
            adjM[li, di] = 1
            adjM[di, li] = 1

        for acol in ('actor_1_name', 'actor_2_name', 'actor_3_name'):
            actor = row.get(acol)
            if pd.notna(actor) and actor in actor_to_idx:
                ai = a_off + actor_to_idx[actor]
                adjM[li, ai] = 1
                adjM[ai, li] = 1


def build_edges_v3(adjM, movies, director_to_idx, actor_to_idx, link_to_idx,
                   n_m, n_d, n_a):
    """
    V3: M-D, M-L, L-A   chain A-L-M-D
    """
    d_off, a_off, l_off = n_m, n_m + n_d, n_m + n_d + n_a

    for movie_idx, row in movies.iterrows():
        director = row['director_name']
        if director in director_to_idx:
            di = d_off + director_to_idx[director]
            adjM[movie_idx, di] = 1
            adjM[di, movie_idx] = 1

        link = row['movie_imdb_link']
        if link in link_to_idx:
            li = l_off + link_to_idx[link]
            adjM[movie_idx, li] = 1
            adjM[li, movie_idx] = 1

            for acol in ('actor_1_name', 'actor_2_name', 'actor_3_name'):
                actor = row.get(acol)
                if pd.notna(actor) and actor in actor_to_idx:
                    ai = a_off + actor_to_idx[actor]
                    adjM[li, ai] = 1
                    adjM[ai, li] = 1


def build_edges_v4(adjM, movies, director_to_idx, actor_to_idx, link_to_idx,
                   n_m, n_d, n_a):
    """
    V4: M-A, M-L, L-D   chain D-L-M-A
    """
    d_off, a_off, l_off = n_m, n_m + n_d, n_m + n_d + n_a

    for movie_idx, row in movies.iterrows():
        for acol in ('actor_1_name', 'actor_2_name', 'actor_3_name'):
            actor = row.get(acol)
            if pd.notna(actor) and actor in actor_to_idx:
                ai = a_off + actor_to_idx[actor]
                adjM[movie_idx, ai] = 1
                adjM[ai, movie_idx] = 1

        link = row['movie_imdb_link']
        if link in link_to_idx:
            li = l_off + link_to_idx[link]
            adjM[movie_idx, li] = 1
            adjM[li, movie_idx] = 1

            director = row['director_name']
            if director in director_to_idx:
                di = d_off + director_to_idx[director]
                adjM[li, di] = 1
                adjM[di, li] = 1


EDGE_BUILDERS = {
    'v1': build_edges_v1,
    'v2': build_edges_v2,
    'v3': build_edges_v3,
    'v4': build_edges_v4,
}


# ---------------------------------------------------------------------
# Skip-node metapaths
# ---------------------------------------------------------------------
# v2–v4 lists are permuted so compressed semantic metapaths appear in the
# same order as v1 per center type. That matches run_IMDB_skip.SEMANTIC_METAPATHS
# (expected_metapaths / on-disk stems) for all variants.
SKIP_METAPATHS = {
    'v1': [
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
    'v2': [
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
    'v3': [
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
    'v4': [
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
# Feature construction
# ---------------------------------------------------------------------
def build_features(adjM, movies, num_movies, num_directors, num_actors, num_links):
    """
    Build 4 aligned feature blocks:
      movie_X, director_X, actor_X, link_X
    """
    vectorizer = CountVectorizer(min_df=2)
    movie_X = vectorizer.fit_transform(
        movies['plot_keywords'].fillna('').apply(lambda x: str(x).replace('|', ' ')).values
    )

    d_off = num_movies
    a_off = num_movies + num_directors
    l_off = num_movies + num_directors + num_actors

    # director <- movie
    dir_to_movie = adjM[d_off:d_off + num_directors, :num_movies].astype(float)
    dir_deg = dir_to_movie.sum(axis=1, keepdims=True)
    dir_deg[dir_deg == 0] = 1.0
    dir_to_movie_norm = dir_to_movie / dir_deg
    director_X = scipy.sparse.csr_matrix(dir_to_movie_norm).dot(movie_X)

    # actor <- movie
    act_to_movie = adjM[a_off:a_off + num_actors, :num_movies].astype(float)
    act_deg = act_to_movie.sum(axis=1, keepdims=True)
    act_deg[act_deg == 0] = 1.0
    act_to_movie_norm = act_to_movie / act_deg
    actor_X = scipy.sparse.csr_matrix(act_to_movie_norm).dot(movie_X)

    # link <- movie
    link_to_movie = adjM[l_off:l_off + num_links, :num_movies].astype(float)
    link_deg = link_to_movie.sum(axis=1, keepdims=True)
    link_deg[link_deg == 0] = 1.0
    link_to_movie_norm = link_to_movie / link_deg
    link_X = scipy.sparse.csr_matrix(link_to_movie_norm).dot(movie_X)

    return (
        scipy.sparse.csr_matrix(movie_X),
        scipy.sparse.csr_matrix(director_X),
        scipy.sparse.csr_matrix(actor_X),
        scipy.sparse.csr_matrix(link_X),
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='IMDB skip-node preprocessing (full traversal then compress)')
    parser.add_argument('--variant', choices=['v1', 'v2', 'v3', 'v4'], required=True)
    parser.add_argument('--deduplicate', action='store_true',
                        help='Deduplicate identical compressed semantic paths before saving')
    args = parser.parse_args()

    variant = args.variant
    save_prefix = f'data/preprocessed/IMDB_preprocessed_skip_full_{variant}/'
    num_ntypes = 4

    print(f'=== IMDB full-traversal-then-compress preprocessing: {variant} ===')

    # -------------------------------------------------------------
    # Load and clean raw data
    # -------------------------------------------------------------
    movies = (
        pd.read_csv('data/raw/IMDB/movie_metadata.csv', encoding='utf-8')
        .drop_duplicates(subset=['movie_imdb_link'])
        .dropna(axis=0, subset=['actor_1_name', 'director_name', 'movie_imdb_link'])
        .reset_index(drop=True)
    )

    labels = np.zeros(len(movies), dtype=int)
    for movie_idx, genres in movies['genres'].items():
        labels[movie_idx] = -1
        for genre in genres.split('|'):
            if genre == 'Action':
                labels[movie_idx] = 0
                break
            elif genre == 'Comedy':
                labels[movie_idx] = 1
                break
            elif genre == 'Drama':
                labels[movie_idx] = 2
                break

    unwanted_idx = np.where(labels == -1)[0]
    movies = movies.drop(unwanted_idx).reset_index(drop=True)
    labels = np.delete(labels, unwanted_idx, 0)

    directors = sorted(set(movies['director_name'].dropna()))
    actors = sorted(set(
        movies['actor_1_name'].dropna().tolist() +
        movies['actor_2_name'].dropna().tolist() +
        movies['actor_3_name'].dropna().tolist()
    ))
    imdb_links = sorted(set(movies['movie_imdb_link'].dropna()))

    director_to_idx = {x: i for i, x in enumerate(directors)}
    actor_to_idx = {x: i for i, x in enumerate(actors)}
    link_to_idx = {x: i for i, x in enumerate(imdb_links)}

    n_m = len(movies)
    n_d = len(directors)
    n_a = len(actors)
    n_l = len(imdb_links)
    dim = n_m + n_d + n_a + n_l

    print(f'Entities: movies={n_m}, directors={n_d}, actors={n_a}, links={n_l}')

    # -------------------------------------------------------------
    # Type mask
    # -------------------------------------------------------------
    type_mask = np.zeros(dim, dtype=int)
    type_mask[n_m:n_m + n_d] = 1
    type_mask[n_m + n_d:n_m + n_d + n_a] = 2
    type_mask[n_m + n_d + n_a:] = 3

    # -------------------------------------------------------------
    # Adjacency per variation
    # -------------------------------------------------------------
    adjM = np.zeros((dim, dim), dtype=int)
    EDGE_BUILDERS[variant](adjM, movies, director_to_idx, actor_to_idx, link_to_idx, n_m, n_d, n_a)
    print(f'Undirected edges in graph: {adjM.sum() // 2}')

    # -------------------------------------------------------------
    # Features
    # -------------------------------------------------------------
    movie_X, director_X, actor_X, link_X = build_features(adjM, movies, n_m, n_d, n_a, n_l)
    features_by_type = [movie_X, director_X, actor_X, link_X]

    # -------------------------------------------------------------
    # Metapaths
    # -------------------------------------------------------------
    raw_specs = SKIP_METAPATHS[variant]

    for ntype in range(num_ntypes):
        print(f'\nNode type {ntype} metapaths:')
        for spec in raw_specs[ntype]:
            full, skip_pos, semantic = parse_skip_metapath(spec)
            print(f'  spec={spec} | full={full} | semantic={semantic}')

    # -------------------------------------------------------------
    # Save directories
    # -------------------------------------------------------------
    for i in range(num_ntypes):
        pathlib.Path(save_prefix + str(i)).mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Preprocess each center node type
    # -------------------------------------------------------------
    for ntype in tqdm(range(num_ntypes), desc='node type'):
        print(f'\n--- Processing node type {ntype} ---')

        neighbor_pairs_info = get_metapath_neighbor_pairs_full(
            adjM,
            type_mask,
            raw_specs[ntype]
        )

        G_list = get_networkx_graph(neighbor_pairs_info, type_mask, ntype)

        # SAVE USING SEMANTIC METAPATH NAME
        for G, mp_info in zip(G_list, neighbor_pairs_info):
            semantic_name = '-'.join(map(str, mp_info['semantic']))
            nx.write_adjlist(G, save_prefix + f'{ntype}/' + semantic_name + '.adjlist')

        # SAVE COMPRESSED PATH ARRAYS USING SEMANTIC METAPATH NAME
        all_idx_arrays = get_edge_metapath_idx_array(
            neighbor_pairs_info,
            deduplicate=args.deduplicate
        )

        for mp_info, idx_arr in zip(neighbor_pairs_info, all_idx_arrays):
            semantic_name = '-'.join(map(str, mp_info['semantic']))
            np.save(save_prefix + f'{ntype}/' + semantic_name + '_idx.npy', idx_arr)

    # -------------------------------------------------------------
    # Save global artifacts
    # -------------------------------------------------------------
    scipy.sparse.save_npz(save_prefix + 'adjM.npz', scipy.sparse.csr_matrix(adjM))

    for i in range(num_ntypes):
        scipy.sparse.save_npz(save_prefix + f'features_{i}.npz', features_by_type[i])

    np.save(save_prefix + 'node_types.npy', type_mask)
    np.save(save_prefix + 'labels.npy', labels)

    rand_seed = 1566911444
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size=int(0.1 * len(labels)),
        random_state=rand_seed
    )
    train_idx, test_idx = train_test_split(
        train_idx,
        test_size=int(0.2 * len(labels)),
        random_state=rand_seed
    )

    train_idx.sort()
    val_idx.sort()
    test_idx.sort()

    np.savez(
        save_prefix + 'train_val_test_idx.npz',
        val_idx=val_idx,
        train_idx=train_idx,
        test_idx=test_idx
    )

    print(f'\nDone. Saved to {save_prefix}')
    print(f'train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}')


if __name__ == '__main__':
    main()