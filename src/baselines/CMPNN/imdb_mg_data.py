"""IMDB movie→genre LP for CMPNN: four graph variants (dedicated structural layouts)."""
import numpy as np
import pandas as pd

NUM_GENRES = 3
MAX_DISTINCT_WRONG_GENRES = 2


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


def _offsets_mg(movies):
    directors = sorted(set(movies['director_name'].dropna().tolist()))
    actors = sorted(set(
        movies[['actor_1_name']].dropna()['actor_1_name'].tolist()
    ))
    M = len(movies)
    Dn = len(directors)
    An = len(actors)
    off_d = M
    off_a = M + Dn
    off_l = M + Dn + An
    off_g = off_l + M
    N = off_g + NUM_GENRES
    return M, Dn, An, off_d, off_a, off_l, off_g, N


def build_triplets_mg(movies, variant, train_pos):
    """Structure edges + train movie→genre only. Directed triplets."""
    M, _Dn, _An, off_d, off_a, off_l, off_g, N = _offsets_mg(movies)

    directors = sorted(set(movies['director_name'].dropna().tolist()))
    actors = sorted(set(
        movies[['actor_1_name']].dropna()['actor_1_name'].tolist()
    ))

    def d_idx(name):
        return off_d + directors.index(name)

    def a_idx(name):
        return off_a + actors.index(name)

    triplets = []
    train_mg = set(map(tuple, train_pos.tolist()))

    if variant == 'v1':
        r_md, r_ma, r_ml, r_mg = 0, 1, 2, 3
        rel_mg = r_mg
        for i, row in movies.iterrows():
            i = int(i)
            di = d_idx(row['director_name'])
            triplets.append([i, di, r_md])
            for col in ['actor_1_name']:
                ai = a_idx(row[col])
                triplets.append([i, ai, r_ma])
            li = off_l + i
            triplets.append([i, li, r_ml])
        for m, g in train_mg:
            triplets.append([int(m), off_g + int(g), r_mg])
    elif variant == 'v2':
        r_ld, r_la, r_ml, r_mg = 0, 1, 2, 3
        rel_mg = r_mg
        for i, row in movies.iterrows():
            i = int(i)
            li = off_l + i
            di = d_idx(row['director_name'])
            triplets.append([li, di, r_ld])
            for col in ['actor_1_name']:
                ai = a_idx(row[col])
                triplets.append([li, ai, r_la])
            triplets.append([i, li, r_ml])
        for m, g in train_mg:
            triplets.append([int(m), off_g + int(g), r_mg])
    elif variant == 'v3':
        r_ml, r_la, r_md, r_mg = 0, 1, 2, 3
        rel_mg = r_mg
        for i, row in movies.iterrows():
            i = int(i)
            li = off_l + i
            triplets.append([i, li, r_ml])
            for col in ['actor_1_name']:
                ai = a_idx(row[col])
                triplets.append([li, ai, r_la])
            di = d_idx(row['director_name'])
            triplets.append([i, di, r_md])
        for m, g in train_mg:
            triplets.append([int(m), off_g + int(g), r_mg])
    elif variant == 'v4':
        r_ma, r_ml, r_ld, r_mg = 0, 1, 2, 3
        rel_mg = r_mg
        for i, row in movies.iterrows():
            i = int(i)
            for col in ['actor_1_name']:
                ai = a_idx(row[col])
                triplets.append([i, ai, r_ma])
            li = off_l + i
            triplets.append([i, li, r_ml])
            di = d_idx(row['director_name'])
            triplets.append([li, di, r_ld])
        for m, g in train_mg:
            triplets.append([int(m), off_g + int(g), r_mg])
    else:
        raise ValueError(f"variant must be v1–v4, got {variant}")

    triplets = np.asarray(triplets, dtype=np.int64)
    num_relation = 4
    meta = {
        'num_entity': N,
        'num_relation': num_relation,
        'rel_mg': rel_mg,
        'off_g': off_g,
        'num_genres': NUM_GENRES,
        'counts': {'train_pos': len(train_mg)},
        'neg_note': f'max {MAX_DISTINCT_WRONG_GENRES} distinct wrong genres',
    }
    return triplets, meta


def _sample_k_negs_genre(pos_pairs, k, rng):
    N = len(pos_pairs)
    neg = np.zeros((N, k), dtype=np.int64)
    for i, (m, g_true) in enumerate(pos_pairs):
        cand = [x for x in range(NUM_GENRES) if x != int(g_true)]
        chosen = rng.choice(cand, size=k, replace=(k > len(cand)))
        neg[i] = chosen
    return neg


def preprocess_imdb_mg(csv_path, shared_npz, variant, neg_k, seed):
    movies = read_imdb_frame(csv_path)
    shared = np.load(shared_npz)
    train_pos = shared['train_pos'].astype(np.int64)
    val_pos = shared['val_pos'].astype(np.int64)
    test_pos = shared['test_pos'].astype(np.int64)

    triplets, meta = build_triplets_mg(movies, variant, train_pos)

    rng = np.random.RandomState(seed)
    k = min(int(neg_k), MAX_DISTINCT_WRONG_GENRES)

    train_neg = _sample_k_negs_genre(train_pos, k, rng)
    val_neg = _sample_k_negs_genre(val_pos, k, rng)
    test_neg = _sample_k_negs_genre(test_pos, k, rng)

    meta['neg_k'] = k
    meta['counts'].update({
        'val_pos': len(val_pos),
        'test_pos': len(test_pos),
    })

    splits = {
        'train_pos': train_pos,
        'train_neg': train_neg,
        'val_pos': val_pos,
        'val_neg': val_neg,
        'test_pos': test_pos,
        'test_neg': test_neg,
    }
    return triplets, meta, splits
