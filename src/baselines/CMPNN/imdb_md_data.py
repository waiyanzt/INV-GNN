"""IMDB movie→director LP for CMPNN (v1 and v3 only; no genre nodes)."""
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
    return df


def _offsets(movies):
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
    N = off_l + M
    return M, Dn, An, off_d, off_a, off_l, N


def build_triplets_md(movies, variant, train_pos):
    """Structure edges + train movie→director only. Directed triplets."""
    M, _Dn, _An, off_d, off_a, off_l, N = _offsets(movies)

    directors = sorted(set(movies['director_name'].dropna().tolist()))
    actors = sorted(set(
        movies[['actor_1_name']].dropna()['actor_1_name'].tolist()
    ))

    def d_idx(name):
        return off_d + directors.index(name)

    def a_idx(name):
        return off_a + actors.index(name)

    triplets = []
    train_md = set(map(tuple, train_pos.tolist()))

    if variant == 'v1':
        r_ma = 0
        r_ml = 1
        r_md = 2
        rel_md = r_md
        for i, row in movies.iterrows():
            i = int(i)
            for col in ['actor_1_name']:
                ai = a_idx(row[col])
                triplets.append([i, ai, r_ma])
            li = off_l + i
            triplets.append([i, li, r_ml])
        for m, d in train_md:
            di = d_idx(movies.iloc[m]['director_name'])
            triplets.append([int(m), di, r_md])
    elif variant == 'v3':
        r_ml = 0
        r_la = 1
        r_md = 2
        rel_md = r_md
        for i, row in movies.iterrows():
            i = int(i)
            li = off_l + i
            triplets.append([i, li, r_ml])
            for col in ['actor_1_name']:
                ai = a_idx(row[col])
                triplets.append([li, ai, r_la])
        for m, d in train_md:
            di = d_idx(movies.iloc[m]['director_name'])
            triplets.append([int(m), di, r_md])
    else:
        raise ValueError(f"variant must be v1 or v3, got {variant}")

    triplets = np.asarray(triplets, dtype=np.int64)
    num_relation = rel_md + 1
    meta = {
        'num_entity': N,
        'num_relation': num_relation,
        'rel_md': rel_md,
        'off_d': off_d,
        'sizes': {'M': M, 'D': _Dn, 'A': _An},
        'counts': {'train_pos': len(train_md)},
    }
    return triplets, meta


def _sample_k_negs(pos_pairs, D_all, true_set, k, rng):
    N = len(pos_pairs)
    neg = np.zeros((N, k), dtype=np.int64)
    for i, (m, d_true) in enumerate(pos_pairs):
        cand = [x for x in D_all if (m, x) not in true_set]
        chosen = rng.choice(cand, size=k, replace=(k > len(cand)))
        neg[i] = chosen
    return neg


def preprocess_imdb_md(csv_path, shared_npz, variant, neg_k, seed):
    movies = read_imdb_frame(csv_path)
    shared = np.load(shared_npz)
    train_pos = shared['train_pos'].astype(np.int64)
    val_pos = shared['val_pos'].astype(np.int64)
    test_pos = shared['test_pos'].astype(np.int64)

    triplets, meta = build_triplets_md(movies, variant, train_pos)

    rng = np.random.RandomState(seed)
    D_all = np.arange(meta['sizes']['D'], dtype=np.int64)
    train_true = set(map(tuple, train_pos.tolist()))
    val_true = set(map(tuple, val_pos.tolist()))
    test_true = set(map(tuple, test_pos.tolist()))
    all_true = train_true | val_true | test_true

    k = min(int(neg_k), max(1, len(D_all) - 1))

    train_neg = _sample_k_negs(train_pos, D_all, all_true, k, rng)
    val_neg = _sample_k_negs(val_pos, D_all, all_true, k, rng)
    test_neg = _sample_k_negs(test_pos, D_all, all_true, k, rng)

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
