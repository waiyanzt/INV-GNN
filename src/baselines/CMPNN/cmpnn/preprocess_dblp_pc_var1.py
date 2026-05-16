"""Standalone preprocessing for DBLP paper-conference LP (variant 1).

Reads raw DBLP files, builds entity ID maps, constructs KG triplets with
4 relation types (PA, PT, PC, PR), splits paper-conf edges into
train/val/test, and writes triplets.tsv, splits_pc.npz, and stats.json.
"""
import glob
import re
import os
import json
import argparse
import numpy as np
import pandas as pd


def _read_txt_int(path, names):
    return pd.read_csv(path, sep='\t', header=None, names=names).astype(np.int64)


def _read_tsv_any(path, names):
    df = pd.read_csv(path, sep='\t', header=None, names=names, engine='python')
    return df.astype(str)


def _coerce_two_int_cols(df, col0, col1, out_names):
    a = pd.to_numeric(df[col0], errors='coerce')
    b = pd.to_numeric(df[col1], errors='coerce')
    out = pd.DataFrame({out_names[0]: a, out_names[1]: b}).dropna().astype(np.int64)
    return out


def _load_author_label(raw_dir):
    candidates = [
        os.path.join(raw_dir, 'author_label.txt'),
        os.path.join(raw_dir, 'author_labels.txt'),
        os.path.join(raw_dir, 'author_label.tsv'),
    ]
    candidates += sorted(set(
        glob.glob(os.path.join(raw_dir, '*author*label*.*'))
    ))
    candidates += sorted(set(
        glob.glob(os.path.join(raw_dir, '*label*author*.*'))
    ))

    best = None
    best_rows = 0
    for path in candidates:
        if not os.path.exists(path):
            continue
        df = _read_tsv_any(path, ['c0', 'c1'])
        if df.shape[1] < 2:
            continue
        for i in range(df.shape[1]):
            for j in range(df.shape[1]):
                if i == j:
                    continue
                tmp = _coerce_two_int_cols(df, df.columns[i], df.columns[j],
                                           ['author_id', 'label'])
                if len(tmp) > best_rows:
                    best = tmp
                    best_rows = len(tmp)

    if best is None or best_rows == 0:
        raise ValueError(
            f'Could not find a numeric author-label mapping file in raw_dir ({raw_dir}).'
        )
    out = best.drop_duplicates().reset_index(drop=True)
    print(f'[author_label] cols=({out.columns.tolist()}) rows={len(out)}')
    return out


def preprocess(raw_dir, out_dir, seed, val_ratio, test_ratio):
    raw_dir = os.path.expanduser(raw_dir)
    os.makedirs(out_dir, exist_ok=True)

    pa_path = os.path.join(raw_dir, 'paper_author.txt')
    pc_path = os.path.join(raw_dir, 'paper_conf.txt')
    pt_path = os.path.join(raw_dir, 'paper_term.txt')

    for p in [pa_path, pc_path, pt_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f'Missing raw file: {p}')

    pa = _read_txt_int(pa_path, ['paper_id', 'author_id'])
    pc = _read_txt_int(pc_path, ['paper_id', 'conf_id'])
    pt = _read_txt_int(pt_path, ['paper_id', 'term_id'])
    author = _load_author_label(raw_dir)

    pa = pa.drop_duplicates().reset_index(drop=True)
    pc = pc.drop_duplicates().reset_index(drop=True)
    pt = pt.drop_duplicates().reset_index(drop=True)

    a2r = author.set_index('author_id')['label']
    pr = pa.assign(area_id=pa['author_id'].map(a2r)).dropna()
    pr = pr[['paper_id', 'area_id']].drop_duplicates().astype(np.int64)

    papers = sorted(np.unique(np.concatenate([
        pa['paper_id'].values, pc['paper_id'].values, pt['paper_id'].values
    ])))
    authors = sorted(pa['author_id'].unique())
    terms = sorted(pt['term_id'].unique())
    confs = sorted(pc['conf_id'].unique())
    areas = sorted(pr['area_id'].unique())

    a_map = {int(x): i for i, x in enumerate(authors)}
    p_map = {int(x): i for i, x in enumerate(papers)}
    t_map = {int(x): i for i, x in enumerate(terms)}
    c_map = {int(x): i for i, x in enumerate(confs)}
    r_map = {int(x): i for i, x in enumerate(areas)}

    nP, nA, nT, nC, nR = len(papers), len(authors), len(terms), len(confs), len(areas)
    offP = 0
    offA = nP
    offT = nP + nA
    offC = nP + nA + nT
    offR = nP + nA + nT + nC
    num_entity = nP + nA + nT + nC + nR

    REL_PA, REL_PT, REL_PC, REL_PR = 0, 1, 2, 3
    num_relation = 4

    def map_edges(df, hcol, tcol, hmap, tmap, hoff, toff, rel_id):
        h = df[hcol].map(lambda x: int(hmap[int(x)])).to_numpy().astype(np.int64)
        t = df[tcol].map(lambda x: int(tmap[int(x)])).to_numpy().astype(np.int64)
        r = np.full(len(h), rel_id, dtype=np.int64)
        return np.stack([h + hoff, t + toff, r], axis=1)

    trip_pa = map_edges(pa, 'paper_id', 'author_id', p_map, a_map, offP, offA, REL_PA)
    trip_pt = map_edges(pt, 'paper_id', 'term_id', p_map, t_map, offP, offT, REL_PT)
    trip_pc = map_edges(pc, 'paper_id', 'conf_id', p_map, c_map, offP, offC, REL_PC)
    pr_p_map = {int(x): p_map[int(x)] for x in pr['paper_id'].unique() if int(x) in p_map}
    trip_pr = map_edges(pr, 'paper_id', 'area_id', pr_p_map, r_map, offP, offR, REL_PR)

    triplets = np.concatenate([trip_pa, trip_pt, trip_pc, trip_pr], axis=0)

    pc_local = np.stack([
        pc['paper_id'].map(lambda x: int(p_map[int(x)])).to_numpy().astype(np.int64),
        pc['conf_id'].map(lambda x: int(c_map[int(x)])).to_numpy().astype(np.int64),
    ], axis=1)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(pc_local))
    n_total = len(pc_local)
    n_test = max(1, int(round(test_ratio * n_total)))
    n_val = max(1, int(round(val_ratio * n_total)))
    n_train = n_total - n_test - n_val

    if n_train <= 0:
        raise ValueError('Split ratios too large; training set becomes empty.')

    train_pos = pc_local[perm[:n_train]]
    val_pos = pc_local[perm[n_train:n_train + n_val]]
    test_pos = pc_local[perm[n_train + n_val:]]

    trip_path = os.path.join(out_dir, 'triplets.tsv')
    split_path = os.path.join(out_dir, 'splits_pc.npz')
    stats_path = os.path.join(out_dir, 'stats.json')

    np.savetxt(trip_path, triplets, fmt='%d', delimiter='\t')
    np.savez_compressed(split_path, train_pos=train_pos, val_pos=val_pos, test_pos=test_pos)

    stats = {
        'num_entity': int(num_entity),
        'num_relation': int(num_relation),
        'offsets': {'P': offP, 'A': offA, 'T': offT, 'C': offC, 'R': offR},
        'sizes': {'P': nP, 'A': nA, 'T': nT, 'C': nC, 'R': nR},
        'triplets': int(len(triplets)),
        'train': int(len(train_pos)),
        'val': int(len(val_pos)),
        'test': int(len(test_pos)),
    }
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f'Wrote: {trip_path}, {split_path}, {stats_path}')
    print(f'Stats: {stats}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--seed', type=int, default=1566911444)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--test_ratio', type=float, default=0.2)
    args = ap.parse_args()
    preprocess(args.raw_dir, args.out_dir, args.seed, args.val_ratio, args.test_ratio)


if __name__ == '__main__':
    main()
