#!/usr/bin/env python3
"""
DBLP RGCN preprocessing — **non-skip** variants (one area channel each, MAGNN DBLP1–3):

  * **v1 (DBLP1 / Area–Paper):** ``paper-area`` only.
  * **v2 (DBLP2 / Area–Venue):** ``conference-area`` only (conf ↔ area via shared papers).
  * **v3 (DBLP3 / Area–Author):** ``author-area`` only (author ↔ area via shared papers).

Shared across variants: ``author-paper``, ``paper-term``, ``paper-conference`` using **train
positives only** for P–C (no val/test venue edges in the graph), aligned with MAGNN LP /
``preprocess_DBLP_rgcn_skip.py``.

Negative pairs live in the shared splits NPZ; ``run_DBLP_rgcn.py`` subsamples to at most
``--neg-per-paper`` per split (default **3**, MAGNN-aligned). No extra preprocessing step for that.

to run: python preprocess_DBLP_rgcn.py --variant v1,v2,v3

"""
import os, argparse, shutil, random
from collections import defaultdict
import numpy as np
import pandas as pd
import torch

SEED = 1566911444

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def _parse_variants(s):
    vals = [x.strip().lower() for x in str(s).split(',') if x.strip()]
    good = {'v1','v2','v3'}
    bad = [v for v in vals if v not in good]
    if bad:
        raise SystemExit(f'Unknown variant(s): {bad}; expected subset of {sorted(good)}')
    return vals

def load_tables(raw_dir):
    al = pd.read_csv(os.path.join(raw_dir,'author_label.txt'), sep='\t',
                     names=['author_id','label','author_name'], header=None, encoding='utf-8')
    pa = pd.read_csv(os.path.join(raw_dir,'paper_author.txt'), sep='\t',
                     names=['paper_id','author_id'], header=None, encoding='utf-8')
    pc = pd.read_csv(os.path.join(raw_dir,'paper_conf.txt'), sep='\t',
                     names=['paper_id','conf_id'], header=None, encoding='utf-8')
    pt = pd.read_csv(os.path.join(raw_dir,'paper_term.txt'), sep='\t',
                     names=['paper_id','term_id'], header=None, encoding='utf-8')
    return al, pa, pc, pt

def map_pairs_npz(shared_npz, Pa_map, Co_map):
    Z = np.load(shared_npz)
    def M(X, name):
        df = pd.DataFrame(X, columns=['paper_id','conf_id'])
        df['paper_id'] = df['paper_id'].map(Pa_map)
        df['conf_id']  = df['conf_id'].map(Co_map)
        out = df.dropna().to_numpy(dtype=np.int64)
        dropped = len(df) - len(out)
        if dropped:
            print(f"[warn] {name}: dropped {dropped}", flush=True)
        return out
    out = {
        'train_pos': M(Z['train_pos'], 'train_pos'),
        'val_pos'  : M(Z['val_pos'], 'val_pos'),
        'test_pos' : M(Z['test_pos'], 'test_pos'),
        'train_neg': M(Z['train_neg'], 'train_neg'),
        'val_neg'  : M(Z['val_neg'], 'val_neg'),
        'test_neg' : M(Z['test_neg'], 'test_neg'),
    }
    paper_subset = Z['paper_subset'].tolist() if 'paper_subset' in Z.files else None
    return out, paper_subset

def filter_tables(al, pa, pc, pt, paper_subset=None, min_conf=0):
    valid_authors = set(al['author_id'])
    pa = pa[pa['author_id'].isin(valid_authors)].reset_index(drop=True)

    valid_papers = set(pa['paper_id'])
    pc = pc[pc['paper_id'].isin(valid_papers)].reset_index(drop=True)
    pt = pt[pt['paper_id'].isin(valid_papers)].reset_index(drop=True)
    pa = pa[pa['paper_id'].isin(valid_papers)].reset_index(drop=True)

    if min_conf > 0:
        big = pc['conf_id'].value_counts().loc[lambda x: x >= min_conf].index
        pc = pc[pc['conf_id'].isin(big)].reset_index(drop=True)
        valid_papers = set(pc['paper_id'])
        pt = pt[pt['paper_id'].isin(valid_papers)].reset_index(drop=True)
        pa = pa[pa['paper_id'].isin(valid_papers)].reset_index(drop=True)

    if paper_subset is not None:
        keep = set(paper_subset) & set(pc['paper_id'])
        pc = pc[pc['paper_id'].isin(keep)].reset_index(drop=True)
        pa = pa[pa['paper_id'].isin(keep)].reset_index(drop=True)
        pt = pt[pt['paper_id'].isin(keep)].reset_index(drop=True)

    return al, pa, pc, pt

def build_base_tables(al, pa, pc):
    a2r = al.set_index('author_id')['label']
    pr_raw = (
        pa.assign(area_id=pa['author_id'].map(a2r))[['paper_id','area_id']]
        .drop_duplicates().dropna().reset_index(drop=True)
    )
    area_counts = pr_raw.groupby('paper_id')['area_id'].nunique()
    bad_papers = set(area_counts[area_counts != 1].index.tolist())
    return pr_raw, bad_papers

def build_maps(pa, pc, pt, pr):
    Au_ids = sorted(pa['author_id'].unique())
    Pa_ids = sorted(pc['paper_id'].unique())
    Te_ids = sorted(pt['term_id'].unique())
    Co_ids = sorted(pc['conf_id'].unique())
    Ar_ids = sorted(pr['area_id'].unique())
    Au_map = {a:i for i,a in enumerate(Au_ids)}
    Pa_map = {p:i for i,p in enumerate(Pa_ids)}
    Te_map = {t:i for i,t in enumerate(Te_ids)}
    Co_map = {c:i for i,c in enumerate(Co_ids)}
    Ar_map = {r:i for i,r in enumerate(Ar_ids)}
    return Au_map, Pa_map, Te_map, Co_map, Ar_map

def preprocess_one(variant, raw_dir, shared_npz, min_conf, out_dir_base):
    out_dir = os.path.join(out_dir_base, f'DBLP_rgcn_{variant}')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== preprocess original RGCN {variant} ===", flush=True)

    al, pa, pc, pt = load_tables(raw_dir)
    tmp_Pa_map = {p:i for i,p in enumerate(sorted(pa['paper_id'].unique()))}
    tmp_Co_map = {c:i for i,c in enumerate(sorted(pc['conf_id'].unique()))}
    _, paper_subset = map_pairs_npz(shared_npz, tmp_Pa_map, tmp_Co_map)

    al, pa, pc, pt = filter_tables(al, pa, pc, pt, paper_subset=paper_subset, min_conf=min_conf)

    pr_raw, bad_papers = build_base_tables(al, pa, pc)
    if bad_papers:
        print(f"Removing {len(bad_papers)} papers violating 1 paper -> 1 area", flush=True)
        pa = pa[~pa['paper_id'].isin(bad_papers)].reset_index(drop=True)
        pc = pc[~pc['paper_id'].isin(bad_papers)].reset_index(drop=True)
        pt = pt[~pt['paper_id'].isin(bad_papers)].reset_index(drop=True)
        pr_raw, bad2 = build_base_tables(al, pa, pc)
        assert len(bad2) == 0

    pr = pr_raw.drop_duplicates(subset=['paper_id']).reset_index(drop=True)
    Au_map, Pa_map, Te_map, Co_map, Ar_map = build_maps(pa, pc, pt, pr)
    splits, _ = map_pairs_npz(shared_npz, Pa_map, Co_map)

    print(f"Authors={len(Au_map)} Papers={len(Pa_map)} Terms={len(Te_map)} Confs={len(Co_map)} Areas={len(Ar_map)}", flush=True)

    # Paper–venue edges in the graph: train positives only (link-prediction–fair; matches skip RGCN).
    train_pc = splits['train_pos'].astype(np.int64)

    A2P_src, A2P_dst = [], []
    for p, a in pa[['paper_id','author_id']].itertuples(index=False):
        if p in Pa_map and a in Au_map:
            A2P_src.append(Au_map[a]); A2P_dst.append(Pa_map[p])

    P2T_src, P2T_dst = [], []
    for p, t in pt[['paper_id','term_id']].itertuples(index=False):
        if p in Pa_map and t in Te_map:
            P2T_src.append(Pa_map[p]); P2T_dst.append(Te_map[t])

    graph_data = {
        'author-paper': (np.array(A2P_src, dtype=np.int64), np.array(A2P_dst, dtype=np.int64)),
        'paper-term': (np.array(P2T_src, dtype=np.int64), np.array(P2T_dst, dtype=np.int64)),
        'paper-conference': (train_pc[:, 0], train_pc[:, 1]),
    }
    print(f"   paper-conference edges in graph: {len(train_pc)} (train positives only)", flush=True)

    if variant == 'v1':
        P2R_src, P2R_dst = [], []
        for p, r in pr[['paper_id','area_id']].itertuples(index=False):
            if p in Pa_map and r in Ar_map:
                P2R_src.append(Pa_map[p]); P2R_dst.append(Ar_map[r])
        graph_data['paper-area'] = (np.array(P2R_src, dtype=np.int64), np.array(P2R_dst, dtype=np.int64))

    elif variant == 'v2':
        cr = (
            pc[['paper_id','conf_id']]
            .drop_duplicates()
            .merge(pr[['paper_id','area_id']], on='paper_id')
            [['conf_id','area_id']]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        C2R_src, C2R_dst = [], []
        for c, r in cr.itertuples(index=False):
            if c in Co_map and r in Ar_map:
                C2R_src.append(Co_map[c]); C2R_dst.append(Ar_map[r])
        graph_data['conference-area'] = (np.array(C2R_src, dtype=np.int64), np.array(C2R_dst, dtype=np.int64))

    elif variant == 'v3':
        ar = (
            pa[['paper_id','author_id']]
            .drop_duplicates()
            .merge(pr[['paper_id','area_id']], on='paper_id')
            [['author_id','area_id']]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        A2R_src, A2R_dst = [], []
        for a, r in ar.itertuples(index=False):
            if a in Au_map and r in Ar_map:
                A2R_src.append(Au_map[a]); A2R_dst.append(Ar_map[r])
        graph_data['author-area'] = (np.array(A2R_src, dtype=np.int64), np.array(A2R_dst, dtype=np.int64))

    meta = {
        'num_nodes': {
            'author': len(Au_map),
            'paper': len(Pa_map),
            'term': len(Te_map),
            'conference': len(Co_map),
            'area': len(Ar_map),
        },
        'splits': {k: torch.tensor(v, dtype=torch.long) for k, v in splits.items()},
        'variant': variant,
        'paper_conf_edges': 'train_only',
    }

    torch.save(graph_data, os.path.join(out_dir, 'graph_data.pt'))
    torch.save(meta, os.path.join(out_dir, 'meta.pt'))
    print(f"Saved to {out_dir}", flush=True)

def main():
    ap = argparse.ArgumentParser(description="Preprocess DBLP RGCN original variants")
    ap.add_argument('--variant', default='v1,v2,v3',
                    help='One of v1,v2,v3 or a comma-separated list like v1,v2,v3')
    ap.add_argument('--raw-dir', default='data/raw/DBLP/')
    ap.add_argument('--shared-npz', default='data/preprocessed/DBLP_shared_splits/DBLP_pc_shared_splits.npz')
    ap.add_argument('--min-conf', type=int, default=0)
    ap.add_argument('--out-dir', default='data/preprocessed')
    args = ap.parse_args()

    set_seed()
    for v in _parse_variants(args.variant):
        preprocess_one(v, args.raw_dir, args.shared_npz, args.min_conf, args.out_dir)

if __name__ == "__main__":
    main()