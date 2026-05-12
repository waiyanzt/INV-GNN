#!/usr/bin/env python3
"""
DBLP paper–venue link prediction preprocessing with canonical skip-node alignment
across three heterogeneous-graph variants (v1–v3). Outputs MAGNN-LP pickles/adjlists
consumed by ``run_DBLP_skip.py`` (``use_skip_conf_area=True``).

Node types: 0=Author, 1=Paper, 2=Term, 3=Conf, 4=Area.

On-disk metapaths (fixed channel count for the skip runner):
  mode 0: 1-0-1, 1-2-1, 1-4-1, 1-3-1
  mode 1: 3-1-0-1-3, 3-1-3

Variants (where the *raw* area edge attaches):
  v1: Paper–Area
  v2: Conf–Area (induced from train papers + canonical paper→area)
  v3: Author–Area (induced from papers + canonical paper→area)

``SKIP_METAPATHS`` below lists, for each variant, the **same-length** mode-0 / mode-1
menus: row index *i* is the same *role* across v1,v2,v3. Bracketed entries (e.g. ``[3]``)
mark node types that appear on the long walk but are **skipped** in the stored MAGNN
path — we still write **one** ``1-4-1`` (and one ``3-1-0-1-3``) file per variant, built by
projecting the variant’s long walks to (paper,area,paper) and (conf,area,conf).

Usage:
  python preprocess_DBLP_skip.py --variant v1
  python preprocess_DBLP_skip.py --variant all    # v1, v2, v3 sequentially
"""

from __future__ import annotations

import argparse
import os
import pickle
import pathlib
import random
import shutil
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


# -----------------------------------------------------------------------------
# Cross-variant skip metapath spec (documentation + alignment).
# Mode 0: six conceptual rows per variant — rows 0,1,5 map to 1-0-1, 1-2-1, 1-3-1;
# rows 2–4 describe long / skip forms that feed the single 1-4-1 channel on disk.
# Mode 1: C-P-A-P-C (3-1-0-1-3) and C-P-C (3-1-3).
# -----------------------------------------------------------------------------
SKIP_METAPATHS: Dict[str, List[List[Any]]] = {
    'v1': [
        [
            [1, 0, 1],
            [1, 2, 1],
            [1, 0, [1], 4, [1], 0, 1],
            [1, 4, 1],
            [1, 3, [1], 4, [1], 3, 1],
            [1, 3, 1],
        ],
        [
            [3, 1, 0, 1, 3],
            [3, 1, 3],
        ],
    ],
    'v2': [
        [
            [1, 0, 1],
            [1, 2, 1],
            [1, 3, 4, 3, 1],
            [1, [3], 4, [3], 1],
            [1, 0, [1], [3], 4, [3], [1], 0, 1],
            [1, 3, 1],
        ],
        [
            [3, 1, 0, 1, 3],
            [3, 1, 3],
        ],
    ],
    'v3': [
        [
            [1, 0, 1],
            [1, 2, 1],
            [1, 0, 4, 0, 1],
            [1, [0], 4, [0], 1],
            [1, 3, [1], [0], 4, [0], [1], 3, 1],
            [1, 3, 1],
        ],
        [
            [3, 1, 0, 1, 3],
            [3, 1, 3],
        ],
    ],
}

RAW = 'data/raw/DBLP/'
SHARED = 'data/preprocessed/DBLP_shared_splits/DBLP_pc_shared_splits.npz'
SEED = 1566911444
MIN_CONF = 0

OUT_PREFIX = {
    'v1': 'data/preprocessed/DBLP_lp_pc_skip_full_v1/',
    'v2': 'data/preprocessed/DBLP_lp_pc_skip_full_v2/',
    'v3': 'data/preprocessed/DBLP_lp_pc_skip_full_v3/',
}
# If >0, deterministically subsample large skip-instance tables to this many rows.
# Set to 0 to keep all instances (may be very large / slow / memory-heavy).
MAX_SKIP_INSTANCES = 0
MAX_CHANNEL_INSTANCES = 0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def sort_rows(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    keys = [arr[:, i] for i in range(arr.shape[1] - 1, -1, -1)]
    return arr[np.lexsort(keys)]


def assert_type_pattern(arr: np.ndarray, type_mask: np.ndarray, pattern, name: str):
    if arr.size == 0:
        return
    actual = np.stack([type_mask[arr[:, i]] for i in range(arr.shape[1])], axis=1)
    expected = np.array(pattern, dtype=actual.dtype)
    bad = np.where(np.any(actual != expected, axis=1))[0]
    if len(bad) > 0:
        idx = int(bad[0])
        raise ValueError(
            f"{name} has invalid row at index {idx}: "
            f"row={arr[idx].tolist()} "
            f"types={actual[idx].tolist()} "
            f"expected={pattern}"
        )


def save_per_target_pickle_and_adjlist(
    out_dir: str,
    semantic_name: str,
    arr: np.ndarray,
    target_count: int,
    target_global_offset: int,
):
    os.makedirs(out_dir, exist_ok=True)
    arr = sort_rows(arr)

    per_target = {}
    left = right = 0

    for tgt in range(target_count):
        tgt_global = target_global_offset + tgt
        while right < len(arr) and arr[right, 0] == tgt_global:
            right += 1
        per_target[tgt] = arr[left:right, ::-1]
        left = right

    with open(os.path.join(out_dir, f'{semantic_name}_idx.pickle'), 'wb') as f:
        pickle.dump(per_target, f)

    left = right = 0
    with open(os.path.join(out_dir, f'{semantic_name}.adjlist'), 'w') as f:
        for tgt in range(target_count):
            tgt_global = target_global_offset + tgt
            while right < len(arr) and arr[right, 0] == tgt_global:
                right += 1
            nbrs = arr[left:right, -1] - target_global_offset
            if len(nbrs) > 0:
                f.write(f'{tgt} ' + ' '.join(map(str, nbrs.tolist())) + '\n')
            else:
                f.write(f'{tgt}\n')
            left = right


def build_3hop_from_center_to_end(
    center_to_end: dict, center_offset: int, end_offset: int
) -> np.ndarray:
    rows = []
    for center_local, end_list in tqdm(center_to_end.items(), leave=False):
        if len(end_list) == 0:
            continue
        n = len(end_list)
        e1 = np.repeat(end_list, n)
        e2 = np.tile(end_list, n)
        rows.append(
            np.stack([
                end_offset + e1,
                center_offset + np.full(n * n, center_local, dtype=np.int32),
                end_offset + e2
            ], axis=1)
        )
    if rows:
        return np.concatenate(rows, axis=0).astype(np.int32)
    return np.empty((0, 3), dtype=np.int32)


def _dedupe_sort_paper_area_paper_rows(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr.astype(np.int32)
    u = np.unique(arr, axis=0)
    return sort_rows(u.astype(np.int32))


def maybe_cap_rows(arr: np.ndarray, max_rows: int, seed: int, tag: str) -> np.ndarray:
    """
    Deterministically cap large metapath instance tables to avoid OOM during preprocessing.
    """
    if max_rows <= 0 or arr.shape[0] <= max_rows:
        return arr
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(arr.shape[0], size=max_rows, replace=False))
    out = arr[keep]
    print(f"   [cap] {tag}: {arr.shape[0]} -> {out.shape[0]}")
    return out


def report_arr_stats(name: str, arr: np.ndarray):
    if arr.size == 0 or arr.ndim < 2:
        print(f"   {name}: rows=0 | unique endpoint pairs=0")
        print(f"    idx shape: {arr.shape} | unique endpoint pairs: 0")
        return
    pairs = np.unique(arr[:, [0, -1]], axis=0)
    print(f"   {name}: rows={arr.shape[0]} | unique endpoint pairs={len(pairs)}")
    print(f"    idx shape: {arr.shape} | unique endpoint pairs: {len(pairs)}")


def parse_skip_metapath_spec(spec):
    full = []
    skip_pos = set()
    for i, x in enumerate(spec):
        if isinstance(x, list):
            if len(x) != 1:
                raise ValueError(f"Skip notation must be single-item list, got {x}")
            full.append(int(x[0]))
            skip_pos.add(i)
        else:
            full.append(int(x))
    full = tuple(full)
    semantic = tuple(full[i] for i in range(len(full)) if i not in skip_pos)
    return full, skip_pos, semantic


def print_metapath_debug(spec, arr):
    full, skip_pos, semantic = parse_skip_metapath_spec(spec)
    if arr.size == 0 or arr.ndim < 2:
        npairs = 0
    else:
        npairs = len(np.unique(arr[:, [0, -1]], axis=0))
    print(f"  Full metapath: {full}")
    print(f"  Skip positions: {skip_pos if skip_pos else 'none'}")
    print(f"  Semantic metapath: {semantic}")
    print(f"    -> {npairs} neighbor pairs found")


def build_paper_area_paper_skip_from_pcrcp(
    adjM: np.ndarray,
    offP: int,
    offC: int,
    offR: int,
    P: int,
    C: int,
    R: int,
) -> np.ndarray:
    """v2: P–C–R–C–P on adjM → project to (P,R,P), confs skipped (full walk, no subsampling)."""
    conf_paper_list = {i: adjM[offC + i, offP:offP + P].nonzero()[0] for i in range(C)}
    area_conf_list = {i: adjM[offR + i, offC:offC + C].nonzero()[0] for i in range(R)}
    rows = []
    for rloc in range(R):
        c_list = area_conf_list[rloc]
        if len(c_list) == 0:
            continue
        rg = offR + rloc
        for c1 in c_list:
            for c2 in c_list:
                c1i = int(c1)
                c2i = int(c2)
                p1s = conf_paper_list.get(c1i)
                p2s = conf_paper_list.get(c2i)
                if p1s is None or p2s is None or len(p1s) == 0 or len(p2s) == 0:
                    continue
                c1g = offC + c1i
                c2g = offC + c2i
                for p1 in p1s:
                    for p2 in p2s:
                        rows.append([int(offP + p1), int(c1g), int(rg), int(c2g), int(offP + p2)])
    if not rows:
        return np.empty((0, 3), dtype=np.int32)
    full = np.array(rows, dtype=np.int32)
    tri = full[:, [0, 2, 4]]
    tri_rev = tri[:, [2, 1, 0]]
    both = np.vstack([tri, tri_rev])
    return _dedupe_sort_paper_area_paper_rows(both)


def build_paper_area_paper_skip_from_parap(
    adjM: np.ndarray,
    offA: int,
    offP: int,
    offR: int,
    A: int,
    P: int,
    R: int,
) -> np.ndarray:
    """v3: P–A–R–A–P → project to (P,R,P), authors skipped (full walk, no subsampling)."""
    author_paper_list = {i: adjM[offA + i, offP:offP + P].nonzero()[0] for i in range(A)}
    area_author_list = {i: adjM[offR + i, offA:offA + A].nonzero()[0] for i in range(R)}
    rows = []
    for rloc in range(R):
        a_list = area_author_list[rloc]
        if len(a_list) == 0:
            continue
        rg = offR + rloc
        for a1 in a_list:
            for a2 in a_list:
                a1i, a2i = int(a1), int(a2)
                p1s = author_paper_list.get(a1i)
                p2s = author_paper_list.get(a2i)
                if p1s is None or p2s is None or len(p1s) == 0 or len(p2s) == 0:
                    continue
                a1g = offA + a1i
                a2g = offA + a2i
                for p1 in p1s:
                    for p2 in p2s:
                        rows.append([int(offP + p1), int(a1g), int(rg), int(a2g), int(offP + p2)])
    if not rows:
        return np.empty((0, 3), dtype=np.int32)
    full = np.array(rows, dtype=np.int32)
    tri = full[:, [0, 2, 4]]
    tri_rev = tri[:, [2, 1, 0]]
    both = np.vstack([tri, tri_rev])
    return _dedupe_sort_paper_area_paper_rows(both)


def build_conf_area_conf_skip_from_pcprpc(
    paper_conf: dict,
    paper_area: dict,
    offC: int,
    offR: int,
) -> np.ndarray:
    """v1: C–P–R–P–C via train P–C + canonical P–R → (C,R,C)."""
    area_conf = defaultdict(set)
    for p_local, confs in paper_conf.items():
        areas = paper_area.get(p_local, np.array([], dtype=np.int32))
        if len(areas) == 0 or len(confs) == 0:
            continue
        for r in areas:
            for c in confs:
                area_conf[int(r)].add(int(c))
    rows = []
    for r_local, confs in area_conf.items():
        conf_list = np.array(sorted(confs), dtype=np.int32)
        if len(conf_list) == 0:
            continue
        n = len(conf_list)
        c1 = np.repeat(conf_list, n)
        c2 = np.tile(conf_list, n)
        rg = offR + int(r_local)
        rows.append(np.stack([offC + c1, np.full(n * n, rg, dtype=np.int32), offC + c2], axis=1))
    if not rows:
        return np.empty((0, 3), dtype=np.int32)
    arr = np.concatenate(rows, axis=0).astype(np.int32)
    return _dedupe_sort_paper_area_paper_rows(arr)


def build_conf_area_conf_skip_from_crcp(
    adjM: np.ndarray,
    offP: int,
    offC: int,
    offR: int,
    P: int,
    C: int,
    R: int,
) -> np.ndarray:
    """v2: C–R–C on adjM; expand to (C,R,C) triples for confs sharing an area."""
    area_conf_list = {i: adjM[offR + i, offC:offC + C].nonzero()[0] for i in range(R)}
    rows = []
    for rloc in range(R):
        c_list = area_conf_list[rloc]
        if len(c_list) == 0:
            continue
        rg = offR + rloc
        for c1 in c_list:
            for c2 in c_list:
                rows.append([int(offC + int(c1)), int(rg), int(offC + int(c2))])
    if not rows:
        return np.empty((0, 3), dtype=np.int32)
    arr = np.array(rows, dtype=np.int32)
    return _dedupe_sort_paper_area_paper_rows(arr)


def build_conf_area_conf_skip_from_cparapc(
    adjM: np.ndarray,
    offA: int,
    offP: int,
    offC: int,
    offR: int,
    A: int,
    P: int,
    C: int,
    R: int,
) -> np.ndarray:
    """v3: C–P–A–R–A–P–C → (C,R,C) with full walk enumeration (no subsampling)."""
    conf_paper_list = {i: adjM[offC + i, offP:offP + P].nonzero()[0] for i in range(C)}
    area_author_list = {i: adjM[offR + i, offA:offA + A].nonzero()[0] for i in range(R)}
    author_paper_list = {i: adjM[offA + i, offP:offP + P].nonzero()[0] for i in range(A)}
    paper_conf_list = {i: adjM[offP + i, offC:offC + C].nonzero()[0] for i in range(P)}
    rows = []
    for rloc in range(R):
        a_list = area_author_list[rloc]
        if len(a_list) == 0:
            continue
        rg = offR + rloc
        for a1 in a_list:
            for a2 in a_list:
                a1i, a2i = int(a1), int(a2)
                p1s = author_paper_list.get(a1i)
                p2s = author_paper_list.get(a2i)
                if p1s is None or p2s is None or len(p1s) == 0 or len(p2s) == 0:
                    continue
                for p1 in p1s:
                    c1s = paper_conf_list.get(int(p1), np.array([], dtype=np.int32))
                    if len(c1s) == 0:
                        continue
                    for p2 in p2s:
                        c2s = paper_conf_list.get(int(p2), np.array([], dtype=np.int32))
                        if len(c2s) == 0:
                            continue
                        for c1 in c1s:
                            for c2 in c2s:
                                rows.append([int(offC + c1), int(rg), int(offC + c2)])
    if not rows:
        return np.empty((0, 3), dtype=np.int32)
    arr = np.array(rows, dtype=np.int32)
    return _dedupe_sort_paper_area_paper_rows(arr)


def build_conf_paper_author_paper_conf(
    conf_paper: dict,
    paper_author: dict,
    author_paper: dict,
    paper_conf: dict,
    offC: int,
    offP: int,
    offA: int,
) -> np.ndarray:
    """
    Canonical conf-centric semantic channel: C-P-A-P-C (3-1-0-1-3).

    Output rows are full-path node ids:
      [C1_global, P1_global, A_global, P2_global, C2_global]
    The first column is the "target" (C1) and the last column is the neighbor (C2),
    matching how MAGNN adjlists are written per-target.
    """
    rows = []
    for c1_local, p1_list in tqdm(conf_paper.items(), leave=False):
        if len(p1_list) == 0:
            continue
        c1g = offC + int(c1_local)
        for p1_local in p1_list:
            p1_local = int(p1_local)
            a_list = paper_author.get(p1_local)
            if a_list is None or len(a_list) == 0:
                continue
            p1g = offP + p1_local
            for a_local in a_list:
                a_local = int(a_local)
                p2_list = author_paper.get(a_local)
                if p2_list is None or len(p2_list) == 0:
                    continue
                ag = offA + a_local
                for p2_local in p2_list:
                    p2_local = int(p2_local)
                    c2_list = paper_conf.get(p2_local)
                    if c2_list is None or len(c2_list) == 0:
                        continue
                    p2g = offP + p2_local
                    for c2_local in c2_list:
                        rows.append([c1g, p1g, ag, p2g, offC + int(c2_local)])
    if not rows:
        return np.empty((0, 5), dtype=np.int32)
    return sort_rows(np.array(rows, dtype=np.int32))


def build_variant_adjM(
    variant: str,
    pa: pd.DataFrame,
    pt: pd.DataFrame,
    pc: pd.DataFrame,
    canonical_pr: pd.DataFrame,
    train_pos_local: np.ndarray,
    train_paper_raw_ids: set,
    Au_map: dict,
    Pa_map: dict,
    Te_map: dict,
    Co_map: dict,
    Ar_map: dict,
    A: int,
    P: int,
    T: int,
    C: int,
) -> np.ndarray:
    R = len(Ar_map)
    N = A + P + T + C + R
    offA = 0
    offP = A
    offT = A + P
    offC = A + P + T
    offR = A + P + T + C

    adjM = np.zeros((N, N), dtype=np.int32)

    for p, a in pa[['paper_id', 'author_id']].itertuples(index=False):
        if a in Au_map and p in Pa_map:
            u = offA + Au_map[a]
            v = offP + Pa_map[p]
            adjM[u, v] = 1
            adjM[v, u] = 1

    for p, t in pt[['paper_id', 'term_id']].itertuples(index=False):
        if p in Pa_map and t in Te_map:
            u = offP + Pa_map[p]
            v = offT + Te_map[t]
            adjM[u, v] = 1
            adjM[v, u] = 1

    for p_local, c_local in train_pos_local:
        u = offP + int(p_local)
        v = offC + int(c_local)
        adjM[u, v] = 1
        adjM[v, u] = 1

    if variant == 'v1':
        for p, r in canonical_pr[['paper_id', 'area_id']].itertuples(index=False):
            if p in Pa_map and r in Ar_map:
                u = offP + Pa_map[p]
                v = offR + Ar_map[r]
                adjM[u, v] = 1
                adjM[v, u] = 1

    elif variant == 'v2':
        cr = (
            pc[pc['paper_id'].isin(train_paper_raw_ids)][['paper_id', 'conf_id']].drop_duplicates()
            .merge(canonical_pr[['paper_id', 'area_id']].drop_duplicates(), on='paper_id')
            [['conf_id', 'area_id']]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        for c, r in cr.itertuples(index=False):
            if c in Co_map and r in Ar_map:
                u = offC + Co_map[c]
                v = offR + Ar_map[r]
                adjM[u, v] = 1
                adjM[v, u] = 1

    elif variant == 'v3':
        ar = (
            pa[['paper_id', 'author_id']].drop_duplicates()
            .merge(canonical_pr[['paper_id', 'area_id']].drop_duplicates(), on='paper_id')
            [['author_id', 'area_id']]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        for a, r in ar.itertuples(index=False):
            if a in Au_map and r in Ar_map:
                u = offA + Au_map[a]
                v = offR + Ar_map[r]
                adjM[u, v] = 1
                adjM[v, u] = 1
    else:
        raise ValueError(f'Unknown variant {variant}')

    return adjM


def _validate_skip_metapaths():
    for v, modes in SKIP_METAPATHS.items():
        if v not in OUT_PREFIX:
            raise ValueError(f'SKIP_METAPATHS key {v!r} missing from OUT_PREFIX')
        if len(modes) != 2:
            raise ValueError(f'{v}: expected 2 modes')
        n0 = len(modes[0])
        for other in SKIP_METAPATHS:
            if len(SKIP_METAPATHS[other][0]) != n0:
                raise ValueError('mode-0 SKIP_METAPATHS rows must match length across variants')
            if len(SKIP_METAPATHS[other][1]) != len(modes[1]):
                raise ValueError('mode-1 SKIP_METAPATHS rows must match length across variants')


def preprocess_one_variant(variant: str) -> None:
    if variant not in OUT_PREFIX:
        raise ValueError(f'Unknown variant {variant!r}')

    OUT = os.path.abspath(OUT_PREFIX[variant])
    set_seed(SEED)

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT, exist_ok=True)

    spec = SKIP_METAPATHS[variant]
    print(f'=== DBLP LP skip preprocessing | {variant} ===')
    print(f'    SKIP_METAPATHS[{variant!r}]: mode0 rows={len(spec[0])}, mode1 rows={len(spec[1])} '
          f'(see module docstring for disk mapping)')

    print('1) load raw tables')
    author = pd.read_csv(
        os.path.join(RAW, 'author_label.txt'),
        sep='\t',
        names=['author_id', 'label', 'author_name'],
        header=None,
        encoding='utf-8'
    )
    pa = pd.read_csv(
        os.path.join(RAW, 'paper_author.txt'),
        sep='\t',
        names=['paper_id', 'author_id'],
        header=None,
        encoding='utf-8'
    )
    pc = pd.read_csv(
        os.path.join(RAW, 'paper_conf.txt'),
        sep='\t',
        names=['paper_id', 'conf_id'],
        header=None,
        encoding='utf-8'
    )
    pt = pd.read_csv(
        os.path.join(RAW, 'paper_term.txt'),
        sep='\t',
        names=['paper_id', 'term_id'],
        header=None,
        encoding='utf-8'
    )

    print('2) filter to labeled authors + papers with venues')
    valid_authors = set(author['author_id'])
    pa = pa[pa['author_id'].isin(valid_authors)].reset_index(drop=True)

    valid_papers = set(pa['paper_id'])
    pc = pc[pc['paper_id'].isin(valid_papers)].reset_index(drop=True)
    pt = pt[pt['paper_id'].isin(valid_papers)].reset_index(drop=True)
    pa = pa[pa['paper_id'].isin(valid_papers)].reset_index(drop=True)

    big_confs = pc['conf_id'].value_counts().loc[lambda x: x >= MIN_CONF].index
    pc = pc[pc['conf_id'].isin(big_confs)].reset_index(drop=True)

    valid_papers = set(pc['paper_id'])
    pt = pt[pt['paper_id'].isin(valid_papers)].reset_index(drop=True)
    pa = pa[pa['paper_id'].isin(valid_papers)].reset_index(drop=True)

    shared = np.load(SHARED)
    if 'paper_subset' in shared.files:
        keep_papers = set(shared['paper_subset'].tolist()) & valid_papers
        pc = pc[pc['paper_id'].isin(keep_papers)].reset_index(drop=True)
        pa = pa[pa['paper_id'].isin(keep_papers)].reset_index(drop=True)
        pt = pt[pt['paper_id'].isin(keep_papers)].reset_index(drop=True)
        valid_papers = set(pc['paper_id'])
        print(f'   Enforced paper subset from SHARED: {len(valid_papers)} papers')

    print('3) build canonical paper->area relation')
    a2r = author.set_index('author_id')['label']
    pr_raw = (
        pa.assign(area_id=pa['author_id'].map(a2r))
        [['paper_id', 'area_id']]
        .drop_duplicates()
        .dropna()
        .reset_index(drop=True)
    )

    area_counts = pr_raw.groupby('paper_id')['area_id'].nunique()
    bad_papers = set(area_counts[area_counts != 1].index.tolist())
    if bad_papers:
        print(f'   Removing {len(bad_papers)} papers violating 1 paper -> 1 area')
        pa = pa[~pa['paper_id'].isin(bad_papers)].reset_index(drop=True)
        pc = pc[~pc['paper_id'].isin(bad_papers)].reset_index(drop=True)
        pt = pt[~pt['paper_id'].isin(bad_papers)].reset_index(drop=True)

        pr_raw = (
            pa.assign(area_id=pa['author_id'].map(a2r))
            [['paper_id', 'area_id']]
            .drop_duplicates()
            .dropna()
            .reset_index(drop=True)
        )
        area_counts = pr_raw.groupby('paper_id')['area_id'].nunique()
        assert (area_counts == 1).all(), 'Still not 1 paper -> 1 area after filtering.'

    canonical_pr = pr_raw.drop_duplicates(subset=['paper_id']).reset_index(drop=True)
    print(f'   Canonical paper-area edges: {len(canonical_pr)}')

    print('4) reindex per type + offsets')
    Au_ids = sorted(pa['author_id'].unique())
    Pa_ids = sorted(set(pc['paper_id']))
    Te_ids = sorted(pt['term_id'].unique())
    Co_ids = sorted(pc['conf_id'].unique())
    Ar_ids = sorted(canonical_pr['area_id'].unique())

    Au_map = {a: i for i, a in enumerate(Au_ids)}
    Pa_map = {p: i for i, p in enumerate(Pa_ids)}
    Te_map = {t: i for i, t in enumerate(Te_ids)}
    Co_map = {c: i for i, c in enumerate(Co_ids)}
    Ar_map = {r: i for i, r in enumerate(Ar_ids)}

    A = len(Au_map)
    P = len(Pa_map)
    T = len(Te_map)
    C = len(Co_map)
    R = len(Ar_map)
    N = A + P + T + C + R

    offA = 0
    offP = A
    offT = A + P
    offC = A + P + T
    offR = A + P + T + C

    print(f'   Authors={A}, Papers={P}, Terms={T}, Confs={C}, Areas={R}, Total={N}')

    print('5) load shared train/val/test splits')

    def map_pairs(arr, Pmap, Cmap, name):
        df = pd.DataFrame(arr, columns=['paper_id', 'conf_id'])
        df['paper_id'] = df['paper_id'].map(Pmap)
        df['conf_id'] = df['conf_id'].map(Cmap)
        out = df.dropna().to_numpy(dtype=np.int32)
        dropped = len(arr) - len(out)
        if dropped:
            print(f'   [warn] {name}: dropped {dropped} pairs')
        return out

    train_pos = map_pairs(shared['train_pos'], Pa_map, Co_map, 'train_pos')
    val_pos = map_pairs(shared['val_pos'], Pa_map, Co_map, 'val_pos')
    test_pos = map_pairs(shared['test_pos'], Pa_map, Co_map, 'test_pos')
    train_neg = map_pairs(shared['train_neg'], Pa_map, Co_map, 'train_neg')
    val_neg = map_pairs(shared['val_neg'], Pa_map, Co_map, 'val_neg')
    test_neg = map_pairs(shared['test_neg'], Pa_map, Co_map, 'test_neg')
    train_paper_raw_ids = {p for p, p_local in Pa_map.items() if int(p_local) in set(train_pos[:, 0].tolist())}

    print('6) build raw adjacency matrix')
    type_mask = np.zeros(N, dtype=np.int32)
    type_mask[offP:offP + P] = 1
    type_mask[offT:offT + T] = 2
    type_mask[offC:offC + C] = 3
    type_mask[offR:offR + R] = 4

    adjM = build_variant_adjM(
        variant=variant,
        pa=pa,
        pt=pt,
        pc=pc,
        canonical_pr=canonical_pr,
        train_pos_local=train_pos,
        train_paper_raw_ids=train_paper_raw_ids,
        Au_map=Au_map,
        Pa_map=Pa_map,
        Te_map=Te_map,
        Co_map=Co_map,
        Ar_map=Ar_map,
        A=A, P=P, T=T, C=C
    )
    print(f'   Undirected edges: {adjM.sum() // 2}')

    print('7) build local neighbor lists')
    paper_author = defaultdict(list)
    author_paper = defaultdict(list)
    paper_term = defaultdict(list)
    term_paper = defaultdict(list)
    paper_conf = defaultdict(list)
    conf_paper = defaultdict(list)
    paper_area = defaultdict(list)

    for p, a in pa[['paper_id', 'author_id']].itertuples(index=False):
        if p in Pa_map and a in Au_map:
            pl = Pa_map[p]
            al = Au_map[a]
            paper_author[pl].append(al)
            author_paper[al].append(pl)

    for p, t in pt[['paper_id', 'term_id']].itertuples(index=False):
        if p in Pa_map and t in Te_map:
            pl = Pa_map[p]
            tl = Te_map[t]
            paper_term[pl].append(tl)
            term_paper[tl].append(pl)

    for p_local, c_local in train_pos:
        pl = int(p_local)
        cl = int(c_local)
        paper_conf[pl].append(cl)
        conf_paper[cl].append(pl)

    for p, r in canonical_pr[['paper_id', 'area_id']].itertuples(index=False):
        if p in Pa_map and r in Ar_map:
            paper_area[Pa_map[p]].append(Ar_map[r])

    paper_author = {k: np.array(sorted(set(v)), dtype=np.int32) for k, v in paper_author.items()}
    author_paper = {k: np.array(sorted(set(v)), dtype=np.int32) for k, v in author_paper.items()}
    paper_term = {k: np.array(sorted(set(v)), dtype=np.int32) for k, v in paper_term.items()}
    term_paper = {k: np.array(sorted(set(v)), dtype=np.int32) for k, v in term_paper.items()}
    paper_conf = {k: np.array(sorted(set(v)), dtype=np.int32) for k, v in paper_conf.items()}
    conf_paper = {k: np.array(sorted(set(v)), dtype=np.int32) for k, v in conf_paper.items()}
    paper_area = {k: np.array(sorted(set(v)), dtype=np.int32) for k, v in paper_area.items()}

    area_paper = defaultdict(list)
    for p_local, areas in paper_area.items():
        for r_local in areas:
            area_paper[r_local].append(p_local)
    area_paper = {k: np.array(sorted(set(v)), dtype=np.int32) for k, v in area_paper.items()}

    pathlib.Path(os.path.join(OUT, '0')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(OUT, '1')).mkdir(parents=True, exist_ok=True)

    print('8) build paper-centric artifacts')
    arr_101 = build_3hop_from_center_to_end(author_paper, offA, offP)
    arr_101 = maybe_cap_rows(arr_101, MAX_CHANNEL_INSTANCES, SEED + 101, f'{variant} 1-0-1')
    print_metapath_debug(spec[0][0], arr_101)
    assert_type_pattern(arr_101, type_mask, [1, 0, 1], '1-0-1')
    report_arr_stats('1-0-1', arr_101)
    save_per_target_pickle_and_adjlist(os.path.join(OUT, '0'), '1-0-1', arr_101, P, offP)

    arr_121 = build_3hop_from_center_to_end(term_paper, offT, offP)
    arr_121 = maybe_cap_rows(arr_121, MAX_CHANNEL_INSTANCES, SEED + 121, f'{variant} 1-2-1')
    print_metapath_debug(spec[0][1], arr_121)
    assert_type_pattern(arr_121, type_mask, [1, 2, 1], '1-2-1')
    report_arr_stats('1-2-1', arr_121)
    save_per_target_pickle_and_adjlist(os.path.join(OUT, '0'), '1-2-1', arr_121, P, offP)

    if variant == 'v1':
        arr_141 = build_3hop_from_center_to_end(area_paper, offR, offP)
    elif variant == 'v2':
        arr_141 = build_paper_area_paper_skip_from_pcrcp(adjM, offP, offC, offR, P, C, R)
    elif variant == 'v3':
        arr_141 = build_paper_area_paper_skip_from_parap(adjM, offA, offP, offR, A, P, R)
    else:
        raise ValueError(variant)
    arr_141 = maybe_cap_rows(arr_141, MAX_SKIP_INSTANCES, SEED + 141, f'{variant} 1-4-1')
    assert_type_pattern(arr_141, type_mask, [1, 4, 1], '1-4-1')
    # Show all equivalent full/skip expressions for this aligned semantic slot.
    for s in spec[0][2:5]:
        print_metapath_debug(s, arr_141)
    report_arr_stats('1-4-1', arr_141)
    save_per_target_pickle_and_adjlist(os.path.join(OUT, '0'), '1-4-1', arr_141, P, offP)

    arr_131 = build_3hop_from_center_to_end(conf_paper, offC, offP)
    arr_131 = maybe_cap_rows(arr_131, MAX_CHANNEL_INSTANCES, SEED + 131, f'{variant} 1-3-1')
    print_metapath_debug(spec[0][5], arr_131)
    assert_type_pattern(arr_131, type_mask, [1, 3, 1], '1-3-1')
    report_arr_stats('1-3-1', arr_131)
    save_per_target_pickle_and_adjlist(os.path.join(OUT, '0'), '1-3-1', arr_131, P, offP)

    print('9) build conf-centric artifacts')
    arr_31013 = build_conf_paper_author_paper_conf(
        conf_paper=conf_paper,
        paper_author=paper_author,
        author_paper=author_paper,
        paper_conf=paper_conf,
        offC=offC,
        offP=offP,
        offA=offA,
    )
    arr_31013 = maybe_cap_rows(arr_31013, MAX_SKIP_INSTANCES, SEED + 31013, f'{variant} 3-1-0-1-3')
    assert_type_pattern(arr_31013, type_mask, [3, 1, 0, 1, 3], '3-1-0-1-3')
    print_metapath_debug(spec[1][0], arr_31013)
    report_arr_stats('3-1-0-1-3', arr_31013)
    save_per_target_pickle_and_adjlist(os.path.join(OUT, '1'), '3-1-0-1-3', arr_31013, C, offC)

    arr_313 = build_3hop_from_center_to_end(paper_conf, offP, offC)
    arr_313 = maybe_cap_rows(arr_313, MAX_CHANNEL_INSTANCES, SEED + 313, f'{variant} 3-1-3')
    print_metapath_debug(spec[1][1], arr_313)
    assert_type_pattern(arr_313, type_mask, [3, 1, 3], '3-1-3')
    report_arr_stats('3-1-3', arr_313)
    save_per_target_pickle_and_adjlist(os.path.join(OUT, '1'), '3-1-3', arr_313, C, offC)

    print('10) save global files')
    sp.save_npz(os.path.join(OUT, 'adjM.npz'), sp.csr_matrix(adjM))
    np.save(os.path.join(OUT, 'node_types.npy'), type_mask)

    np.savez(
        os.path.join(OUT, 'train_val_test_pos_paper_conf.npz'),
        train_pos=train_pos, val_pos=val_pos, test_pos=test_pos
    )
    np.savez(
        os.path.join(OUT, 'train_val_test_neg_paper_conf.npz'),
        train_neg=train_neg, val_neg=val_neg, test_neg=test_neg
    )

    with open(os.path.join(OUT, 'node_maps.pkl'), 'wb') as f:
        pickle.dump({
            'author_idx': Au_map,
            'paper_idx': Pa_map,
            'term_idx': Te_map,
            'conf_idx': Co_map,
            'area_idx': Ar_map,
            'offsets': {'A': offA, 'P': offP, 'T': offT, 'C': offC, 'R': offR},
        }, f)

    print(f'Done! Saved to {OUT}\n')


def main():
    _validate_skip_metapaths()
    parser = argparse.ArgumentParser(description='DBLP LP canonical skip-node preprocessing (v1–v3)')
    parser.add_argument(
        '--variant',
        choices=['v1', 'v2', 'v3', 'all'],
        required=True,
        help="Graph variant, or 'all' to run v1, v2, v3 in order.",
    )
    parser.add_argument(
        '--max-skip-instances',
        type=int,
        default=0,
        help='Cap rows for large channels (1-4-1 and 3-1-0-1-3); <=0 disables cap (default: 0).',
    )
    parser.add_argument(
        '--max-channel-instances',
        type=int,
        default=0,
        help='Optional cap for non-skip channels (1-0-1,1-2-1,1-3-1,3-1-3); <=0 disables.',
    )
    args = parser.parse_args()
    global MAX_SKIP_INSTANCES
    global MAX_CHANNEL_INSTANCES
    MAX_SKIP_INSTANCES = int(args.max_skip_instances)
    MAX_CHANNEL_INSTANCES = int(args.max_channel_instances)

    if args.variant == 'all':
        for v in ('v1', 'v2', 'v3'):
            preprocess_one_variant(v)
    else:
        preprocess_one_variant(args.variant)


if __name__ == '__main__':
    main()
