"""
Movie-node features aligned with MAGNN skip semantic metapaths (center type movie).

Uses the same neighbor-pair discovery as ``MAGNN/preprocess_IMDB_skip.py`` for the
three movie-row specs in ``SKIP_METAPATHS['v*'][0]``, yielding channels
``M`` (BoW), ``MDM``, ``MAM``, ``MLM`` — identical key set across variants v1..v4.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

from .data import IMDB_TYPE_ID_TO_KEY

_MAGNN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'MAGNN'))


def _maggn_skip_imports():
    if _MAGNN_DIR not in sys.path:
        sys.path.insert(0, _MAGNN_DIR)
    from preprocess_IMDB_skip import (  # noqa: WPS433
        SKIP_METAPATHS,
        get_metapath_neighbor_pairs_semantic,
        parse_skip_metapath,
    )

    return SKIP_METAPATHS, get_metapath_neighbor_pairs_semantic, parse_skip_metapath


def semantic_tuple_to_feat_key(semantic: tuple) -> str:
    return ''.join(IMDB_TYPE_ID_TO_KEY[int(t)] for t in semantic)


def _pairs_to_rownorm_csr(
    pairs_dict: dict,
    type_mask: np.ndarray,
    n_m: int,
) -> sp.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    for (src, dst) in pairs_dict.keys():
        s, d = int(src), int(dst)
        if int(type_mask[s]) != 0 or int(type_mask[d]) != 0:
            continue
        if s >= n_m or d >= n_m:
            continue
        rows.append(s)
        cols.append(d)
    if not rows:
        return sp.csr_matrix((n_m, n_m), dtype=np.float64)
    data = np.ones(len(rows), dtype=np.float64)
    mat = sp.csr_matrix((data, (rows, cols)), shape=(n_m, n_m), dtype=np.float64)
    # Older SciPy builds don't expose eliminate_duplicates on csr_matrix.
    mat.sum_duplicates()
    sums = np.asarray(mat.sum(axis=1)).ravel()
    inv = np.zeros_like(sums)
    nz = sums > 0
    inv[nz] = 1.0 / sums[nz]
    return sp.diags(inv) @ mat


def build_imdb_skip_semantic_movie_feats(
    base_dir: str,
    variant: int,
    movie_feat: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Parameters
    ----------
    base_dir
        Preprocessed skip bundle directory (contains ``adjM.npz``, ``node_types.npy``).
    variant
        Integer 1..4 (selects ``SKIP_METAPATHS['v{N}']``).
    movie_feat
        Movie BoW tensor ``[n_m, bow_dim]`` (same order as global movie indices 0..n_m-1).
    device
        Target device for returned tensors.

    Returns
    -------
    dict
        Keys ``M``, ``MAM``, ``MDM``, ``MLM`` (``M`` is a copy of ``movie_feat`` on device).
    """
    SKIP_METAPATHS, get_metapath_neighbor_pairs_semantic, parse_skip_metapath = (
        _maggn_skip_imports()
    )
    vkey = f'v{int(variant)}'
    if vkey not in SKIP_METAPATHS:
        raise ValueError(f'variant must be 1..4, got {variant!r}')

    specs_movie = SKIP_METAPATHS[vkey][0]
    type_mask = np.load(os.path.join(base_dir, 'node_types.npy'))
    n_m = int((type_mask == 0).sum())
    if movie_feat.shape[0] != n_m:
        raise ValueError(
            f'movie_feat rows {movie_feat.shape[0]} != n_movies {n_m} from node_types',
        )

    adj = sp.load_npz(os.path.join(base_dir, 'adjM.npz'))
    adj_dense = adj.toarray()

    neighbor_outs = get_metapath_neighbor_pairs_semantic(
        adj_dense, type_mask, list(specs_movie),
    )

    movie_np = movie_feat.detach().cpu().numpy().astype(np.float64)
    feats: dict[str, torch.Tensor] = {
        'M': movie_feat.to(device=device, dtype=torch.float32),
    }

    for spec, pairs_dict in zip(specs_movie, neighbor_outs):
        _, _, semantic = parse_skip_metapath(spec)
        key = semantic_tuple_to_feat_key(semantic)
        smat = _pairs_to_rownorm_csr(pairs_dict, type_mask, n_m)
        agg = smat @ movie_np
        feats[key] = torch.as_tensor(agg, dtype=torch.float32, device=device)

    return feats
