"""
Data bridge: load preprocessed IMDB data and convert it into the
DGL heterograph + feature format that SeHGNN expects.
"""

import os

import dgl
import numpy as np
import scipy.sparse as sp
import torch

# MAGNN type ids: 0=movie, 1=director, 2=actor, 3=keyword
# SeHGNN type keys: M, D, A, L
IMDB_TYPE_ID_TO_KEY = {0: 'M', 1: 'D', 2: 'A', 3: 'L'}


def _offsets_from_type_mask(type_mask):
    """Return (counts, offsets) per type id present in type_mask."""
    counts = {}
    offsets = {}
    off = 0
    for tid in sorted(set(type_mask)):
        n = int((type_mask == tid).sum())
        counts[tid] = n
        offsets[tid] = off
        off += n
    return counts, offsets


def load_imdb_graph(base_dir, feats_type=2):
    """
    Load an IMDB variant directory and return a DGL heterograph.

    feats_type=2: movies keep BoW features, all other types get identity.
    """
    adjM = sp.load_npz(os.path.join(base_dir, 'adjM.npz'))
    type_mask = np.load(os.path.join(base_dir, 'node_types.npy'))

    counts, offsets = _offsets_from_type_mask(type_mask)
    coo = adjM.tocoo()
    num_types = len(counts)

    node_counts = {IMDB_TYPE_ID_TO_KEY[tid]: counts[tid] for tid in counts}

    new_edges = {}
    for src_tid in range(num_types):
        for dst_tid in range(num_types):
            src_off = offsets[src_tid]
            dst_off = offsets[dst_tid]
            src_cnt = counts[src_tid]
            dst_cnt = counts[dst_tid]

            mask = ((coo.row >= dst_off) & (coo.row < dst_off + dst_cnt) &
                    (coo.col >= src_off) & (coo.col < src_off + src_cnt))
            if mask.sum() == 0:
                continue

            local_row = coo.row[mask] - dst_off
            local_col = coo.col[mask] - src_off

            src_key = IMDB_TYPE_ID_TO_KEY[src_tid]
            dst_key = IMDB_TYPE_ID_TO_KEY[dst_tid]
            etype_name = f'{src_key}-{dst_key}'
            new_edges[(src_key, etype_name, dst_key)] = (
                local_col.astype(np.int64),
                local_row.astype(np.int64),
            )

    g = dgl.heterograph(new_edges, num_nodes_dict=node_counts)

    f0 = sp.load_npz(os.path.join(base_dir, 'features_0.npz'))
    movie_feat = torch.FloatTensor(f0.toarray())

    if feats_type == 2:
        g.nodes['M'].data['M'] = movie_feat
        for tid, key in IMDB_TYPE_ID_TO_KEY.items():
            if tid == 0:
                continue
            if key in g.ntypes:
                g.nodes[key].data[key] = torch.eye(node_counts[key])
    else:
        raise ValueError(f'Only feats_type=2 is implemented, got {feats_type}')

    return g, node_counts, type_mask


def load_imdb_nc_labels(base_dir):
    """Load movie-genre NC labels and splits from a preprocessed IMDB variant."""
    labels = np.load(os.path.join(base_dir, 'labels.npy'))
    idx = np.load(os.path.join(base_dir, 'train_val_test_idx.npz'))

    train_nid = idx['train_idx']
    val_nid = idx['val_idx']
    test_nid = idx['test_idx']

    num_classes = len(np.unique(labels))

    return torch.LongTensor(labels), train_nid, val_nid, test_nid, num_classes
