"""Data loaders for SlotGAT HGB-format datasets."""

import json
import os
import sys
import numpy as np
import scipy.sparse as sp


def load_data(prefix='IMDB_var1', multi_labels=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, '..')
    sys.path.insert(0, base_dir)
    from scripts.data_loader import data_loader

    data_path = os.path.join(base_dir, 'data', prefix)
    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Data directory not found: {data_path}\n"
            f"Run preprocess_IMDB.py first to generate HGB-format data."
        )

    dl = data_loader(data_path)

    features = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th)

    adjM = sum(dl.links['data'].values())

    labels_path = os.path.join(data_path, 'labels.npy')
    splits_path = os.path.join(data_path, 'train_val_test_idx.npz')

    if not os.path.isfile(labels_path) or not os.path.isfile(splits_path):
        raise FileNotFoundError(
            f"Fixed split artifacts not found in {data_path}.\n"
            f"Re-run preprocess_IMDB.py to generate labels.npy and "
            f"train_val_test_idx.npz."
        )

    labels = np.load(labels_path)
    split_data = np.load(splits_path)
    train_idx = np.sort(split_data['train_idx'])
    val_idx = np.sort(split_data['val_idx'])
    test_idx = np.sort(split_data['test_idx'])

    train_val_test_idx = {
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
    }

    return features, adjM, labels, train_val_test_idx, dl


def load_data_lp(prefix='DBLP_var1_lp'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, '..')
    sys.path.insert(0, base_dir)
    from scripts.data_loader import data_loader

    data_path = os.path.join(base_dir, 'data', prefix)
    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Data directory not found: {data_path}\n"
            f"Run preprocess_DBLP_lp.py first to generate LP artifacts."
        )

    dl = data_loader(data_path)

    features = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th)

    adjM = sum(dl.links['data'].values())

    splits_path = os.path.join(data_path, 'lp_splits.npz')
    meta_path = os.path.join(data_path, 'lp_meta.json')
    if not os.path.isfile(splits_path) or not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"LP artifacts not found in {data_path}.\n"
            f"Expected files: lp_splits.npz and lp_meta.json. "
            f"Re-run preprocess_DBLP_lp.py."
        )

    split_data = np.load(splits_path)
    lp_splits = {
        'train_pos': split_data['train_pos'],
        'val_pos': split_data['val_pos'],
        'test_pos': split_data['test_pos'],
        'train_neg': split_data['train_neg'],
        'val_neg': split_data['val_neg'],
        'test_neg': split_data['test_neg'],
    }
    with open(meta_path, 'r', encoding='utf-8') as f:
        lp_meta = json.load(f)

    return features, adjM, dl, lp_splits, lp_meta
