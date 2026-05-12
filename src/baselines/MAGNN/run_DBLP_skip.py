#!/usr/bin/env python3
"""
MAGNN link prediction on DBLP paper–venue with unified skip preprocess (v1–v3).

Examples:
  python preprocess_DBLP_skip.py --variant all
  python run_DBLP_skip.py --variants v1 --seeds 1566911444 --epoch 2
  python run_DBLP_skip.py --variants v1,v2,v3 --seeds 1566911444 \\
      --epoch 100 --save-postfix DBLP_skip
  # Later, same cwd (score CSV names must match save_postfix + variant + seed):
  python run_DBLP_skip.py --compare-only --compare v1,v2 \\
      --seeds 1566911444,20241017 --save-postfix DBLP_skip
  python run_DBLP_skip.py --compare-only \\
      --score-csv-a path/to/a_scores.csv --score-csv-b path/to/b_scores.csv
"""
import argparse
import os
import sys
import itertools
import time

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from model import MAGNN_lp
from utils.data import load_DBLP_lp_pc_var1_data
from utils.dblp_lp_eval import subsample_test_neg_per_paper
from utils.pytorchtools import EarlyStopping
from utils.tools import index_generator, parse_minibatch_LastFM


def set_seed(seed: int) -> None:
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


# Shared training hyperparameters (match other DBLP runners)
num_ntype = 5
dropout_rate = 0.5
lr = 0.005
weight_decay = 0.001


EXPECTED_METAPATHS_SKIP = [
    [(1, 0, 1), (1, 2, 1), (1, 4, 1), (1, 3, 1)],
    [(3, 1, 0, 1, 3), (3, 1, 3)],
]

# NOTE: edge types are the canonical ones used by the DBLP paper-conf KG:
# 0:A→P, 1:P→A, 2:P→T, 3:T→P, 4:P→C, 5:C→P, 6:P→R, 7:R→P
# Skip mode uses the canonical conf-centric semantic channel 3-1-0-1-3 (C-P-A-P-C),
# matching the normal DBLP runner’s conference-centric metapath.
ETYPES_LISTS_SKIP = [
    [[1, 0], [2, 3], [4, 5], [6, 7]],
    [[5, 1, 0, 4], [5, 4]],
]
USE_MASKS_SKIP = [[True] * 4, [True] * 2]
NO_MASKS_SKIP = [[False] * 4, [False] * 2]


def run_model_DBLP_skip(
    feats_type: int,
    hidden_dim: int,
    num_heads: int,
    attn_vec_dim: int,
    rnn_type: str,
    num_epochs: int,
    patience: int,
    batch_size: int,
    neighbor_samples: int,
    repeat: int,
    save_postfix: str,
    threshold: float,
    K: int,
    neg_mult: int,
    base_seed: int,
    preprocessed_base: str,
    test_neg_per_paper: int = 3,
    debug_metapaths: bool = False,
):
    print("→ Loading preprocessed DBLP paper–venue (skip) data ...")
    t0 = time.time()
    (adjlists, edge_metapath_indices_list, _, type_mask, pos_splits, neg_splits, num_paper, num_conf) = load_DBLP_lp_pc_var1_data(
        EXPECTED_METAPATHS_SKIP, base=preprocessed_base
    )
    print(f"   loaded in {time.time() - t0:.2f}s")

    if debug_metapaths:
        print("→ Debug metapath instance tensor non-emptiness")
        expected_metapaths = EXPECTED_METAPATHS_SKIP
        for mode in range(len(expected_metapaths)):
            for mi, meta in enumerate(expected_metapaths[mode]):
                edge_idx = edge_metapath_indices_list[mode][mi]
                # In this MAGNN codebase, edge_metapath_indices can be stored either as:
                # - a dict: target -> ndarray [n_paths_for_target, L]
                # - a list: per-target ndarrays
                # - a single ndarray [total_paths, L]
                non_empty = 0
                total_paths = 0
                max_paths = 0
                if edge_idx is None:
                    pass
                elif isinstance(edge_idx, dict):
                    for _, v in edge_idx.items():
                        if v is None:
                            continue
                        vv = np.asarray(v)
                        n = int(vv.shape[0]) if vv.ndim >= 2 else 0
                        if n > 0:
                            non_empty += 1
                            total_paths += n
                            if n > max_paths:
                                max_paths = n
                elif isinstance(edge_idx, (list, tuple)):
                    for v in edge_idx:
                        if v is None:
                            continue
                        vv = np.asarray(v)
                        n = int(vv.shape[0]) if vv.ndim >= 2 else 0
                        if n > 0:
                            non_empty += 1
                            total_paths += n
                            if n > max_paths:
                                max_paths = n
                else:
                    edge_arr = np.asarray(edge_idx)
                    if edge_arr.ndim >= 2 and edge_arr.shape[0] > 0:
                        # endpoints are first and last columns (target-centric by construction)
                        tgt = edge_arr[:, 0]
                        counts = np.bincount(tgt.astype(np.int64))
                        non_empty = int((counts > 0).sum())
                        total_paths = int(edge_arr.shape[0])
                        max_paths = int(counts.max()) if len(counts) else 0
                denom = num_paper if mode == 0 else num_conf
                print(
                    f"   mode={mode} metapath[{mi}]={meta} L={len(meta)} "
                    f"non_empty_targets={non_empty}/{denom} total_paths={total_paths} "
                    f"max_paths_per_target={max_paths}"
                )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"→ Device: {device}")

    # Features
    print("→ Building features ...")
    features_list, in_dims = [], []
    if feats_type == 0:
        for i in range(num_ntype):
            dim = int((type_mask == i).sum())
            in_dims.append(dim)
            idx = np.vstack((np.arange(dim), np.arange(dim)))
            features_list.append(
                torch.sparse_coo_tensor(
                    torch.LongTensor(idx),
                    torch.FloatTensor(np.ones(dim)),
                    torch.Size([dim, dim]),
                    device=device,
                )
            )
    else:
        for i in range(num_ntype):
            in_dims.append(10)
            features_list.append(torch.zeros(((type_mask == i).sum(), 10), device=device))

    train_pos = pos_splits["train_pos_paper_conf"]
    val_pos = pos_splits["val_pos_paper_conf"]
    test_pos = pos_splits["test_pos_paper_conf"]
    train_neg = neg_splits["train_neg_paper_conf"]
    val_neg = neg_splits["val_neg_paper_conf"]
    test_neg = neg_splits["test_neg_paper_conf"]

    if test_neg_per_paper and test_neg_per_paper > 0:
        n_full = len(test_neg)
        test_neg = subsample_test_neg_per_paper(test_neg, test_neg_per_paper, base_seed)
        print(f"→ Test negatives subsampled for evaluation: {n_full} -> {len(test_neg)} (max {test_neg_per_paper} per paper)")

    print(f"→ Targets: papers={num_paper}, confs={num_conf}")
    print(f"→ Splits: train_pos={len(train_pos)}, val_pos={len(val_pos)}, test_pos={len(test_pos)}")
    print(f"           train_neg={len(train_neg)}, val_neg={len(val_neg)}, test_neg(eval)={len(test_neg)}")
    print(f"→ Metapaths per mode: {[len(m) for m in EXPECTED_METAPATHS_SKIP]} (etypes_lists: {[len(e) for e in ETYPES_LISTS_SKIP]})")
    print(f"→ GNN layers (K): {K}")

    rng = np.random.default_rng(base_seed)
    val_neg_fixed = val_neg.copy()
    if len(val_neg_fixed) > 0:
        val_neg_fixed = val_neg_fixed[rng.permutation(len(val_neg_fixed))]
    val_neg_cursor = 0

    auc_list, ap_list = [], []
    prec_list, rec_list, f1_list, acc_list = [], [], [], []
    hits1_list, hits3_list, hits5_list, mrr_list = [], [], [], []
    train_wall_list, epochs_ran_list = [], []

    os.makedirs("checkpoint", exist_ok=True)

    for rep in range(repeat):
        print(f"\n===== Run {rep + 1}/{repeat} =====")
        rep_t0 = time.perf_counter()
        net = MAGNN_lp(
            [4, 2],
            8,
            ETYPES_LISTS_SKIP,
            in_dims,
            hidden_dim,
            hidden_dim,
            num_heads,
            attn_vec_dim,
            rnn_type,
            dropout_rate,
            num_layers=K,
        ).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        print(f"→ Model params: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}")

        ckpt_path = f"checkpoint/checkpoint_{save_postfix}_K{K}.pt"
        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path=ckpt_path)
        train_pos_idx_generator = index_generator(batch_size=batch_size, num_data=len(train_pos))
        val_idx_generator = index_generator(batch_size=batch_size, num_data=len(val_pos), shuffle=False)

        epochs_ran = num_epochs
        for epoch in range(num_epochs):
            epoch_t0 = time.time()
            net.train()
            iters = train_pos_idx_generator.num_iterations()
            pbar = tqdm(range(iters), desc=f"Epoch {epoch:03d} [train]", leave=False)
            running_loss = 0.0

            for it in pbar:
                pos_idx = train_pos_idx_generator.next()
                pos_idx.sort()
                train_pos_batch = train_pos[pos_idx].tolist()
                neg_bs = neg_mult * len(pos_idx)
                replace_flag = neg_bs > len(train_neg)
                neg_idx = rng.choice(len(train_neg), neg_bs, replace=replace_flag)
                neg_idx.sort()
                train_neg_batch = train_neg[neg_idx].tolist()

                pos_g_lists, pos_indices_lists, pos_idx_batch_mapped_lists = parse_minibatch_LastFM(
                    adjlists,
                    edge_metapath_indices_list,
                    train_pos_batch,
                    device,
                    neighbor_samples,
                    USE_MASKS_SKIP,
                    num_paper,
                )
                neg_g_lists, neg_indices_lists, neg_idx_batch_mapped_lists = parse_minibatch_LastFM(
                    adjlists,
                    edge_metapath_indices_list,
                    train_neg_batch,
                    device,
                    neighbor_samples,
                    NO_MASKS_SKIP,
                    num_paper,
                )

                [pos_emb_left, pos_emb_right], _ = net(
                    (pos_g_lists, features_list, type_mask, pos_indices_lists, pos_idx_batch_mapped_lists)
                )
                [neg_emb_left, neg_emb_right], _ = net(
                    (neg_g_lists, features_list, type_mask, neg_indices_lists, neg_idx_batch_mapped_lists)
                )
                pos_emb_left = pos_emb_left.view(-1, 1, pos_emb_left.shape[1])
                pos_emb_right = pos_emb_right.view(-1, pos_emb_right.shape[1], 1)
                neg_emb_left = neg_emb_left.view(-1, 1, neg_emb_left.shape[1])
                neg_emb_right = neg_emb_right.view(-1, neg_emb_right.shape[1], 1)

                pos_out = torch.bmm(pos_emb_left, pos_emb_right)
                neg_out = -torch.bmm(neg_emb_left, neg_emb_right)
                train_loss = -(F.logsigmoid(pos_out).mean() + F.logsigmoid(neg_out).mean())

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                running_loss += float(train_loss.item())
                if (it + 1) % max(1, iters // 5) == 0:
                    pbar.set_postfix(loss=f"{running_loss / (it + 1):.4f}")

            # validation
            net.eval()
            val_loss_vals = []
            v_iters = val_idx_generator.num_iterations()
            vbar = tqdm(range(v_iters), desc=f"Epoch {epoch:03d} [val]  ", leave=False)
            with torch.no_grad():
                for _ in vbar:
                    val_idx = val_idx_generator.next()
                    val_pos_batch = val_pos[val_idx].tolist()
                    neg_bs = neg_mult * len(val_idx)
                    if len(val_neg_fixed) == 0:
                        val_neg_batch = []
                    else:
                        end = val_neg_cursor + neg_bs
                        if end <= len(val_neg_fixed):
                            val_neg_batch = val_neg_fixed[val_neg_cursor:end]
                            val_neg_cursor = end % len(val_neg_fixed)
                        else:
                            take1 = len(val_neg_fixed) - val_neg_cursor
                            take2 = neg_bs - take1
                            part1 = val_neg_fixed[val_neg_cursor:]
                            part2 = val_neg_fixed[:take2]
                            val_neg_batch = np.concatenate([part1, part2], axis=0)
                            val_neg_cursor = take2 % len(val_neg_fixed)
                        val_neg_batch = val_neg_batch.tolist()

                    val_pos_g, val_pos_i, val_pos_m = parse_minibatch_LastFM(
                        adjlists,
                        edge_metapath_indices_list,
                        val_pos_batch,
                        device,
                        neighbor_samples,
                        NO_MASKS_SKIP,
                        num_paper,
                    )
                    val_neg_g, val_neg_i, val_neg_m = parse_minibatch_LastFM(
                        adjlists,
                        edge_metapath_indices_list,
                        val_neg_batch,
                        device,
                        neighbor_samples,
                        NO_MASKS_SKIP,
                        num_paper,
                    )
                    [pos_L, pos_R], _ = net((val_pos_g, features_list, type_mask, val_pos_i, val_pos_m))
                    [neg_L, neg_R], _ = net((val_neg_g, features_list, type_mask, val_neg_i, val_neg_m))

                    pos_L = pos_L.view(-1, 1, pos_L.shape[1])
                    pos_R = pos_R.view(-1, pos_R.shape[1], 1)
                    neg_L = neg_L.view(-1, 1, neg_L.shape[1])
                    neg_R = neg_R.view(-1, neg_R.shape[1], 1)

                    pos_out = torch.bmm(pos_L, pos_R)
                    neg_out = -torch.bmm(neg_L, neg_R)
                    vloss = -(F.logsigmoid(pos_out).mean() + F.logsigmoid(neg_out).mean())
                    val_loss_vals.append(float(vloss.item()))
                    vbar.set_postfix(loss=f"{np.mean(val_loss_vals):.4f}")

            val_loss = float(np.mean(val_loss_vals)) if val_loss_vals else 0.0
            print(f"Epoch {epoch:03d} | Val_Loss {val_loss:.4f} | Time(s) {time.time() - epoch_t0:.2f}")
            early_stopping(torch.tensor(val_loss), net)
            if early_stopping.early_stop:
                print("Early stopping!")
                epochs_ran = epoch + 1
                break

        train_wall = float(time.perf_counter() - rep_t0)
        train_wall_list.append(train_wall)
        epochs_ran_list.append(float(epochs_ran))
        print(f"→ Training finished: train_wall_sec={train_wall:.2f} epochs_ran={epochs_ran}")

        # Test
        print("→ Testing best checkpoint ...")
        net.load_state_dict(torch.load(ckpt_path, map_location=device))
        net.eval()

        pos_prob, neg_prob = [], []
        test_pos_idx_gen = index_generator(batch_size=batch_size, num_data=len(test_pos), shuffle=False)
        tbar = tqdm(range(test_pos_idx_gen.num_iterations()), desc="[test:pos] ", leave=False)
        with torch.no_grad():
            for _ in tbar:
                idx = test_pos_idx_gen.next()
                batch_pos = test_pos[idx].tolist()
                g_pos, i_pos, m_pos = parse_minibatch_LastFM(
                    adjlists,
                    edge_metapath_indices_list,
                    batch_pos,
                    device,
                    neighbor_samples,
                    NO_MASKS_SKIP,
                    num_paper,
                )
                [L, R], _ = net((g_pos, features_list, type_mask, i_pos, m_pos))
                L = L.view(-1, 1, L.shape[1])
                R = R.view(-1, R.shape[1], 1)
                out = torch.bmm(L, R).flatten()
                pos_prob.append(torch.sigmoid(out))

        test_neg_idx_gen = index_generator(batch_size=batch_size, num_data=len(test_neg), shuffle=False)
        tbar = tqdm(range(test_neg_idx_gen.num_iterations()), desc="[test:neg] ", leave=True)
        with torch.no_grad():
            for _ in tbar:
                idx = test_neg_idx_gen.next()
                batch_neg = test_neg[idx].tolist()
                g_neg, i_neg, m_neg = parse_minibatch_LastFM(
                    adjlists,
                    edge_metapath_indices_list,
                    batch_neg,
                    device,
                    neighbor_samples,
                    NO_MASKS_SKIP,
                    num_paper,
                )
                [L, R], _ = net((g_neg, features_list, type_mask, i_neg, m_neg))
                L = L.view(-1, 1, L.shape[1])
                R = R.view(-1, R.shape[1], 1)
                out = torch.bmm(L, R).flatten()
                neg_prob.append(torch.sigmoid(out))

        y_pos = torch.cat(pos_prob).cpu().numpy()
        y_neg = torch.cat(neg_prob).cpu().numpy()
        pairs_pos = test_pos
        pairs_neg = test_neg
        pairs_all = np.vstack((pairs_pos, pairs_neg))
        scores_all = np.concatenate((y_pos, y_neg))

        pd.DataFrame(
            {
                "paper_id": pairs_all[:, 0].astype(int),
                "conf_id": pairs_all[:, 1].astype(int),
                "score": scores_all,
                "label": np.concatenate([np.ones(len(pairs_pos), dtype=np.int64), np.zeros(len(pairs_neg), dtype=np.int64)]),
            }
        ).to_csv(f"{save_postfix}_scores.csv", index=False)

        y_proba_test = np.concatenate([y_pos, y_neg])
        y_true_test = np.array([1] * len(pairs_pos) + [0] * len(pairs_neg))
        auc = float(roc_auc_score(y_true_test, y_proba_test))
        ap = float(average_precision_score(y_true_test, y_proba_test))

        # simple ranking metrics (same as other runners)
        from collections import defaultdict

        cand = defaultdict(list)
        for (p, c), s in zip(pairs_pos, y_pos):
            cand[int(p)].append((float(s), int(c), 1))
        for (p, c), s in zip(pairs_neg, y_neg):
            cand[int(p)].append((float(s), int(c), 0))

        hits1 = hits3 = hits5 = 0
        rr_sum = 0.0
        num_papers = 0
        for _pid, items in cand.items():
            if not items:
                continue
            items.sort(key=lambda x: x[0], reverse=True)
            ranks = [i + 1 for i, (_, _, t) in enumerate(items) if t == 1]
            if not ranks:
                continue
            r = min(ranks)
            num_papers += 1
            rr_sum += 1.0 / r
            hits1 += int(r <= 1)
            hits3 += int(r <= 3)
            hits5 += int(r <= 5)
        h1 = hits1 / max(num_papers, 1)
        h3 = hits3 / max(num_papers, 1)
        h5 = hits5 / max(num_papers, 1)
        mrr = rr_sum / max(num_papers, 1)

        th = float(threshold)
        y_pred = (y_proba_test >= th).astype(int)
        TP = int(((y_pred == 1) & (y_true_test == 1)).sum())
        TN = int(((y_pred == 0) & (y_true_test == 0)).sum())
        FP = int(((y_pred == 1) & (y_true_test == 0)).sum())
        FN = int(((y_pred == 0) & (y_true_test == 1)).sum())
        prec = TP / max(TP + FP, 1)
        rec = TP / max(TP + FN, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        acc = (TP + TN) / max(TP + TN + FP + FN, 1)

        print(f"Test: Hits@1={h1:.6f} Hits@3={h3:.6f} Hits@5={h5:.6f} MRR={mrr:.6f} | AUC={auc:.6f} AP={ap:.6f}")
        auc_list.append(auc)
        ap_list.append(ap)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)
        acc_list.append(acc)
        hits1_list.append(h1)
        hits3_list.append(h3)
        hits5_list.append(h5)
        mrr_list.append(mrr)

    return {
        "auc_mean": float(np.mean(auc_list)),
        "auc_std": float(np.std(auc_list, ddof=0)),
        "ap_mean": float(np.mean(ap_list)),
        "ap_std": float(np.std(ap_list, ddof=0)),
        "precision_mean": float(np.mean(prec_list)),
        "precision_std": float(np.std(prec_list, ddof=0)),
        "recall_mean": float(np.mean(rec_list)),
        "recall_std": float(np.std(rec_list, ddof=0)),
        "f1_mean": float(np.mean(f1_list)),
        "f1_std": float(np.std(f1_list, ddof=0)),
        "accuracy_mean": float(np.mean(acc_list)),
        "accuracy_std": float(np.std(acc_list, ddof=0)),
        "hits1_mean": float(np.mean(hits1_list)),
        "hits1_std": float(np.std(hits1_list, ddof=0)),
        "hits3_mean": float(np.mean(hits3_list)),
        "hits3_std": float(np.std(hits3_list, ddof=0)),
        "hits5_mean": float(np.mean(hits5_list)),
        "hits5_std": float(np.std(hits5_list, ddof=0)),
        "mrr_mean": float(np.mean(mrr_list)),
        "mrr_std": float(np.std(mrr_list, ddof=0)),
        "train_wall_sec_mean": float(np.mean(train_wall_list)) if train_wall_list else 0.0,
        "train_wall_sec_std": float(np.std(train_wall_list, ddof=0)) if train_wall_list else 0.0,
        "epochs_ran_mean": float(np.mean(epochs_ran_list)) if epochs_ran_list else 0.0,
        "epochs_ran_std": float(np.std(epochs_ran_list, ddof=0)) if epochs_ran_list else 0.0,
    }

SKIP_BASE = {
    'v1': 'data/preprocessed/DBLP_lp_pc_skip_full_v1/',
    'v2': 'data/preprocessed/DBLP_lp_pc_skip_full_v2/',
    'v3': 'data/preprocessed/DBLP_lp_pc_skip_full_v3/',
}


def _parse_variants(s):
    out = [x.strip().lower() for x in s.split(',') if x.strip()]
    for v in out:
        if v not in SKIP_BASE:
            raise SystemExit('Unknown variant {!r}; expected one of {}'.format(v, ','.join(SKIP_BASE)))
    return out


def _metrics_from_stats(st):
    """One training run (repeat=1): *_mean fields are the scalar test metrics."""
    return {
        'AUC': float(st['auc_mean']),
        'AP': float(st['ap_mean']),
        'Precision': float(st['precision_mean']),
        'Recall': float(st['recall_mean']),
        'F1': float(st['f1_mean']),
        'Accuracy': float(st['accuracy_mean']),
        'Hits@1': float(st['hits1_mean']),
        'Hits@3': float(st['hits3_mean']),
        'Hits@5': float(st['hits5_mean']),
        'MRR': float(st['mrr_mean']),
        'Train Time (s)': float(st.get('train_wall_sec_mean', 0.0)),
        'Epochs': float(st.get('epochs_ran_mean', 0.0)),
    }


def summarize_dblp_lp_over_seeds(metrics_runs):
    keys = [
        'AUC', 'AP', 'Precision', 'Recall', 'F1', 'Accuracy',
        'Hits@1', 'Hits@3', 'Hits@5', 'MRR', 'Train Time (s)', 'Epochs',
    ]
    print('\n===== Summary over seeds (mean ± std) =====')
    for k in keys:
        arr = np.array([r[k] for r in metrics_runs], dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue
        if k in ('Train Time (s)', 'Epochs'):
            print('{:<22}: {:.2f} ± {:.2f}'.format(k, float(arr.mean()), float(arr.std(ddof=0))))
        else:
            print('{:<22}: {:.6f} ± {:.6f}'.format(k, float(arr.mean()), float(arr.std(ddof=0))))


def kendall_lp_scores_csv(path_a, path_b):
    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)
    m = a.merge(b, on=['paper_id', 'conf_id'], suffixes=('_a', '_b'), how='inner')
    if len(m) < 2:
        return float('nan'), len(m)
    tau, _ = kendalltau(m['score_a'], m['score_b'], nan_policy='omit')
    return float(tau), len(m)


def _load_test_pos_set_for_variant(variant):
    base = SKIP_BASE[variant]
    pos_path = os.path.join(base, 'train_val_test_pos_paper_conf.npz')
    if not os.path.isfile(pos_path):
        return set()
    z = np.load(pos_path)
    return set((int(p), int(c)) for p, c in z['test_pos'])


def _hits_by_paper(df, pos_set=None):
    d = df.copy()
    if 'label' not in d.columns:
        if pos_set is None:
            return pd.DataFrame(columns=['paper_id', 'hit1', 'hit3'])
        d['label'] = [
            1 if (int(p), int(c)) in pos_set else 0
            for p, c in zip(d['paper_id'].astype(int), d['conf_id'].astype(int))
        ]
    out = []
    for pid, grp in d.groupby('paper_id', sort=True):
        g = grp.sort_values('score', ascending=False).reset_index(drop=True)
        pos_idx = g.index[g['label'] == 1].tolist()
        if not pos_idx:
            continue
        rank = int(min(pos_idx)) + 1
        out.append({'paper_id': int(pid), 'hit1': int(rank <= 1), 'hit3': int(rank <= 3)})
    return pd.DataFrame(out)


def kendall_lp_scores_and_hits(path_a, path_b, pos_set_a=None, pos_set_b=None):
    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)
    m = a.merge(b, on=['paper_id', 'conf_id'], suffixes=('_a', '_b'), how='inner')
    out = {'overall_tau': float('nan'), 'overall_n': len(m), 'h1_tau': float('nan'), 'h3_tau': float('nan'), 'hits_n': 0}
    if len(m) >= 2:
        tau, _ = kendalltau(m['score_a'], m['score_b'], nan_policy='omit')
        out['overall_tau'] = float(tau) if np.isfinite(tau) else float('nan')
    ha = _hits_by_paper(a, pos_set=pos_set_a).rename(columns={'hit1': 'hit1_a', 'hit3': 'hit3_a'})
    hb = _hits_by_paper(b, pos_set=pos_set_b).rename(columns={'hit1': 'hit1_b', 'hit3': 'hit3_b'})
    hh = ha.merge(hb, on='paper_id', how='inner')
    out['hits_n'] = len(hh)
    if len(hh) >= 2:
        t1, _ = kendalltau(hh['hit1_a'], hh['hit1_b'], nan_policy='omit')
        t3, _ = kendalltau(hh['hit3_a'], hh['hit3_b'], nan_policy='omit')
        out['h1_tau'] = float(t1) if np.isfinite(t1) else float('nan')
        out['h3_tau'] = float(t3) if np.isfinite(t3) else float('nan')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MAGNN DBLP paper–venue (skip preprocess v1–v3)')
    ap.add_argument('--feats-type', type=int, default=0)
    ap.add_argument('--hidden-dim', type=int, default=64)
    ap.add_argument('--num-heads', type=int, default=8)
    ap.add_argument('--attn-vec-dim', type=int, default=128)
    ap.add_argument('--rnn-type', default='RotatE0')
    ap.add_argument('--epoch', type=int, default=100)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--samples', type=int, default=100)
    ap.add_argument('--save-postfix', default='DBLP_skip')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--K', type=int, default=3)
    ap.add_argument('--neg-mult', type=int, default=3)
    ap.add_argument('--test-neg-per-paper', type=int, default=3)
    ap.add_argument('--seeds', default='1566911444,20241017,20251017')
    ap.add_argument('--variants', default='v1,v2,v3',
                    help='Comma-separated subset of v1,v2,v3.')
    ap.add_argument('--compare', default='',
                    help='Optional pair for Kendall tau on test LP scores CSV, e.g. v1,v2.')
    ap.add_argument('--compare-only', action='store_true',
                    help='Skip training; only Kendall tau on existing *_scores.csv files.')
    ap.add_argument('--score-csv-a', default='',
                    help='With --compare-only: first scores CSV (one pair; ignores variants).')
    ap.add_argument('--score-csv-b', default='',
                    help='With --compare-only: second scores CSV.')
    ap.add_argument(
        '--debug-metapaths',
        action='store_true',
        help='After loading, print for each metapath the non-empty target count and total path instances.',
    )
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]

    if args.compare_only:
        if args.score_csv_a and args.score_csv_b:
            kk = kendall_lp_scores_and_hits(args.score_csv_a, args.score_csv_b)
            print('Kendall tau overall: {:.6f} (n={})'.format(kk['overall_tau'], kk['overall_n']))
            print('Kendall tau Hits@1: {:.6f} (hits_n={})'.format(kk['h1_tau'], kk['hits_n']))
            print('Kendall tau Hits@3: {:.6f} (hits_n={})'.format(kk['h3_tau'], kk['hits_n']))
            print('  A: {}'.format(args.score_csv_a))
            print('  B: {}'.format(args.score_csv_b))
            sys.exit(0)
        vars_for_kendall = _parse_variants(args.compare) if args.compare.strip() else _parse_variants(args.variants)
        if len(vars_for_kendall) < 2:
            sys.exit('--compare-only needs at least two variants via --compare or --variants.')
        print('########## Comparison Kendall-Tau (mean ± std over seeds) ##########')
        for va, vb in itertools.combinations(vars_for_kendall, 2):
            taus, h1_taus, h3_taus = [], [], []
            pos_a = _load_test_pos_set_for_variant(va)
            pos_b = _load_test_pos_set_for_variant(vb)
            for seed in seeds:
                pa = '{}_{}_seed{}_scores.csv'.format(args.save_postfix, va, seed)
                pb = '{}_{}_seed{}_scores.csv'.format(args.save_postfix, vb, seed)
                if not os.path.isfile(pa) or not os.path.isfile(pb):
                    sys.exit('Missing CSV: {!r} or {!r}'.format(pa, pb))
                kk = kendall_lp_scores_and_hits(pa, pb, pos_set_a=pos_a, pos_set_b=pos_b)
                print(
                    '{} vs {} | seed {} | n_pairs={} | tau={:.6f} | hits_n={} | h@1_tau={:.6f} | h@3_tau={:.6f}'.format(
                        va, vb, seed, kk['overall_n'], kk['overall_tau'], kk['hits_n'], kk['h1_tau'], kk['h3_tau']
                    )
                )
                if np.isfinite(kk['overall_tau']):
                    taus.append(float(kk['overall_tau']))
                if np.isfinite(kk['h1_tau']):
                    h1_taus.append(float(kk['h1_tau']))
                if np.isfinite(kk['h3_tau']):
                    h3_taus.append(float(kk['h3_tau']))
            arr = np.array(taus, dtype=float)
            if len(arr):
                msg = '{} vs {} | overall {:.6f} ± {:.6f}'.format(va, vb, float(arr.mean()), float(arr.std(ddof=0)))
                if h1_taus:
                    a1 = np.array(h1_taus, dtype=float)
                    msg += ' | H@1 {:.6f} ± {:.6f}'.format(float(a1.mean()), float(a1.std(ddof=0)))
                if h3_taus:
                    a3 = np.array(h3_taus, dtype=float)
                    msg += ' | H@3 {:.6f} ± {:.6f}'.format(float(a3.mean()), float(a3.std(ddof=0)))
                print(msg)
            else:
                print('{} vs {} | mean ± std = N/A'.format(va, vb))
        sys.exit(0)

    variants = _parse_variants(args.variants)
    os.makedirs('checkpoint', exist_ok=True)
    if args.neg_mult != 3:
        print('WARNING: overriding --neg-mult to 3 for skip runs (requested setting).')
    fixed_neg_mult = 3

    score_paths = {}
    by_variant_metrics = {}

    for ver in variants:
        print('\n########## DBLP skip variant {} | {} seeds ##########'.format(ver, len(seeds)))
        score_paths[ver] = []
        base = SKIP_BASE[ver]
        metrics_runs = []
        for seed in seeds:
            sp = '{}_{}_seed{}'.format(args.save_postfix, ver, seed)
            print('----------------------------------------------------------------')
            print('Variant {} | seed {} | data {}'.format(ver, seed, base))
            print('----------------------------------------------------------------')
            set_seed(seed)
            stats = run_model_DBLP_skip(
                feats_type=args.feats_type,
                hidden_dim=args.hidden_dim,
                num_heads=args.num_heads,
                attn_vec_dim=args.attn_vec_dim,
                rnn_type=args.rnn_type,
                num_epochs=args.epoch,
                patience=args.patience,
                batch_size=args.batch_size,
                neighbor_samples=args.samples,
                repeat=1,
                save_postfix=sp,
                threshold=args.threshold,
                K=args.K,
                neg_mult=fixed_neg_mult,
                base_seed=seed,
                preprocessed_base=base,
                test_neg_per_paper=args.test_neg_per_paper,
                debug_metapaths=args.debug_metapaths,
            )
            metrics_runs.append(_metrics_from_stats(stats))
            score_paths[ver].append('{}_scores.csv'.format(sp))
        by_variant_metrics[ver] = metrics_runs
        summarize_dblp_lp_over_seeds(metrics_runs)

    print('\nDBLP Link Prediction Summary Over Seeds (mean ± std)')
    print('Variant | Precision | Recall | F1 | Hits@1 | Hits@3 | MRR | Train Time (s) | Epochs')
    for ver in variants:
        mets = by_variant_metrics[ver]
        def mstd(key):
            arr = np.array([m[key] for m in mets], dtype=float)
            return float(arr.mean()), float(arr.std(ddof=0))
        p_m, p_s = mstd('Precision')
        r_m, r_s = mstd('Recall')
        f_m, f_s = mstd('F1')
        h1_m, h1_s = mstd('Hits@1')
        h3_m, h3_s = mstd('Hits@3')
        mrr_m, mrr_s = mstd('MRR')
        tw_m, tw_s = mstd('Train Time (s)')
        ep_m, ep_s = mstd('Epochs')
        print(
            '{} | {:.4f} ± {:.4f} | {:.4f} ± {:.4f} | {:.4f} ± {:.4f} | '
            '{:.4f} ± {:.4f} | {:.4f} ± {:.4f} | {:.4f} ± {:.4f} | {:.2f} ± {:.2f} | {:.2f} ± {:.2f}'.format(
                ver, p_m, p_s, r_m, r_s, f_m, f_s, h1_m, h1_s, h3_m, h3_s, mrr_m, mrr_s, tw_m, tw_s, ep_m, ep_s
            )
        )

    if args.compare.strip():
        pair = _parse_variants(args.compare)
        if len(pair) != 2:
            raise SystemExit('--compare expects exactly two variants, e.g. v1,v2')
        va, vb = pair[0], pair[1]
        if va not in score_paths or vb not in score_paths:
            raise SystemExit('--compare requires both variants in --variants')

        print('\n########## Kendall tau (test LP scores) | {} vs {} ##########'.format(va, vb))
        pos_a = _load_test_pos_set_for_variant(va)
        pos_b = _load_test_pos_set_for_variant(vb)
        taus, h1_taus, h3_taus = [], [], []
        for i, seed in enumerate(seeds):
            pa, pb = score_paths[va][i], score_paths[vb][i]
            kk = kendall_lp_scores_and_hits(pa, pb, pos_set_a=pos_a, pos_set_b=pos_b)
            print(
                'seed {} | n_pairs={} | tau={:.6f} | hits_n={} | h@1_tau={:.6f} | h@3_tau={:.6f}'.format(
                    seed, kk['overall_n'], kk['overall_tau'], kk['hits_n'], kk['h1_tau'], kk['h3_tau']
                )
            )
            if np.isfinite(kk['overall_tau']):
                taus.append(float(kk['overall_tau']))
            if np.isfinite(kk['h1_tau']):
                h1_taus.append(float(kk['h1_tau']))
            if np.isfinite(kk['h3_tau']):
                h3_taus.append(float(kk['h3_tau']))
        if taus:
            at = np.array(taus, dtype=float)
            print('Mean Kendall tau overall: {:.6f} (std {:.6f})'.format(float(at.mean()), float(at.std(ddof=0))))
        if h1_taus:
            a1 = np.array(h1_taus, dtype=float)
            print('Mean Kendall tau Hits@1: {:.6f} (std {:.6f})'.format(float(a1.mean()), float(a1.std(ddof=0))))
        if h3_taus:
            a3 = np.array(h3_taus, dtype=float)
            print('Mean Kendall tau Hits@3: {:.6f} (std {:.6f})'.format(float(a3.mean()), float(a3.std(ddof=0))))
