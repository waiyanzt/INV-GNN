#!/usr/bin/env python3
"""
MAGNN link prediction on DBLP paper–venue using **train-only P–C** preprocess
(``preprocess_DBLP_pc_trainpc.py`` → ``DBLP_lp_pc_var{k}_train_pc/``).

Covers var1 (``run_DBLP_pc.py``), var2 (``run_DBLP_pc_t.py``), var3 (``run_DBLP_pc_a.py``)
metapaths and hyperparameters unchanged from those scripts. 

Examples (from ``MAGNN/``):
  python preprocess_DBLP_pc_trainpc.py --variants v1,v2,v3
  python run_DBLP_pc_trainpc.py --variants v1,v2,v3 --seeds 1566911444,20241017
  python run_DBLP_pc_trainpc.py --kendall-only --variants v1,v2,v3 --save-postfix DBLP_pc_trainpc 
"""
from __future__ import annotations

import argparse
import itertools
import os
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from model import MAGNN_lp
from utils.data import (
    load_DBLP_lp_pc_var1_data,
    load_DBLP_lp_pc_var2_data,
    load_DBLP_lp_pc_var3_data,
)
from utils.dblp_lp_eval import subsample_test_neg_per_paper
from utils.pytorchtools import EarlyStopping
from utils.tools import index_generator, parse_minibatch_LastFM

# Shared training hyperparameters (match run_DBLP_pc*.py)
num_ntype = 5
dropout_rate = 0.5
lr = 0.005
weight_decay = 0.001

PRE_BASE = {
    "v1": "data/preprocessed/DBLP_lp_pc_var1_train_pc/",
    "v2": "data/preprocessed/DBLP_lp_pc_var2_train_pc/",
    "v3": "data/preprocessed/DBLP_lp_pc_var3_train_pc/",
}

MAGNN_CONFIG: Dict[str, Dict[str, Any]] = {
    "v1": {
        "loader": load_DBLP_lp_pc_var1_data,
        "num_etypes": 8,
        "expected_metapaths": [
            [(1, 0, 1), (1, 2, 1), (1, 4, 1), (1, 3, 1)],
            [(3, 1, 0, 1, 3), (3, 1, 3)],
        ],
        "etypes_lists": [
            [[1, 0], [2, 3], [4, 5], [6, 7]],
            [[5, 1, 0, 4], [5, 4]],
        ],
        "use_masks": [[True] * 4, [True] * 2],
        "no_masks": [[False] * 4, [False] * 2],
    },
    "v2": {
        "loader": load_DBLP_lp_pc_var2_data,
        "num_etypes": 8,
        "expected_metapaths": [
            [(1, 0, 1), (1, 2, 1), (1, 3, 1), (1, 3, 4, 3, 1)],
            [(3, 1, 0, 1, 3), (3, 1, 3)],
        ],
        "etypes_lists": [
            [[1, 0], [2, 3], [4, 5], [4, 6, 7, 5]],
            [[5, 1, 0, 4], [5, 4]],
        ],
        "use_masks": [[True] * 4, [True] * 2],
        "no_masks": [[False] * 4, [False] * 2],
    },
    "v3": {
        "loader": load_DBLP_lp_pc_var3_data,
        "num_etypes": 8,
        "expected_metapaths": [
            [(1, 0, 1), (1, 2, 1), (1, 0, 4, 0, 1), (1, 3, 1)],
            [(3, 1, 0, 1, 3), (3, 1, 3)],
        ],
        "etypes_lists": [
            [[1, 0], [2, 3], [1, 6, 7, 0], [4, 5]],
            [[5, 1, 0, 4], [5, 4]],
        ],
        "use_masks": [[True] * 4, [True] * 2],
        "no_masks": [[False] * 4, [False] * 2],
    },
}


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


def kendall_two_csvs(path_a: str, path_b: str) -> Tuple[float, int]:
    """Overall Kendall tau on merged (paper_id, conf_id) scores."""
    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)
    m = a.merge(b, on=["paper_id", "conf_id"], suffixes=("_a", "_b"))
    if len(m) < 2:
        return float("nan"), len(m)
    tau, _ = kendalltau(m["score_a"], m["score_b"], nan_policy="omit")
    return (float(tau) if np.isfinite(tau) else float("nan"), len(m))


def _dblr_test_pos_set(variant: str) -> set[tuple[int, int]]:
    base = PRE_BASE[variant]
    pos_path = os.path.join(base, "train_val_test_pos_paper_conf.npz")
    if not os.path.isfile(pos_path):
        return set()
    z = np.load(pos_path)
    return set((int(p), int(c)) for p, c in z["test_pos"])


def _hits_by_paper(df: pd.DataFrame, pos_set: set[tuple[int, int]] | None = None) -> pd.DataFrame:
    d = df.copy()
    if "label" not in d.columns:
        if pos_set is None:
            return pd.DataFrame(columns=["paper_id", "hit1", "hit3"])
        d["label"] = [
            1 if (int(p), int(c)) in pos_set else 0
            for p, c in zip(d["paper_id"].astype(int), d["conf_id"].astype(int))
        ]
    out = []
    for pid, grp in d.groupby("paper_id", sort=True):
        g = grp.sort_values("score", ascending=False).reset_index(drop=True)
        pos_idx = g.index[g["label"] == 1].tolist()
        if not pos_idx:
            continue
        rank = int(min(pos_idx)) + 1
        out.append({"paper_id": int(pid), "hit1": int(rank <= 1), "hit3": int(rank <= 3)})
    return pd.DataFrame(out)


def kendall_with_hits_csvs(
    path_a: str, path_b: str, pos_set_a: set[tuple[int, int]] | None = None, pos_set_b: set[tuple[int, int]] | None = None
) -> Dict[str, float]:
    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)
    m = a.merge(b, on=["paper_id", "conf_id"], suffixes=("_a", "_b"))
    out = {"overall_tau": float("nan"), "overall_n": float(len(m)), "h1_tau": float("nan"), "h3_tau": float("nan"), "hits_n": 0.0}
    if len(m) >= 2:
        tau, _ = kendalltau(m["score_a"], m["score_b"], nan_policy="omit")
        out["overall_tau"] = float(tau) if np.isfinite(tau) else float("nan")

    ha = _hits_by_paper(a, pos_set=pos_set_a).rename(columns={"hit1": "hit1_a", "hit3": "hit3_a"})
    hb = _hits_by_paper(b, pos_set=pos_set_b).rename(columns={"hit1": "hit1_b", "hit3": "hit3_b"})
    hh = ha.merge(hb, on="paper_id", how="inner")
    out["hits_n"] = float(len(hh))
    if len(hh) >= 2:
        t1, _ = kendalltau(hh["hit1_a"], hh["hit1_b"], nan_policy="omit")
        t3, _ = kendalltau(hh["hit3_a"], hh["hit3_b"], nan_policy="omit")
        out["h1_tau"] = float(t1) if np.isfinite(t1) else float("nan")
        out["h3_tau"] = float(t3) if np.isfinite(t3) else float("nan")
    return out


def run_one_variant(
    variant: str,
    seed: int,
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
    preprocessed_base: str | None,
    test_neg_per_paper: int = 3,
) -> Dict[str, Any]:
    cfg = MAGNN_CONFIG[variant]
    loader: Callable = cfg["loader"]
    expected_metapaths = cfg["expected_metapaths"]
    etypes_lists = cfg["etypes_lists"]
    use_masks = cfg["use_masks"]
    no_masks = cfg["no_masks"]
    num_etypes = cfg["num_etypes"]

    _base = preprocessed_base or PRE_BASE[variant]
    print(f"→ Loading preprocessed DBLP paper–venue data ({variant}) from {_base!r} ...")
    t0 = time.time()
    (adjlists, edge_metapath_indices_list, _, type_mask, pos_splits, neg_splits, num_paper, num_conf) = loader(
        expected_metapaths, base=_base
    )
    print(f"   loaded in {time.time()-t0:.2f}s")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"→ Device: {device}")

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
            dim = 10
            in_dims.append(dim)
            features_list.append(torch.zeros(((type_mask == i).sum(), 10)).to(device))

    train_pos = pos_splits["train_pos_paper_conf"]
    val_pos = pos_splits["val_pos_paper_conf"]
    test_pos = pos_splits["test_pos_paper_conf"]
    train_neg = neg_splits["train_neg_paper_conf"]
    val_neg = neg_splits["val_neg_paper_conf"]
    test_neg = neg_splits["test_neg_paper_conf"]
    if test_neg_per_paper and test_neg_per_paper > 0:
        n_full = len(test_neg)
        test_neg = subsample_test_neg_per_paper(test_neg, test_neg_per_paper, seed)
        print(
            f"→ Test negatives subsampled for ranking/AUC: {n_full} -> {len(test_neg)} pairs "
            f"(max {test_neg_per_paper} negatives per paper; train/val unchanged)"
        )

    print(f"→ Targets: papers={num_paper}, confs={num_conf}")
    print(f"→ Splits: train_pos={len(train_pos)}, val_pos={len(val_pos)}, test_pos={len(test_pos)}")
    print(f"           train_neg={len(train_neg)}, val_neg={len(val_neg)}, test_neg(eval)={len(test_neg)}")
    print(f"→ Metapaths per mode: {[len(m) for m in expected_metapaths]} "
          f"(etypes_lists: {[len(e) for e in etypes_lists]})")
    print(f"→ GNN layers (K): {K}")

    rng = np.random.default_rng(seed)

    val_neg_fixed = val_neg.copy()
    if len(val_neg_fixed) > 0:
        val_neg_fixed = val_neg_fixed[rng.permutation(len(val_neg_fixed))]

    auc_list, ap_list = [], []
    prec_list, rec_list, f1_list, acc_list = [], [], [], []
    hits1_list, hits3_list, hits5_list, mrr_list = [], [], [], []
    train_wall_list: List[float] = []
    epochs_ran_list: List[int] = []

    os.makedirs("checkpoint", exist_ok=True)

    for rep in range(repeat):
        print(f"\n===== Variant={variant} Seed={seed} Run {rep+1}/{repeat} =====")
        tag_base = f"{save_postfix}_{variant}_seed{seed}"
        if repeat > 1:
            tag_base = f"{tag_base}_rep{rep}"
        val_neg_cursor = 0
        net = MAGNN_lp(
            [4, 2],
            num_etypes,
            etypes_lists,
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

        ckpt_path = f"checkpoint/checkpoint_{tag_base}_K{K}.pt"
        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path=ckpt_path)
        train_pos_idx_generator = index_generator(batch_size=batch_size, num_data=len(train_pos))
        val_idx_generator = index_generator(batch_size=batch_size, num_data=len(val_pos), shuffle=False)

        train_t0 = time.perf_counter()
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
                    use_masks,
                    num_paper,
                )
                neg_g_lists, neg_indices_lists, neg_idx_batch_mapped_lists = parse_minibatch_LastFM(
                    adjlists,
                    edge_metapath_indices_list,
                    train_neg_batch,
                    device,
                    neighbor_samples,
                    no_masks,
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

                running_loss += train_loss.item()
                if (it + 1) % max(1, iters // 5) == 0:
                    pbar.set_postfix(loss=f"{running_loss / (it+1):.4f}")

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
                    val_neg_batch = val_neg_batch.tolist() if len(val_neg_fixed) > 0 else []

                    val_pos_g, val_pos_i, val_pos_m = parse_minibatch_LastFM(
                        adjlists,
                        edge_metapath_indices_list,
                        val_pos_batch,
                        device,
                        neighbor_samples,
                        no_masks,
                        num_paper,
                    )
                    val_neg_g, val_neg_i, val_neg_m = parse_minibatch_LastFM(
                        adjlists,
                        edge_metapath_indices_list,
                        val_neg_batch,
                        device,
                        neighbor_samples,
                        no_masks,
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
                    val_loss_vals.append(vloss.item())
                    vbar.set_postfix(loss=f"{np.mean(val_loss_vals):.4f}")

            val_loss = float(np.mean(val_loss_vals)) if val_loss_vals else 0.0
            wall_epoch = time.perf_counter() - train_t0
            print(
                f"Epoch {epoch:03d} | Val_Loss {val_loss:.4f} | "
                f"EpochWall(s) {time.time()-epoch_t0:.2f} | CumTrainWall(s) {wall_epoch:.2f}"
            )
            early_stopping(torch.tensor(val_loss), net)
            if early_stopping.early_stop:
                print("Early stopping!")
                epochs_ran = epoch + 1
                break

        train_wall_list.append(time.perf_counter() - train_t0)
        epochs_ran_list.append(epochs_ran)
        print(
            f"→ Training finished: total_train_wall_sec={train_wall_list[-1]:.2f} "
            f"epochs_ran={epochs_ran} (max_epochs={num_epochs})"
        )

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
                    no_masks,
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
                    no_masks,
                    num_paper,
                )
                [L, R], _ = net((g_neg, features_list, type_mask, i_neg, m_neg))
                L = L.view(-1, 1, L.shape[1])
                R = R.view(-1, R.shape[1], 1)
                out = torch.bmm(L, R).flatten()
                neg_prob.append(torch.sigmoid(out))

        y_proba_test = torch.cat(pos_prob + neg_prob).cpu().numpy()
        y_true_test = np.array([1] * len(test_pos) + [0] * len(test_neg))
        auc = roc_auc_score(y_true_test, y_proba_test)
        ap = average_precision_score(y_true_test, y_proba_test)

        y_pos = torch.cat(pos_prob).cpu().numpy()
        y_neg = torch.cat(neg_prob).cpu().numpy()
        pairs_pos = test_pos
        pairs_neg = test_neg

        pairs_all = np.vstack((pairs_pos, pairs_neg))
        scores_all = np.concatenate((y_pos, y_neg))
        csv_name = f"{tag_base}_scores.csv"
        pd.DataFrame(
            {
                "paper_id": pairs_all[:, 0].astype(int),
                "conf_id": pairs_all[:, 1].astype(int),
                "score": scores_all,
                "label": np.concatenate(
                    [np.ones(len(pairs_pos), dtype=np.int64), np.zeros(len(pairs_neg), dtype=np.int64)]
                ),
            }
        ).to_csv(csv_name, index=False)
        print(f"→ Saved scores for Kendall τ: {csv_name}")

        cand = defaultdict(list)
        for (p, c), s in zip(pairs_pos, y_pos):
            cand[int(p)].append((float(s), int(c), 1))
        for (p, c), s in zip(pairs_neg, y_neg):
            cand[int(p)].append((float(s), int(c), 0))

        hits1 = hits3 = hits5 = 0
        rr_sum = 0.0
        num_papers = 0
        for _, items in cand.items():
            if not items:
                continue
            items.sort(key=lambda x: x[0], reverse=True)
            ranks = [i + 1 for i, (_, _, t) in enumerate(items) if t == 1]
            if not ranks:
                continue
            r = min(ranks)
            num_papers += 1
            rr_sum += 1.0 / r
            if r <= 1:
                hits1 += 1
            if r <= 3:
                hits3 += 1
            if r <= 5:
                hits5 += 1

        print("Argmax (ranking) on test:")
        # Hits@k: best rank r among sorted-by-score pairs per paper; Hits@k iff r<=k (see utils/dblp_lp_eval.py).
        _h1 = hits1 / num_papers
        _h3 = hits3 / num_papers
        _h5 = hits5 / num_papers
        _mrr = rr_sum / num_papers
        if test_neg_per_paper and test_neg_per_paper <= 4:
            print(
                f"Hits@1 = {_h1:.6f}  Hits@3 = {_h3:.6f}  "
                f"MRR = {_mrr:.6f}  (Hits@5 omitted: ≤{1 + test_neg_per_paper} candidates/paper ⇒ Hits@5 is uninformative)"
            )
        else:
            print(f"Hits@1 = {_h1:.6f}  Hits@3 = {_h3:.6f}  Hits@5 = {_h5:.6f}  MRR = {_mrr:.6f}")
        hits1_list.append(_h1)
        hits3_list.append(_h3)
        hits5_list.append(_h5)
        mrr_list.append(_mrr)

        th = threshold
        y_pred_test = (y_proba_test >= th).astype(int)
        TP = np.sum((y_pred_test == 1) & (y_true_test == 1))
        TN = np.sum((y_pred_test == 0) & (y_true_test == 0))
        FP = np.sum((y_pred_test == 1) & (y_true_test == 0))
        FN = np.sum((y_pred_test == 0) & (y_true_test == 1))
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

        print(f"Confusion matrix @ threshold {th:.2f}: TP={TP} TN={TN} FP={FP} FN={FN}")
        print("Link Prediction Test")
        print(f"AUC={auc:.6f} AP={ap:.6f} Precision={prec:.6f} Recall={rec:.6f} F1={f1:.6f} Acc={acc:.6f}")

        auc_list.append(auc)
        ap_list.append(ap)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)
        acc_list.append(acc)

    out = {
        "auc_mean": float(np.mean(auc_list)),
        "auc_std": float(np.std(auc_list)),
        "ap_mean": float(np.mean(ap_list)),
        "ap_std": float(np.std(ap_list)),
        "precision_mean": float(np.mean(prec_list)),
        "precision_std": float(np.std(prec_list)),
        "recall_mean": float(np.mean(rec_list)),
        "recall_std": float(np.std(rec_list)),
        "f1_mean": float(np.mean(f1_list)),
        "f1_std": float(np.std(f1_list)),
        "accuracy_mean": float(np.mean(acc_list)),
        "accuracy_std": float(np.std(acc_list)),
        "hits1_mean": float(np.mean(hits1_list)),
        "hits1_std": float(np.std(hits1_list)),
        "hits3_mean": float(np.mean(hits3_list)),
        "hits3_std": float(np.std(hits3_list)),
        "hits5_mean": float(np.mean(hits5_list)),
        "hits5_std": float(np.std(hits5_list)),
        "mrr_mean": float(np.mean(mrr_list)),
        "mrr_std": float(np.std(mrr_list)),
        "train_wall_sec_mean": float(np.mean(train_wall_list)),
        "train_wall_sec_std": float(np.std(train_wall_list)),
        "epochs_ran_mean": float(np.mean(epochs_ran_list)),
        "epochs_ran_std": float(np.std(epochs_ran_list)),
    }
    return out


def _parse_variants(s: str) -> List[str]:
    out = [x.strip().lower() for x in str(s).split(",") if x.strip()]
    for v in out:
        if v not in MAGNN_CONFIG:
            raise SystemExit(f"Unknown variant {v!r}; expected one of {','.join(MAGNN_CONFIG)}")
    return out


def run_kendall_only(variants: List[str], seeds: List[int], save_postfix: str) -> None:
    pairs = list(itertools.combinations(variants, 2))
    if not pairs:
        raise SystemExit("Need at least two variants for Kendall comparison.")
    for va, vb in pairs:
        print(f"\n=== Kendall τ | {va} vs {vb} (scores merged on paper_id, conf_id) ===")
        taus, h1_taus, h3_taus = [], [], []
        pos_a = _dblr_test_pos_set(va)
        pos_b = _dblr_test_pos_set(vb)
        for seed in seeds:
            pa = f"{save_postfix}_{va}_seed{seed}_scores.csv"
            pb = f"{save_postfix}_{vb}_seed{seed}_scores.csv"
            if not os.path.isfile(pa) or not os.path.isfile(pb):
                print(f"  [skip seed {seed}] missing {pa!r} or {pb!r}")
                continue
            kk = kendall_with_hits_csvs(pa, pb, pos_set_a=pos_a, pos_set_b=pos_b)
            print(
                f"  seed {seed}: n={int(kk['overall_n'])} tau={kk['overall_tau']:.6f} | "
                f"hits_n={int(kk['hits_n'])} h@1_tau={kk['h1_tau']:.6f} h@3_tau={kk['h3_tau']:.6f}"
            )
            taus.append(kk["overall_tau"])
            h1_taus.append(kk["h1_tau"])
            h3_taus.append(kk["h3_tau"])
        arr = np.array(taus, dtype=float)
        finite = arr[np.isfinite(arr)]
        if len(finite):
            print(f"  mean ± std over seeds: {finite.mean():.6f} ± {finite.std(ddof=0):.6f}")
        else:
            print("  (no finite tau values)")
        for nm, vals in [("Hits@1 tau", h1_taus), ("Hits@3 tau", h3_taus)]:
            a2 = np.array(vals, dtype=float)
            f2 = a2[np.isfinite(a2)]
            if len(f2):
                print(f"  {nm} mean ± std over seeds: {f2.mean():.6f} ± {f2.std(ddof=0):.6f}")
            else:
                print(f"  {nm}: (no finite tau values)")


def _fmt_pm(mean: float, std: float, decimals: int = 4) -> str:
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def main():
    ap = argparse.ArgumentParser(description="MAGNN DBLP paper–venue (train-only P–C preprocess), var1–v3.")
    ap.add_argument("--variants", default="v1,v2,v3", help="Comma-separated: v1,v2,v3")
    ap.add_argument("--feats-type", type=int, default=0)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--attn-vec-dim", type=int, default=128)
    ap.add_argument("--rnn-type", default="RotatE0")
    ap.add_argument("--epoch", type=int, default=100)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--save-postfix", default="DBLP_pc_trainpc")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--seeds", default="1566911444,20241017,20251017")
    ap.add_argument("--neg-mult", type=int, default=3)
    ap.add_argument(
        "--test-neg-per-paper",
        type=int,
        default=3,
        help="Max test negatives per paper for ranking/AUC CSV (default 3). Use 0 for full split (slow).",
    )
    ap.add_argument(
        "--preprocessed-base",
        default="",
        help="Override data dir for a single run (default: per-variant train_pc path).",
    )
    ap.add_argument(
        "--kendall-only",
        action="store_true",
        help="Skip training; compare existing score CSVs (pairwise τ between variants).",
    )
    args = ap.parse_args()

    variants = _parse_variants(args.variants)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    if args.kendall_only:
        run_kendall_only(variants, seeds, args.save_postfix)
        return

    pre_override = args.preprocessed_base.strip() or None

    all_rows: List[Tuple[int, str, Dict[str, Any]]] = []
    for v in variants:
        for seed in seeds:
            print(f"\n########## Variant={v} Seed={seed} ##########")
            set_seed(seed)
            stats = run_one_variant(
                v,
                seed,
                args.feats_type,
                args.hidden_dim,
                args.num_heads,
                args.attn_vec_dim,
                args.rnn_type,
                args.epoch,
                args.patience,
                args.batch_size,
                args.samples,
                args.repeat,
                args.save_postfix,
                args.threshold,
                args.K,
                args.neg_mult,
                pre_override if len(variants) == 1 else None,
                args.test_neg_per_paper,
            )
            all_rows.append((seed, v, stats))
            print(
                f"TrainWallSec={stats['train_wall_sec_mean']:.2f} "
                f"EpochsRan={stats['epochs_ran_mean']:.1f} "
                f"| MRR={stats['mrr_mean']:.4f}"
            )

    print("\n Summary (variant × seed)")
    for seed, v, st in all_rows:
        print(
            f"seed={seed} variant={v}: "
            f"AUC={st['auc_mean']:.4f} MRR={st['mrr_mean']:.4f} "
            f"train_s={st['train_wall_sec_mean']:.1f} epochs={st['epochs_ran_mean']:.0f}"
        )

    # Aggregate over seeds per variant (table-friendly mean ± std).
    by_variant: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for _, v, st in all_rows:
        by_variant[v].append(st)

    print("\n DBLP Link Prediction Summary Over Seeds (mean ± std)")
    print(" Variant | Precision | Recall | F1 | Hits@1 | Hits@3 | MRR | Train Time (s) | Epochs")
    for v in variants:
        rows = by_variant.get(v, [])
        if not rows:
            continue

        def mstd(key: str) -> Tuple[float, float]:
            vals = np.array([float(r[key]) for r in rows], dtype=float)
            return float(np.mean(vals)), float(np.std(vals, ddof=0))

        p_m, p_s = mstd("precision_mean")
        r_m, r_s = mstd("recall_mean")
        f_m, f_s = mstd("f1_mean")
        h1_m, h1_s = mstd("hits1_mean")
        h3_m, h3_s = mstd("hits3_mean")
        mrr_m, mrr_s = mstd("mrr_mean")
        tw_m, tw_s = mstd("train_wall_sec_mean")
        ep_m, ep_s = mstd("epochs_ran_mean")

        print(
            f" {v:>7} | "
            f"{_fmt_pm(p_m, p_s, 4)} | "
            f"{_fmt_pm(r_m, r_s, 4)} | "
            f"{_fmt_pm(f_m, f_s, 4)} | "
            f"{_fmt_pm(h1_m, h1_s, 4)} | "
            f"{_fmt_pm(h3_m, h3_s, 4)} | "
            f"{_fmt_pm(mrr_m, mrr_s, 4)} | "
            f"{_fmt_pm(tw_m, tw_s, 2)} | "
            f"{_fmt_pm(ep_m, ep_s, 2)}"
        )

    # Pairwise Kendall-Tau comparison over seeds from saved score CSVs.
    if len(variants) >= 2:
        print("\n Comparison Kendall-Tau (mean ± std over seeds)")
        for va, vb in itertools.combinations(variants, 2):
            taus, h1_taus, h3_taus = [], [], []
            pos_a = _dblr_test_pos_set(va)
            pos_b = _dblr_test_pos_set(vb)
            for seed in seeds:
                pa = f"{args.save_postfix}_{va}_seed{seed}_scores.csv"
                pb = f"{args.save_postfix}_{vb}_seed{seed}_scores.csv"
                if not os.path.isfile(pa) or not os.path.isfile(pb):
                    continue
                kk = kendall_with_hits_csvs(pa, pb, pos_set_a=pos_a, pos_set_b=pos_b)
                if np.isfinite(kk["overall_tau"]):
                    taus.append(float(kk["overall_tau"]))
                if np.isfinite(kk["h1_tau"]):
                    h1_taus.append(float(kk["h1_tau"]))
                if np.isfinite(kk["h3_tau"]):
                    h3_taus.append(float(kk["h3_tau"]))
            if taus:
                arr = np.array(taus, dtype=float)
                msg = f" {va} vs {vb}: overall {_fmt_pm(float(arr.mean()), float(arr.std(ddof=0)), 4)}"
                if h1_taus:
                    h1a = np.array(h1_taus, dtype=float)
                    msg += f" | H@1 {_fmt_pm(float(h1a.mean()), float(h1a.std(ddof=0)), 4)}"
                if h3_taus:
                    h3a = np.array(h3_taus, dtype=float)
                    msg += f" | H@3 {_fmt_pm(float(h3a.mean()), float(h3a.std(ddof=0)), 4)}"
                print(msg)
            else:
                print(f" {va} vs {vb}: N/A (missing score files or non-finite tau)")


if __name__ == "__main__":
    main()
