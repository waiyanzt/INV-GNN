import time
import argparse
import os
import random
import pathlib
from itertools import combinations

import torch
import torch.nn.functional as F
import numpy as np
import dgl
import networkx as nx
from scipy.special import softmax
from scipy.stats import kendalltau
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.validation import check_array

from utils.pytorchtools import EarlyStopping
from utils.data import load_IMDB_data
from utils.tools import evaluate_results_nc
from model import MAGNN_nc

# Params
out_dim = 3
dropout_rate = 0.5
lr = 0.005
weight_decay = 0.001

# ---------------------------------------------------------------------------
# Semantic metapaths (SAME for all variants after skip-node preprocessing)
# Must match semantic filenames emitted by preprocess_IMDB_skip.py (v1 channel order).
# ---------------------------------------------------------------------------
SEMANTIC_METAPATHS = [
    [(0, 1, 0), (0, 3, 1, 3, 0), (0, 2, 0), (0, 3, 2, 3, 0), (0, 3, 0)],
    [
        (1, 0, 1),
        (1, 3, 0, 3, 1),
        (1, 0, 2, 0, 1),
        (1, 3, 2, 3, 1),
        (1, 3, 0, 2, 0, 3, 1),
        (1, 0, 3, 2, 3, 0, 1),
        (1, 0, 3, 0, 1),
        (1, 3, 1),
    ],
    [
        (2, 0, 2),
        (2, 3, 0, 3, 2),
        (2, 0, 1, 0, 2),
        (2, 0, 3, 1, 3, 0, 2),
        (2, 3, 0, 1, 0, 3, 2),
        (2, 3, 1, 3, 2),
        (2, 0, 3, 0, 2),
        (2, 3, 2),
    ],
    [(3, 0, 3), (3, 0, 1, 0, 3), (3, 1, 3), (3, 0, 2, 0, 3), (3, 2, 3)],
]

# Semantic edge types (universal across all variants):
# 0:M→D  1:D→M  2:M→A  3:A→M  4:M→L  5:L→M
# 6:D→L  7:L→D  8:A→L  9:L→A
NUM_EDGE_TYPES = 10

SEMANTIC_ETYPES = [
    [
        [0, 1],
        [4, 7, 6, 5],
        [2, 3],
        [4, 9, 8, 5],
        [4, 5],
    ],
    [
        [1, 0],
        [6, 5, 4, 7],
        [1, 2, 3, 0],
        [6, 9, 8, 7],
        [6, 5, 2, 3, 4, 7],
        [1, 4, 9, 8, 5, 0],
        [1, 4, 5, 0],
        [6, 7],
    ],
    [
        [3, 2],
        [8, 5, 4, 9],
        [3, 0, 1, 2],
        [3, 4, 7, 6, 5, 2],
        [8, 5, 0, 1, 4, 9],
        [8, 7, 6, 9],
        [3, 4, 5, 2],
        [8, 9],
    ],
    [
        [5, 4],
        [5, 0, 1, 4],
        [7, 6],
        [5, 2, 3, 4],
        [9, 8],
    ],
]

DATA_PREFIX = {
    'v1': 'data/preprocessed/IMDB_preprocessed_skip_full_v1',
    'v2': 'data/preprocessed/IMDB_preprocessed_skip_full_v2',
    'v3': 'data/preprocessed/IMDB_preprocessed_skip_full_v3',
    'v4': 'data/preprocessed/IMDB_preprocessed_skip_full_v4',
}


def _pair_set_from_idx_array(arr: np.ndarray):
    if arr.size == 0 or arr.ndim < 2:
        return set()
    pairs = np.unique(arr[:, [0, -1]], axis=0)
    return set((int(a), int(b)) for a, b in pairs)


def debug_compare_variant_to_v1(variant: str):
    if variant == 'v1':
        return
    base_ref = pathlib.Path(DATA_PREFIX['v1'])
    base_cur = pathlib.Path(DATA_PREFIX[variant])
    print("\n[debug] Comparing semantic channels vs v1 using endpoint-pair sets")
    for ntype, metas in enumerate(SEMANTIC_METAPATHS):
        for meta in metas:
            stem = '-'.join(map(str, meta))
            p_ref = base_ref / str(ntype) / f"{stem}_idx.npy"
            p_cur = base_cur / str(ntype) / f"{stem}_idx.npy"
            if (not p_ref.exists()) or (not p_cur.exists()):
                print(f"[debug] ntype={ntype} meta={stem} | missing file in v1 or {variant}")
                continue
            a = np.load(str(p_ref))
            b = np.load(str(p_cur))
            sa = _pair_set_from_idx_array(a)
            sb = _pair_set_from_idx_array(b)
            inter = len(sa & sb)
            union = len(sa | sb)
            jac = (inter / union) if union else 1.0
            same = (sa == sb)
            print(
                "[debug] ntype={} meta={} | v1_pairs={} {}_pairs={} | jaccard={:.4f} | exact={}".format(
                    ntype, stem, len(sa), variant, len(sb), jac, same
                )
            )


def set_seed(seed: int) -> None:
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


def nc_ranking_metrics(logits: np.ndarray, y_true: np.ndarray, num_classes: int = 3):
    prob = softmax(logits, axis=1)
    ranks = []
    for i in range(len(y_true)):
        order = np.argsort(-prob[i])
        t = int(y_true[i])
        r = int(np.where(order == t)[0][0]) + 1
        ranks.append(r)
    ranks = np.array(ranks, dtype=np.float64)
    return (
        float((ranks <= 1).mean()),
        float((ranks <= 3).mean()),
        float((ranks <= min(5, num_classes)).mean()),
        float(np.mean(1.0 / ranks)),
    )


def nc_sklearn_metrics(logits: np.ndarray, y_true: np.ndarray, num_classes: int = 3, th: float = 0.5):
    prob = softmax(logits, axis=1)
    pred = np.argmax(logits, axis=1)
    out = {
        "Accuracy": float(accuracy_score(y_true, pred)),
        "Precision_macro": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "Recall_macro": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "F1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "Precision_micro": float(precision_score(y_true, pred, average="micro", zero_division=0)),
        "Recall_micro": float(recall_score(y_true, pred, average="micro", zero_division=0)),
        "F1_micro": float(f1_score(y_true, pred, average="micro", zero_division=0)),
    }
    try:
        out["AUC_ovr"] = float(roc_auc_score(y_true, prob, multi_class="ovr"))
    except ValueError:
        out["AUC_ovr"] = float("nan")
    Y = label_binarize(y_true, classes=np.arange(num_classes))
    try:
        out["AP_macro_ovr"] = float(average_precision_score(Y, prob, average="macro"))
    except ValueError:
        out["AP_macro_ovr"] = float("nan")
    y_pred_thr = (prob >= th).astype(int).ravel()
    out["Precision_thr_flat"] = float(precision_score(Y.ravel(), y_pred_thr, zero_division=0))
    out["Recall_thr_flat"] = float(recall_score(Y.ravel(), y_pred_thr, zero_division=0))
    out["F1_thr_flat"] = float(f1_score(Y.ravel(), y_pred_thr, zero_division=0))
    out["Accuracy_thr_flat"] = float(accuracy_score(Y.ravel(), y_pred_thr))
    out["threshold"] = th
    return out


def run_model_IMDB(
    feats_type,
    num_layers,
    hidden_dim,
    num_heads,
    attn_vec_dim,
    rnn_type,
    num_epochs,
    patience,
    save_postfix,
    data_prefix,
    seed: int,
    max_metapath_instances: int = 250000,
):
    set_seed(seed)
    nx_G_lists, edge_metapath_indices_lists, features_list, adjM, type_mask, labels, train_val_test_idx = load_IMDB_data(
        data_prefix, expected_metapaths=SEMANTIC_METAPATHS
    )
    # Drop metapaths with zero path instances for this variant, otherwise RotatE
    # internals can hit empty tensors (e.g. IndexError on final_r_vec[-1, ...]).
    run_etypes = []
    run_num_metapaths_list = []
    for ntype in range(len(SEMANTIC_METAPATHS)):
        kept_graphs = []
        kept_indices = []
        kept_etypes = []
        dropped = []
        for mp_idx, (g, idx_arr, etypes, meta) in enumerate(
            zip(
                nx_G_lists[ntype],
                edge_metapath_indices_lists[ntype],
                SEMANTIC_ETYPES[ntype],
                SEMANTIC_METAPATHS[ntype],
            )
        ):
            has_instances = (idx_arr.ndim >= 1 and idx_arr.shape[0] > 0)
            if has_instances:
                if max_metapath_instances and idx_arr.shape[0] > max_metapath_instances:
                    orig_n = idx_arr.shape[0]
                    edge_list = sorted((int(u), int(v)) for u, v in g.edges())
                    if len(edge_list) != orig_n:
                        print(
                            "Skip capping {} for node type {}: edge/index mismatch {} vs {}".format(
                                "-".join(map(str, meta)),
                                ntype,
                                len(edge_list),
                                orig_n,
                            )
                        )
                    else:
                        rng = np.random.default_rng(seed + 10007 * ntype + mp_idx)
                        keep = np.sort(rng.choice(orig_n, size=max_metapath_instances, replace=False))
                        idx_arr = idx_arr[keep]
                        # Keep graph edges aligned with capped idx rows.
                        g_cap = nx.MultiDiGraph()
                        g_cap.add_nodes_from(g.nodes())
                        g_cap.add_edges_from([edge_list[int(i)] for i in keep])
                        g = g_cap
                        print(
                            "Capping metapath {} for node type {}: {} -> {} instances".format(
                                "-".join(map(str, meta)),
                                ntype,
                                orig_n,
                                idx_arr.shape[0],
                            )
                        )
                kept_graphs.append(g)
                kept_indices.append(idx_arr)
                kept_etypes.append(etypes)
            else:
                dropped.append((mp_idx, meta))
        if dropped:
            print(
                "Dropping {} empty metapaths for node type {}: {}".format(
                    len(dropped),
                    ntype,
                    ", ".join(
                        "#{}:{}".format(i, "-".join(map(str, m)))
                        for i, m in dropped
                    ),
                )
            )
        if not kept_indices:
            raise RuntimeError(
                "All metapaths are empty for node type {} under {}. "
                "Try reducing/adjusting union metapaths for this variant.".format(ntype, data_prefix)
            )
        nx_G_lists[ntype] = kept_graphs
        edge_metapath_indices_lists[ntype] = kept_indices
        run_etypes.append(kept_etypes)
        run_num_metapaths_list.append(len(kept_indices))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features_list = [torch.FloatTensor(features.todense()).to(device) for features in features_list]
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1:
        in_dims = [features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(1, len(features_list)):
            features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2:
        in_dims = [features.shape[0] for features in features_list]
        in_dims[0] = features_list[0].shape[1]
        for i in range(1, len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    edge_metapath_indices_lists = [[torch.LongTensor(indices).to(device) for indices in indices_list] for indices_list in
                                   edge_metapath_indices_lists]
    labels = torch.LongTensor(labels).to(device)
    g_lists = []
    for nx_G_list in nx_G_lists:
        g_lists.append([])
        for nx_G in nx_G_list:
            g = dgl.DGLGraph(multigraph=True)
            g.add_nodes(nx_G.number_of_nodes())
            edges = sorted((int(u), int(v)) for u, v in nx_G.edges())
            if edges:
                u, v = zip(*edges)
                g.add_edges(u, v)
            g_lists[-1].append(g.to(device))
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    net = MAGNN_nc(
        num_layers,
        run_num_metapaths_list,
        NUM_EDGE_TYPES,
        run_etypes,
        in_dims,
        hidden_dim,
        out_dim,
        num_heads,
        attn_vec_dim,
        rnn_type,
        dropout_rate,
    )
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    target_node_indices = np.where(type_mask == 0)[0]

    os.makedirs("checkpoint", exist_ok=True)
    ckpt_path = "checkpoint/checkpoint_{}_seed{}.pt".format(save_postfix, seed)
    early_stopping = EarlyStopping(patience=patience, verbose=True, save_path=ckpt_path)
    dur1 = []
    dur2 = []
    dur3 = []
    train_wall_t0 = time.time()
    epochs_ran = 0
    for epoch in range(num_epochs):
        t0 = time.time()

        net.train()
        logits, embeddings = net((g_lists, features_list, type_mask, edge_metapath_indices_lists), target_node_indices)
        logp = F.log_softmax(logits, 1)
        train_loss = F.nll_loss(logp[train_idx], labels[train_idx])

        t1 = time.time()
        dur1.append(t1 - t0)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        t2 = time.time()
        dur2.append(t2 - t1)

        net.eval()
        with torch.no_grad():
            logits, embeddings = net((g_lists, features_list, type_mask, edge_metapath_indices_lists), target_node_indices)
            logp = F.log_softmax(logits, 1)
            val_loss = F.nll_loss(logp[val_idx], labels[val_idx])

        t3 = time.time()
        dur3.append(t3 - t2)

        tr_logits_np = logits[train_idx].detach().cpu().numpy()
        va_logits_np = logits[val_idx].detach().cpu().numpy()
        train_acc = accuracy_score(labels[train_idx].cpu().numpy(), np.argmax(tr_logits_np, axis=1))
        val_acc = accuracy_score(labels[val_idx].cpu().numpy(), np.argmax(va_logits_np, axis=1))
        print(
            "Epoch {:05d} | Train_Loss {:.3f} | Train_Acc {:.3f} | Val_Loss {:.3f} | Val_Acc {:.3f} | Time1(s) {:.3f} | Time2(s) {:.3f} | Time3(s) {:.3f}".format(
                epoch,
                train_loss.item(),
                train_acc,
                val_loss.item(),
                val_acc,
                np.mean(dur1),
                np.mean(dur2),
                np.mean(dur3),
            )
        )
        early_stopping(val_loss, net)
        epochs_ran = epoch + 1
        if early_stopping.early_stop:
            print('Early stopping!')
            break
    train_wall_sec = time.time() - train_wall_t0

    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.eval()
    with torch.no_grad():
        logits, embeddings = net((g_lists, features_list, type_mask, edge_metapath_indices_lists), target_node_indices)
        train_logits = logits[train_idx].detach().cpu().numpy()
        val_logits = logits[val_idx].detach().cpu().numpy()
        test_logits = logits[test_idx].detach().cpu().numpy()
        y_tr = labels[train_idx].cpu().numpy()
        y_va = labels[val_idx].cpu().numpy()
        y_te = labels[test_idx].cpu().numpy()
        eval_out = evaluate_results_nc(
            embeddings[test_idx].cpu().numpy(), y_te, num_classes=out_dim
        )

    if len(eval_out) == 8:
        svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std, svm_preds, knn_preds = eval_out
    else:
        svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std = eval_out
        p_te = softmax(test_logits, axis=1)
        svm_preds = [p_te.copy() for _ in range(4)]
        knn_preds = p_te.copy()

    train_acc = accuracy_score(y_tr, np.argmax(train_logits, axis=1))
    val_acc = accuracy_score(y_va, np.argmax(val_logits, axis=1))
    test_acc = accuracy_score(y_te, np.argmax(test_logits, axis=1))

    test_cls = nc_sklearn_metrics(test_logits, y_te, out_dim, th=0.5)
    h1, h3, h5, mrr = nc_ranking_metrics(test_logits, y_te, out_dim)

    print("----------------------------------------------------------------")
    print("=== IMDB skip-node classification (test) | seed {} ===".format(seed))
    print(
        "Train_Acc {:.3f} | Val_Acc {:.3f} | Test_Acc {:.3f}".format(train_acc, val_acc, test_acc)
    )
    print(
        "AUC_ovr={:.3f}  AP_macro_ovr={:.3f}  P_macro={:.3f}  R_macro={:.3f}  F1_macro={:.3f}".format(
            test_cls["AUC_ovr"],
            test_cls["AP_macro_ovr"],
            test_cls["Precision_macro"],
            test_cls["Recall_macro"],
            test_cls["F1_macro"],
        )
    )
    print(
        "P_micro={:.3f}  R_micro={:.3f}  F1_micro={:.3f}".format(
            test_cls["Precision_micro"], test_cls["Recall_micro"], test_cls["F1_micro"]
        )
    )
    print(
        "Hits@1={:.3f}  Hits@3={:.3f}  Hits@5={:.3f}  MRR={:.3f}  (class-ranking; C={})".format(
            h1, h3, h5, mrr, out_dim
        )
    )
    print(
        "Thr {:.2f} (flat OVR): P={:.3f} R={:.3f} F1={:.3f} Acc_flat={:.3f}".format(
            test_cls["threshold"],
            test_cls["Precision_thr_flat"],
            test_cls["Recall_thr_flat"],
            test_cls["F1_thr_flat"],
            test_cls["Accuracy_thr_flat"],
        )
    )
    print("SVM tests summary")
    print(
        "Macro-F1: "
        + ", ".join(
            [
                "{:.3f}~{:.3f} ({:.1f})".format(macro_f1[0], macro_f1[1], train_size)
                for macro_f1, train_size in zip(svm_macro_f1_list, [0.8, 0.6, 0.4, 0.2])
            ]
        )
    )
    print(
        "Micro-F1: "
        + ", ".join(
            [
                "{:.3f}~{:.3f} ({:.1f})".format(micro_f1[0], micro_f1[1], train_size)
                for micro_f1, train_size in zip(svm_micro_f1_list, [0.8, 0.6, 0.4, 0.2])
            ]
        )
    )
    print("K-means: NMI {:.3f}~{:.3f}  ARI {:.3f}~{:.3f}".format(nmi_mean, nmi_std, ari_mean, ari_std))

    metrics = {
        **test_cls,
        "Hits@1": h1,
        "Hits@3": h3,
        "Hits@5": h5,
        "MRR": mrr,
        "Test_acc_argmax": float(test_acc),
        "NMI_mean": float(nmi_mean),
        "ARI_mean": float(ari_mean),
        "Train Time (s)": float(train_wall_sec),
        "Epochs": float(epochs_ran),
    }

    return {
        "metrics": metrics,
        "train_logits": train_logits,
        "val_logits": val_logits,
        "test_logits": test_logits,
        "svm_preds": svm_preds,
        "knn_preds": knn_preds,
        "train_pred": np.argmax(train_logits, axis=1),
        "val_pred": np.argmax(val_logits, axis=1),
        "test_pred": np.argmax(test_logits, axis=1),
        "train_labels": y_tr,
        "val_labels": y_va,
        "test_labels": y_te,
        "svm_macro_f1_list": svm_macro_f1_list,
        "svm_micro_f1_list": svm_micro_f1_list,
    }


@torch.no_grad()
def eval_only_imdb_logits(
    feats_type,
    num_layers,
    hidden_dim,
    num_heads,
    attn_vec_dim,
    rnn_type,
    save_postfix,
    data_prefix,
    seed: int,
    max_metapath_instances: int = 0,
):
    """
    Load checkpoint + run a forward pass to produce logits for Kendall τ.
    Does not train. Used by --compare-only.
    """
    set_seed(seed)
    nx_G_lists, edge_metapath_indices_lists, features_list, _adjM, type_mask, labels, train_val_test_idx = load_IMDB_data(
        data_prefix, expected_metapaths=SEMANTIC_METAPATHS
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    features_list = [torch.FloatTensor(features.todense()).to(device) for features in features_list]
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1:
        in_dims = [features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(1, len(features_list)):
            features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2:
        in_dims = [features.shape[0] for features in features_list]
        in_dims[0] = features_list[0].shape[1]
        for i in range(1, len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    else:
        raise ValueError(f"Unsupported feats_type={feats_type}")

    def _maybe_cap(idx_arr: np.ndarray) -> np.ndarray:
        if max_metapath_instances and max_metapath_instances > 0 and idx_arr.shape[0] > max_metapath_instances:
            return idx_arr[:max_metapath_instances]
        return idx_arr

    edge_metapath_indices_lists = [
        [torch.LongTensor(_maybe_cap(indices)).to(device) for indices in indices_list] for indices_list in edge_metapath_indices_lists
    ]
    labels_t = torch.LongTensor(labels).to(device)

    g_lists = []
    for nx_G_list in nx_G_lists:
        g_lists.append([])
        for nx_G in nx_G_list:
            g = dgl.DGLGraph(multigraph=True)
            g.add_nodes(nx_G.number_of_nodes())
            edges = sorted((int(u), int(v)) for u, v in nx_G.edges())
            if edges:
                src, dst = zip(*edges)
                g.add_edges(src, dst)
            g_lists[-1].append(g.to(device))

    target_node_indices = np.where(type_mask == 0)[0]
    num_metapaths_list = [len(SEMANTIC_METAPATHS[i]) for i in range(len(SEMANTIC_METAPATHS))]
    net = MAGNN_nc(
        num_layers,
        num_metapaths_list,
        NUM_EDGE_TYPES,
        SEMANTIC_ETYPES,
        in_dims,
        hidden_dim,
        out_dim,
        num_heads,
        attn_vec_dim,
        rnn_type,
        dropout_rate,
    ).to(device)

    ckpt_path = "checkpoint/checkpoint_{}_seed{}.pt".format(save_postfix, seed)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.eval()

    logits, _emb = net((g_lists, features_list, type_mask, edge_metapath_indices_lists), target_node_indices)
    logits_np = logits.detach().cpu().numpy()
    train_idx = train_val_test_idx["train_idx"]
    val_idx = train_val_test_idx["val_idx"]
    test_idx = train_val_test_idx["test_idx"]
    y = labels_t.detach().cpu().numpy()

    return [
        logits_np[train_idx],
        logits_np[val_idx],
        logits_np[test_idx],
        y[train_idx],
        y[val_idx],
        y[test_idx],
        None,  # svm_preds not available in compare-only
    ]


def result_to_legacy_list(res):
    return [
        res["train_logits"],
        res["val_logits"],
        res["test_logits"],
        res["train_labels"],
        res["val_labels"],
        res["test_labels"],
        res["svm_preds"],
    ]


def summarize_imdb_metrics(runs):
    summary_fields = [
        ("Accuracy", "Accuracy"),
        ("Precision", "Precision_macro"),
        ("Recall", "Recall_macro"),
        ("Micro-F1", "F1_micro"),
        ("Macro-F1", "F1_macro"),
        ("Train Time (s)", "Train Time (s)"),
        ("Epoch", "Epochs"),
    ]
    print("\n===== Summary over seeds (mean +/- std) =====")
    for label, key in summary_fields:
        arr = np.array([r["metrics"][key] for r in runs], dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue
        if label in ("Train Time (s)", "Epoch"):
            print("{:<22}: {:.2f} +/- {:.2f}".format(label, arr.mean(), arr.std(ddof=0)))
        else:
            print("{:<22}: {:.3f} +/- {:.3f}".format(label, arr.mean(), arr.std(ddof=0)))


def robustness(pred1, pred2):
    train_robust_score = accuracy_score(pred1[0], pred2[0])
    val_robust_score = accuracy_score(pred1[1], pred2[1])
    test_robust_score = accuracy_score(pred1[2], pred2[2])
    return [train_robust_score, val_robust_score, test_robust_score]


def kendall_tau_single(pred1, pred2):
    taus = []
    for i in range(pred1.shape[0]):
        tau, _ = kendalltau(pred1[i], pred2[i], variant="b", nan_policy="omit")
        taus.append(tau)
    taus = np.array(taus)
    return np.nanmean(taus)


def kendall_tau(pred1, pred2):
    taus = []
    for i in range(pred1[0].shape[0]):
        tau, _ = kendalltau(pred1[0][i], pred2[0][i], variant="b", nan_policy="omit")
        taus.append(tau)
    train_tau = np.nanmean(np.array(taus))

    taus = []
    for i in range(pred1[1].shape[0]):
        tau, _ = kendalltau(pred1[1][i], pred2[1][i], variant="b", nan_policy="omit")
        taus.append(tau)
    val_tau = np.nanmean(np.array(taus))

    taus = []
    for i in range(pred1[2].shape[0]):
        tau, _ = kendalltau(pred1[2][i], pred2[2][i], variant="b", nan_policy="omit")
        taus.append(tau)
    test_tau = np.nanmean(np.array(taus))
    return [train_tau, val_tau, test_tau]


def robustness_sliced_full(pred1, pred2):
    a0 = np.argmax(pred1[0], axis=1)
    b0 = np.argmax(pred2[0], axis=1)
    a1 = np.argmax(pred1[1], axis=1)
    b1 = np.argmax(pred2[1], axis=1)
    a2 = np.argmax(pred1[2], axis=1)
    b2 = np.argmax(pred2[2], axis=1)
    for split_name, a, b, gt in [("Train", a0, b0, pred1[3]),
                                  ("Val", a1, b1, pred1[4]),
                                  ("Test", a2, b2, pred1[5])]:
        both_fail = one_fail = both_ok = 0
        bf_match = of_match = bo_match = 0
        for p1, p2, p3 in zip(a, b, gt):
            if p1 != p3 and p2 != p3:
                both_fail += 1
                if p1 == p2:
                    bf_match += 1
            elif (p1 != p3) != (p2 != p3):
                one_fail += 1
                if p1 == p2:
                    of_match += 1
            else:
                both_ok += 1
                if p1 == p2:
                    bo_match += 1
        print('{}_Robust (both fail) {:d}/{:d} = {:.3f} | (one fails) {:d}/{:d} = {:.3f} | (both ok) {:d}/{:d} = {:.3f}'.format(
            split_name,
            bf_match, both_fail, bf_match / max(both_fail, 1),
            of_match, one_fail, of_match / max(one_fail, 1),
            bo_match, both_ok, bo_match / max(both_ok, 1)))


def robustness_sliced(pred1, pred2):
    a0 = np.argmax(pred1[0], axis=1)
    b0 = np.argmax(pred2[0], axis=1)
    a1 = np.argmax(pred1[1], axis=1)
    b1 = np.argmax(pred2[1], axis=1)
    a2 = np.argmax(pred1[2], axis=1)
    b2 = np.argmax(pred2[2], axis=1)
    for split_name, a, b, gt in [("Train", a0, b0, pred1[3]),
                                  ("Val", a1, b1, pred1[4]),
                                  ("Test", a2, b2, pred1[5])]:
        either_fail = both_ok = 0
        ef_match = bo_match = 0
        for p1, p2, p3 in zip(a, b, gt):
            if p1 != p3 or p2 != p3:
                either_fail += 1
                if p1 == p2:
                    ef_match += 1
            else:
                both_ok += 1
                if p1 == p2:
                    bo_match += 1
        print('{}_Robust (either fail) {:d}/{:d} = {:.3f} | (both ok) {:d}/{:d} = {:.3f}'.format(
            split_name,
            ef_match, either_fail, ef_match / max(either_fail, 1),
            bo_match, both_ok, bo_match / max(both_ok, 1)))


def _parse_int_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_variant_list(s):
    out = [x.strip().lower() for x in s.split(",") if x.strip()]
    for v in out:
        if v not in ("v1", "v2", "v3", "v4"):
            raise ValueError("Unknown variant {!r}; expected v1,v2,v3,v4".format(v))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='MAGNN IMDB skip-node: all variants share semantic metapaths')
    ap.add_argument('--feats-type', type=int, default=1,
                    help='0=loaded feats; 1=target only (zero others); '
                         '2=target + id; 3=all id. Default 1.')
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--hidden-dim', type=int, default=64)
    ap.add_argument('--num-heads', type=int, default=8)
    ap.add_argument('--attn-vec-dim', type=int, default=128)
    ap.add_argument('--rnn-type', default='RotatE0')
    ap.add_argument('--epoch', type=int, default=100)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--seeds', type=str, default='1566911444,20241017,20251017')
    ap.add_argument('--variants', type=str, default='v1,v2,v3,v4',
                    help='Comma-separated: v1,v2,v3,v4.')
    ap.add_argument('--compare', type=str, default='',
                    help='Pair for Kendall tau, e.g. v1,v2.')
    ap.add_argument(
        '--compare-only',
        action='store_true',
        help='Skip training. Load checkpoints and compute Kendall tau (all pairs unless --compare is set).',
    )
    ap.add_argument('--save-postfix', default='IMDB_skip')
    ap.add_argument('--max-metapath-instances', type=int, default=250000,
                    help='Cap per-metapath instance rows loaded to GPU to avoid OOM; set <=0 to disable.')
    ap.add_argument('--debug-compare-v1', action='store_true',
                    help='Print per-metapath endpoint-pair overlap (current variant vs v1) before training.')

    args = ap.parse_args()
    seeds = _parse_int_list(args.seeds)
    variants = _parse_variant_list(args.variants)

    os.makedirs('checkpoint', exist_ok=True)

    if args.compare_only:
        # Load logits from checkpoints and compute Kendall τ without retraining.
        logits_by_variant = {v: [] for v in variants}
        for ver in variants:
            print("\n########## Compare-only | variant {} | {} seeds ##########".format(ver, len(seeds)))
            data_prefix = DATA_PREFIX[ver]
            postfix = "imdb_skip_{}".format(ver)
            for seed in seeds:
                print('----------------------------------------------------------------')
                print("Variant {} | seed {} | data {}".format(ver, seed, data_prefix))
                print('----------------------------------------------------------------')
                if args.debug_compare_v1:
                    debug_compare_variant_to_v1(ver)
                legacy = eval_only_imdb_logits(
                    args.feats_type,
                    args.layers,
                    args.hidden_dim,
                    args.num_heads,
                    args.attn_vec_dim,
                    args.rnn_type,
                    postfix,
                    data_prefix,
                    seed,
                    max_metapath_instances=(args.max_metapath_instances if args.max_metapath_instances > 0 else 0),
                )
                logits_by_variant[ver].append(legacy)

        pairs = []
        if args.compare.strip():
            pair = _parse_variant_list(args.compare)
            if len(pair) != 2:
                raise SystemExit('--compare expects exactly two variants, e.g. v1,v2')
            pairs = [(pair[0], pair[1])]
        else:
            pairs = list(combinations(variants, 2))

        print("\n########## Kendall tau (logits) | compare-only ##########")
        for a, b in pairs:
            print("\n=== {} vs {} ===".format(a, b))
            train_taus, val_taus, test_taus = [], [], []
            for i, seed in enumerate(seeds):
                print("--- seed {} ---".format(seed))
                la = logits_by_variant[a][i]
                lb = logits_by_variant[b][i]
                check_array(la[1], ensure_2d=True)
                check_array(lb[1], ensure_2d=True)
                tau = kendall_tau(la, lb)
                train_taus.append(tau[0])
                val_taus.append(tau[1])
                test_taus.append(tau[2])
                print(
                    'Train_Kendall_Tau {:.3f} | Val_Kendall_Tau {:.3f} | Test_Kendall_Tau {:.3f}'.format(
                        tau[0], tau[1], tau[2]
                    )
                )

            def _mean_std0(x):
                x = np.array(x, dtype=float)
                return float(x.mean()), float(x.std(ddof=0)) if len(x) > 1 else 0.0

            mtr, str_ = _mean_std0(train_taus)
            mva, sva = _mean_std0(val_taus)
            mte, ste = _mean_std0(test_taus)
            print(
                "Summary (mean ± std): Train {:.3f} ± {:.3f} | Val {:.3f} ± {:.3f} | Test {:.3f} ± {:.3f}".format(
                    mtr, str_, mva, sva, mte, ste
                )
            )
        raise SystemExit(0)

    all_variant_runs = {}
    for ver in variants:
        print("\n########## Skip-node variant {} | {} seeds ##########".format(ver, len(seeds)))
        runs = []
        for seed in seeds:
            data_prefix = DATA_PREFIX[ver]
            postfix = "imdb_skip_{}".format(ver)
            print('----------------------------------------------------------------')
            print("Variant {} | seed {} | data {}".format(ver, seed, data_prefix))
            print('----------------------------------------------------------------')
            if args.debug_compare_v1:
                debug_compare_variant_to_v1(ver)
            res = run_model_IMDB(
                args.feats_type,
                args.layers,
                args.hidden_dim,
                args.num_heads,
                args.attn_vec_dim,
                args.rnn_type,
                args.epoch,
                args.patience,
                postfix,
                data_prefix,
                seed,
                max_metapath_instances=(args.max_metapath_instances if args.max_metapath_instances > 0 else 0),
            )
            runs.append(res)
        summarize_imdb_metrics(runs)
        all_variant_runs[ver] = runs

    if args.compare.strip():
        pair = _parse_variant_list(args.compare)
        if len(pair) != 2:
            raise SystemExit('--compare expects exactly two variants, e.g. v1,v2')
        a, b = pair[0], pair[1]
        if a not in all_variant_runs or b not in all_variant_runs:
            missing = [v for v in (a, b) if v not in all_variant_runs]
            raise SystemExit(
                '--compare requires {} in --variants; missing: {}'.format(
                    ','.join(pair), ','.join(missing)))

        print("\n########## Kendall tau (logits) | {} vs {} ##########".format(a, b))
        train_taus, val_taus, test_taus = [], [], []
        svm_8, svm_6, svm_4, svm_2 = [], [], [], []
        for i, seed in enumerate(seeds):
            print("--- seed {} ---".format(seed))
            ra = all_variant_runs[a][i]
            rb = all_variant_runs[b][i]
            la = result_to_legacy_list(ra)
            lb = result_to_legacy_list(rb)
            check_array(la[1], ensure_2d=True)
            check_array(lb[1], ensure_2d=True)
            tau = kendall_tau(la, lb)
            train_taus.append(tau[0])
            val_taus.append(tau[1])
            test_taus.append(tau[2])
            print(
                'Train_Kendall_Tau {:.3f} | Val_Kendall_Tau {:.3f} | Test_Kendall_Tau {:.3f}'.format(
                    tau[0], tau[1], tau[2]
                )
            )
            sp_a, sp_b = ra["svm_preds"], rb["svm_preds"]
            for j, bucket in enumerate([svm_8, svm_6, svm_4, svm_2]):
                check_array(sp_a[j], ensure_2d=True)
                check_array(sp_b[j], ensure_2d=True)
                bucket.append(kendall_tau_single(sp_a[j], sp_b[j]))
            print(
                'SVM tau (0.8/0.6/0.4/0.2): {:.3f} {:.3f} {:.3f} {:.3f}'.format(
                    svm_8[-1], svm_6[-1], svm_4[-1], svm_2[-1]
                )
            )
            print("--- Robustness (sliced full) ---")
            robustness_sliced_full(la, lb)
            print("--- Robustness (sliced) ---")
            robustness_sliced(la, lb)

        def _mean_std(x):
            x = np.array(x, dtype=float)
            return x.mean(), x.std(ddof=1) if len(x) > 1 else 0.0

        print("\n===== Kendall tau summary across seeds =====")
        for name, arr in [
            ('Train tau', train_taus),
            ('Val tau', val_taus),
            ('Test tau', test_taus),
            ('SVM tau 0.8', svm_8),
            ('SVM tau 0.6', svm_6),
            ('SVM tau 0.4', svm_4),
            ('SVM tau 0.2', svm_2),
        ]:
            m, s = _mean_std(arr)
            print('{}: {:.3f} +/- {:.3f}'.format(name, m, s))