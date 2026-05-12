import os
import time
import copy
import random
import argparse
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
import torch.sparse
import dgl

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from scipy.stats import kendalltau

from utils.pytorchtools import EarlyStopping
from utils.data import load_IMDB_data
from utils.tools import evaluate_results_nc
from model import MAGNN_nc


# =========================
# Global params
# =========================
out_dim = 3
dropout_rate = 0.5
lr = 0.005
weight_decay = 0.001


# =========================
# Variation config
# =========================
# These match your existing runner's version/prefix setup.
# v1 -> IMDB_preprocessed_star
# v2 -> IMDB_preprocessed_star_t
# v3 -> IMDB_preprocessed_star_t_2
# v4 -> IMDB_preprocessed_star_t_3

elist_v1 = [
    [[0, 1], [2, 3], [4, 5]],
    [[1, 0], [1, 2, 3, 0], [1, 4, 5, 0]],
    [[3, 0, 1, 2], [3, 2], [3, 4, 5, 2]],
    [[5, 0, 1, 4], [5, 2, 3, 4], [5, 4]],
]

# 0:M→L, 1:L→M, 2:A→L, 3:L→A, 4:D→L, 5:L→D
elist_v2 = [
    [[0, 1], [0, 5, 4, 1], [0, 3, 2, 1]],
    [[4, 1, 0, 5], [4, 5], [4, 3, 2, 5]],
    [[2, 1, 0, 3], [2, 5, 4, 3], [2, 3]],
    [[1, 0], [5, 4], [3, 2]],
]
# 0:M→D, 1:D→M, 2:M→L, 3:L→M, 4:A→L, 5:L→A
elist_v3 = [
    [[0, 1], [2, 3], [2, 5, 4, 3]],
    [[1, 0], [1, 2, 3, 0], [1, 2, 4, 5, 3, 0]],
    [[4, 3, 0, 1, 2, 5], [4, 3, 2, 5], [4, 5]],
    [[3, 0, 1, 2], [3, 2], [5, 4]],
]

# 0:M→A, 1:A→M, 2:M→L, 3:L→M, 4:D→L, 5:L→D
elist_v4 = [
    [[0, 1], [2, 3], [2, 5, 4, 3]],
    [[4, 3, 0, 1, 2, 5], [4, 3, 2, 5], [4, 5]],
    [[1, 0], [1, 2, 3, 0], [1, 2, 5, 4, 3, 0]],
    [[3, 0, 1, 2], [3, 2], [5, 4]],
]

DATA_PREFIX_DICT = {
    "v1": "data/preprocessed/IMDB_preprocessed_star",
    "v2": "data/preprocessed/IMDB_preprocessed_star_t",
    "v3": "data/preprocessed/IMDB_preprocessed_star_t_2",
    "v4": "data/preprocessed/IMDB_preprocessed_star_t_3",
}

SAVE_POSTFIX_DICT = {
    "v1": "imdb_v1",
    "v2": "imdb_v2",
    "v3": "imdb_v3",
    "v4": "imdb_v4",
}

E_LISTS_DICT = {
    "v1": elist_v1,
    "v2": elist_v2,
    "v3": elist_v3,
    "v4": elist_v4,
}


# =========================
# Utilities
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def mean_std(arr):
    arr = np.array(arr, dtype=float)
    if len(arr) == 1:
        return arr.mean(), 0.0
    return arr.mean(), arr.std(ddof=1)


def format_mean_std(arr):
    m, s = mean_std(arr)
    return f"{m:.4f} ± {s:.4f}"


def kendall_tau_logits(logits1, logits2):
    """
    logits1/logits2: shape [N, C]
    Compute average Kendall tau across samples.
    """
    taus = []
    for i in range(logits1.shape[0]):
        tau, _ = kendalltau(logits1[i], logits2[i], variant="b", nan_policy="omit")
        if not np.isnan(tau):
            taus.append(tau)
    if len(taus) == 0:
        return np.nan
    return float(np.mean(taus))


def compute_classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def print_metric_summary(metric_store, split_name):
    print(f"\n===== {split_name} Summary over seeds (mean ± std) =====")
    for k, vals in metric_store.items():
        print("{:<12}: {}".format(k, format_mean_std(vals)))


# =========================
# Main single-variation run
# =========================
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
    etypes_lists,
):
    nx_G_lists, edge_metapath_indices_lists, features_list, adjM, type_mask, labels, train_val_test_idx = load_IMDB_data(data_prefix)

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
            features_list[i] = torch.sparse_coo_tensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse_coo_tensor(indices, values, torch.Size([dim, dim])).to(device)
    else:
        raise ValueError(f"Unsupported feats_type={feats_type}")

    edge_metapath_indices_lists = [
        [torch.LongTensor(indices).to(device) for indices in indices_list]
        for indices_list in edge_metapath_indices_lists
    ]

    labels = torch.LongTensor(labels).to(device)

    g_lists = []
    for nx_G_list in nx_G_lists:
        temp_list = []
        for nx_G in nx_G_list:
            g = dgl.DGLGraph(multigraph=True)
            g.add_nodes(nx_G.number_of_nodes())

            edges = sorted((int(u), int(v)) for u, v in nx_G.edges())
            if len(edges) > 0:
                src, dst = zip(*edges)
                g.add_edges(src, dst)

            g = g.to(device)  
            temp_list.append(g)
        g_lists.append(temp_list)

    print("feature device:", features_list[0].device)
    print("label device:", labels.device)
    print("edge idx device:", edge_metapath_indices_lists[0][0].device)
    print("graph device:", g_lists[0][0].device)

    def debug_etypes_vs_indices(edge_metapath_indices_lists, etypes_lists, version_name):
        print(f"\n===== Debugging {version_name} =====")
        for ntype in range(len(edge_metapath_indices_lists)):
            idx_lists = edge_metapath_indices_lists[ntype]
            et_lists = etypes_lists[ntype]

            print(f"Node type {ntype}: data metapaths={len(idx_lists)}, etype specs={len(et_lists)}")

            for j in range(min(len(idx_lists), len(et_lists))):
                idx_arr = idx_lists[j]
                path_len = idx_arr.shape[1]
                needed = path_len - 1
                got = len(et_lists[j]) if et_lists[j] is not None else None
                print(
                    f"  metapath {j}: idx shape={tuple(idx_arr.shape)}, "
                    f"path_len={path_len}, needs={needed}, got={got}, etypes={et_lists[j]}"
                )

            if len(idx_lists) != len(et_lists):
                print("  >>> COUNT MISMATCH <<<")

    debug_etypes_vs_indices(edge_metapath_indices_lists, etypes_lists, save_postfix)

    train_idx = train_val_test_idx["train_idx"]
    val_idx = train_val_test_idx["val_idx"]
    test_idx = train_val_test_idx["test_idx"]

    target_node_indices = np.where(type_mask == 0)[0]

    net = MAGNN_nc(
        num_layers,
        [3, 3, 3, 3],
        6,
        etypes_lists,
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

    checkpoint_path = f"checkpoint/checkpoint_{save_postfix}.pt"
    early_stopping = EarlyStopping(patience=patience, verbose=True, save_path=checkpoint_path)

    dur1, dur2, dur3 = [], [], []
    train_t0 = time.perf_counter()
    epochs_ran = num_epochs

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

        train_logits_np = logits[train_idx].detach().cpu().numpy()
        val_logits_np = logits[val_idx].detach().cpu().numpy()

        y_train = labels[train_idx].detach().cpu().numpy()
        y_val = labels[val_idx].detach().cpu().numpy()

        train_pred = np.argmax(train_logits_np, axis=1)
        val_pred = np.argmax(val_logits_np, axis=1)

        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)

        print(
            "Epoch {:05d} | Train_Loss {:.4f} | Train_Acc {:.4f} | Val_Loss {:.4f} | Val_Acc {:.4f} | Time1 {:.4f} | Time2 {:.4f} | Time3 {:.4f}".format(
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
        if early_stopping.early_stop:
            print("Early stopping!")
            epochs_ran = epoch + 1
            break

    net.load_state_dict(torch.load(checkpoint_path))
    net.eval()

    with torch.no_grad():
        logits, embeddings = net((g_lists, features_list, type_mask, edge_metapath_indices_lists), target_node_indices)

        # keep evaluate_results_nc since your old runner uses it
        svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std = evaluate_results_nc(
            embeddings[test_idx].cpu().numpy(),
            labels[test_idx].cpu().numpy(),
            num_classes=out_dim,
        )

    train_logits = logits[train_idx].detach().cpu().numpy()
    val_logits = logits[val_idx].detach().cpu().numpy()
    test_logits = logits[test_idx].detach().cpu().numpy()

    y_train = labels[train_idx].detach().cpu().numpy()
    y_val = labels[val_idx].detach().cpu().numpy()
    y_test = labels[test_idx].detach().cpu().numpy()

    train_pred = np.argmax(train_logits, axis=1)
    val_pred = np.argmax(val_logits, axis=1)
    test_pred = np.argmax(test_logits, axis=1)

    train_metrics = compute_classification_metrics(y_train, train_pred)
    val_metrics = compute_classification_metrics(y_val, val_pred)
    test_metrics = compute_classification_metrics(y_test, test_pred)

    print("\nFinal metrics:")
    print(
        "Train | Acc {:.4f} | Prec {:.4f} | Recall {:.4f} | Micro-F1 {:.4f} | Macro-F1 {:.4f}".format(
            train_metrics["accuracy"],
            train_metrics["precision"],
            train_metrics["recall"],
            train_metrics["micro_f1"],
            train_metrics["macro_f1"],
        )
    )
    print(
        "Val   | Acc {:.4f} | Prec {:.4f} | Recall {:.4f} | Micro-F1 {:.4f} | Macro-F1 {:.4f}".format(
            val_metrics["accuracy"],
            val_metrics["precision"],
            val_metrics["recall"],
            val_metrics["micro_f1"],
            val_metrics["macro_f1"],
        )
    )
    print(
        "Test  | Acc {:.4f} | Prec {:.4f} | Recall {:.4f} | Micro-F1 {:.4f} | Macro-F1 {:.4f}".format(
            test_metrics["accuracy"],
            test_metrics["precision"],
            test_metrics["recall"],
            test_metrics["micro_f1"],
            test_metrics["macro_f1"],
        )
    )

    train_wall_sec = float(time.perf_counter() - train_t0)
    print(f"Training summary | train_wall_sec={train_wall_sec:.2f} | epochs_ran={epochs_ran}")

    return {
        "train_logits": train_logits,
        "val_logits": val_logits,
        "test_logits": test_logits,
        "train_pred": train_pred,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "svm_macro_f1_list": svm_macro_f1_list,
        "svm_micro_f1_list": svm_micro_f1_list,
        "nmi_mean": nmi_mean,
        "nmi_std": nmi_std,
        "ari_mean": ari_mean,
        "ari_std": ari_std,
        "train_wall_sec": train_wall_sec,
        "epochs_ran": int(epochs_ran),
    }


# =========================
# Multi-variation experiment
# =========================
def run_experiment(args):
    if not os.path.exists("checkpoint"):
        os.makedirs("checkpoint")

    selected_versions = args.versions
    for v in selected_versions:
        if v not in DATA_PREFIX_DICT:
            raise ValueError(f"Unknown variation: {v}")

    print("\nRunning versions:", selected_versions)
    print("Seeds:", args.seeds)

    # store metrics per variation over seeds
    aggregate = {
        v: {
            "train": {"accuracy": [], "precision": [], "recall": [], "micro_f1": [], "macro_f1": []},
            "val":   {"accuracy": [], "precision": [], "recall": [], "micro_f1": [], "macro_f1": []},
            "test":  {"accuracy": [], "precision": [], "recall": [], "micro_f1": [], "macro_f1": []},
            "train_wall_sec": [],
            "epochs_ran": [],
        }
        for v in selected_versions
    }

    # store logits for pairwise Kendall tau
    pairwise_tau_store = {
        f"{a} vs {b}": {"train": [], "val": [], "test": []}
        for a, b in combinations(selected_versions, 2)
    }

    for seed in args.seeds:
        print("\n" + "=" * 80)
        print(f"SEED {seed}")
        print("=" * 80)

        set_seed(seed)
        seed_results = {}

        for version in selected_versions:
            data_prefix = DATA_PREFIX_DICT[version]
            save_postfix = f"{SAVE_POSTFIX_DICT[version]}_seed{seed}"
            etypes_list = E_LISTS_DICT[version]

            print("\n" + "-" * 80)
            print(f"Running {version} | data_prefix={data_prefix}")
            print("-" * 80)

            result = run_model_IMDB(
                feats_type=args.feats_type,
                num_layers=args.layers,
                hidden_dim=args.hidden_dim,
                num_heads=args.num_heads,
                attn_vec_dim=args.attn_vec_dim,
                rnn_type=args.rnn_type,
                num_epochs=args.epoch,
                patience=args.patience,
                save_postfix=save_postfix,
                data_prefix=data_prefix,
                etypes_lists=etypes_list,
            )
            seed_results[version] = result

            for split in ["train", "val", "test"]:
                for metric_name in aggregate[version][split]:
                    aggregate[version][split][metric_name].append(result[f"{split}_metrics"][metric_name])
            aggregate[version]["train_wall_sec"].append(float(result["train_wall_sec"]))
            aggregate[version]["epochs_ran"].append(float(result["epochs_ran"]))
            print(
                f"{version} seed {seed} | train_wall_sec={result['train_wall_sec']:.2f} "
                f"| epochs_ran={result['epochs_ran']}"
            )

        # pairwise Kendall tau only when 2+ variations selected
        if len(selected_versions) >= 2:
            print("\nPairwise Kendall tau for this seed:")
            for a, b in combinations(selected_versions, 2):
                train_tau = kendall_tau_logits(seed_results[a]["train_logits"], seed_results[b]["train_logits"])
                val_tau = kendall_tau_logits(seed_results[a]["val_logits"], seed_results[b]["val_logits"])
                test_tau = kendall_tau_logits(seed_results[a]["test_logits"], seed_results[b]["test_logits"])

                pair_name = f"{a} vs {b}"
                pairwise_tau_store[pair_name]["train"].append(train_tau)
                pairwise_tau_store[pair_name]["val"].append(val_tau)
                pairwise_tau_store[pair_name]["test"].append(test_tau)

                print(
                    "{} | Train τ {:.4f} | Val τ {:.4f} | Test τ {:.4f}".format(
                        pair_name, train_tau, val_tau, test_tau
                    )
                )

    # =========================
    # Final summaries
    # =========================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY OVER SEEDS")
    print("=" * 80)

    for version in selected_versions:
        print("\n" + "#" * 80)
        print(f"Variation: {version}")
        print("#" * 80)

        print_metric_summary(aggregate[version]["train"], "Train")
        print_metric_summary(aggregate[version]["val"], "Validation")
        print_metric_summary(aggregate[version]["test"], "Test")
        print("Train Time (s):", format_mean_std(aggregate[version]["train_wall_sec"]))
        print("Epochs Ran    :", format_mean_std(aggregate[version]["epochs_ran"]))

    if len(selected_versions) >= 2:
        print("\n" + "=" * 80)
        print("PAIRWISE KENDALL TAU SUMMARY (mean ± std)")
        print("=" * 80)
        for pair_name, store in pairwise_tau_store.items():
            print(
                "{} | Train {} | Val {} | Test {}".format(
                    pair_name,
                    format_mean_std(store["train"]),
                    format_mean_std(store["val"]),
                    format_mean_std(store["test"]),
                )
            )


# =========================
# Argparse
# =========================
def build_parser():
    parser = argparse.ArgumentParser(description="IMDB MAGNN runner for v1/v2/v3/v4")

    parser.add_argument("--feats-type", type=int, default=0, help="Type of node features used")
    parser.add_argument("--layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--attn-vec-dim", type=int, default=128, help="Attention vector dimension")
    parser.add_argument("--rnn-type", type=str, default="RotatE0", help="Type of aggregator")
    parser.add_argument("--epoch", type=int, default=300, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=30, help="Patience for early stopping")

    # run one or multiple versions
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["v1", "v2", "v3", "v4"],
        choices=["v1", "v2", "v3", "v4"],
        help="Which variations to run. Example: --versions v1 or --versions v1 v2"
    )

    # exact seeds
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Seeds to run. Example: --seeds 0 1 2"
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_experiment(args)