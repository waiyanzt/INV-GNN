"""
Kendall tau robustness between two IMDB graph variants (SlotGAT node classification).

Trains SlotGAT per variant (same split conventions as run_IMDB_nc.py), then compares
class-logit rankings between variants on train / val / test nodes using Kendall's tau
(per node, across the 3 genre classes).

Usage:
    cd src/baselines/SlotGAT
    python kendall_tau_IMDB_nc.py --variant 1 2 --seeds 1566911444 20241017
    python kendall_tau_IMDB_nc.py --variant 1 2 --ckpt-suffix _exp1
"""

import argparse
import gc
import os

import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from model import slotGAT
from utils.data import load_data
from utils.pytorchtools import EarlyStopping
from utils.tools import func_args_parse, set_seed, single_feat_net

NUM_CLASSES = 3
VARIANT_NAMES = {
    1: "IMDB_var1",
    2: "IMDB_var2",
    3: "IMDB_var3",
    4: "IMDB_var4",
}


def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if isinstance(mat, np.ndarray):
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)


def build_graph(features_list, adj_m, dl, device):
    """Build DGL homogeneous graph with edge type features."""
    edge2type = {}
    for k in dl.links["data"]:
        for u, v in zip(*dl.links["data"][k].nonzero()):
            edge2type[(u, v)] = k

    for i in range(dl.nodes["total"]):
        if (i, i) not in edge2type:
            edge2type[(i, i)] = len(dl.links["count"])

    count_reverse = 0
    for k in dl.links["data"]:
        has_reverse = 0
        for u, v in zip(*dl.links["data"][k].nonzero()):
            if (v, u) not in edge2type:
                edge2type[(v, u)] = count_reverse + 1 + len(dl.links["count"])
                has_reverse = 1
        count_reverse += has_reverse

    num_etype = len(dl.links["count"]) + 1 + count_reverse

    g = dgl.DGLGraph(adj_m + adj_m.T)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)

    e_feat = []
    count = 0
    count_mappings = {}
    counted_dict = {}
    g_cpu = g.cpu()
    for u, v in zip(*g_cpu.edges()):
        u = u.item()
        v = v.item()
        if not counted_dict.setdefault(edge2type[(u, v)], False):
            count_mappings[edge2type[(u, v)]] = count
            counted_dict[edge2type[(u, v)]] = True
            count += 1
        e_feat.append(count_mappings[edge2type[(u, v)]])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)

    num_ntypes = len(features_list)
    num_nodes = dl.nodes["total"]
    g.num_ntypes = num_ntypes
    g.node_idx_by_ntype = []
    g.node_ntype_indexer = torch.zeros(num_nodes, num_ntypes).to(device)
    g.edge_type_indexer = F.one_hot(e_feat).to(device)

    idx_count = 0
    ntype_count = 0
    for feature in features_list:
        temp = []
        for _ in feature:
            temp.append(idx_count)
            g.node_ntype_indexer[idx_count][ntype_count] = 1
            idx_count += 1
        g.node_idx_by_ntype.append(temp)
        ntype_count += 1

    return g, e_feat, num_etype


def kendall_tau(pred1, pred2):
    """Kendall tau between matching splits: pred[k] are (n_nodes, n_classes) logits."""
    taus = []
    for i in range(pred1[0].shape[0]):
        tau, _ = kendalltau(pred1[0][i], pred2[0][i], variant="b", nan_policy="omit")
        taus.append(tau)
    train_tau = np.nanmean(np.asarray(taus))

    taus = []
    for i in range(pred1[1].shape[0]):
        tau, _ = kendalltau(pred1[1][i], pred2[1][i], variant="b", nan_policy="omit")
        taus.append(tau)
    val_tau = np.nanmean(np.asarray(taus))

    taus = []
    for i in range(pred1[2].shape[0]):
        tau, _ = kendalltau(pred1[2][i], pred2[2][i], variant="b", nan_policy="omit")
        taus.append(tau)
    test_tau = np.nanmean(np.asarray(taus))

    return [train_tau, val_tau, test_tau]


def _mean_std_ddof1(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def train_and_eval_variant(args, variant, seed):
    device = torch.device(
        f"cuda:{args.gpu}" if not args.cpu and torch.cuda.is_available() else "cpu"
    )
    set_seed(seed)

    variant_name = VARIANT_NAMES[variant]
    features_list, adj_m, labels, train_val_test_idx, dl = load_data(
        prefix=variant_name, multi_labels=False
    )

    num_classes = dl.labels_train["num_classes"]
    if num_classes != NUM_CLASSES:
        print(f"Warning: expected {NUM_CLASSES} classes, got {num_classes}")

    features_list = [mat2tensor(f).to(device) for f in features_list]
    in_dims = [f.shape[1] for f in features_list]
    labels = torch.LongTensor(labels).to(device)

    train_idx = np.sort(train_val_test_idx["train_idx"])
    val_idx = np.sort(train_val_test_idx["val_idx"])
    test_idx = np.sort(train_val_test_idx["test_idx"])

    g, e_feat, num_etype = build_graph(features_list, adj_m, dl, device)

    num_ntypes = len(features_list)
    heads = [args.num_heads] * args.num_layers + [1]
    data_recorder = {
        "meta": {"getSAAttentionScore": "False"},
        "data": {},
        "status": "None",
    }
    fargs, fkargs = func_args_parse(
        g,
        args.edge_feats,
        num_etype,
        in_dims,
        args.hidden_dim,
        num_classes,
        args.num_layers,
        heads,
        F.elu,
        args.dropout_feat,
        args.dropout_attn,
        args.slope,
        True,
        0.05,
        num_ntype=num_ntypes,
        eindexer=None,
        aggregator=args.aggregator,
        SAattDim=args.SAattDim,
        dataRecorder=data_recorder,
        vis_data_saver=None,
    )
    net = single_feat_net(slotGAT, *fargs, **fkargs).to(device)

    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    os.makedirs("checkpoint", exist_ok=True)
    ckp_fname = (
        f"checkpoint/slotgat_imdb_kendall_var{variant}_seed{seed}{args.ckpt_suffix}.pt"
    )
    early_stopping = EarlyStopping(
        patience=args.patience, verbose=False, save_path=ckp_fname
    )

    for epoch in tqdm(range(args.epoch), desc=f"kendall var{variant} seed={seed}"):
        net.train()
        logits, _ = net(features_list, e_feat)
        logp = F.log_softmax(logits, 1)
        train_loss = F.nll_loss(logp[train_idx], labels[train_idx])

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        net.eval()
        with torch.no_grad():
            logits, _ = net(features_list, e_feat)
            logp = F.log_softmax(logits, 1)
            val_loss = F.nll_loss(logp[val_idx], labels[val_idx])

        early_stopping(val_loss, net)
        if epoch > args.epoch / 2 and early_stopping.early_stop:
            break

    try:
        state = torch.load(ckp_fname, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckp_fname, map_location=device)
    net.load_state_dict(state, strict=False)

    net.eval()
    with torch.no_grad():
        logits, _ = net(features_list, e_feat)

    train_logits = logits[train_idx].detach().cpu().numpy()
    val_logits = logits[val_idx].detach().cpu().numpy()
    test_logits = logits[test_idx].detach().cpu().numpy()

    y_tr = labels[train_idx].detach().cpu().numpy()
    y_va = labels[val_idx].detach().cpu().numpy()
    y_te = labels[test_idx].detach().cpu().numpy()
    p_tr = np.argmax(train_logits, axis=1)
    p_va = np.argmax(val_logits, axis=1)
    p_te = np.argmax(test_logits, axis=1)

    acc_list = [
        accuracy_score(y_tr, p_tr),
        accuracy_score(y_va, p_va),
        accuracy_score(y_te, p_te),
    ]
    micro_f1_list = [
        f1_score(y_tr, p_tr, average="micro"),
        f1_score(y_va, p_va, average="micro"),
        f1_score(y_te, p_te, average="micro"),
    ]
    macro_f1_list = [
        f1_score(y_tr, p_tr, average="macro"),
        f1_score(y_va, p_va, average="macro"),
        f1_score(y_te, p_te, average="macro"),
    ]

    print(
        f"  Acc train/val/test: {acc_list[0]:.4f} / {acc_list[1]:.4f} / {acc_list[2]:.4f}"
    )

    del net, g
    if not args.cpu and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return [
        train_logits,
        val_logits,
        test_logits,
        None,
        None,
        acc_list,
        micro_f1_list,
        macro_f1_list,
        p_tr,
        p_va,
        p_te,
        y_tr,
        y_va,
        y_te,
        train_idx,
        val_idx,
        test_idx,
    ]


def parse_args():
    p = argparse.ArgumentParser(
        description="SlotGAT Kendall tau between two IMDB NC variants",
    )
    p.add_argument(
        "--variant",
        nargs=2,
        type=int,
        default=[1, 2],
        metavar=("V1", "V2"),
        choices=[1, 2, 3, 4],
        help="Two graph variants to compare (default: 1 2).",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1566911444, 20241017, 20251017],
    )
    p.add_argument(
        "--ckpt-suffix",
        type=str,
        default="",
        help="Extra string before .pt (e.g. _run2). Default: empty.",
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--cpu", action="store_true", default=False)
    p.add_argument("--epoch", type=int, default=100)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--edge-feats", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=0.001)
    p.add_argument("--dropout-feat", type=float, default=0.5)
    p.add_argument("--dropout-attn", type=float, default=0.5)
    p.add_argument("--slope", type=float, default=0.05)
    p.add_argument(
        "--aggregator",
        type=str,
        default="SA",
        choices=["SA", "average", "last_fc", "max", "onedimconv"],
    )
    p.add_argument("--SAattDim", type=int, default=3)
    return p.parse_args()


def _print_latex_metrics(tag, acc_list_of_lists, micro_list_of_lists, macro_list_of_lists):
    """acc_list_of_lists: list over seeds of [train, val, test]."""
    acc_list_of_lists = np.asarray(acc_list_of_lists)
    line = ""
    for j in range(3):
        mean, std = _mean_std_ddof1(acc_list_of_lists[:, j])
        line += f"& {mean:.3f} \\pm {std:.3f} "
    print(f"\nACC {tag} (train / val / test):")
    print(line.rstrip() + r" \\")

    micro_list_of_lists = np.asarray(micro_list_of_lists)
    line = ""
    for j in range(3):
        mean, std = _mean_std_ddof1(micro_list_of_lists[:, j])
        line += f"& {mean:.3f} \\pm {std:.3f} "
    print(f"\nMicro-F1 {tag}:")
    print(line.rstrip() + r" \\")

    macro_list_of_lists = np.asarray(macro_list_of_lists)
    line = ""
    for j in range(3):
        mean, std = _mean_std_ddof1(macro_list_of_lists[:, j])
        line += f"& {mean:.3f} \\pm {std:.3f} "
    print(f"\nMacro-F1 {tag}:")
    print(line.rstrip() + r" \\")


if __name__ == "__main__":
    args = parse_args()
    v1, v2 = args.variant
    print(args)

    train_taus, val_taus, test_taus = [], [], []
    acc_a, acc_b = [], []
    micro_a, micro_b = [], []
    macro_a, macro_b = [], []

    for seed in args.seeds:
        print(f"\n{'=' * 60}")
        print(f"  Seed {seed}: train variant {v1}, then variant {v2}")
        print(f"{'=' * 60}")

        print(f"\n--- Variant {v1} ({VARIANT_NAMES[v1]}) ---")
        out1 = train_and_eval_variant(args, v1, seed)
        acc_a.append(out1[5])
        micro_a.append(out1[6])
        macro_a.append(out1[7])

        print(f"\n--- Variant {v2} ({VARIANT_NAMES[v2]}) ---")
        out2 = train_and_eval_variant(args, v2, seed)
        acc_b.append(out2[5])
        micro_b.append(out2[6])
        macro_b.append(out2[7])

        if not np.array_equal(out1[14], out2[14]):
            raise RuntimeError("train_idx differs between variants; cannot compare train tau")
        if not np.array_equal(out1[15], out2[15]):
            raise RuntimeError("val_idx differs between variants; cannot compare val tau")
        if not np.array_equal(out1[16], out2[16]):
            raise RuntimeError("test_idx differs between variants; cannot compare test tau")
        if not np.array_equal(out1[11], out2[11]):
            raise RuntimeError("train labels differ between variants")
        if not np.array_equal(out1[12], out2[12]):
            raise RuntimeError("val labels differ between variants")
        if not np.array_equal(out1[13], out2[13]):
            raise RuntimeError("test labels differ between variants")

        tau = kendall_tau(out1, out2)
        train_taus.append(tau[0])
        val_taus.append(tau[1])
        test_taus.append(tau[2])
        print(
            f"\n  Kendall tau (seed {seed}): "
            f"train={tau[0]:.4f} | val={tau[1]:.4f} | test={tau[2]:.4f}"
        )

    print(f"\n{'=' * 60}")
    print(f"  Summary: variants {v1} vs {v2}, {len(args.seeds)} seed(s)")
    print(f"{'=' * 60}")

    tau_str = ""
    for name, arr in (("Train", train_taus), ("Val", val_taus), ("Test", test_taus)):
        mean, std = _mean_std_ddof1(arr)
        print(f"  {name} Kendall tau: {mean:.4f} +- {std:.4f}")
        tau_str += f"& {mean:.3f} \\pm {std:.3f} "

    print("\nLaTeX (tau):")
    print(tau_str.rstrip() + r" \\")

    _print_latex_metrics(f"variant {v1}", acc_a, micro_a, macro_a)
    _print_latex_metrics(f"variant {v2}", acc_b, micro_b, macro_b)
