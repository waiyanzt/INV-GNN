#!/usr/bin/env python3
"""
IMDb RGCN runner for 4 skip-node-collapsed variants.
python run_IMDB_rgcn_skip.py --variants v1
python run_IMDB_rgcn_skip.py --variants v1,v2,v3,v4 --compare v1,v4

"""

import os
import sys
import time
import random
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scipy.stats import kendalltau
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv


BASE = {
    "v1": "data/preprocessed/IMDB_rgcn_skip_v1",
    "v2": "data/preprocessed/IMDB_rgcn_skip_v2",
    "v3": "data/preprocessed/IMDB_rgcn_skip_v3",
    "v4": "data/preprocessed/IMDB_rgcn_skip_v4",
}


def set_seed(seed: int):
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


class RGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_relations, num_classes):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, hidden_dim, num_relations, num_bases=min(30, num_relations))
        self.conv2 = RGCNConv(hidden_dim, num_classes, num_relations, num_bases=min(30, num_relations))

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index, data.edge_type))
        x = self.conv2(x, data.edge_index, data.edge_type)
        return x


def load_variant(base_dir, device):
    x = torch.load(os.path.join(base_dir, "x.pt"), map_location=device)
    edge_index = torch.load(os.path.join(base_dir, "edge_index.pt"), map_location=device)
    edge_type = torch.load(os.path.join(base_dir, "edge_type.pt"), map_location=device)
    y = torch.load(os.path.join(base_dir, "y.pt"), map_location=device)
    train_mask = torch.load(os.path.join(base_dir, "train_mask.pt"), map_location=device)
    val_mask = torch.load(os.path.join(base_dir, "val_mask.pt"), map_location=device)
    test_mask = torch.load(os.path.join(base_dir, "test_mask.pt"), map_location=device)
    meta = torch.load(os.path.join(base_dir, "meta.pt"), map_location="cpu")

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data, meta


def evaluate(logits, y_true):
    y_pred = logits.argmax(dim=1).cpu().numpy()
    y_true = y_true.cpu().numpy()
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_micro = f1_score(y_true, y_pred, average="micro")
    return acc, f1_macro, f1_micro


def save_test_scores(save_prefix, logits, test_mask):
    probs = torch.softmax(logits[test_mask], dim=1).detach().cpu().numpy()
    df = pd.DataFrame({
        "node_id": np.where(test_mask.cpu().numpy())[0],
        "score_c0": probs[:, 0],
        "score_c1": probs[:, 1],
        "score_c2": probs[:, 2],
    })
    df.to_csv(f"{save_prefix}_scores.csv", index=False)


def kendall_scores_csv(path_a, path_b):
    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)
    m = a.merge(b, on="node_id", suffixes=("_a", "_b"), how="inner")
    if len(m) < 2:
        return float("nan"), len(m)

    taus = []
    for _, row in m.iterrows():
        va = [row["score_c0_a"], row["score_c1_a"], row["score_c2_a"]]
        vb = [row["score_c0_b"], row["score_c1_b"], row["score_c2_b"]]
        tau, _ = kendalltau(va, vb, nan_policy="omit")
        taus.append(tau)
    return float(np.nanmean(taus)), len(m)


def run_one_variant(args, variant, seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, meta = load_variant(BASE[variant], device)
    model = RGCN(
        in_dim=data.x.shape[1],
        hidden_dim=args.hidden_dim,
        num_relations=meta["num_relations"],
        num_classes=meta["num_classes"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = -1.0
    best_state = None
    patience_ctr = 0

    t_train0 = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)

        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data)
            val_acc, _, _ = evaluate(out[data.val_mask], data.y[data.val_mask])

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= args.patience:
            break

    num_epochs = epoch + 1
    train_time_sec = time.perf_counter() - t_train0

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(data)

    test_acc, test_f1_macro, test_f1_micro = evaluate(out[data.test_mask], data.y[data.test_mask])

    save_prefix = f"{args.save_postfix}_{variant}_seed{seed}"
    save_test_scores(save_prefix, out, data.test_mask)

    print(
        f"epochs={num_epochs} train_time_sec={train_time_sec:.3f} | "
        f"Accuracy={test_acc:.6f} F1_macro={test_f1_macro:.6f} F1_micro={test_f1_micro:.6f}",
        flush=True,
    )

    return {
        "Accuracy": float(test_acc),
        "F1_macro": float(test_f1_macro),
        "F1_micro": float(test_f1_micro),
        "epochs": int(num_epochs),
        "train_time_sec": float(train_time_sec),
    }


def summarize(metrics_runs):
    keys = ["Accuracy", "F1_macro", "F1_micro", "epochs", "train_time_sec"]
    print("\n===== Summary over seeds (mean +/- std) =====")
    for k in keys:
        arr = np.array([m[k] for m in metrics_runs], dtype=float)
        if k == "epochs":
            print(f"{k:<22}: {arr.mean():.2f} +/- {arr.std(ddof=0):.2f}")
        elif k == "train_time_sec":
            print(f"{k:<22}: {arr.mean():.3f} +/- {arr.std(ddof=0):.3f}")
        else:
            print(f"{k:<22}: {arr.mean():.6f} +/- {arr.std(ddof=0):.6f}")


def _parse_variants(s):
    out = [x.strip().lower() for x in s.split(",") if x.strip()]
    for v in out:
        if v not in BASE:
            raise SystemExit(f"Unknown variant {v}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="IMDb RGCN runner")
    ap.add_argument("--variants", default="v1,v2,v3,v4")
    ap.add_argument("--seeds", default="1566911444,20241017,20251017")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--save-postfix", default="IMDB_RGCN_skip")
    ap.add_argument("--compare", default="")
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--score-csv-a", default="")
    ap.add_argument("--score-csv-b", default="")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    if args.compare_only:
        if args.score_csv_a and args.score_csv_b:
            tau, n = kendall_scores_csv(args.score_csv_a, args.score_csv_b)
            print(f"Kendall tau: {tau:.6f} (n={n})")
            sys.exit(0)

        pair = _parse_variants(args.compare) if args.compare.strip() else []
        if len(pair) != 2:
            sys.exit("Need --compare v1,v2 or two score csv files.")

        va, vb = pair
        taus = []
        for seed in seeds:
            pa = f"{args.save_postfix}_{va}_seed{seed}_scores.csv"
            pb = f"{args.save_postfix}_{vb}_seed{seed}_scores.csv"
            tau, n = kendall_scores_csv(pa, pb)
            print(f"seed {seed} | n={n} | tau={tau:.6f}")
            taus.append(tau)
        taus = np.array(taus, dtype=float)
        print(f"Mean Kendall tau: {taus.mean():.6f} (std {taus.std(ddof=1) if len(taus) > 1 else 0.0:.6f})")
        sys.exit(0)

    variants = _parse_variants(args.variants)

    score_paths = {}
    for v in variants:
        print(f"\n########## IMDb RGCN variant {v} ##########")
        metrics_runs = []
        score_paths[v] = []
        for seed in seeds:
            stats = run_one_variant(args, v, seed)
            metrics_runs.append(stats)
            score_paths[v].append(f"{args.save_postfix}_{v}_seed{seed}_scores.csv")
        summarize(metrics_runs)

    if args.compare.strip():
        pair = _parse_variants(args.compare)
        if len(pair) != 2:
            raise SystemExit("--compare expects exactly two variants")
        va, vb = pair
        print(f"\n########## Kendall tau | {va} vs {vb} ##########")
        taus = []
        for i, seed in enumerate(seeds):
            tau, n = kendall_scores_csv(score_paths[va][i], score_paths[vb][i])
            print(f"seed {seed} | n={n} | tau={tau:.6f}")
            taus.append(tau)
        taus = np.array(taus, dtype=float)
        print(f"Mean Kendall tau: {taus.mean():.6f} (std {taus.std(ddof=1) if len(taus) > 1 else 0.0:.6f})")