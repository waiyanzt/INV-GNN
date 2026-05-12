#!/usr/bin/env python3
"""
DBLP paper–conference LP with CMPNN on **universal** skip graph
Examples::

  python preprocess_DBLP_cmpnn_skip.py --variant v1,v2,v3
  python run_DBLP_cmpnn_skip.py --variants v1,v2,v3 --save-postfix DBLP_cmpnn_skip
  python run_DBLP_cmpnn_skip.py --compare-only --variants v1,v2,v3 --save-postfix DBLP_cmpnn_skip
"""
import itertools
import os
import sys
import argparse
import random
import time

_PERF_COUNTER = getattr(time, "perf_counter", time.time)
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as torch_data
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import kendalltau
from tqdm import tqdm
from torchdrug import data

sys.path.insert(0, os.path.dirname(__file__))

from cmpnn.model import CMPNN

SEED = 1566911444

# After preprocess, v1/v2/v3 point to identical graphs (replicated for naming).
BASE = {
    "v1": "data/preprocessed/DBLP_cmpnn_skip_v1",
    "v2": "data/preprocessed/DBLP_cmpnn_skip_v2",
    "v3": "data/preprocessed/DBLP_cmpnn_skip_v3",
}


def set_determinism(seed=SEED):
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


def _parse_variants(s):
    out = [x.strip().lower() for x in str(s).split(",") if x.strip()]
    for v in out:
        if v not in BASE:
            raise SystemExit(f"Unknown variant {v!r}; expected one of {','.join(BASE)}")
    return out


def load_preprocessed(base_dir):
    edge_list = torch.load(os.path.join(base_dir, "edge_list.pt"))
    meta = torch.load(os.path.join(base_dir, "meta.pt"))
    return edge_list, meta


def build_graph(edge_list, num_node, num_relation, device):
    return data.Graph(edge_list=edge_list.to(device), num_node=num_node, num_relation=num_relation)


def subsample_negs_per_paper(neg_edges, k, rng):
    by_paper = defaultdict(list)
    for p, c in neg_edges:
        by_paper[int(p)].append((int(p), int(c)))
    out = []
    for p, items in by_paper.items():
        if len(items) <= k:
            out.extend(items)
        else:
            idx = rng.choice(len(items), size=k, replace=False)
            out.extend([items[i] for i in idx])
    return np.array(out, dtype=np.int64)


class DBLPPCQuery(torch_data.Dataset):
    def __init__(self, pos_pairs, neg_dict, off_p, off_c, rel_pc, neg_k):
        self.pos_pairs = pos_pairs
        self.neg_dict = neg_dict
        self.off_p = off_p
        self.off_c = off_c
        self.rel_pc = rel_pc
        self.neg_k = neg_k

    def __len__(self):
        return len(self.pos_pairs)

    def __getitem__(self, i):
        p_local, c_true = self.pos_pairs[i]
        negs = self.neg_dict.get(int(p_local), [])
        if len(negs) == 0:
            negs = [int(c_true)]
        if len(negs) >= self.neg_k:
            negs = negs[:self.neg_k]
        else:
            negs = negs + [negs[-1]] * (self.neg_k - len(negs))

        h = self.off_p + int(p_local)
        t_true = self.off_c + int(c_true)
        t_neg = self.off_c + np.array(negs, dtype=np.int64)
        return h, t_true, t_neg


def build_neg_dict(neg_edges):
    by_paper = defaultdict(list)
    for p, c in neg_edges:
        by_paper[int(p)].append(int(c))
    return by_paper


def collate_query(batch, rel_pc):
    h, t_true, t_neg = zip(*batch)
    h = torch.tensor(h, dtype=torch.long)
    t_true = torch.tensor(np.array(t_true), dtype=torch.long)
    t_neg = torch.tensor(np.stack(t_neg), dtype=torch.long)

    K = t_neg.shape[1]
    h_index = h.unsqueeze(1).expand(-1, 1 + K)
    t_index = torch.cat([t_true.unsqueeze(1), t_neg], dim=1)
    r_index = torch.full_like(h_index, rel_pc)

    y = torch.zeros_like(h_index, dtype=torch.float)
    y[:, 0] = 1.0
    return h_index, t_index, r_index, y


def neg_logsigmoid_loss(scores):
    pos = scores[:, 0]
    neg = scores[:, 1:]
    return -(F.logsigmoid(pos).mean() + F.logsigmoid(-neg).mean())


def _cmpnn_forward_lp(model, graph, h_index, t_index, r_index, device):
    return model(
        graph,
        h_index.to(device),
        t_index.to(device),
        r_index.to(device),
        all_loss=None,
        metric=None,
    )


def find_best_threshold(model, graph, loader, device):
    model.eval()
    all_scores = []
    all_y = []

    with torch.no_grad():
        for h_index, t_index, r_index, y in tqdm(loader, desc="val-threshold", leave=False):
            score = _cmpnn_forward_lp(model, graph, h_index, t_index, r_index, device)
            all_scores.append(score.cpu())
            all_y.append(y)

    all_scores = torch.cat(all_scores, dim=0).numpy().flatten()
    all_y = torch.cat(all_y, dim=0).numpy().flatten()

    # CMPNN logits can be direction-inverted for LP runs.
    # Pick sign (+1 / -1) by **validation AP** on continuous scores (not thresholded F1):
    # with 1 pos + K negs per query, F1 often plateaus at a degenerate optimum (e.g. predict
    # all candidates positive → P=1/(K+1), R=1, fixed F1), and tiny F1 noise can lock the
    # wrong sign if F1 is used to choose direction first.
    signs = (1.0, -1.0)
    # cover both tiny probabilities and standard thresholds
    thresholds = np.unique(
        np.concatenate(
            [
                np.logspace(-8, -1, 16),
                np.arange(0.05, 0.96, 0.05),
            ]
        )
    )

    def _sigmoid(x):
        x = np.clip(x, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-x))

    best_sign = 1.0
    best_rank_ap = -np.inf
    best_rank_auc = -np.inf
    for sign in signs:
        probs = _sigmoid(sign * all_scores)
        ap = average_precision_score(all_y, probs)
        try:
            auc = roc_auc_score(all_y, sign * all_scores)
        except ValueError:
            auc = float("-inf")
        if ap > best_rank_ap + 1e-9 or (
            abs(ap - best_rank_ap) <= 1e-9 and auc > best_rank_auc + 1e-9
        ):
            best_rank_ap = float(ap)
            best_rank_auc = float(auc) if np.isfinite(auc) else float("-inf")
            best_sign = float(sign)

    probs = _sigmoid(best_sign * all_scores)
    best_t, best_f1 = 0.5, -1.0
    best_prec = -1.0
    for t in thresholds:
        pred = (probs >= t).astype(int)
        f1 = f1_score(all_y, pred, zero_division=0)
        prec = precision_score(all_y, pred, zero_division=0)
        if f1 > best_f1 + 1e-6 or (
            abs(f1 - best_f1) <= 1e-6 and prec > best_prec + 1e-6
        ):
            best_f1 = float(f1)
            best_prec = float(prec)
            best_t = float(t)

    return best_t, best_f1, best_sign, best_rank_ap


def evaluate_test(model, graph, loader, device, threshold, score_sign: float, off_paper: int, off_conf: int):
    """Scores CSV uses **local** paper_id / conf_id (same convention as ``run_CMPNN_DBLP_pc.py``)."""
    model.eval()
    all_scores, all_y, all_h, all_t = [], [], [], []

    with torch.no_grad():
        for h_index, t_index, r_index, y in tqdm(loader, desc="test", leave=False):
            score = _cmpnn_forward_lp(model, graph, h_index, t_index, r_index, device)
            all_scores.append(score.cpu())
            all_y.append(y)
            all_h.append(h_index)
            all_t.append(t_index)

    all_scores = torch.cat(all_scores, dim=0)
    all_y = torch.cat(all_y, dim=0)
    all_h = torch.cat(all_h, dim=0)
    all_t = torch.cat(all_t, dim=0)

    effective_scores = score_sign * all_scores
    probs = torch.sigmoid(effective_scores)
    preds = (probs >= threshold).float()

    flat_y = all_y.numpy().flatten()
    flat_p = preds.numpy().flatten()
    flat_s = effective_scores.numpy().flatten()

    precision = precision_score(flat_y, flat_p, zero_division=0)
    recall = recall_score(flat_y, flat_p, zero_division=0)
    f1 = f1_score(flat_y, flat_p, zero_division=0)
    auc = roc_auc_score(flat_y, flat_s)
    ap = average_precision_score(flat_y, flat_s)

    N = all_scores.shape[0]
    hits1 = hits3 = 0.0
    mrr_sum = 0.0

    for i in range(N):
        sc = effective_scores[i]
        pos_score = sc[0]
        greater = int((sc[1:] > pos_score).sum().item())
        equal = int((sc[1:] == pos_score).sum().item())
        rank = 1.0 + greater + 0.5 * equal

        hits1 += float(rank <= 1)
        hits3 += float(rank <= 3)
        mrr_sum += 1.0 / rank

    rows = []
    for i in range(N):
        p_loc = int((all_h[i, 0] - off_paper).item())
        sc = effective_scores[i].numpy()
        raw_sc = all_scores[i].numpy()
        for j in range(sc.shape[0]):
            c_loc = int((all_t[i, j] - off_conf).item())
            rows.append(
                {
                    "paper_id": p_loc,
                    "conf_id": c_loc,
                    "score": float(sc[j]),
                    "raw_score": float(raw_sc[j]),
                    "label": int(j == 0),
                }
            )

    return {
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "AUC": float(auc),
        "AP": float(ap),
        "Hits@1": float(hits1 / N),
        "Hits@3": float(hits3 / N),
        "MRR": float(mrr_sum / N),
        "scores_df": pd.DataFrame(rows),
    }


def kendall_scores_csv(path_a, path_b):
    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)
    m = a.merge(b, on=["paper_id", "conf_id"], suffixes=("_a", "_b"), how="inner")
    if len(m) < 2:
        return float("nan"), len(m)
    tau, _ = kendalltau(m["score_a"], m["score_b"], nan_policy="omit")
    return float(tau), len(m)


def _hits_by_query(df: pd.DataFrame, query_col: str) -> pd.DataFrame:
    if "label" not in df.columns:
        return pd.DataFrame(columns=[query_col, "hit1", "hit3"])
    out = []
    for q, grp in df.groupby(query_col, sort=True):
        g = grp.sort_values("score", ascending=False).reset_index(drop=True)
        pos_idx = g.index[g["label"] == 1].tolist()
        if not pos_idx:
            continue
        rank = int(min(pos_idx)) + 1
        out.append({query_col: int(q), "hit1": int(rank <= 1), "hit3": int(rank <= 3)})
    return pd.DataFrame(out)


def kendall_csv_with_hits(path_a: str, path_b: str, on_cols: list, query_col: str) -> dict:
    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)
    m = a.merge(b, on=on_cols, suffixes=("_a", "_b"), how="inner")
    out = {
        "overall_tau": float("nan"),
        "overall_n": len(m),
        "h1_tau": float("nan"),
        "h3_tau": float("nan"),
        "hits_n": 0,
    }
    if len(m) >= 2:
        tau, _ = kendalltau(m["score_a"], m["score_b"], nan_policy="omit")
        out["overall_tau"] = float(tau) if np.isfinite(tau) else float("nan")
    ha = _hits_by_query(a, query_col).rename(columns={"hit1": "hit1_a", "hit3": "hit3_a"})
    hb = _hits_by_query(b, query_col).rename(columns={"hit1": "hit1_b", "hit3": "hit3_b"})
    hh = ha.merge(hb, on=query_col, how="inner")
    out["hits_n"] = len(hh)
    if len(hh) >= 2:
        t1, _ = kendalltau(hh["hit1_a"], hh["hit1_b"], nan_policy="omit")
        t3, _ = kendalltau(hh["hit3_a"], hh["hit3_b"], nan_policy="omit")
        out["h1_tau"] = float(t1) if np.isfinite(t1) else float("nan")
        out["h3_tau"] = float(t3) if np.isfinite(t3) else float("nan")
    return out


def _variant_pairs_from_csv_spec(compare_or_variants: str) -> list:
    lst = _parse_variants(compare_or_variants)
    if len(lst) < 2:
        raise SystemExit("Need at least two variants (comma-separated), e.g. v1,v2,v3")
    return list(itertools.combinations(lst, 2))


def run_one_variant(args, variant, seed, preprocessed_base=None):
    """Train/eval one variant. If ``preprocessed_base`` is set, load that folder instead of ``BASE[variant]`` (for pc runner)."""
    set_determinism(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Variant={variant} Seed={seed} Device={device}", flush=True)

    base_path = preprocessed_base if preprocessed_base is not None else BASE[variant]
    edge_list, meta = load_preprocessed(base_path)
    offsets = meta["offsets"]
    num_nodes = meta["num_nodes"]
    relation_map = meta["relation_map"]
    splits = {k: v.cpu().numpy() for k, v in meta["splits"].items()}
    print(
        f"Loaded graph: universal_area_channels={meta.get('universal_area_channels', 'paper_conf')}",
        flush=True,
    )

    splits["train_neg"] = subsample_negs_per_paper(splits["train_neg"], args.neg_k, np.random.RandomState(seed + 11))
    splits["val_neg"] = subsample_negs_per_paper(splits["val_neg"], args.neg_k, np.random.RandomState(seed + 13))
    splits["test_neg"] = subsample_negs_per_paper(splits["test_neg"], args.neg_k, np.random.RandomState(seed + 17))
    print(f"Subsampled negatives to max {args.neg_k} per paper", flush=True)
    if args.neg_k > 0 and args.neg_k <= 4:
        print(
            f"[note] Each query has ≤{1 + args.neg_k} venue candidates; MRR/Hits@1 can approach 1.0. "
            f"Use --neg-k 19 for a harder ranking pool.",
            flush=True,
        )

    graph = build_graph(edge_list=edge_list, num_node=num_nodes["total"], num_relation=len(relation_map), device=device)
    rel_pc = relation_map["paper-conference"]

    train_ds = DBLPPCQuery(
        splits["train_pos"],
        build_neg_dict(splits["train_neg"]),
        offsets["paper"],
        offsets["conference"],
        rel_pc,
        args.neg_k,
    )
    val_ds = DBLPPCQuery(
        splits["val_pos"],
        build_neg_dict(splits["val_neg"]),
        offsets["paper"],
        offsets["conference"],
        rel_pc,
        args.neg_k,
    )
    test_ds = DBLPPCQuery(
        splits["test_pos"],
        build_neg_dict(splits["test_neg"]),
        offsets["paper"],
        offsets["conference"],
        rel_pc,
        args.neg_k,
    )

    collate_fn = lambda b: collate_query(b, rel_pc=rel_pc)

    train_loader = torch_data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)
    val_loader = torch_data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    test_loader = torch_data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)

    hidden_dims = [int(x) for x in args.hidden_dims.split(",") if x.strip()]

    model = CMPNN(
        input_dim=args.input_dim,
        hidden_dims=hidden_dims,
        num_relation=len(relation_map),
        message_func=args.message_func,
        aggregate_func=args.aggregate_func,
        short_cut=args.short_cut,
        layer_norm=args.layer_norm,
        activation="relu",
        concat_hidden=False,
        num_mlp_layer=args.num_mlp_layer,
        dependent=True,
        remove_one_hop=True,
        set_boundary=True,
        rgcn=args.rgcn,
        num_bases=args.num_bases,
        initialization=args.initialization,
        has_readout=args.has_readout,
        readout_type=args.readout_type,
        query_specific_readout=args.query_specific_readout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = -1.0
    bad = 0

    train_t0 = _PERF_COUNTER()
    epochs_ran = args.epochs

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0

        for h_index, t_index, r_index, y in tqdm(train_loader, desc=f"train e{epoch}", leave=False):
            score = _cmpnn_forward_lp(model, graph, h_index, t_index, r_index, device)
            loss = neg_logsigmoid_loss(score)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        best_t, best_val_f1, best_sign, best_val_ap = find_best_threshold(model, graph, val_loader, device)

        print(
            f"Epoch {epoch:03d} | Loss {avg_loss:.4f} | "
            f"ValBestF1 {best_val_f1:.4f} | ValBestAP {best_val_ap:.4f} | "
            f"ValBestT {best_t:.2e} | ScoreSign {int(best_sign):+d} | "
            f"time {time.time()-t0:.1f}s",
            flush=True,
        )

        ckpt = args.ckpt.replace(".pt", f"_{variant}_seed{seed}.pt")
        if best_val_f1 > best_val + 1e-6:
            best_val = best_val_f1
            bad = 0
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            torch.save({"model": model.state_dict(), "threshold": best_t, "score_sign": best_sign}, ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping!", flush=True)
                epochs_ran = epoch + 1
                break

    train_wall_sec = float(_PERF_COUNTER() - train_t0)

    ckpt = args.ckpt.replace(".pt", f"_{variant}_seed{seed}.pt")
    saved = torch.load(ckpt, map_location=device)
    model.load_state_dict(saved["model"])
    best_t = float(saved["threshold"])
    best_sign = float(saved.get("score_sign", 1.0))

    test_results = evaluate_test(
        model,
        graph,
        test_loader,
        device,
        best_t,
        best_sign,
        int(offsets["paper"]),
        int(offsets["conference"]),
    )

    if getattr(args, "save_postfix", ""):
        csv_path = f"{args.save_postfix}_{variant}_seed{seed}_scores.csv"
        test_results["scores_df"].to_csv(csv_path, index=False)
        print(f"Scores saved to {csv_path}", flush=True)

    print(
        f"Best threshold={best_t:.2e} | ScoreSign={int(best_sign):+d} | "
        f"Test Precision={test_results['Precision']:.6f} "
        f"Recall={test_results['Recall']:.6f} "
        f"F1={test_results['F1']:.6f} "
        f"AUC={test_results['AUC']:.6f} "
        f"AP={test_results['AP']:.6f} "
        f"Hits@1={test_results['Hits@1']:.6f} "
        f"Hits@3={test_results['Hits@3']:.6f} "
        f"MRR={test_results['MRR']:.6f} | "
        f"Train Time (s)={train_wall_sec:.2f} Epochs={epochs_ran}",
        flush=True,
    )

    out = {k: v for k, v in test_results.items() if k != "scores_df"}
    out["Train Time (s)"] = float(train_wall_sec)
    out["Epochs"] = float(epochs_ran)
    return out


def summarize(metrics_list):
    if not metrics_list:
        return
    keys = [
        "Precision", "Recall", "F1", "AUC", "AP", "Hits@1", "Hits@3", "MRR",
        "Train Time (s)", "Epochs",
    ]
    print("\n===== Summary over seeds (mean ± std) =====")
    for k in keys:
        if k not in metrics_list[0]:
            continue
        arr = np.array([m[k] for m in metrics_list], dtype=float)
        if k in ("Train Time (s)", "Epochs"):
            print(f"{k:<22}: {arr.mean():.2f} ± {arr.std(ddof=0):.2f}")
        else:
            print(f"{k:<22}: {arr.mean():.6f} ± {arr.std(ddof=0):.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DBLP CMPNN universal skip link prediction")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--variants", default="v1,v2,v3")
    ap.add_argument("--input-dim", type=int, default=32)
    ap.add_argument("--hidden-dims", default="32,32")
    ap.add_argument("--message-func", default="distmult")
    ap.add_argument("--aggregate-func", default="pna")
    ap.add_argument("--short-cut", action="store_true")
    ap.add_argument("--layer-norm", action="store_true")
    ap.add_argument("--num-mlp-layer", type=int, default=2)
    ap.add_argument("--rgcn", action="store_true")
    ap.add_argument("--num-bases", type=int, default=None)
    ap.add_argument("--initialization", default="Query")
    ap.add_argument("--has-readout", action="store_true")
    ap.add_argument("--readout-type", default="mean")
    ap.add_argument("--query-specific-readout", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="LP often needs more epochs than small defaults; tune with val F1.",
    )
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--neg-k",
        type=int,
        default=3,
        help="Max negatives per paper (train/val/test). Small k (e.g. 3) ⇒ few candidates ⇒ MRR/Hits@1 can saturate.",
    )
    ap.add_argument("--ckpt", default="checkpoint/dblp_cmpnn_skip.pt")
    ap.add_argument("--seeds", default="1566911444,20241017,20251017")
    ap.add_argument("--save-postfix", default="DBLP_cmpnn_skip")
    ap.add_argument(
        "--compare",
        default="",
        help="Comma-separated variants for Kendall after training, e.g. v1,v2,v3 (all pairs) or v1,v3.",
    )
    ap.add_argument(
        "--compare-only",
        action="store_true",
        help="Only Kendall from existing CSVs; use --compare or default --variants for variant list.",
    )
    ap.add_argument("--score-csv-a", default="")
    ap.add_argument("--score-csv-b", default="")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    merge_keys = ["paper_id", "conf_id"]
    query_col = "paper_id"

    def _print_kendall_block(va: str, vb: str):
        o_taus, h1_taus, h3_taus = [], [], []
        for sd in seeds:
            pa = f"{args.save_postfix}_{va}_seed{sd}_scores.csv"
            pb = f"{args.save_postfix}_{vb}_seed{sd}_scores.csv"
            if not os.path.isfile(pa) or not os.path.isfile(pb):
                print(f"  seed {sd}: missing {pa} or {pb}", flush=True)
                continue
            kk = kendall_csv_with_hits(pa, pb, merge_keys, query_col)
            print(
                f"  seed {sd} | pair_rows={kk['overall_n']} | overall_τ={kk['overall_tau']:.6f} | "
                f"papers={kk['hits_n']} | Hits@1_τ={kk['h1_tau']:.6f} | Hits@3_τ={kk['h3_tau']:.6f}",
                flush=True,
            )
            if np.isfinite(kk["overall_tau"]):
                o_taus.append(float(kk["overall_tau"]))
            if np.isfinite(kk["h1_tau"]):
                h1_taus.append(float(kk["h1_tau"]))
            if np.isfinite(kk["h3_tau"]):
                h3_taus.append(float(kk["h3_tau"]))
        if o_taus:
            oa = np.array(o_taus, dtype=float)
            msg = f"  Mean overall τ: {oa.mean():.6f} ± {oa.std(ddof=0):.6f}"
            if h1_taus:
                a1 = np.array(h1_taus, dtype=float)
                msg += f" | Hits@1 τ: {a1.mean():.6f} ± {a1.std(ddof=0):.6f}"
            if h3_taus:
                a3 = np.array(h3_taus, dtype=float)
                msg += f" | Hits@3 τ: {a3.mean():.6f} ± {a3.std(ddof=0):.6f}"
            print(msg, flush=True)

    if args.compare_only:
        if args.score_csv_a and args.score_csv_b:
            kk = kendall_csv_with_hits(args.score_csv_a, args.score_csv_b, merge_keys, query_col)
            print(
                f"pair_rows={kk['overall_n']} | overall_τ={kk['overall_tau']:.6f} | "
                f"papers={kk['hits_n']} | Hits@1_τ={kk['h1_tau']:.6f} | Hits@3_τ={kk['h3_tau']:.6f}",
                flush=True,
            )
            sys.exit(0)

        spec = args.compare.strip() if (args.compare and str(args.compare).strip()) else args.variants
        pairs = _variant_pairs_from_csv_spec(spec)
        print(f"\n########## Kendall τ (overall + per-paper Hits@1/Hits@3) | spec={spec!r} ##########")
        for va, vb in pairs:
            print(f"\n--- {va} vs {vb} ---", flush=True)
            _print_kendall_block(va, vb)
        sys.exit(0)

    variants = _parse_variants(args.variant) if args.variant else _parse_variants(args.variants)

    by_v = {}
    for v in variants:
        print(f"\n########## DBLP CMPNN skip variant {v} ##########", flush=True)
        metrics_runs = []
        for seed in seeds:
            stats = run_one_variant(args, v, seed)
            metrics_runs.append(stats)
        by_v[v] = metrics_runs
        summarize(metrics_runs)

    print("\nDBLP CMPNN skip summary | mean ± std over seeds")
    print("Variant | Precision | Recall | F1 | Hits@1 | Hits@3 | MRR | Train Time (s) | Epochs")
    for v in variants:
        mets = by_v[v]

        def mstd(key):
            arr = np.array([x[key] for x in mets], dtype=float)
            return float(arr.mean()), float(arr.std(ddof=0))

        p_m, p_s = mstd("Precision")
        r_m, r_s = mstd("Recall")
        f_m, f_s = mstd("F1")
        h1_m, h1_s = mstd("Hits@1")
        h3_m, h3_s = mstd("Hits@3")
        m_m, m_s = mstd("MRR")
        tw_m, tw_s = mstd("Train Time (s)")
        e_m, e_s = mstd("Epochs")
        print(
            f"{v:>7} | {p_m:.4f} ± {p_s:.4f} | {r_m:.4f} ± {r_s:.4f} | {f_m:.4f} ± {f_s:.4f} | "
            f"{h1_m:.4f} ± {h1_s:.4f} | {h3_m:.4f} ± {h3_s:.4f} | {m_m:.4f} ± {m_s:.4f} | "
            f"{tw_m:.2f} ± {tw_s:.2f} | {e_m:.2f} ± {e_s:.2f}",
            flush=True,
        )

    if args.compare.strip():
        pairs = _variant_pairs_from_csv_spec(args.compare.strip())
        print(f"\n########## Kendall τ after training | compare={args.compare!r} ##########")
        for va, vb in pairs:
            print(f"\n--- {va} vs {vb} ---", flush=True)
            _print_kendall_block(va, vb)