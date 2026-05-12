#!/usr/bin/env python3
from __future__ import annotations

"""
MAGNN IMDB link prediction (skip-node / universal graph).

This runner mirrors `run_IMDB_magnn_lp.py` but loads data from
`data/preprocessed/IMDB_magnn_lp_skip_{task}_{variant}` produced by
`preprocess_IMDB_magnn_lp_skip.py`.

Because preprocessing canonicalizes semantic channels to v1 specs, all variants
should be identical (for fixed seeds), so Kendall-tau cross-variant comparisons
should approach 1.0.
"""

import argparse
import itertools
import os
import pickle
import random
import time
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from model import MAGNN_lp
from utils.tools import index_generator, parse_lp_minibatch

dropout_rate = 0.5
lr = 0.005
weight_decay = 0.001


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


def parse_task(s: str) -> str:
    t = s.strip().lower()
    if t not in ("md", "ml"):
        raise SystemExit("task must be md or ml")
    return t


def parse_variants(s: str, task: str):
    vs = [x.strip().lower() for x in s.split(",") if x.strip()]
    good = {"v1", "v3"} if task == "md" else {"v1", "v2", "v3", "v4"}
    for v in vs:
        if v not in good:
            raise SystemExit(f"invalid variant {v} for task={task}")
    return vs


def _path(task: str, variant: str) -> str:
    return f"data/preprocessed/IMDB_magnn_lp_skip_{task}_{variant}"


def _tail_col(task: str) -> str:
    return {"md": "director_local", "ml": "link_local"}[task]


def _csr_is_sparse_eye(x: sp.csr_matrix) -> bool:
    return x.shape[0] == x.shape[1] and x.nnz == x.shape[0]


def csr_features_to_torch(x: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    if _csr_is_sparse_eye(x):
        n = x.shape[0]
        return torch.eye(n, device=device, dtype=torch.float32)
    return torch.as_tensor(x.toarray(), device=device, dtype=torch.float32)


def feature_in_dim(x: sp.csr_matrix) -> int:
    return int(x.shape[0]) if _csr_is_sparse_eye(x) else int(x.shape[1])


def load_lp_data(base: str):
    type_mask = np.load(os.path.join(base, "node_types.npy"))
    pos = np.load(os.path.join(base, "train_val_test_pos.npz"))
    neg = np.load(os.path.join(base, "train_val_test_neg.npz"))
    with open(os.path.join(base, "config.pkl"), "rb") as f:
        cfg = pickle.load(f)

    adjlists = [[], []]
    mp_dicts = [[], []]
    for mode in (0, 1):
        for mp in cfg["metapaths"][mode]:
            name = "-".join(map(str, mp))
            adjf = os.path.join(base, str(mode), f"{name}.adjlist")
            idxf = os.path.join(base, str(mode), f"{name}_idx.pickle")
            with open(adjf, "r") as f:
                lines = [ln.rstrip("\n") for ln in f]
            with open(idxf, "rb") as f:
                d = pickle.load(f)
            adjlists[mode].append(lines)
            mp_dicts[mode].append(d)

    feats = []
    for t in range(cfg["num_ntypes"]):
        feats.append(sp.load_npz(os.path.join(base, f"features_{t}.npz")))

    return (
        cfg,
        adjlists,
        mp_dicts,
        feats,
        type_mask,
        pos["train_pos"],
        pos["val_pos"],
        pos["test_pos"],
        neg["train_neg"],
        neg["val_neg"],
        neg["test_neg"],
    )


def _mstd(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))


def _precision_recall_f1_best(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float, float]:
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    j = int(np.nanargmax(f1))
    return float(prec[j]), float(rec[j]), float(f1[j])


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


def kendall_with_hits(path_a: str, path_b: str, query_col: str, tail_col: str) -> dict:
    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)
    m = a.merge(b, on=[query_col, tail_col], suffixes=("_a", "_b"), how="inner")
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


def run_one(args, task: str, variant: str, seed: int) -> dict[str, Any]:
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base = _path(task, variant)
    (
        cfg,
        adjlists,
        mp_dicts,
        features_list_raw,
        type_mask,
        train_pos,
        val_pos,
        test_pos,
        train_neg,
        val_neg,
        test_neg,
    ) = load_lp_data(base)
    print(
        f"task={task} variant={variant} seed={seed} | device={device} | "
        f"train_pos={len(train_pos)} val_pos={len(val_pos)} test_pos={len(test_pos)}",
        flush=True,
    )

    features_list = []
    in_dims = []
    for feat in features_list_raw:
        in_dims.append(feature_in_dim(feat))
        features_list.append(csr_features_to_torch(feat, device))

    net = MAGNN_lp(
        [len(cfg["metapaths"][0]), len(cfg["metapaths"][1])],
        cfg["num_etypes"],
        cfg["etypes"],
        in_dims,
        args.hidden_dim,
        args.hidden_dim,
        args.num_heads,
        args.attn_vec_dim,
        args.rnn_type,
        dropout_rate,
        num_layers=args.K,
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    train_gen = index_generator(batch_size=args.batch_size, num_data=len(train_pos))
    val_gen = index_generator(batch_size=args.batch_size, num_data=len(val_pos), shuffle=False)
    train_iters = train_gen.num_iterations()
    val_iters = val_gen.num_iterations()
    best = float("inf")
    bad = 0
    ckpt = f"checkpoint/imdb_magnn_lp_skip_{task}_{variant}_seed{seed}.pt"
    os.makedirs("checkpoint", exist_ok=True)

    train_t0 = time.perf_counter()
    epochs_ran = args.epoch
    for ep in range(args.epoch):
        ep_t0 = time.perf_counter()
        net.train()
        losses = []
        log_every = max(1, train_iters // 5)
        for it in range(train_iters):
            idx = train_gen.next()
            pos = train_pos[idx]
            neg = np.column_stack([np.repeat(pos[:, 0], train_neg.shape[1]), train_neg[idx].ravel()])
            g_pos, idx_pos, map_pos = parse_lp_minibatch(adjlists, mp_dicts, pos.tolist(), device, samples=args.samples)
            g_neg, idx_neg, map_neg = parse_lp_minibatch(adjlists, mp_dicts, neg.tolist(), device, samples=args.samples)
            [pl, pr], _ = net((g_pos, features_list, type_mask, idx_pos, map_pos))
            [nl, nr], _ = net((g_neg, features_list, type_mask, idx_neg, map_neg))
            pl = pl.view(-1, 1, pl.shape[1])
            pr = pr.view(-1, pr.shape[1], 1)
            nl = nl.view(-1, 1, nl.shape[1])
            nr = nr.view(-1, nr.shape[1], 1)
            pos_out = torch.bmm(pl, pr)
            neg_out = -torch.bmm(nl, nr)
            loss = -(F.logsigmoid(pos_out).mean() + F.logsigmoid(neg_out).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if (it + 1) % log_every == 0 or (it + 1) == train_iters:
                elapsed = time.perf_counter() - ep_t0
                print(
                    f"task={task} var={variant} seed={seed} epoch={ep:03d} "
                    f"[train {it+1}/{train_iters}] loss={np.mean(losses):.4f} elapsed={elapsed:.1f}s",
                    flush=True,
                )

        net.eval()
        vals = []
        with torch.no_grad():
            for vit in range(val_iters):
                idx = val_gen.next()
                pos = val_pos[idx]
                neg = np.column_stack([np.repeat(pos[:, 0], val_neg.shape[1]), val_neg[idx].ravel()])
                g_pos, idx_pos, map_pos = parse_lp_minibatch(adjlists, mp_dicts, pos.tolist(), device, samples=args.samples)
                g_neg, idx_neg, map_neg = parse_lp_minibatch(adjlists, mp_dicts, neg.tolist(), device, samples=args.samples)
                [pl, pr], _ = net((g_pos, features_list, type_mask, idx_pos, map_pos))
                [nl, nr], _ = net((g_neg, features_list, type_mask, idx_neg, map_neg))
                pl = pl.view(-1, 1, pl.shape[1])
                pr = pr.view(-1, pr.shape[1], 1)
                nl = nl.view(-1, 1, nl.shape[1])
                nr = nr.view(-1, nr.shape[1], 1)
                pos_out = torch.bmm(pl, pr)
                neg_out = -torch.bmm(nl, nr)
                vals.append((-(F.logsigmoid(pos_out).mean() + F.logsigmoid(neg_out).mean())).item())
        vloss = float(np.mean(vals)) if vals else 0.0
        print(
            f"task={task} var={variant} seed={seed} epoch={ep:03d} "
            f"train={np.mean(losses):.4f} val={vloss:.4f} epoch_time={time.perf_counter()-ep_t0:.1f}s",
            flush=True,
        )
        if vloss < best - 1e-6:
            best = vloss
            bad = 0
            torch.save(net.state_dict(), ckpt)
            print(f"  new best val={best:.4f} -> saved {ckpt}", flush=True)
        else:
            bad += 1
            if bad >= args.patience:
                epochs_ran = ep + 1
                print(f"  early stopping at epoch {ep:03d}", flush=True)
                break

    train_wall = time.perf_counter() - train_t0
    print(f"training done | wall={train_wall:.1f}s | epochs_ran={epochs_ran}", flush=True)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()

    # score positives
    pos_scores = []
    with torch.no_grad():
        gen = index_generator(batch_size=args.batch_size, num_data=len(test_pos), shuffle=False)
        while gen.num_iterations_left() > 0:
            idx = gen.next()
            g, ii, mm = parse_lp_minibatch(adjlists, mp_dicts, test_pos[idx].tolist(), device, samples=args.samples)
            [l, r], _ = net((g, features_list, type_mask, ii, mm))
            o = torch.bmm(l.view(-1, 1, l.shape[1]), r.view(-1, r.shape[1], 1)).flatten()
            pos_scores.append(torch.sigmoid(o).cpu().numpy())
    pos_scores = np.concatenate(pos_scores) if pos_scores else np.array([], dtype=float)

    # score negatives
    neg_pairs = np.column_stack([np.repeat(test_pos[:, 0], test_neg.shape[1]), test_neg.ravel()])
    neg_scores = []
    with torch.no_grad():
        gen = index_generator(batch_size=args.batch_size, num_data=len(neg_pairs), shuffle=False)
        while gen.num_iterations_left() > 0:
            idx = gen.next()
            g, ii, mm = parse_lp_minibatch(adjlists, mp_dicts, neg_pairs[idx].tolist(), device, samples=args.samples)
            [l, r], _ = net((g, features_list, type_mask, ii, mm))
            o = torch.bmm(l.view(-1, 1, l.shape[1]), r.view(-1, r.shape[1], 1)).flatten()
            neg_scores.append(torch.sigmoid(o).cpu().numpy())
    neg_scores = np.concatenate(neg_scores) if neg_scores else np.array([], dtype=float)

    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_prob = np.concatenate([pos_scores, neg_scores])
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    y_pred = (y_prob >= args.threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision_bf1, recall_bf1, f1_bf1 = _precision_recall_f1_best(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)

    # ranking metrics per query
    # Break equal-score ties deterministically to avoid stable-sort bias.
    rng_rank = np.random.default_rng(seed + 1337)
    Kneg = test_neg.shape[1]
    h1 = h3 = h5 = 0
    rr = 0.0
    n_q = 0
    for i in range(len(test_pos)):
        s_true = float(pos_scores[i])
        items = [(s_true, int(test_pos[i, 1]), 1)]
        for j in range(Kneg):
            items.append((float(neg_scores[i * Kneg + j]), int(test_neg[i, j]), 0))
        rng_rank.shuffle(items)
        items.sort(key=lambda x: x[0], reverse=True)
        ranks = [idx + 1 for idx, (_, __, is_pos) in enumerate(items) if is_pos == 1]
        if not ranks:
            continue
        rank = min(ranks)
        n_q += 1
        rr += 1.0 / rank
        h1 += int(rank <= 1)
        h3 += int(rank <= 3)
        h5 += int(rank <= 5)
    hits1, hits3, hits5, mrr = h1 / max(n_q, 1), h3 / max(n_q, 1), h5 / max(n_q, 1), rr / max(n_q, 1)

    tail_col = _tail_col(task)
    rows = []
    for i, (m, t) in enumerate(test_pos):
        rows.append({"movie_local": int(m), tail_col: int(t), "score": float(pos_scores[i]), "label": 1})
        for j in range(Kneg):
            rows.append({"movie_local": int(m), tail_col: int(test_neg[i, j]), "score": float(neg_scores[i * Kneg + j]), "label": 0})
    csv = f"{args.save_postfix}_skip_{task}_{variant}_seed{seed}_scores.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"saved score csv: {csv}", flush=True)

    return {
        "AUC": float(auc),
        "AP": float(ap),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "Precision_best_f1": float(precision_bf1),
        "Recall_best_f1": float(recall_bf1),
        "F1_best_f1": float(f1_bf1),
        "Accuracy": float(acc),
        "Hits@1": float(hits1),
        "Hits@3": float(hits3),
        "Hits@5": float(hits5),
        "MRR": float(mrr),
        "Train Time (s)": float(train_wall),
        "Epochs": float(epochs_ran),
    }


def summarize(runs):
    keys = [
        "AUC",
        "AP",
        "Precision",
        "Recall",
        "F1",
        "Precision_best_f1",
        "Recall_best_f1",
        "F1_best_f1",
        "Accuracy",
        "Hits@1",
        "Hits@3",
        "Hits@5",
        "MRR",
        "Train Time (s)",
        "Epochs",
    ]
    print("\n===== Summary over seeds (mean ± std) =====")
    for k in keys:
        m, s = _mstd([r[k] for r in runs])
        if k in ("Train Time (s)", "Epochs"):
            print(f"{k:<22}: {m:.2f} ± {s:.2f}")
        else:
            print(f"{k:<22}: {m:.6f} ± {s:.6f}")


def main():
    ap = argparse.ArgumentParser(description="MAGNN IMDB link prediction (skip-node, md/ml).")
    ap.add_argument("--task", required=True)
    ap.add_argument("--variants", default="", help="default: v1,v3 for md; v1,v2,v3,v4 for ml")
    ap.add_argument("--epoch", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--attn-vec-dim", type=int, default=128)
    ap.add_argument("--rnn-type", default="RotatE0")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seeds", default="1566911444,20241017,20251017")
    ap.add_argument("--save-postfix", default="IMDB_magnn_lp")
    ap.add_argument("--compare-only", action="store_true")
    args = ap.parse_args()

    task = parse_task(args.task)
    vstr = args.variants.strip() or ("v1,v3" if task == "md" else "v1,v2,v3,v4")
    variants = parse_variants(vstr, task)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    if args.compare_only:
        tcol = _tail_col(task)
        print(f"\nComparison Kendall-Tau (task={task}, skip)")
        for va, vb in itertools.combinations(variants, 2):
            taus = []
            for sd in seeds:
                pa = f"{args.save_postfix}_skip_{task}_{va}_seed{sd}_scores.csv"
                pb = f"{args.save_postfix}_skip_{task}_{vb}_seed{sd}_scores.csv"
                if not os.path.isfile(pa) or not os.path.isfile(pb):
                    print(f"{va} vs {vb} seed {sd}: missing CSV")
                    continue
                kk = kendall_with_hits(pa, pb, query_col="movie_local", tail_col=tcol)
                print(f"{va} vs {vb} seed {sd}: n={kk['overall_n']} tau={kk['overall_tau']:.6f}")
                if np.isfinite(kk["overall_tau"]):
                    taus.append(float(kk["overall_tau"]))
            if taus:
                mm, ss = _mstd(taus)
                print(f"{va} vs {vb}: overall {mm:.6f} ± {ss:.6f}")
        return

    for v in variants:
        print(f"\n########## task={task} variant={v} (skip) ##########")
        runs = []
        for sd in seeds:
            runs.append(run_one(args, task, v, sd))
        summarize(runs)


if __name__ == "__main__":
    main()

