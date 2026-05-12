#!/usr/bin/env python3
"""
  python run_IMDB_rgcn_lp.py --task ml/md --variant v1,v2,v3,v4 
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
import time

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import RelGraphConv
from scipy.stats import kendalltau
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SEED = 1566911444


def set_determinism(seed: int = SEED) -> None:
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


def _parse_task(s: str) -> str:
    t = str(s).strip().lower()
    if t not in {"md", "mg", "ml"}:
        raise SystemExit("task must be md, mg, or ml")
    return t


def _parse_variants(s: str, task: str) -> list[str]:
    out = [x.strip().lower() for x in str(s).split(",") if x.strip()]
    if task == "md":
        good = {"v1", "v3"}
    else:
        good = {"v1", "v2", "v3", "v4"}
    for v in out:
        if v not in good:
            raise SystemExit(f"Invalid variant {v!r} for task={task}; expected subset of {sorted(good)}")
    return out


def base_dir(task: str, variant: str) -> str:
    return f"data/preprocessed/IMDB_rgcn_lp_{task}_{variant}"


def to_homo_with_indexers(g):
    hg = dgl.to_homogeneous(g)
    etypes = hg.edata.get(dgl.ETYPE, hg.edata["_TYPE"]).long()
    ntype = hg.ndata.get(dgl.NTYPE, hg.ndata["_TYPE"]).long()
    nid = hg.ndata.get(dgl.NID, hg.ndata["_ID"]).long()
    indexers = {}
    for name in g.ntypes:
        tid = g.get_ntype_id(name)
        mask = ntype == tid
        homo_idx = mask.nonzero(as_tuple=False).squeeze(1)
        lid = nid[mask]
        order = torch.argsort(lid)
        indexers[name] = homo_idx[order]
    return hg, etypes, indexers


def build_heterograph(graph_data: dict, num_nodes: dict) -> dgl.DGLGraph:
    data = {}

    def bi(srctype, fname, dsttype, rname, key):
        if key not in graph_data:
            return
        u, v = graph_data[key]
        data[(srctype, fname, dsttype)] = (u, v)
        data[(dsttype, rname, srctype)] = (v, u)

    bi("movie", "movie-actor", "actor", "actor-movie", "movie-actor")
    bi("movie", "movie-director", "director", "director-movie", "movie-director")
    bi("movie", "movie-link", "link", "link-movie", "movie-link")
    bi("movie", "movie-genre", "genre", "genre-movie", "movie-genre")
    bi("link", "link-director", "director", "director-link", "link-director")
    bi("link", "link-actor", "actor", "actor-link", "link-actor")

    return dgl.heterograph(data, num_nodes_dict=num_nodes)


class RGCNEncoder(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_rels: int,
        in_dim: int = 128,
        hid_dim: int = 256,
        out_dim: int = 256,
        num_layers: int = 3,
        num_bases: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, in_dim)
        nn.init.xavier_uniform_(self.emb.weight)

        def mk(din, dout, act, self_loop=True, layer_dropout=0.0):
            try:
                return RelGraphConv(
                    din,
                    dout,
                    num_rels,
                    regularizer="basis",
                    num_bases=num_bases,
                    self_loop=self_loop,
                    dropout=layer_dropout,
                    activation=act,
                    low_mem=True,
                )
            except TypeError:
                try:
                    return RelGraphConv(
                        din,
                        dout,
                        num_rels,
                        regularizer="basis",
                        num_bases=num_bases,
                        self_loop=self_loop,
                        dropout=layer_dropout,
                        activation=act,
                    )
                except TypeError:
                    return RelGraphConv(
                        din,
                        dout,
                        num_rels,
                        regularizer="basis",
                        num_bases=num_bases,
                        self_loop=self_loop,
                        activation=act,
                    )

        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(mk(in_dim, out_dim, act=None, layer_dropout=dropout))
        else:
            self.layers.append(mk(in_dim, hid_dim, act=F.relu, layer_dropout=dropout))
            for _ in range(num_layers - 2):
                self.layers.append(mk(hid_dim, hid_dim, act=F.relu, layer_dropout=dropout))
            self.layers.append(mk(hid_dim, out_dim, act=None, layer_dropout=dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, hg, etypes):
        h = self.emb.weight
        for layer in self.layers:
            h = layer(hg, h, etypes)
        return self.dropout(h)


def pairwise_loss(pos_logit, neg_logit):
    return -(F.logsigmoid(pos_logit).mean() + F.logsigmoid(-neg_logit).mean())


def load_preprocessed(path: str):
    return torch.load(os.path.join(path, "graph_data.pt")), torch.load(os.path.join(path, "meta.pt"))


def _tail_ntype(task: str) -> str:
    return {"md": "director", "mg": "genre", "ml": "link"}[task]


def _kendall_merge_keys(task: str) -> list[str]:
    return {
        "md": ["movie_local", "director_local"],
        "mg": ["movie_local", "genre_id"],
        "ml": ["movie_local", "link_local"],
    }[task]


@torch.no_grad()
def eval_ranking(test_pos: np.ndarray, test_neg: np.ndarray, P: torch.Tensor, T: torch.Tensor):
    """Hits@k / MRR: one true tail per movie row vs K negatives (CMPNN-style ranking)."""
    pos_s = torch.sigmoid((P[test_pos[:, 0]] * T[test_pos[:, 1]]).sum(-1)).cpu().numpy()
    K = test_neg.shape[1]
    neg_s = torch.sigmoid(
        (P[np.repeat(test_pos[:, 0], K)] * T[test_neg.ravel()]).sum(-1)
    ).cpu().numpy().reshape(len(test_pos), K)

    hits1 = hits3 = hits5 = 0
    rr_sum = 0.0
    n_q = 0
    for i in range(len(test_pos)):
        s_true = float(pos_s[i])
        items = [(s_true, int(test_pos[i, 1]), 1)]
        for j in range(K):
            items.append((float(neg_s[i, j]), int(test_neg[i, j]), 0))
        items.sort(key=lambda x: x[0], reverse=True)
        ranks = [idx + 1 for idx, (_, _, is_pos) in enumerate(items) if is_pos == 1]
        if not ranks:
            continue
        r = min(ranks)
        n_q += 1
        rr_sum += 1.0 / r
        if r <= 1:
            hits1 += 1
        if r <= 3:
            hits3 += 1
        if r <= 5:
            hits5 += 1

    return (
        hits1 / max(n_q, 1),
        hits3 / max(n_q, 1),
        hits5 / max(n_q, 1),
        rr_sum / max(n_q, 1),
        n_q,
    )


def kendall_csv(path_a: str, path_b: str, on_cols: list[str]):
    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)
    m = a.merge(b, on=on_cols, suffixes=("_a", "_b"), how="inner")
    if len(m) < 2:
        return float("nan"), len(m)
    tau, _ = kendalltau(m["score_a"], m["score_b"], nan_policy="omit")
    return (float(tau) if np.isfinite(tau) else float("nan"), len(m))


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


def kendall_csv_with_hits(path_a: str, path_b: str, on_cols: list[str], query_col: str) -> dict:
    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)
    m = a.merge(b, on=on_cols, suffixes=("_a", "_b"), how="inner")
    out = {"overall_tau": float("nan"), "overall_n": len(m), "h1_tau": float("nan"), "h3_tau": float("nan"), "hits_n": 0}
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


def run_one(args, task: str, variant: str, seed: int) -> dict:
    set_determinism(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"task={task} variant={variant} seed={seed} device={device}", flush=True)

    bdir = base_dir(task, variant)
    graph_data, meta = load_preprocessed(bdir)
    num_nodes = meta["num_nodes"]
    splits = {k: v.cpu().numpy() for k, v in meta["splits"].items()}

    train_pos = splits["train_pos"]
    train_neg = splits["train_neg"]
    val_pos = splits["val_pos"]
    val_neg = splits["val_neg"]
    test_pos = splits["test_pos"]
    test_neg = splits["test_neg"]

    K = train_neg.shape[1]
    print(
        f"Splits: train_pos={len(train_pos)} val_pos={len(val_pos)} test_pos={len(test_pos)} | neg_k={K}",
        flush=True,
    )

    g = build_heterograph(graph_data, num_nodes)
    hg, etypes, indexers = to_homo_with_indexers(g)
    hg = hg.to(device)
    etypes = etypes.to(device)

    idx_movie = indexers["movie"].to(device)
    tail_name = _tail_ntype(task)
    idx_tail = indexers[tail_name].to(device)

    enc = RGCNEncoder(
        num_nodes=hg.num_nodes(),
        num_rels=int(etypes.max().item()) + 1,
        in_dim=args.in_dim,
        hid_dim=args.hid_dim,
        out_dim=args.out_dim,
        num_layers=args.layers,
        num_bases=args.num_bases,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    iters = max(1, int(np.ceil(len(train_pos) / args.batch_size)))
    best_val = float("inf")
    bad = 0
    train_t0 = time.perf_counter()
    epochs_ran = args.epochs

    ckpt = args.ckpt.replace(".pt", f"_{task}_{variant}_seed{seed}.pt")

    for epoch in range(args.epochs):
        enc.train()
        perm = np.random.permutation(len(train_pos))
        running = 0.0
        t0 = time.time()

        for it in range(iters):
            sl = perm[it * args.batch_size : (it + 1) * args.batch_size]
            if len(sl) == 0:
                continue
            pos = train_pos[sl]
            neg_m = train_neg[sl]
            B, Kb = neg_m.shape
            neg_flat = np.column_stack([np.repeat(pos[:, 0], Kb), neg_m.ravel()])

            opt.zero_grad()
            H = enc(hg, etypes)
            P = H[idx_movie]
            Tn = H[idx_tail]
            pos_logit = (P[pos[:, 0]] * Tn[pos[:, 1]]).sum(-1)
            neg_logit = (P[neg_flat[:, 0]] * Tn[neg_flat[:, 1]]).sum(-1)
            loss = pairwise_loss(pos_logit, neg_logit) + args.emb_reg * enc.emb.weight.pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 2.0)
            opt.step()
            running += loss.item()

        enc.eval()
        with torch.no_grad():
            H = enc(hg, etypes)
            P = H[idx_movie]
            Tn = H[idx_tail]
            vp = (P[val_pos[:, 0]] * Tn[val_pos[:, 1]]).sum(-1)
            vn_list = []
            for j in range(len(val_pos)):
                for k in range(val_neg.shape[1]):
                    vn_list.append((val_pos[j, 0], val_neg[j, k]))
            vn_arr = np.array(vn_list, dtype=np.int64)
            vn = (P[vn_arr[:, 0]] * Tn[vn_arr[:, 1]]).sum(-1)
            y = torch.cat([torch.ones_like(vp), torch.zeros_like(vn)])
            yhat = torch.cat([vp, vn])
            vloss = bce(yhat, y).item()

        print(
            f"Epoch {epoch:03d} | TrainLoss {running/max(1,iters):.4f} | ValLoss {vloss:.4f} | "
            f"{time.time()-t0:.1f}s",
            flush=True,
        )

        if vloss < best_val - 1e-6:
            best_val = vloss
            bad = 0
            os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)
            torch.save(enc.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.", flush=True)
                epochs_ran = epoch + 1
                break

    train_wall = time.perf_counter() - train_t0
    print(
        f"Training finished: wall_sec={train_wall:.2f} epochs_ran={epochs_ran} (max_epochs={args.epochs})",
        flush=True,
    )

    enc.load_state_dict(torch.load(ckpt, map_location=device))
    enc.eval()
    with torch.no_grad():
        H = enc(hg, etypes)
        P = H[idx_movie]
        Tn = H[idx_tail]

        pp = torch.sigmoid((P[test_pos[:, 0]] * Tn[test_pos[:, 1]]).sum(-1)).cpu().numpy()
        pn = torch.sigmoid(
            (P[np.repeat(test_pos[:, 0], test_neg.shape[1])] * Tn[test_neg.ravel()]).sum(-1)
        ).cpu().numpy()

    y_true = np.concatenate([np.ones(len(pp)), np.zeros(len(pn))])
    y_prob = np.concatenate([pp, pn])
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    th = args.th
    y_pred = (y_prob >= th).astype(int)
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    h1, h3, h5, mrr, _ = eval_ranking(test_pos, test_neg, P, Tn)

    print(f"AUC={auc:.6f} AP={ap:.6f}", flush=True)
    print(f"Threshold={th:.3f} | TP={TP} TN={TN} FP={FP} FN={FN}", flush=True)
    print(f"Precision={prec:.6f} Recall={rec:.6f} F1={f1:.6f} Acc={acc:.6f}", flush=True)
    print(f"Hits@1={h1:.6f} Hits@3={h3:.6f} Hits@5={h5:.6f} MRR={mrr:.6f}", flush=True)

    merge_keys = _kendall_merge_keys(task)
    col_movie, col_tail = merge_keys[0], merge_keys[1]
    rows = []
    for i in range(len(test_pos)):
        rows.append({col_movie: int(test_pos[i, 0]), col_tail: int(test_pos[i, 1]), "score": float(pp[i]), "label": 1})
    Kte = test_neg.shape[1]
    for i in range(len(test_pos)):
        for j in range(Kte):
            rows.append(
                {
                    col_movie: int(test_pos[i, 0]),
                    col_tail: int(test_neg[i, j]),
                    "score": float(pn[i * Kte + j]),
                    "label": 0,
                }
            )
    csv_path = f"{args.save_postfix}_{task}_{variant}_seed{seed}_scores.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[Saved] {csv_path}", flush=True)

    return {
        "AUC": float(auc),
        "AP": float(ap),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
        "Accuracy": float(acc),
        "Hits@1": float(h1),
        "Hits@3": float(h3),
        "Hits@5": float(h5),
        "MRR": float(mrr),
        "Train Time (s)": float(train_wall),
        "Epochs": float(epochs_ran),
    }


def summarize(metrics_list: list[dict]) -> None:
    keys = [
        "AUC",
        "AP",
        "Precision",
        "Recall",
        "F1",
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
        arr = np.array([m[k] for m in metrics_list], dtype=float)
        if k in ("Train Time (s)", "Epochs"):
            print(f"{k:<22}: {arr.mean():.2f} ± {arr.std(ddof=0):.2f}")
        else:
            print(f"{k:<22}: {arr.mean():.6f} ± {arr.std(ddof=0):.6f}")


def main():
    ap = argparse.ArgumentParser(description="IMDB RGCN link prediction (CMPNN-aligned tasks).")
    ap.add_argument("--task", type=_parse_task, required=True)
    ap.add_argument(
        "--variant",
        default=None,
        help="Single variant (e.g. v1) or comma list (e.g. v1,v2,v3,v4). Same as --variants if comma-separated.",
    )
    ap.add_argument("--variants", default="v1,v2,v3,v4", help="Comma-separated variants (used if --variant is omitted).")
    ap.add_argument("--in-dim", type=int, default=128)
    ap.add_argument("--hid-dim", type=int, default=256)
    ap.add_argument("--out-dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--num-bases", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--emb-reg", type=float, default=1e-6)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--th", type=float, default=0.5)
    ap.add_argument("--ckpt", default="checkpoint/imdb_rgcn_lp.pt")
    ap.add_argument("--seeds", default="1566911444,20241017,20251017")
    ap.add_argument("--save-postfix", default="IMDB_rgcn_lp")
    ap.add_argument("--compare-only", action="store_true")
    args = ap.parse_args()

    task = args.task
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    merge_keys = _kendall_merge_keys(task)

    if args.compare_only:
        variants = _parse_variants(args.variants, task)
        print(f"\n########## Kendall τ | task={task} (pairwise) ##########")
        for va, vb in itertools.combinations(variants, 2):
            taus, h1_taus, h3_taus = [], [], []
            for sd in seeds:
                pa = f"{args.save_postfix}_{task}_{va}_seed{sd}_scores.csv"
                pb = f"{args.save_postfix}_{task}_{vb}_seed{sd}_scores.csv"
                if not os.path.isfile(pa) or not os.path.isfile(pb):
                    print(f"  {va} vs {vb} seed {sd}: missing CSV")
                    continue
                kk = kendall_csv_with_hits(pa, pb, merge_keys, query_col=merge_keys[0])
                print(
                    f"  {va} vs {vb} seed {sd}: n={kk['overall_n']} tau={kk['overall_tau']:.6f} | "
                    f"hits_n={kk['hits_n']} h@1_tau={kk['h1_tau']:.6f} h@3_tau={kk['h3_tau']:.6f}"
                )
                if np.isfinite(kk["overall_tau"]):
                    taus.append(float(kk["overall_tau"]))
                if np.isfinite(kk["h1_tau"]):
                    h1_taus.append(float(kk["h1_tau"]))
                if np.isfinite(kk["h3_tau"]):
                    h3_taus.append(float(kk["h3_tau"]))
            if taus:
                a = np.array(taus, dtype=float)
                msg = f"  {va} vs {vb}: overall {a.mean():.6f} ± {a.std(ddof=0):.6f}"
                if h1_taus:
                    a1 = np.array(h1_taus, dtype=float)
                    msg += f" | H@1 {a1.mean():.6f} ± {a1.std(ddof=0):.6f}"
                if h3_taus:
                    a3 = np.array(h3_taus, dtype=float)
                    msg += f" | H@3 {a3.mean():.6f} ± {a3.std(ddof=0):.6f}"
                print(msg)
        return

    # --variant v1,v2 was wrongly wrapped as one folder name; always split via _parse_variants.
    if args.variant:
        variants = _parse_variants(args.variant, task)
    else:
        variants = _parse_variants(args.variants, task)
    os.makedirs("checkpoint", exist_ok=True)

    by_v: dict[str, list[dict]] = {}
    for v in variants:
        print(f"\n########## IMDB RGCN LP {task} {v} ##########", flush=True)
        runs = []
        for sd in seeds:
            runs.append(run_one(args, task, v, sd))
        by_v[v] = runs
        summarize(runs)

    print(f"\nIMDB RGCN LP summary | task={task} | mean ± std over seeds")
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
            f"{tw_m:.2f} ± {tw_s:.2f} | {e_m:.2f} ± {e_s:.2f}"
        )


if __name__ == "__main__":
    main()
