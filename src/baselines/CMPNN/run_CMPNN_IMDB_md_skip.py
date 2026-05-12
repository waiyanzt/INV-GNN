#!/usr/bin/env python3
"""CMPNN IMDB movie→director LP on **universal skip** graph (``preprocess_IMDB_cmpnn_lp_skip.py``).
"""
import itertools
import os
import sys
import time
import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils import data as torch_data
from tqdm import tqdm
import torchdrug

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "MAGNN"))
from cmpnn.model import CMPNN
from utils.pytorchtools import EarlyStopping

# Dataset + forward only from non-skip (collate/eval there use 1D batches; CMPNN needs 2D when num_relation>0).
from run_CMPNN_IMDB_md import IMDB_MD_Query, _cmpnn_forward_lp, set_seed  # noqa: E402


def collate_query_md_2d(batch, rel_md):
    """(B, 1+K) head/tail layout for relational CMPNN (``negative_sample_to_tail``)."""
    b = list(zip(*batch))
    h = torch.tensor(b[0], dtype=torch.long)
    t_true = torch.tensor(b[1], dtype=torch.long)
    t_neg = torch.stack([torch.tensor(x, dtype=torch.long) for x in b[2]])
    B, K = t_neg.size()
    # repeat (not expand): expanded views alias storage and break DataLoader pin_memory
    h_index = h.unsqueeze(1).repeat(1, 1 + K)
    t_index = torch.cat([t_true.unsqueeze(1), t_neg], dim=1)
    r_index = torch.full_like(h_index, rel_md, dtype=torch.long)
    y = torch.zeros_like(h_index, dtype=torch.float)
    y[:, 0] = 1.0
    return h_index, t_index, r_index, y


def _precision_recall_f1_best(y_true: np.ndarray, y_proba: np.ndarray):
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    j = int(np.nanargmax(f1))
    return float(prec[j]), float(rec[j]), float(f1[j])


def neg_logsigmoid_loss_2d(scores):
    pos = scores[..., 0]
    neg = scores[..., 1:]
    return -(F.logsigmoid(pos).mean() + F.logsigmoid(-neg).mean())


@torch.no_grad()
def evaluate_2d(model, graph, loader, device):
    model.eval()
    all_y, all_p = [], []
    hits1 = hits3 = hits5 = 0
    rr_sum = 0.0
    n_q = 0
    for h_index, t_index, r_index, y in loader:
        scores = _cmpnn_forward_lp(model, graph, h_index, t_index, r_index, device)
        probs = torch.sigmoid(scores)
        B, L = probs.shape
        K = L - 1
        if B > 0 and K > 0:
            s_true = probs[:, 0]
            rank = (probs[:, 1:] >= s_true.unsqueeze(1)).sum(dim=1) + 1
            hits1 += (rank <= 1).sum().item()
            hits3 += (rank <= 3).sum().item()
            hits5 += (rank <= 5).sum().item()
            rr_sum += (1.0 / rank.float()).sum().item()
            n_q += B
        all_y.append(y.flatten())
        all_p.append(probs.flatten().detach().cpu().numpy())
    y_all = torch.cat(all_y).numpy()
    p_all = np.concatenate(all_p)
    return {
        "auc": roc_auc_score(y_all, p_all),
        "ap": average_precision_score(y_all, p_all),
        "hits1": hits1 / max(n_q, 1),
        "hits3": hits3 / max(n_q, 1),
        "hits5": hits5 / max(n_q, 1),
        "mrr": rr_sum / max(n_q, 1),
    }


@torch.no_grad()
def evaluate_full_md_2d(model, graph, test_pos, test_neg, off_d, rel_md, batch_size, device, threshold):
    model.eval()
    K = test_neg.shape[1]
    all_scores, all_labels, all_pairs = [], [], []
    hits1 = hits3 = hits5 = 0
    rr_sum = 0.0
    n_q = 0
    for start in range(0, len(test_pos), batch_size):
        pos_batch = test_pos[start : start + batch_size]
        neg_batch = test_neg[start : start + batch_size]
        B = len(pos_batch)
        h_list, t_list = [], []
        for j in range(B):
            m = int(pos_batch[j, 0])
            d_true = int(pos_batch[j, 1])
            h_list.append(np.full(1 + K, m, dtype=np.int64))
            t_list.append(np.concatenate([[off_d + d_true], off_d + neg_batch[j].astype(np.int64)]))
        h_index = torch.tensor(np.stack(h_list), dtype=torch.long)
        t_index = torch.tensor(np.stack(t_list), dtype=torch.long)
        r_index = torch.full_like(h_index, rel_md)
        scores = _cmpnn_forward_lp(model, graph, h_index, t_index, r_index, device)
        probs = torch.sigmoid(scores)
        rank = (probs[:, 1:] >= probs[:, 0:1]).sum(dim=1) + 1
        hits1 += (rank <= 1).sum().item()
        hits3 += (rank <= 3).sum().item()
        hits5 += (rank <= 5).sum().item()
        rr_sum += (1.0 / rank.float()).sum().item()
        n_q += B
        probs_np = probs.cpu().numpy()
        for ki in range(B):
            all_scores.append(probs_np[ki, 0])
            all_labels.append(1)
            all_pairs.append((int(pos_batch[ki, 0]), int(pos_batch[ki, 1])))
            for jj in range(K):
                all_scores.append(probs_np[ki, 1 + jj])
                all_labels.append(0)
                all_pairs.append((int(pos_batch[ki, 0]), int(neg_batch[ki, jj])))
    y_true = np.array(all_labels)
    y_proba = np.array(all_scores)
    auc = roc_auc_score(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = TP / max(TP + FP, 1)
    rec = TP / max(TP + FN, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    prec_bf1, rec_bf1, f1_bf1 = _precision_recall_f1_best(y_true, y_proba)
    acc = (TP + TN) / max(TP + TN + FP + FN, 1)
    pairs_arr = np.array(all_pairs)
    scores_df = pd.DataFrame(
        {
            "movie_local": pairs_arr[:, 0],
            "director_local": pairs_arr[:, 1],
            "label": y_true,
            "prob": y_proba,
        }
    )
    return {
        "auc": auc,
        "ap": ap,
        "hits1": hits1 / max(n_q, 1),
        "hits3": hits3 / max(n_q, 1),
        "hits5": hits5 / max(n_q, 1),
        "mrr": rr_sum / max(n_q, 1),
        "top1_accuracy": hits1 / max(n_q, 1),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "precision_best_f1": prec_bf1,
        "recall_best_f1": rec_bf1,
        "f1_best_f1": f1_bf1,
        "accuracy": acc,
        "confusion": (TP, TN, FP, FN),
        "scores_df": scores_df,
    }


def base_dir(variant: str) -> str:
    return f"data/preprocessed/IMDB_cmpnn_lp_skip_md_{variant}"


def load_preprocessed(variant: str):
    b = base_dir(variant)
    edge_list = torch.load(os.path.join(b, "edge_list.pt"), map_location="cpu")
    meta = torch.load(os.path.join(b, "meta.pt"), map_location="cpu")
    splits = {k: v.cpu().numpy() for k, v in meta["splits"].items()}
    meta_np = {k: meta[k] for k in meta if k != "splits"}
    return np.asarray(edge_list.numpy(), dtype=np.int64), meta_np, splits


def kendall_scores(path_a: str, path_b: str, on_cols):
    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)
    m = a.merge(b, on=on_cols, suffixes=("_a", "_b"), how="inner")
    if len(m) < 2:
        return float("nan"), len(m)
    tau, _ = kendalltau(m["score_a"], m["score_b"], nan_policy="omit")
    return (float(tau) if np.isfinite(tau) else float("nan"), len(m))


def run_model_skip(
    variant,
    neg_k,
    input_dim,
    hidden_dim,
    num_layers,
    num_epochs,
    patience,
    batch_size,
    num_workers,
    threshold,
    save_postfix,
    base_seed,
    gpu,
    use_cpu,
    eval_only,
    checkpoint_path,
):
    t0 = time.time()
    triplets, meta, splits = load_preprocessed(variant)
    preprocess_sec = time.time() - t0
    print(
        f"-> Loaded skip preprocess {preprocess_sec:.1f}s | entities={meta['num_entity']} "
        f"rel={meta['num_relation']} triplets={len(triplets)}"
    )
    off_d = meta["off_d"]
    rel_md = meta["rel_md"]
    print(f"-> off_d={off_d} rel_md={rel_md} variant={variant} (folder label) neg_k={meta.get('neg_k', neg_k)}")

    if use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cuda:0")

    graph = torchdrug.data.Graph(
        torch.as_tensor(triplets, dtype=torch.long),
        num_node=int(meta["num_entity"]),
        num_relation=int(meta["num_relation"]),
    )
    graph = graph.to(device)

    train_ds = IMDB_MD_Query(splits["train_pos"], splits["train_neg"], off_d)
    val_ds = IMDB_MD_Query(splits["val_pos"], splits["val_neg"], off_d)
    collate_fn = lambda b: collate_query_md_2d(b, rel_md=rel_md)
    pin_mem = (not use_cpu) and torch.cuda.is_available()
    dl_kw = dict(num_workers=num_workers, pin_memory=pin_mem)
    if num_workers > 0:
        dl_kw["persistent_workers"] = True
    train_loader = torch_data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, **dl_kw
    )
    val_loader = torch_data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, **dl_kw
    )

    if hidden_dim != input_dim:
        raise ValueError("hidden_dim must equal input_dim")

    model = CMPNN(
        input_dim=input_dim,
        hidden_dims=[hidden_dim] * num_layers,
        num_relation=int(meta["num_relation"]),
        message_func="distmult",
        aggregate_func="pna",
        short_cut=True,
        layer_norm=True,
        dependent=False,
        set_boundary=True,
        remove_one_hop=True,
        activation="relu",
        initialization="Query",
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    os.makedirs("checkpoint", exist_ok=True)
    ckpt_path = f"checkpoint/checkpoint_{save_postfix}.pt"
    early_stopping = EarlyStopping(patience=patience, verbose=False, save_path=ckpt_path)

    train_wall_sec = 0.0
    epochs_ran = 0
    if eval_only:
        print("-> eval-only, skip training")
    else:
        train_t0 = time.perf_counter()
        for epoch in range(num_epochs):
            model.train()
            losses = []
            for h_index, t_index, r_index, _y in tqdm(train_loader, desc=f"Epoch {epoch}"):
                scores = _cmpnn_forward_lp(model, graph, h_index, t_index, r_index, device)
                loss = neg_logsigmoid_loss_2d(scores)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            vm = evaluate_2d(model, graph, val_loader, device)
            print(f"Epoch {epoch} loss={np.mean(losses):.4f} val_AP={vm['ap']:.4f} val_AUC={vm['auc']:.4f}")
            early_stopping(-vm["ap"], model)
            epochs_ran = epoch + 1
            if early_stopping.early_stop:
                print("Early stopping")
                break
        train_wall_sec = float(time.perf_counter() - train_t0)

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    elif not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    else:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    print("-> Testing best checkpoint ...")
    results = evaluate_full_md_2d(
        model, graph, splits["test_pos"], splits["test_neg"], off_d, rel_md, batch_size, device, threshold
    )
    df = results["scores_df"].copy()
    df["score"] = df["prob"].astype(float)
    df.to_csv(f"{save_postfix}_scores.csv", index=False)

    print(
        f"    Hits@1={results['hits1']:.4f} Hits@3={results['hits3']:.4f} MRR={results['mrr']:.4f} "
        f"AUC={results['auc']:.4f} AP={results['ap']:.4f}"
    )
    print(
        f"    Prec={results['precision_best_f1']:.4f} Rec={results['recall_best_f1']:.4f} "
        f"F1={results['f1_best_f1']:.4f} (pairwise, best τ) | "
        f"Prec@τ={threshold}={results['precision']:.4f} Rec@τ={threshold}={results['recall']:.4f} "
        f"F1@τ={threshold}={results['f1']:.4f}"
    )
    print(f"    Train wall (s)={train_wall_sec:.2f}  Epochs={epochs_ran}  Preprocess load (s)={preprocess_sec:.2f}")

    out = {k: v for k, v in results.items() if k != "scores_df"}
    out["preprocess_sec"] = float(preprocess_sec)
    out["train_wall_sec"] = float(train_wall_sec)
    out["epochs_ran"] = float(epochs_ran)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="CMPNN IMDB MD LP — universal skip graph",
        epilog="Build data with preprocess_IMDB_cmpnn_lp_skip.py (RGCN-aligned skip graph from "
        "preprocess_IMDB_rgcn_lp_skip.build_universal_lp_graph; --task md, --shared-npz, --neg-k). "
        "This runner only loads preprocessed dirs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--variants", default="v1,v3")
    ap.add_argument("--input-dim", type=int, default=32)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--epoch", type=int, default=100, help="Max epochs (early stopping on val AP)")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training/eval minibatch size. Universal skip graphs are edge-heavy; CMPNN+PNA "
        "often needs ≤32 on a 40GB GPU (try 16 if OOM). Raise to 64–128 only if memory allows.",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (0 disables multiprocessing prefetching)",
    )
    ap.add_argument("--neg-k", type=int, default=19)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seeds", default="1566911444,20241017,20251017")
    ap.add_argument("--save-postfix", default="IMDB_cmpnn_md_skip")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument(
        "--compare-only",
        action="store_true",
        help="Only Kendall τ between existing score CSVs (needs prior runs).",
    )
    args = ap.parse_args()

    variants = [x.strip().lower() for x in args.variants.split(",") if x.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    merge_keys = ["movie_local", "director_local"]

    if args.compare_only:
        print("########## Kendall τ (pairwise) | IMDB CMPNN MD skip ##########")
        for va, vb in itertools.combinations(variants, 2):
            for sd in seeds:
                pa = f"{args.save_postfix}_{va}_seed{sd}_scores.csv"
                pb = f"{args.save_postfix}_{vb}_seed{sd}_scores.csv"
                if not os.path.isfile(pa) or not os.path.isfile(pb):
                    print(f"  {va} vs {vb} seed {sd}: missing {pa} or {pb}")
                    continue
                tau, n = kendall_scores(pa, pb, merge_keys)
                print(f"  {va} vs {vb} seed {sd}: n={n} Kendall τ={tau:.6f}")
        sys.exit(0)

    if args.eval_only and args.checkpoint is None:
        ap.error("--eval-only requires --checkpoint")

    for v in variants:
        print("\n" + "#" * 20 + f" variant={v} " + "#" * 20)
        for s in seeds:
            print("=" * 60)
            print(f" Seed {s}")
            set_seed(s)
            run_model_skip(
                variant=v,
                neg_k=args.neg_k,
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.layers,
                num_epochs=args.epoch,
                patience=args.patience,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                threshold=args.threshold,
                save_postfix=f"{args.save_postfix}_{v}_seed{s}",
                base_seed=s,
                gpu=args.gpu,
                use_cpu=args.cpu,
                eval_only=args.eval_only,
                checkpoint_path=args.checkpoint,
            )

    print("\n########## Kendall τ after training ##########")
    for va, vb in itertools.combinations(variants, 2):
        for sd in seeds:
            pa = f"{args.save_postfix}_{va}_seed{sd}_scores.csv"
            pb = f"{args.save_postfix}_{vb}_seed{sd}_scores.csv"
            if os.path.isfile(pa) and os.path.isfile(pb):
                tau, n = kendall_scores(pa, pb, merge_keys)
                print(f"  {va} vs {vb} seed {sd}: n={n} τ={tau:.6f}")
