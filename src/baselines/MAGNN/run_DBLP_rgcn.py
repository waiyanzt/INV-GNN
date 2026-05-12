#!/usr/bin/env python3
"""
RGCN paper–venue link prediction on DBLP
to run:  python run_DBLP_rgcn.py --variants v1,v2,v3 --save-postfix DBLP_pv_rgcn
         python run_DBLP_rgcn.py --compare-only --variants v1,v2,v3 --save-postfix DBLP_pv_rgcn

"""
import os, argparse, time, random
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import kendalltau

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import RelGraphConv
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, accuracy_score
)

SEED = 1566911444

BASE = {
    'v1': 'data/preprocessed/DBLP_rgcn_v1',
    'v2': 'data/preprocessed/DBLP_rgcn_v2',
    'v3': 'data/preprocessed/DBLP_rgcn_v3',
}

def set_determinism(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try: torch.use_deterministic_algorithms(True)
    except Exception: pass

def _parse_variants(s):
    out = [x.strip().lower() for x in str(s).split(',') if x.strip()]
    for v in out:
        if v not in BASE:
            raise SystemExit(f'Unknown variant {v!r}; expected one of {",".join(BASE)}')
    return out

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

def to_homo_with_indexers(g):
    hg = dgl.to_homogeneous(g)
    etypes = hg.edata.get(dgl.ETYPE, hg.edata['_TYPE']).long()
    ntype  = hg.ndata.get(dgl.NTYPE, hg.ndata['_TYPE']).long()
    nid    = hg.ndata.get(dgl.NID,   hg.ndata['_ID']).long()
    indexers = {}
    for name in g.ntypes:
        tid = g.get_ntype_id(name)
        mask = (ntype == tid)
        homo_idx = mask.nonzero(as_tuple=False).squeeze(1)
        lid = nid[mask]
        order = torch.argsort(lid)
        indexers[name] = homo_idx[order]
    return hg, etypes, indexers

class RGCNEncoder(nn.Module):
    def __init__(self, num_nodes, num_rels,
                 in_dim=128, hid_dim=256, out_dim=256,
                 num_layers=3, num_bases=16, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, in_dim)
        nn.init.xavier_uniform_(self.emb.weight)

        def mk(din, dout, act, self_loop=True, dropout=0.0):
            try:
                return RelGraphConv(din, dout, num_rels,
                                    regularizer='basis', num_bases=num_bases,
                                    self_loop=self_loop, dropout=dropout,
                                    activation=act, low_mem=True)
            except TypeError:
                return RelGraphConv(din, dout, num_rels,
                                    regularizer='basis', num_bases=num_bases,
                                    self_loop=self_loop, dropout=dropout,
                                    activation=act)

        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(mk(in_dim, out_dim, act=None, dropout=dropout))
        else:
            self.layers.append(mk(in_dim, hid_dim, act=F.relu, dropout=dropout))
            for _ in range(num_layers - 2):
                self.layers.append(mk(hid_dim, hid_dim, act=F.relu, dropout=dropout))
            self.layers.append(mk(hid_dim, out_dim, act=None, dropout=dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, hg, etypes):
        h = self.emb.weight
        for layer in self.layers:
            h = layer(hg, h, etypes)
        return self.dropout(h)

def pairwise_loss(pos_logit, neg_logit):
    return -(F.logsigmoid(pos_logit).mean() + F.logsigmoid(-neg_logit).mean())

@torch.no_grad()
def eval_ranking_pc(test_pos, test_neg, P, C):
    pos_s = torch.sigmoid((P[test_pos[:,0]] * C[test_pos[:,1]]).sum(-1)).cpu().numpy()
    neg_s = torch.sigmoid((P[test_neg[:,0]] * C[test_neg[:,1]]).sum(-1)).cpu().numpy()
    cand = defaultdict(list)
    for (p,c), s in zip(test_pos, pos_s): cand[int(p)].append((float(s), int(c), 1))
    for (p,c), s in zip(test_neg, neg_s): cand[int(p)].append((float(s), int(c), 0))
    h1=h3=h5=0; rr=0.0; n=0
    for _, items in cand.items():
        if not items: continue
        items.sort(key=lambda x: x[0], reverse=True)
        ranks = [i+1 for i,(_,_,t) in enumerate(items) if t==1]
        if not ranks: continue
        r = min(ranks); n+=1; rr += 1.0/r
        if r<=1: h1+=1
        if r<=3: h3+=1
        if r<=5: h5+=1
    return (h1/n if n else 0.0, h3/n if n else 0.0, h5/n if n else 0.0, rr/n if n else 0.0, n)

def build_graph(graph_data, num_nodes, variant):
    data = {
        ('author','author-paper','paper'): graph_data['author-paper'],
        ('paper','paper-author','author'): (graph_data['author-paper'][1], graph_data['author-paper'][0]),
        ('paper','paper-conference','conference'): graph_data['paper-conference'],
        ('conference','conference-paper','paper'): (graph_data['paper-conference'][1], graph_data['paper-conference'][0]),
        ('paper','paper-term','term'): graph_data['paper-term'],
        ('term','term-paper','paper'): (graph_data['paper-term'][1], graph_data['paper-term'][0]),
    }
    if variant == 'v1':
        data[('paper', 'paper-area', 'area')] = graph_data['paper-area']
        data[('area', 'area-paper', 'paper')] = (graph_data['paper-area'][1], graph_data['paper-area'][0])
    elif variant == 'v2':
        data[('conference', 'conference-area', 'area')] = graph_data['conference-area']
        data[('area', 'area-conference', 'conference')] = (
            graph_data['conference-area'][1],
            graph_data['conference-area'][0],
        )
    elif variant == 'v3':
        data[('author', 'author-area', 'area')] = graph_data['author-area']
        data[('area', 'area-author', 'author')] = (graph_data['author-area'][1], graph_data['author-area'][0])
    return dgl.heterograph(data, num_nodes_dict=num_nodes)

def load_preprocessed(base_dir):
    return torch.load(os.path.join(base_dir, 'graph_data.pt')), torch.load(os.path.join(base_dir, 'meta.pt'))

def run_one(args, variant, seed):
    set_determinism(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Variant={variant} Seed={seed} Device={device}", flush=True)

    graph_data, meta = load_preprocessed(BASE[variant])
    num_nodes = meta['num_nodes']
    S = {k: v.cpu().numpy() for k, v in meta['splits'].items()}

    if getattr(args, "neg_per_paper", 0) and args.neg_per_paper > 0:
        k = int(args.neg_per_paper)
        S["train_neg"] = subsample_negs_per_paper(S["train_neg"], k, np.random.RandomState(int(seed) + 11))
        S["val_neg"] = subsample_negs_per_paper(S["val_neg"], k, np.random.RandomState(int(seed) + 13))
        S["test_neg"] = subsample_negs_per_paper(S["test_neg"], k, np.random.RandomState(int(seed) + 17))
        print(f"Subsampled negatives to max {k} per paper (train/val/test), run seed={seed}", flush=True)
        if k <= 4:
            print(
                f"[note] Ranking (Hits@k/MRR) uses ≤{1 + k} venues per test paper (1 true + {k} negs). "
                f"MRR can sit at 1.0 if the model always beats those few distractors — that is expected, not a bug. "
                f"For a harder ranking benchmark use e.g. --neg-per-paper 19.",
                flush=True,
            )

    print(f"Splits: train_pos={len(S['train_pos'])}  val_pos={len(S['val_pos'])}  test_pos={len(S['test_pos'])}", flush=True)
    print(f"        train_neg={len(S['train_neg'])}  val_neg={len(S['val_neg'])}  test_neg={len(S['test_neg'])}", flush=True)

    g = build_graph(graph_data, num_nodes, variant)
    hg, etypes, indexers = to_homo_with_indexers(g)
    N_total = hg.num_nodes(); R_total = int(etypes.max().item()) + 1
    print(f"Graph: N={N_total}  E={hg.num_edges()}  node types={g.ntypes}  etypes={g.etypes}", flush=True)

    hg = hg.to(device); etypes = etypes.to(device)
    idxP = indexers['paper'].to(device)
    idxC = indexers['conference'].to(device)

    enc = RGCNEncoder(N_total, R_total,
                      in_dim=args.in_dim, hid_dim=args.hid_dim, out_dim=args.out_dim,
                      num_layers=args.layers, num_bases=args.num_bases, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_pos = S['train_pos']
    train_neg = S['train_neg']
    neg_by_paper = {}
    order = np.lexsort((train_neg[:,1], train_neg[:,0]))
    train_neg = train_neg[order]
    i = 0
    while i < len(train_neg):
        p = int(train_neg[i,0]); j = i
        while j < len(train_neg) and int(train_neg[j,0]) == p:
            j += 1
        neg_by_paper[p] = train_neg[i:j]
        i = j

    iters = int(np.ceil(len(train_pos) / args.batch_size))
    best_val = float('inf'); bad = 0; bce = nn.BCEWithLogitsLoss()
    train_t0 = time.perf_counter()
    epochs_ran = args.epochs

    for epoch in range(args.epochs):
        enc.train()
        perm = np.random.permutation(len(train_pos))
        running = 0.0; t0 = time.time()
        pbar = tqdm(range(iters), desc=f"{variant} Epoch {epoch:03d}", leave=True)

        for it in pbar:
            sl = perm[it*args.batch_size : (it+1)*args.batch_size]
            pos = train_pos[sl]
            neg_blocks = [neg_by_paper[int(p)] for p in pos[:,0] if int(p) in neg_by_paper]
            neg = np.concatenate(neg_blocks, axis=0) if len(neg_blocks) > 0 else train_neg[:0]

            opt.zero_grad()
            H = enc(hg, etypes)
            P = H[idxP]; C = H[idxC]
            pos_logit = (P[pos[:,0]] * C[pos[:,1]]).sum(-1)
            neg_logit = (P[neg[:,0]] * C[neg[:,1]]).sum(-1)
            loss = pairwise_loss(pos_logit, neg_logit) + args.emb_reg * enc.emb.weight.pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 2.0)
            opt.step()
            running += loss.item()
            if (it+1) % max(1, iters//5) == 0:
                pbar.set_postfix(loss=f"{running/(it+1):.4f}")

        enc.eval()
        with torch.no_grad():
            H = enc(hg, etypes); P = H[idxP]; C = H[idxC]
            vp = (P[S['val_pos'][:,0]] * C[S['val_pos'][:,1]]).sum(-1)
            vn = (P[S['val_neg'][:,0]] * C[S['val_neg'][:,1]]).sum(-1)
            y  = torch.cat([torch.ones_like(vp), torch.zeros_like(vn)])
            yhat = torch.cat([vp, vn])
            vloss = bce(yhat, y).item()

        print(f"Epoch {epoch:03d} | TrainLoss {running/max(1,iters):.4f} | ValLoss {vloss:.4f} | {time.time()-t0:.1f}s", flush=True)
        ckpt = args.ckpt.replace('.pt', f'_{variant}_seed{seed}.pt')
        if vloss < best_val - 1e-6:
            best_val = vloss; bad = 0
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
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

    ckpt = args.ckpt.replace('.pt', f'_{variant}_seed{seed}.pt')
    enc.load_state_dict(torch.load(ckpt, map_location=device))
    enc.eval()
    with torch.no_grad():
        H = enc(hg, etypes); P = H[idxP]; C = H[idxC]
        pp = torch.sigmoid((P[S['test_pos'][:,0]] * C[S['test_pos'][:,1]]).sum(-1)).cpu().numpy()
        pn = torch.sigmoid((P[S['test_neg'][:,0]] * C[S['test_neg'][:,1]]).sum(-1)).cpu().numpy()
        y_true = np.concatenate([np.ones_like(pp), np.zeros_like(pn)])
        y_prob = np.concatenate([pp, pn])
        th = args.th
        y_pred = (y_prob >= th).astype(int)

    TP = int(((y_pred==1)&(y_true==1)).sum()); FN = int(((y_pred==0)&(y_true==1)).sum())
    TN = int(((y_pred==0)&(y_true==0)).sum()); FP = int(((y_pred==1)&(y_true==0)).sum())

    if getattr(args, 'save_postfix', ''):
        rows = []
        tp = S['test_pos']
        tn = S['test_neg']
        for i in range(len(tp)):
            rows.append({
                'paper_id': int(tp[i, 0]),
                'conf_id': int(tp[i, 1]),
                'score': float(pp[i]),
                'label': 1,
            })
        for j in range(len(tn)):
            rows.append({
                'paper_id': int(tn[j, 0]),
                'conf_id': int(tn[j, 1]),
                'score': float(pn[j]),
                'label': 0,
            })
        csv_path = f"{args.save_postfix}_{variant}_seed{seed}_scores.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"[Saved] Per-pair scores: {csv_path}", flush=True)

    auc = roc_auc_score(y_true, y_prob)
    ap  = average_precision_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    acc  = accuracy_score(y_true, y_pred)
    H1,H3,H5,MRR,n = eval_ranking_pc(S['test_pos'], S['test_neg'], P, C)

    print(f"AUC={auc:.6f}  AP={ap:.6f}", flush=True)
    print(f"Chosen threshold (fixed) = {th:.3f}", flush=True)
    print(f"Confusion matrix : TP={TP}  TN={TN}  FP={FP}  FN={FN} (P={TP+FN}, N={TN+FP})", flush=True)
    print(f"Precision={prec:.6f}  Recall={rec:.6f}  F1={f1:.6f}  Acc={acc:.6f}", flush=True)
    print(f"Hits@1={H1:.6f}  Hits@3={H3:.6f}  Hits@5={H5:.6f}  MRR={MRR:.6f}", flush=True)

    return {
        "AUC": float(auc), "AP": float(ap), "Precision": float(prec),
        "Recall": float(rec), "F1": float(f1), "Accuracy": float(acc),
        "Hits@1": float(H1), "Hits@3": float(H3), "Hits@5": float(H5), "MRR": float(MRR),
        "Train Time (s)": float(train_wall),
        "Epochs": float(epochs_ran),
    }

def summarize(metrics_list):
    keys = [
        'AUC', 'AP', 'Precision', 'Recall', 'F1', 'Accuracy',
        'Hits@1', 'Hits@3', 'Hits@5', 'MRR', 'Train Time (s)', 'Epochs',
    ]
    print("\n===== Summary over seeds (mean +/- std) =====")
    for k in keys:
        arr = np.array([m[k] for m in metrics_list], dtype=float)
        if k in ('Train Time (s)', 'Epochs'):
            print(f"{k:<22}: {arr.mean():.2f} +/- {arr.std(ddof=0):.2f}")
        else:
            print(f"{k:<22}: {arr.mean():.6f} +/- {arr.std(ddof=0):.6f}")


def _hits_by_query(df: pd.DataFrame, query_col: str) -> pd.DataFrame:
    if 'label' not in df.columns:
        return pd.DataFrame(columns=[query_col, 'hit1', 'hit3'])
    out = []
    for q, grp in df.groupby(query_col, sort=True):
        g = grp.sort_values('score', ascending=False).reset_index(drop=True)
        pos_idx = g.index[g['label'] == 1].tolist()
        if not pos_idx:
            continue
        rank = int(min(pos_idx)) + 1
        out.append({query_col: int(q), 'hit1': int(rank <= 1), 'hit3': int(rank <= 3)})
    return pd.DataFrame(out)


def kendall_csv_with_hits(path_a: str, path_b: str, on_cols: list, query_col: str) -> dict:
    """Overall Kendall τ on scores; Hits@1 / Hits@3 τ = Kendall on per-paper binary hit vectors."""
    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)
    m = a.merge(b, on=on_cols, suffixes=('_a', '_b'), how='inner')
    out = {
        'overall_tau': float('nan'),
        'overall_n': len(m),
        'h1_tau': float('nan'),
        'h3_tau': float('nan'),
        'hits_n': 0,
    }
    if len(m) >= 2:
        tau, _ = kendalltau(m['score_a'], m['score_b'], nan_policy='omit')
        out['overall_tau'] = float(tau) if np.isfinite(tau) else float('nan')
    ha = _hits_by_query(a, query_col).rename(columns={'hit1': 'hit1_a', 'hit3': 'hit3_a'})
    hb = _hits_by_query(b, query_col).rename(columns={'hit1': 'hit1_b', 'hit3': 'hit3_b'})
    hh = ha.merge(hb, on=query_col, how='inner')
    out['hits_n'] = len(hh)
    if len(hh) >= 2:
        t1, _ = kendalltau(hh['hit1_a'], hh['hit1_b'], nan_policy='omit')
        t3, _ = kendalltau(hh['hit3_a'], hh['hit3_b'], nan_policy='omit')
        out['h1_tau'] = float(t1) if np.isfinite(t1) else float('nan')
        out['h3_tau'] = float(t3) if np.isfinite(t3) else float('nan')
    return out


def _variant_pairs_from_csv_spec(compare_or_variants: str) -> list:
    lst = _parse_variants(compare_or_variants)
    if len(lst) < 2:
        raise SystemExit('Need at least two variants (comma-separated), e.g. v1,v2,v3')
    return list(itertools.combinations(lst, 2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RGCN LP on DBLP original variants")
    ap.add_argument('--variant', default=None)
    ap.add_argument('--variants', default='v1,v2,v3')
    ap.add_argument('--in-dim', type=int, default=128)
    ap.add_argument('--hid-dim', type=int, default=256)
    ap.add_argument('--out-dim', type=int, default=256)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--num-bases', type=int, default=16)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-5)
    ap.add_argument('--emb-reg', type=float, default=1e-6)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument(
        '--neg-per-paper',
        type=int,
        default=3,
        help='Max negatives per paper (train/val/test). Small k (e.g. 3) ⇒ MRR/Hits@1 can saturate at 1.0; use 19 for a harder ranking pool.',
    )
    ap.add_argument('--th', type=float, default=0.5)
    ap.add_argument('--ckpt', default='checkpoint/rgcn.pt')
    ap.add_argument('--seeds', default='1566911444,20241017,20251017')
    ap.add_argument('--save-postfix', default='DBLP_pv_rgcn')
    ap.add_argument(
        '--compare',
        default='',
        help='Comma-separated variants for Kendall after training, e.g. v1,v2,v3 (all pairs) or v1,v3.',
    )
    ap.add_argument(
        '--compare-only',
        action='store_true',
        help='Only Kendall from existing CSVs; use --compare or default --variants for variant list.',
    )
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    merge_keys = ['paper_id', 'conf_id']
    query_col = 'paper_id'

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
            if np.isfinite(kk['overall_tau']):
                o_taus.append(float(kk['overall_tau']))
            if np.isfinite(kk['h1_tau']):
                h1_taus.append(float(kk['h1_tau']))
            if np.isfinite(kk['h3_tau']):
                h3_taus.append(float(kk['h3_tau']))
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
        spec = args.compare.strip() if (args.compare and str(args.compare).strip()) else args.variants
        pairs = _variant_pairs_from_csv_spec(spec)
        print(f"\n########## Kendall τ (overall + per-paper Hits@1/Hits@3) | spec={spec!r} ##########")
        for va, vb in pairs:
            print(f"\n--- {va} vs {vb} ---", flush=True)
            _print_kendall_block(va, vb)

    else:
        variants = [args.variant] if args.variant else _parse_variants(args.variants)
        by_v = {}
        for variant in variants:
            print(f"\n########## RGCN {variant} ##########", flush=True)
            mets = []
            for sd in seeds:
                mets.append(run_one(args, variant, sd))
            by_v[variant] = mets
            summarize(mets)

        print("\nDBLP RGCN (non-skip) summary | mean ± std over seeds")
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

        if args.compare.strip():
            pairs = _variant_pairs_from_csv_spec(args.compare.strip())
            print(f"\n########## Kendall τ after training | compare={args.compare!r} ##########")
            for va, vb in pairs:
                print(f"\n--- {va} vs {vb} ---", flush=True)
                _print_kendall_block(va, vb)