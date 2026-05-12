#!/usr/bin/env python3
"""
python run_IMDB_rgcn.py --variants v1,v2,v3,v4 --compare v1,v4

"""
import os
import sys
import argparse
import random
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import RelGraphConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import kendalltau

SEED = 1566911444

BASE = {
    'v1': 'data/preprocessed/IMDB_rgcn_v1',
    'v2': 'data/preprocessed/IMDB_rgcn_v2',
    'v3': 'data/preprocessed/IMDB_rgcn_v3',
    'v4': 'data/preprocessed/IMDB_rgcn_v4',
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
    out = [x.strip().lower() for x in str(s).split(',') if x.strip()]
    for v in out:
        if v not in BASE:
            raise SystemExit(f"Unknown variant {v!r}; expected one of {','.join(BASE)}")
    return out


def to_homo_with_indexers(g):
    hg = dgl.to_homogeneous(g)
    etypes = hg.edata.get(dgl.ETYPE, hg.edata['_TYPE']).long()
    ntype = hg.ndata.get(dgl.NTYPE, hg.ndata['_TYPE']).long()
    nid = hg.ndata.get(dgl.NID, hg.ndata['_ID']).long()

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
                 in_dim=128, hid_dim=128, out_dim=128,
                 num_layers=2, num_bases=8, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, in_dim)
        nn.init.xavier_uniform_(self.emb.weight)

        def mk(din, dout, act, self_loop=True, dropout=0.0):
            try:
                return RelGraphConv(
                    din, dout, num_rels,
                    regularizer='basis', num_bases=num_bases,
                    self_loop=self_loop, dropout=dropout,
                    activation=act, low_mem=True
                )
            except TypeError:
                return RelGraphConv(
                    din, dout, num_rels,
                    regularizer='basis', num_bases=num_bases,
                    self_loop=self_loop, dropout=dropout,
                    activation=act
                )

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


class Classifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def load_preprocessed(base_dir):
    graph_data = torch.load(os.path.join(base_dir, 'graph_data.pt'))
    meta = torch.load(os.path.join(base_dir, 'meta.pt'))
    return graph_data, meta


def build_graph(graph_data, num_nodes, variant):
    data = {}

    if variant == 'v1':
        data[('actor', 'actor-movie', 'movie')] = graph_data['actor-movie']
        data[('movie', 'movie-actor', 'actor')] = (graph_data['actor-movie'][1], graph_data['actor-movie'][0])

        data[('movie', 'movie-link', 'link')] = graph_data['movie-link']
        data[('link', 'link-movie', 'movie')] = (graph_data['movie-link'][1], graph_data['movie-link'][0])

        data[('movie', 'movie-director', 'director')] = graph_data['movie-director']
        data[('director', 'director-movie', 'movie')] = (graph_data['movie-director'][1], graph_data['movie-director'][0])

    elif variant == 'v2':
        data[('actor', 'actor-link', 'link')] = graph_data['actor-link']
        data[('link', 'link-actor', 'actor')] = (graph_data['actor-link'][1], graph_data['actor-link'][0])

        data[('link', 'link-movie', 'movie')] = graph_data['link-movie']
        data[('movie', 'movie-link', 'link')] = (graph_data['link-movie'][1], graph_data['link-movie'][0])

        data[('link', 'link-director', 'director')] = graph_data['link-director']
        data[('director', 'director-link', 'link')] = (graph_data['link-director'][1], graph_data['link-director'][0])

    elif variant == 'v3':
        data[('actor', 'actor-link', 'link')] = graph_data['actor-link']
        data[('link', 'link-actor', 'actor')] = (graph_data['actor-link'][1], graph_data['actor-link'][0])

        data[('link', 'link-movie', 'movie')] = graph_data['link-movie']
        data[('movie', 'movie-link', 'link')] = (graph_data['link-movie'][1], graph_data['link-movie'][0])

        data[('movie', 'movie-director', 'director')] = graph_data['movie-director']
        data[('director', 'director-movie', 'movie')] = (graph_data['movie-director'][1], graph_data['movie-director'][0])

    elif variant == 'v4':
        data[('actor', 'actor-movie', 'movie')] = graph_data['actor-movie']
        data[('movie', 'movie-actor', 'actor')] = (graph_data['actor-movie'][1], graph_data['actor-movie'][0])

        data[('movie', 'movie-link', 'link')] = graph_data['movie-link']
        data[('link', 'link-movie', 'movie')] = (graph_data['movie-link'][1], graph_data['movie-link'][0])

        data[('link', 'link-director', 'director')] = graph_data['link-director']
        data[('director', 'director-link', 'link')] = (graph_data['link-director'][1], graph_data['link-director'][0])

    else:
        raise ValueError(variant)

    return dgl.heterograph(data, num_nodes_dict=num_nodes)


@torch.no_grad()
def eval_split(logits, labels, idx):
    y_true = labels[idx].cpu().numpy()
    y_pred = logits[idx].argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return {
        'Accuracy': float(acc),
        'Precision': float(prec),
        'Recall': float(rec),
        'Micro-F1': float(micro_f1),
        'Macro-F1': float(macro_f1),
    }


@torch.no_grad()
def save_movie_scores(save_path, logits, labels, test_idx):
    probs = torch.softmax(logits[test_idx], dim=1).cpu().numpy()
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    df = pd.DataFrame({
        'movie_id': test_idx.cpu().numpy().astype(int),
        'pred': pred.astype(int),
        'confidence': conf.astype(float),
        'label': labels[test_idx].cpu().numpy().astype(int),
    })
    df.to_csv(save_path, index=False)


def kendall_scores_csv(path_a, path_b):
    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)
    m = a.merge(b, on='movie_id', suffixes=('_a', '_b'), how='inner')
    if len(m) < 2:
        return float('nan'), len(m)
    tau, _ = kendalltau(m['confidence_a'], m['confidence_b'], nan_policy='omit')
    return float(tau), len(m)


def run_one_variant(args, variant, seed):
    set_determinism(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Variant={variant} Seed={seed} Device={device}", flush=True)

    graph_data, meta = load_preprocessed(BASE[variant])
    num_nodes = meta['num_nodes']
    labels = meta['labels']
    train_idx = meta['train_idx']
    val_idx = meta['val_idx']
    test_idx = meta['test_idx']

    g = build_graph(graph_data, num_nodes, variant)
    hg, etypes, indexers = to_homo_with_indexers(g)

    hg = hg.to(device)
    etypes = etypes.to(device)

    idx_movie = indexers['movie'].to(device)
    labels = labels.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)

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
    clf = Classifier(args.out_dim, int(labels.max().item()) + 1).to(device)

    params = list(enc.parameters()) + list(clf.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    best_val = -1.0
    bad = 0

    t_train0 = time.perf_counter()
    for epoch in range(args.epochs):
        enc.train()
        clf.train()
        t0 = time.time()

        H = enc(hg, etypes)
        movie_H = H[idx_movie]
        logits = clf(movie_H)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 2.0)
        opt.step()

        enc.eval()
        clf.eval()
        with torch.no_grad():
            H = enc(hg, etypes)
            movie_H = H[idx_movie]
            logits = clf(movie_H)
            val_metrics = eval_split(logits, labels, val_idx)

        print(
            f"Epoch {epoch:03d} | Loss {loss.item():.4f} | "
            f"ValAcc {val_metrics['Accuracy']:.4f} | "
            f"ValMacroF1 {val_metrics['Macro-F1']:.4f} | "
            f"{time.time()-t0:.1f}s",
            flush=True
        )

        ckpt = args.ckpt.replace('.pt', f'_{variant}_seed{seed}.pt')
        if val_metrics['Macro-F1'] > best_val + 1e-6:
            best_val = val_metrics['Macro-F1']
            bad = 0
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            torch.save({'enc': enc.state_dict(), 'clf': clf.state_dict()}, ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.", flush=True)
                break

    num_epochs = epoch + 1
    train_time_sec = time.perf_counter() - t_train0

    ckpt = args.ckpt.replace('.pt', f'_{variant}_seed{seed}.pt')
    state = torch.load(ckpt, map_location=device)
    enc.load_state_dict(state['enc'])
    clf.load_state_dict(state['clf'])
    enc.eval()
    clf.eval()

    with torch.no_grad():
        H = enc(hg, etypes)
        movie_H = H[idx_movie]
        logits = clf(movie_H)
        test_metrics = eval_split(logits, labels, test_idx)

    if getattr(args, 'save_postfix', ''):
        csv_path = f"{args.save_postfix}_{variant}_seed{seed}_scores.csv"
        save_movie_scores(csv_path, logits, labels, test_idx)
        print(f"[Saved] {csv_path}", flush=True)

    print(
        f"Test Accuracy={test_metrics['Accuracy']:.6f} "
        f"Precision={test_metrics['Precision']:.6f} "
        f"Recall={test_metrics['Recall']:.6f} "
        f"Micro-F1={test_metrics['Micro-F1']:.6f} "
        f"Macro-F1={test_metrics['Macro-F1']:.6f} "
        f"| epochs={num_epochs} train_time_sec={train_time_sec:.3f}",
        flush=True
    )
    test_metrics["epochs"] = int(num_epochs)
    test_metrics["train_time_sec"] = float(train_time_sec)
    return test_metrics


def summarize(metrics_list):
    keys = ['Accuracy', 'Precision', 'Recall', 'Micro-F1', 'Macro-F1', 'epochs', 'train_time_sec']
    print("\n===== Summary over seeds (mean +/- std) =====")
    for k in keys:
        arr = np.array([m[k] for m in metrics_list], dtype=float)
        if k == "epochs":
            print(f"{k:<22}: {arr.mean():.2f} +/- {arr.std(ddof=0):.2f}")
        elif k == "train_time_sec":
            print(f"{k:<22}: {arr.mean():.3f} +/- {arr.std(ddof=0):.3f}")
        else:
            print(f"{k:<22}: {arr.mean():.6f} +/- {arr.std(ddof=0):.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="IMDB RGCN node classification on 4 raw variants")
    ap.add_argument('--variant', default=None)
    ap.add_argument('--variants', default='v1,v2,v3,v4')
    ap.add_argument('--in-dim', type=int, default=128)
    ap.add_argument('--hid-dim', type=int, default=128)
    ap.add_argument('--out-dim', type=int, default=128)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--num-bases', type=int, default=8)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-5)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--ckpt', default='checkpoint/imdb_rgcn.pt')
    ap.add_argument('--seeds', default='1566911444,20241017,20251017')
    ap.add_argument('--save-postfix', default='IMDB_rgcn')
    ap.add_argument('--compare', default='')
    ap.add_argument('--compare-only', action='store_true')
    ap.add_argument('--score-csv-a', default='')
    ap.add_argument('--score-csv-b', default='')
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]

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
        print("\n===== Summary over seeds (mean +/- std) =====")
        print(f"{'Kendall tau':<22}: {taus.mean():.6f} +/- {taus.std(ddof=0):.6f}")
        sys.exit(0)

    variants = [args.variant] if args.variant else _parse_variants(args.variants)

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
        print("\n===== Summary over seeds (mean +/- std) =====")
        print(f"{'Kendall tau':<22}: {taus.mean():.6f} +/- {taus.std(ddof=0):.6f}")