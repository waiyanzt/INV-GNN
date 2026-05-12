# RGCN link prediction on DBLP paper–venue
# VAR-1: area connected to Paper
import os, argparse, time, random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import RelGraphConv
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, accuracy_score
)

from rgcn_dblp_pc_utils import subsample_negs_per_paper

SEED = 1566911444

def set_determinism(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try: torch.use_deterministic_algorithms(True)
    except Exception: pass

def load_tables(raw_dir):
    al = pd.read_csv(os.path.join(raw_dir,'author_label.txt'), sep='\t',
                     names=['author_id','label','author_name'],
                     header=None, encoding='utf-8')
    pa = pd.read_csv(os.path.join(raw_dir,'paper_author.txt'), sep='\t',
                     names=['paper_id','author_id'], header=None, encoding='utf-8')
    pc = pd.read_csv(os.path.join(raw_dir,'paper_conf.txt'), sep='\t',
                     names=['paper_id','conf_id'], header=None, encoding='utf-8')
    pt = pd.read_csv(os.path.join(raw_dir,'paper_term.txt'), sep='\t',
                     names=['paper_id','term_id'], header=None, encoding='utf-8')
    return al, pa, pc, pt

def filter_and_maps_var1(al, pa, pc, pt, paper_subset=None, min_conf=0):
    # keep only authors that have labels
    valid_authors = set(al['author_id'])
    pa = pa[pa['author_id'].isin(valid_authors)].reset_index(drop=True)

    # keep only papers that appear in PA
    valid_papers = set(pa['paper_id'])
    pc = pc[pc['paper_id'].isin(valid_papers)].reset_index(drop=True)
    pt = pt[pt['paper_id'].isin(valid_papers)].reset_index(drop=True)
    pa = pa[pa['paper_id'].isin(valid_papers)].reset_index(drop=True)

    # filter very small conferences if requested
    if min_conf > 0:
        big = pc['conf_id'].value_counts().loc[lambda x: x>=min_conf].index
        pc = pc[pc['conf_id'].isin(big)].reset_index(drop=True)
        valid_papers = set(pc['paper_id'])
        pt = pt[pt['paper_id'].isin(valid_papers)].reset_index(drop=True)
        pa = pa[pa['paper_id'].isin(valid_papers)].reset_index(drop=True)

    # restrict to subset of papers (if shared splits specify it)
    if paper_subset is not None:
        keep = set(paper_subset) & set(pc['paper_id'])
        pc = pc[pc['paper_id'].isin(keep)].reset_index(drop=True)
        pa = pa[pa['paper_id'].isin(keep)].reset_index(drop=True)
        pt = pt[pt['paper_id'].isin(keep)].reset_index(drop=True)

    # paper -> area (via author labels)
    a2r = al.set_index('author_id')['label']
    pr = (pa.assign(area_id=pa['author_id'].map(a2r))
            [['paper_id','area_id']].drop_duplicates().dropna().reset_index(drop=True))

    # id universes
    Au_ids = sorted(pa['author_id'].unique())
    Pa_ids = sorted(pc['paper_id'].unique())
    Te_ids = sorted(pt['term_id'].unique())
    Co_ids = sorted(pc['conf_id'].unique())
    Ar_ids = sorted(pr['area_id'].unique())

    Au_map = {a:i for i,a in enumerate(Au_ids)}
    Pa_map = {p:i for i,p in enumerate(Pa_ids)}
    Te_map = {t:i for i,t in enumerate(Te_ids)}
    Co_map = {c:i for i,c in enumerate(Co_ids)}
    Ar_map = {r:i for i,r in enumerate(Ar_ids)}
    return (pa, pc, pt, pr), (Au_map, Pa_map, Te_map, Co_map, Ar_map)

def map_pairs_npz(shared_npz, Pa_map, Co_map):
    Z = np.load(shared_npz)
    def M(X):
        df = pd.DataFrame(X, columns=['paper_id','conf_id'])
        df['paper_id'] = df['paper_id'].map(Pa_map)
        df['conf_id']  = df['conf_id'].map(Co_map)
        return df.dropna().to_numpy(dtype=np.int64)
    out = {
        'train_pos': M(Z['train_pos']),
        'val_pos'  : M(Z['val_pos']),
        'test_pos' : M(Z['test_pos']),
        'train_neg': M(Z['train_neg']),
        'val_neg'  : M(Z['val_neg']),
        'test_neg' : M(Z['test_neg']),
    }
    out['all_pos'] = np.vstack([out['train_pos'], out['val_pos'], out['test_pos']])
    return out, (Z['paper_subset'].tolist() if 'paper_subset' in Z.files else None)

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
    # Paper → rank conferences
    from collections import defaultdict
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

def run_one(args, seed):
    set_determinism(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Seed {seed} | Device: {device}")

    Z = np.load(args.shared_npz)
    paper_subset = Z['paper_subset'].tolist() if 'paper_subset' in Z.files else None

    al, pa, pc, pt = load_tables(args.raw_dir)
    (pa_m, pc_m, pt_m, pr_m), maps = filter_and_maps_var1(
        al, pa, pc, pt, paper_subset=paper_subset, min_conf=args.min_conf
    )
    Au_map, Pa_map, Te_map, Co_map, Ar_map = maps
    S, _ = map_pairs_npz(args.shared_npz, Pa_map, Co_map)

    if getattr(args, "neg_per_paper", 0) and args.neg_per_paper > 0:
        k = int(args.neg_per_paper)
        S["train_neg"] = subsample_negs_per_paper(
            S["train_neg"], k, np.random.RandomState(int(seed) + 11)
        )
        S["val_neg"] = subsample_negs_per_paper(
            S["val_neg"], k, np.random.RandomState(int(seed) + 13)
        )
        S["test_neg"] = subsample_negs_per_paper(
            S["test_neg"], k, np.random.RandomState(int(seed) + 17)
        )
        print(f"Subsampled negatives to max {k} per paper (train/val/test), run seed={seed}")

    print(f"Splits: train_pos={len(S['train_pos'])}  val_pos={len(S['val_pos'])}  test_pos={len(S['test_pos'])}")
    print(f"        train_neg={len(S['train_neg'])}  val_neg={len(S['val_neg'])}  test_neg={len(S['test_neg'])}")

    def m(df, col, mp): return df[col].map(mp).to_numpy(dtype=np.int64)
    A2P = (m(pa_m,'author_id',Au_map), m(pa_m,'paper_id',Pa_map))
    P2R = (m(pr_m,'paper_id',Pa_map),  m(pr_m,'area_id',Ar_map))
    P2C_all = (m(pc_m,'paper_id',Pa_map),  m(pc_m,'conf_id',Co_map))
    P2T = (m(pt_m,'paper_id',Pa_map),  m(pt_m,'term_id',Te_map))

    data = {
        ('author','author-paper','paper'): A2P,
        ('paper','paper-author','author'): (A2P[1], A2P[0]),
        ('paper','paper-area','area'): P2R,
        ('area','area-paper','paper'): (P2R[1], P2R[0]),
        ('paper','paper-conference','conference'): P2C_all,
        ('conference','conference-paper','paper'): (P2C_all[1], P2C_all[0]),
        ('paper','paper-term','term'): P2T,
        ('term','term-paper','paper'): (P2T[1], P2T[0]),
    }

    num_nodes = {
        'author': len(Au_map), 'paper': len(Pa_map), 'term': len(Te_map),
        'conference': len(Co_map), 'area': len(Ar_map)
    }
    g = dgl.heterograph(data, num_nodes_dict=num_nodes)
    hg, etypes, indexers = to_homo_with_indexers(g)
    N_total = hg.num_nodes(); R_total = int(etypes.max().item()) + 1
    print(f"Graph: N={N_total}  E={hg.num_edges()}  node types={g.ntypes}  etypes={g.etypes}")

    hg = hg.to(device); etypes = etypes.to(device)
    idxP = indexers['paper'].to(device)
    idxC = indexers['conference'].to(device)

    enc = RGCNEncoder(N_total, R_total,
                      in_dim=args.in_dim, hid_dim=args.hid_dim, out_dim=args.out_dim,
                      num_layers=args.layers, num_bases=args.num_bases, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Per-paper negatives (same protocol as var2/var3): each pos row uses that paper's train negs
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

    for epoch in range(args.epochs):
        enc.train()
        perm = np.random.permutation(len(train_pos))
        running = 0.0; t0 = time.time()
        pbar = tqdm(range(iters), desc=f"Epoch {epoch:03d}", leave=False)

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

        print(f"Epoch {epoch:03d} | TrainLoss {running/max(1,iters):.4f} | ValLoss {vloss:.4f} | {time.time()-t0:.1f}s")
        ckpt = args.ckpt.replace('.pt', f'_seed{seed}.pt')
        if vloss < best_val - 1e-6:
            best_val = vloss; bad = 0
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            torch.save(enc.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    # Test
    ckpt = args.ckpt.replace('.pt', f'_seed{seed}.pt')
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

    # save per-pair scores CSV (for Kendall τ or audits)
    if getattr(args, 'save_postfix', ''):
        pairs_all = np.vstack([S['test_pos'], S['test_neg']])
        scores_all = np.concatenate([pp, pn])
        df_out = pd.DataFrame({
            'paper_id': pairs_all[:,0].astype(int),
            'conf_id':  pairs_all[:,1].astype(int),
            'score':    scores_all.astype(float)
        })
        csv_path = f"{args.save_postfix}_seed{seed}_scores.csv"
        df_out.to_csv(csv_path, index=False)
        print(f"[Saved] Per-pair scores: {csv_path}")

    print("=== RGCN (RelGraphConv) VAR-1 (terms→venue from TRAIN, area→paper; P–C all observed): Test ===")
    auc = roc_auc_score(y_true, y_prob)
    ap  = average_precision_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    acc  = accuracy_score(y_true, y_pred)
    print(f"AUC={auc:.6f}  AP={ap:.6f}")
    print(f"Chosen threshold (fixed) = {th:.3f}")
    print(f"Confusion matrix : TP={TP}  TN={TN}  FP={FP}  FN={FN} (P={TP+FN}, N={TN+FP})")
    print(f"Precision={prec:.6f}  Recall={rec:.6f}  F1={f1:.6f}  Acc={acc:.6f}")

    H1,H3,H5,MRR,n = eval_ranking_pc(S['test_pos'], S['test_neg'], P, C)
    n_neg_per = len(S["test_neg"]) // max(1, len(S["test_pos"]))
    print(f"Ranking (paper→confs; each query 1 true + ~{n_neg_per} neg from split) over {n} papers:")
    print(f"Hits@1={H1:.6f}  Hits@3={H3:.6f}  Hits@5={H5:.6f}  MRR={MRR:.6f}")

    return {
        "AUC": float(auc), "AP": float(ap), "Precision": float(prec),
        "Recall": float(rec), "F1": float(f1), "Accuracy": float(acc),
        "Hits@1": float(H1), "Hits@3": float(H3), "Hits@5": float(H5), "MRR": float(MRR)
    }

def summarize(metrics_list):
    keys = ['AUC','AP','Precision','Recall','F1','Accuracy','Hits@1','Hits@3','Hits@5','MRR']
    means, stds = {}, {}
    for k in keys:
        arr = np.array([m[k] for m in metrics_list], dtype=float)
        means[k] = arr.mean()
        stds[k]  = arr.std(ddof=0)
    print("\n===== Summary over seeds =====")
    for k in keys:
        print(f"{k:<9}: {means[k]:.6f} ± {stds[k]:.6f}")

def run_three_seeds(args):
    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    all_metrics = []
    for sd in seeds:
        all_metrics.append(run_one(args, sd))
    summarize(all_metrics)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RGCN LP on DBLP VAR-1 (area→paper; terms→venue from TRAIN)")
    ap.add_argument('--raw-dir', default='data/raw/DBLP/')
    ap.add_argument('--shared-npz', dest='shared_npz',
                    default='data/preprocessed/DBLP_shared_splits/DBLP_pc_shared_splits.npz')
    ap.add_argument('--min-conf', type=int, default=0)

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
        '--neg-per-paper', type=int, default=19,
        help='Max negative (paper,conf) pairs per paper in train/val/test after loading npz '
             '(matches small-K LP setups). Use 0 for full lists in the npz (often |C|-1).',
    )
    ap.add_argument('--th', type=float, default=0.5)

    ap.add_argument('--ckpt', default='checkpoint/rgcn_var1.pt')
    ap.add_argument('--seeds', default='1566911444,1566911445,1566911446',
                    help='Comma-separated seeds; negatives stay fixed order across seeds')
    ap.add_argument('--save-postfix', default='DBLP_pv_rgcn_var1')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    run_three_seeds(args)
