"""CMPNN link-prediction on DBLP paper-conference (var 1)."""

import os
import sys
import time
import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as torch_data
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import torchdrug

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MAGNN'))

from cmpnn.model import CMPNN
from utils.pytorchtools import EarlyStopping


# ------------------------------------------------------------------ helpers

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
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


def _read_tsv(path, names):
    return pd.read_csv(path, sep='\t', header=None, names=names)


def _map_pairs(arr, p_map, c_map):
    df = pd.DataFrame({'paper_id': arr[:, 0], 'conf_id': arr[:, 1]})
    df['paper_id'] = df['paper_id'].map(p_map)
    df['conf_id'] = df['conf_id'].map(c_map)
    df = df.dropna()
    return df.values.astype(np.int64)


def _sample_k_negs(pos_pairs, C_all, true_set, k, rng):
    neg = np.zeros((len(pos_pairs), k), dtype=np.int64)
    for i, (p, c) in enumerate(pos_pairs):
        cands = [cc for cc in C_all if (p, cc) not in true_set]
        replace = len(cands) < k
        neg[i] = rng.choice(cands, size=k, replace=replace)
    return neg


# ----------------------------------------------------------- preprocessing

def preprocess_dblp(raw_dir, shared_npz, neg_k, seed):
    """Build KG triplets and pos/neg splits entirely in memory."""
    rng = np.random.RandomState(seed)

    # ---- read raw DBLP tables ----
    al = _read_tsv(os.path.join(raw_dir, 'author_label.txt'),
                   ['author_id', 'label', 'author_name'])
    pa = _read_tsv(os.path.join(raw_dir, 'paper_author.txt'),
                   ['paper_id', 'author_id'])
    pc = _read_tsv(os.path.join(raw_dir, 'paper_conf.txt'),
                   ['paper_id', 'conf_id'])
    pt = _read_tsv(os.path.join(raw_dir, 'paper_term.txt'),
                   ['paper_id', 'term_id'])

    # ---- load shared splits ----
    sp = np.load(shared_npz)
    train_pos_raw = sp['train_pos']
    val_pos_raw   = sp['val_pos']
    test_pos_raw  = sp['test_pos']
    paper_subset  = set(sp['paper_subset'].tolist())

    # ---- filter to paper subset ----
    valid_authors = set(al['author_id'])
    pa = pa[pa['author_id'].isin(valid_authors)].reset_index(drop=True)
    pa = pa[pa['paper_id'].isin(paper_subset)].reset_index(drop=True)
    pc = pc[pc['paper_id'].isin(paper_subset)].reset_index(drop=True)
    pt = pt[pt['paper_id'].isin(paper_subset)].reset_index(drop=True)

    # ---- area_id from author labels ----
    author_label = dict(zip(al['author_id'], al['label']))

    # ---- entity ID maps ----
    authors = sorted(pa['author_id'].unique())
    papers  = sorted(paper_subset)
    terms   = sorted(pt['term_id'].unique())
    confs   = sorted(pc['conf_id'].unique())
    areas   = sorted(al['label'].unique())

    a_map = {v: i for i, v in enumerate(authors)}
    p_map = {v: i for i, v in enumerate(papers)}
    t_map = {v: i for i, v in enumerate(terms)}
    c_map = {v: i for i, v in enumerate(confs)}
    r_map = {v: i for i, v in enumerate(areas)}

    offA = 0
    offP = len(authors)
    offT = offP + len(papers)
    offC = offT + len(terms)
    offR = offC + len(confs)
    num_entity = offR + len(areas)

    print(f"Entities: A={len(authors)} P={len(papers)} T={len(terms)} "
          f"C={len(confs)} R={len(areas)}  total={num_entity}")

    # ---- build KG triplets ----
    triplets = []

    # P-A edges  (relation 0)
    for _, row in pa.iterrows():
        pid = p_map.get(row['paper_id'])
        aid = a_map.get(row['author_id'])
        if pid is not None and aid is not None:
            triplets.append([offP + pid, offA + aid, 0])

    # P-T edges  (relation 1)
    for _, row in pt.iterrows():
        pid = p_map.get(row['paper_id'])
        tid = t_map.get(row['term_id'])
        if pid is not None and tid is not None:
            triplets.append([offP + pid, offT + tid, 1])

    # P-R edges  (relation 2): paper -> area via author labels  (var1)
    paper_areas = {}
    for _, row in pa.iterrows():
        pid_raw = row['paper_id']
        aid_raw = row['author_id']
        if pid_raw in p_map and aid_raw in author_label:
            area = author_label[aid_raw]
            paper_areas.setdefault(pid_raw, set()).add(area)
    for pid_raw, area_set in paper_areas.items():
        for area in area_set:
            if area in r_map:
                triplets.append([offP + p_map[pid_raw], offR + r_map[area], 2])

    # P-C edges  (relation 3): only train_pos -- no leakage
    train_mapped = _map_pairs(train_pos_raw, p_map, c_map)
    for p_local, c_local in train_mapped:
        triplets.append([offP + p_local, offC + c_local, 3])

    triplets = np.array(triplets, dtype=np.int64)
    print(f"KG triplets: {len(triplets)}")

    # ---- map split pairs to local IDs ----
    train_pos = _map_pairs(train_pos_raw, p_map, c_map)
    val_pos   = _map_pairs(val_pos_raw,   p_map, c_map)
    test_pos  = _map_pairs(test_pos_raw,  p_map, c_map)

    # ---- negative sampling ----
    all_pos   = np.concatenate([train_pos, val_pos, test_pos], axis=0)
    true_set  = set(map(tuple, all_pos.tolist()))
    C_all     = list(range(len(confs)))

    train_neg = _sample_k_negs(train_pos, C_all, true_set, neg_k, rng)
    val_neg   = _sample_k_negs(val_pos,   C_all, true_set, neg_k, rng)
    test_neg  = _sample_k_negs(test_pos,  C_all, true_set, neg_k, rng)

    rel_pc       = 3
    num_relation = 4

    meta = {
        'num_entity':   num_entity,
        'num_relation': num_relation,
        'rel_pc':       rel_pc,
        'offA': offA, 'offP': offP, 'offT': offT, 'offC': offC, 'offR': offR,
        'num_confs':    len(confs),
        'counts': {
            'triplets':  len(triplets),
            'train_pos': len(train_pos),
            'val_pos':   len(val_pos),
            'test_pos':  len(test_pos),
        },
    }

    splits = {
        'train_pos': train_pos, 'train_neg': train_neg,
        'val_pos':   val_pos,   'val_neg':   val_neg,
        'test_pos':  test_pos,  'test_neg':  test_neg,
    }

    print(f"Splits: train={len(train_pos)}  val={len(val_pos)}  "
          f"test={len(test_pos)}  neg_k={neg_k}")
    return triplets, meta, splits


# --------------------------------------------------------------- dataset

class DBLP_PC_Query(torch_data.Dataset):
    def __init__(self, pos, neg, offP, offC):
        self.pL     = pos[:, 0]
        self.c_true = pos[:, 1]
        self.c_neg  = neg
        self.offP   = offP
        self.offC   = offC

    def __len__(self):
        return len(self.pL)

    def __getitem__(self, i):
        h      = self.offP + int(self.pL[i])
        t_true = self.offC + self.c_true[i]
        t_neg  = self.offC + self.c_neg[i]
        return h, t_true, t_neg


def collate_query(batch, rel_pc):
    h, t_true, t_neg = zip(*batch)
    h      = torch.tensor(h, dtype=torch.long)
    t_true = torch.tensor(np.array(t_true), dtype=torch.long)
    t_neg  = torch.tensor(np.array(t_neg), dtype=torch.long)
    K = t_neg.shape[1]
    h_index = h.unsqueeze(1).expand(-1, 1 + K)
    t_index = torch.cat([t_true.unsqueeze(1), t_neg], dim=1)
    r_index = torch.full_like(h_index, rel_pc)
    y = torch.zeros_like(h_index, dtype=torch.float)
    y[:, 0] = 1.0
    return h_index, t_index, r_index, y


def neg_logsigmoid_loss(scores):
    pos = scores[..., 0]
    neg = scores[..., 1:]
    return -(F.logsigmoid(pos).mean() + F.logsigmoid(-neg).mean())


def _cmpnn_forward_lp(model, graph, h_index, t_index, r_index, device):
    """Forward pass for link prediction.

    CMPNN internally masks direct edges via remove_one_hop when
    all_loss is None (eval) or not None (train).  We pass all_loss=None
    so that the graph is *not* modified (edges stay); the model still
    removes the query edge during message passing.
    """
    score = model(graph, h_index.to(device), t_index.to(device),
                  r_index.to(device), all_loss=None, metric=None)
    return score


# -------------------------------------------------------------- evaluate

def evaluate(model, graph, loader, device):
    model.eval()
    all_scores = []
    all_y = []
    with torch.no_grad():
        for h_index, t_index, r_index, y in tqdm(loader, desc='eval',
                                                   leave=False):
            score = _cmpnn_forward_lp(model, graph, h_index, t_index,
                                      r_index, device)
            all_scores.append(score.cpu())
            all_y.append(y)
    all_scores = torch.cat(all_scores, dim=0)
    all_y      = torch.cat(all_y, dim=0)

    N = all_scores.shape[0]
    hits1 = hits3 = hits5 = mrr_sum = 0.0
    for i in range(N):
        sc = all_scores[i]
        pos_score = sc[0]
        rank = int((sc >= pos_score).sum().item())
        hits1   += float(rank <= 1)
        hits3   += float(rank <= 3)
        hits5   += float(rank <= 5)
        mrr_sum += 1.0 / rank

    flat_y = all_y.numpy().flatten()
    flat_s = all_scores.numpy().flatten()
    auc = roc_auc_score(flat_y, flat_s)
    ap  = average_precision_score(flat_y, flat_s)

    return {
        'auc':   auc,
        'ap':    ap,
        'hits1': hits1 / N,
        'hits3': hits3 / N,
        'hits5': hits5 / N,
        'mrr':   mrr_sum / N,
    }


def evaluate_full(model, graph, test_pos, test_neg, offP, offC, rel_pc,
                  batch_size, device, threshold):
    """Compute ranking metrics (primary) and threshold-based binarisation.

    For each query paper we score 1 positive + K negative conferences and
    derive hits@1/3/5, MRR, AUC, AP.  A sigmoid threshold is applied for
    precision / recall / F1 / accuracy.
    """
    model.eval()
    N = len(test_pos)
    K = test_neg.shape[1]

    all_scores = []
    all_y = []

    with torch.no_grad():
        for start in tqdm(range(0, N, batch_size), desc='test', leave=False):
            end = min(start + batch_size, N)
            pos_batch = test_pos[start:end]
            neg_batch = test_neg[start:end]

            B = end - start
            h      = torch.tensor(offP + pos_batch[:, 0], dtype=torch.long)
            t_true = torch.tensor(offC + pos_batch[:, 1], dtype=torch.long)
            t_neg  = torch.tensor(offC + neg_batch,       dtype=torch.long)

            h_index = h.unsqueeze(1).expand(-1, 1 + K)
            t_index = torch.cat([t_true.unsqueeze(1), t_neg], dim=1)
            r_index = torch.full_like(h_index, rel_pc)

            score = _cmpnn_forward_lp(model, graph, h_index, t_index,
                                      r_index, device)
            all_scores.append(score.cpu())

            y = torch.zeros(B, 1 + K, dtype=torch.float)
            y[:, 0] = 1.0
            all_y.append(y)

    all_scores = torch.cat(all_scores, dim=0)
    all_y      = torch.cat(all_y, dim=0)

    # ---- ranking metrics per query ----
    hits1 = hits3 = hits5 = mrr_sum = 0.0
    ranks = []
    for i in range(N):
        sc = all_scores[i]
        pos_score = sc[0]
        rank = int((sc >= pos_score).sum().item())
        ranks.append(rank)
        hits1   += float(rank <= 1)
        hits3   += float(rank <= 3)
        hits5   += float(rank <= 5)
        mrr_sum += 1.0 / rank

    # ---- global AUC / AP ----
    flat_y = all_y.numpy().flatten()
    flat_s = all_scores.numpy().flatten()
    auc = roc_auc_score(flat_y, flat_s)
    ap  = average_precision_score(flat_y, flat_s)

    # ---- threshold-based metrics ----
    probs  = torch.sigmoid(all_scores)
    preds  = (probs >= threshold).float()
    flat_p = preds.numpy().flatten()

    tp = float(((flat_p == 1) & (flat_y == 1)).sum())
    fp = float(((flat_p == 1) & (flat_y == 0)).sum())
    fn = float(((flat_p == 0) & (flat_y == 1)).sum())
    tn = float(((flat_p == 0) & (flat_y == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2.0 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = ((tp + tn) / (tp + fp + fn + tn)
                 if (tp + fp + fn + tn) > 0 else 0.0)

    # ---- scores DataFrame for CSV output ----
    rows = []
    for i in range(N):
        p_local  = int(test_pos[i, 0])
        c_true   = int(test_pos[i, 1])
        sc       = all_scores[i].numpy()
        prob     = probs[i].numpy()
        pred_bin = preds[i].numpy()

        score_pos = float(sc[0])
        prob_pos  = float(prob[0])
        pred_pos  = int(pred_bin[0])
        rank_i    = ranks[i]

        neg_confs  = test_neg[i].tolist()
        neg_scores = sc[1:].tolist()

        rows.append({
            'paper':      p_local,
            'conf_true':  c_true,
            'score_pos':  score_pos,
            'prob_pos':   prob_pos,
            'pred_pos':   pred_pos,
            'rank':       rank_i,
            'hit1':       int(rank_i <= 1),
            'hit3':       int(rank_i <= 3),
            'hit5':       int(rank_i <= 5),
            'neg_confs':  neg_confs,
            'neg_scores': neg_scores,
        })

    scores_df = pd.DataFrame(rows)

    return {
        'auc':       auc,
        'ap':        ap,
        'hits1':     hits1 / N,
        'hits3':     hits3 / N,
        'hits5':     hits5 / N,
        'mrr':       mrr_sum / N,
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'accuracy':  accuracy,
        'scores_df': scores_df,
    }


# --------------------------------------------------------- train / eval

def run_model_DBLP_pc(raw_dir, shared_npz, neg_k, input_dim, hidden_dim,
                      num_layers, num_epochs, patience, batch_size, threshold,
                      save_postfix, base_seed, gpu, use_cpu, eval_only,
                      checkpoint_path):
    set_seed(base_seed)

    # ---- device ----
    if use_cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{gpu}')
    print(f"Device: {device}")

    # ---- data ----
    triplets, meta, splits = preprocess_dblp(raw_dir, shared_npz, neg_k,
                                             base_seed)

    edge_list = torch.tensor(triplets, dtype=torch.long)
    graph = torchdrug.data.Graph(edge_list,
                                 num_node=meta['num_entity'],
                                 num_relation=meta['num_relation'])
    graph = graph.to(device)

    offP   = meta['offP']
    offC   = meta['offC']
    rel_pc = meta['rel_pc']

    # ---- dataloaders ----
    train_ds = DBLP_PC_Query(splits['train_pos'], splits['train_neg'],
                             offP, offC)
    val_ds   = DBLP_PC_Query(splits['val_pos'], splits['val_neg'],
                             offP, offC)

    collate_fn = lambda b: collate_query(b, rel_pc=meta['rel_pc'])

    train_loader = torch_data.DataLoader(train_ds, batch_size=batch_size,
                                         shuffle=True,
                                         collate_fn=collate_fn)
    val_loader   = torch_data.DataLoader(val_ds, batch_size=batch_size,
                                         shuffle=False,
                                         collate_fn=collate_fn)

    # ---- model ----
    model = CMPNN(
        input_dim=input_dim,
        hidden_dims=[hidden_dim] * num_layers,
        message_func='distmult',
        aggregate_func='pna',
        short_cut=True,
        layer_norm=True,
        dependent=False,
        set_boundary=True,
        remove_one_hop=True,
        activation='relu',
        initialization='Query',
    )
    model = model.to(device)

    os.makedirs('checkpoint', exist_ok=True)
    ckpt_path = (checkpoint_path or
                 f"checkpoint/checkpoint_{save_postfix}_seed{base_seed}.pt")

    if eval_only:
        print(f"Loading checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        early_stopping = EarlyStopping(patience=patience, verbose=True,
                                       save_path=ckpt_path)

        for epoch in range(num_epochs):
            t0 = time.time()
            model.train()
            epoch_loss = 0.0
            for h_index, t_index, r_index, y in tqdm(train_loader,
                                                      desc=f'train e{epoch}',
                                                      leave=False):
                score = _cmpnn_forward_lp(model, graph, h_index, t_index,
                                          r_index, device)
                loss = neg_logsigmoid_loss(score)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(train_loader), 1)

            val_metrics = evaluate(model, graph, val_loader, device)
            val_ap  = val_metrics['ap']
            val_auc = val_metrics['auc']
            elapsed = time.time() - t0

            print(f"Epoch {epoch:03d} | loss {avg_loss:.4f} | "
                  f"val_auc {val_auc:.4f}  val_ap {val_ap:.4f} | "
                  f"hits@1 {val_metrics['hits1']:.4f}  "
                  f"mrr {val_metrics['mrr']:.4f} | "
                  f"time {elapsed:.1f}s")

            early_stopping(-val_ap, model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # ---- test ----
    test_results = evaluate_full(
        model, graph,
        splits['test_pos'], splits['test_neg'],
        offP, offC, rel_pc,
        batch_size, device, threshold,
    )

    print("\n=== Test Results ===")
    print(f"AUC:       {test_results['auc']:.4f}")
    print(f"AP:        {test_results['ap']:.4f}")
    print(f"Hits@1:    {test_results['hits1']:.4f}")
    print(f"Hits@3:    {test_results['hits3']:.4f}")
    print(f"Hits@5:    {test_results['hits5']:.4f}")
    print(f"MRR:       {test_results['mrr']:.4f}")
    print(f"Precision: {test_results['precision']:.4f}")
    print(f"Recall:    {test_results['recall']:.4f}")
    print(f"F1:        {test_results['f1']:.4f}")
    print(f"Accuracy:  {test_results['accuracy']:.4f}")

    csv_path = f"{save_postfix}_seed{base_seed}_scores.csv"
    test_results['scores_df'].to_csv(csv_path, index=False)
    print(f"Scores saved to {csv_path}")

    return {k: v for k, v in test_results.items() if k != 'scores_df'}


# ----------------------------------------------------------------- main

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='CMPNN-LP on DBLP paper-conf')
    ap.add_argument('--raw-dir', type=str,
                    default=os.path.join(os.path.dirname(__file__),
                                         '..', 'MAGNN', 'data', 'raw',
                                         'DBLP'))
    ap.add_argument('--shared-npz', type=str, required=True)
    ap.add_argument('--input-dim', type=int, default=32)
    ap.add_argument('--hidden-dim', type=int, default=32)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--epoch', type=int, default=20)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--neg-k', type=int, default=19)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--seeds', type=str,
                    default='1566911444,20241017,20251017')
    ap.add_argument('--save-postfix', type=str, default='DBLP_cmpnn_pc')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--checkpoint', type=str, default=None)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    all_results = []

    for seed in seeds:
        print(f"\n{'=' * 60}")
        print(f"Seed: {seed}")
        print(f"{'=' * 60}")
        result = run_model_DBLP_pc(
            raw_dir=args.raw_dir,
            shared_npz=args.shared_npz,
            neg_k=args.neg_k,
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.layers,
            num_epochs=args.epoch,
            patience=args.patience,
            batch_size=args.batch_size,
            threshold=args.threshold,
            save_postfix=args.save_postfix,
            base_seed=seed,
            gpu=args.gpu,
            use_cpu=args.cpu,
            eval_only=args.eval_only,
            checkpoint_path=args.checkpoint,
        )
        all_results.append(result)

    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("Summary across seeds")
        print(f"{'=' * 60}")
        for key in ['auc', 'ap', 'hits1', 'hits3', 'hits5', 'mrr',
                     'precision', 'recall', 'f1', 'accuracy']:
            vals = [r[key] for r in all_results]
            m, s = np.mean(vals), np.std(vals)
            print(f"{key:12s}: {m:.4f} +/- {s:.4f}")
